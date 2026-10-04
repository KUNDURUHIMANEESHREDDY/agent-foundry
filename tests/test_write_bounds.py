"""The write cap and the token ceiling must stay ordered.

`filesystem.write` caps a single call's blast radius. `max_tokens` is a
cumulative budget for a whole run. For both to mean anything the inner bound has
to sit below the outer — and it did not. Measured at the old 512KB:

    write cap          512KB  ~131,000 tokens of payload
    coder's budget      60K   ~240,000 bytes, overhead aside

so the cap could not fire at any budget a shipping spec grants. A limit that
never fires is decoration, and an untested one rots.

These tests assert the ordering for every spec that can actually write, so
raising either number in isolation is caught.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from factory.capabilities.fs_write import MAX_WRITE_BYTES, FilesystemWrite
from factory.models.base import CHARS_PER_TOKEN, Message, ToolCall
from factory.models.fake import ScriptedAdapter
from factory.runtime.context import estimate_request_tokens
from factory.spec.loader import load_spec

AGENTS = Path(__file__).resolve().parents[1] / "agents"


def payload_budget_bytes(spec) -> int:
    """How many bytes of tool-argument payload this spec's budget allows.

    Measured the way the runtime charges it: the whole request estimated, minus
    everything that is not payload. Deliberately an over-estimate of the headroom
    available to any one call, so the ordering assertion is conservative.
    """
    overhead = estimate_request_tokens(
        [
            Message(role="system", content=spec.system_prompt),
            Message(role="user", content="write something"),
        ],
        [
            {
                "name": "filesystem.write",
                "description": "Write a UTF-8 text file in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ],
    )
    return max(0, (spec.limits.max_tokens - overhead) * CHARS_PER_TOKEN)


WRITABLE = [
    p for p in sorted(AGENTS.glob("*.yaml"))
    if "filesystem.write" in load_spec(p).capabilities
]


class TestThereIsSomethingToTest:
    def test_at_least_one_shipped_spec_can_write(self):
        assert WRITABLE, "no shipped spec requests filesystem.write"

    def test_the_oversized_write_case_exists(self):
        """The eval suite must actually exercise the cap."""
        from factory.eval.runner import load_all

        cases = {
            c.id
            for suite in load_all(AGENTS.parent / "evals")
            for c in suite.cases
        }
        assert "rejects-oversized-write" in cases


class TestBoundsAreOrdered:
    @pytest.mark.parametrize("path", WRITABLE, ids=lambda p: p.stem)
    def test_the_write_cap_is_reachable_for_this_spec(self, path):
        """The invariant that was violated.

        If the cap sat above the budget, no run of this spec could ever reach it
        and the capability's own limit would never fire.
        """
        spec = load_spec(path)
        budget = payload_budget_bytes(spec)

        assert MAX_WRITE_BYTES < budget, (
            f"{spec.name}: write cap {MAX_WRITE_BYTES:,}B is above the "
            f"{budget:,}B its budget allows, so it can never fire. Lower the "
            f"cap or raise max_tokens — but decide deliberately."
        )

    def test_a_payload_between_the_two_bounds_hits_the_capability(self, tmp_path):
        """Above the cap, below the budget: the capability must refuse.

        This is the window `rejects-oversized-write` uses, and the only place
        the cap is distinguishable from the budget.
        """
        spec = load_spec(AGENTS / "coder.yaml")
        budget = payload_budget_bytes(spec)
        between = (MAX_WRITE_BYTES + budget) // 2

        assert MAX_WRITE_BYTES < between < budget, "test setup: no usable window"

        writer = FilesystemWrite(root=tmp_path)
        result = writer.invoke(path="big.txt", content="x" * between)

        assert result["ok"] is False
        assert "over" in result["error"].lower()

    def test_the_cap_error_names_the_limit(self, tmp_path):
        writer = FilesystemWrite(root=tmp_path)
        result = writer.invoke(path="big.txt", content="x" * (MAX_WRITE_BYTES + 1))

        assert result["ok"] is False
        assert str(MAX_WRITE_BYTES) in result["error"]

    def test_a_payload_under_the_cap_is_written(self, tmp_path):
        writer = FilesystemWrite(root=tmp_path)
        content = "x" * (MAX_WRITE_BYTES - 1024)

        result = writer.invoke(path="ok.txt", content=content)

        assert result["ok"] is True
        assert result["bytes_written"] == len(content)


class TestAppendIsNotAWayAroundIt:
    """The cap is on the resulting file, so `append` cannot exceed it.

    Worth asserting rather than assuming: it is tempting to read `append` as an
    escape hatch for legitimately large files. It is not — the check is on the
    post-append size.
    """

    def test_append_within_the_limit_succeeds(self, tmp_path):
        writer = FilesystemWrite(root=tmp_path)
        half = "x" * (MAX_WRITE_BYTES // 2 - 16)

        first = writer.invoke(path="big.txt", content=half)
        second = writer.invoke(path="big.txt", content=half, append=True)

        assert first["ok"] is True
        assert second["ok"] is True
        assert (tmp_path / "big.txt").stat().st_size == 2 * len(half)

    def test_append_past_the_limit_is_refused(self, tmp_path):
        writer = FilesystemWrite(root=tmp_path)

        assert writer.invoke(path="big.txt", content="x" * MAX_WRITE_BYTES)["ok"] is True

        result = writer.invoke(path="big.txt", content="more", append=True)

        assert result["ok"] is False
        assert "append would exceed" in result["error"]

    def test_the_file_is_left_untouched_by_a_refused_append(self, tmp_path):
        writer = FilesystemWrite(root=tmp_path)
        original = "x" * MAX_WRITE_BYTES
        writer.invoke(path="big.txt", content=original)

        writer.invoke(path="big.txt", content="more", append=True)

        assert (tmp_path / "big.txt").read_text(encoding="utf-8") == original


class TestTheOldCapWouldHaveFailedThis:
    """Guard against a regression back to an unreachable cap."""

    def test_512kb_was_never_reachable(self):
        old_cap = 512 * 1024
        spec = load_spec(AGENTS / "coder.yaml")

        assert old_cap > payload_budget_bytes(spec), (
            "if a 512KB cap is now reachable, this test no longer describes the "
            "original problem and the ordering note in MAX_WRITE_BYTES is stale"
        )

    def test_the_cap_is_not_worthless(self):
        """A cap of zero, or one below a single character, is not a limit."""
        assert MAX_WRITE_BYTES > 1024