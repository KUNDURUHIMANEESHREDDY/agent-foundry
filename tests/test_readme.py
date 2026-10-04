"""Every relative markdown link must resolve.

A dangling anchor is worse than no link: it reads as documentation and points
nowhere. The README is edited often and each edit adds cross-references, so the
check is mechanical.

External URLs are not fetched — that belongs in a link checker, and a test that
depends on the network fails for reasons unrelated to the code.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


def slugify(heading: str) -> str:
    """Approximate GitHub's heading slugs: lowercase, drop punctuation, hyphenate."""
    s = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return re.sub(r"\s+", "-", s)


def headings() -> set[str]:
    out: set[str] = set()
    in_fence = False
    for line in README.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            # A `#` inside a fenced block is code, not a heading.
            continue
        m = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if m:
            out.add(slugify(m.group(1)))
    return out


def anchors() -> list[str]:
    return re.findall(
        r"\[[^\]]+\]\(#([^)]+)\)", README.read_text(encoding="utf-8")
    )


class TestReadmeLinks:
    def test_readme_exists(self):
        assert README.is_file()

    def test_it_has_internal_links_to_check(self):
        """Otherwise the tests below pass vacuously."""
        assert len(anchors()) >= 5

    @pytest.mark.parametrize("anchor", sorted(set(anchors())))
    def test_anchor_resolves(self, anchor):
        assert anchor in headings(), f"#{anchor} has no heading"

    def test_no_duplicate_heading_slugs(self):
        """A duplicate slug makes GitHub append -1, silently breaking one link."""
        text = README.read_text(encoding="utf-8")
        in_fence = False
        seen: dict[str, int] = {}
        for line in text.splitlines():
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
            if m:
                slug = slugify(m.group(1))
                seen[slug] = seen.get(slug, 0) + 1

        duplicates = {s: n for s, n in seen.items() if n > 1}
        assert not duplicates, f"duplicate heading slugs: {duplicates}"


def relative_file_links() -> list[str]:
    """Markdown links to a sibling file, e.g. `[LICENSE](LICENSE)`.

    Not anchors and not URLs -- a path relative to the README. These break
    silently: the renderer shows the link text and nothing else.
    """
    text = README.read_text(encoding="utf-8")
    out = []
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        out.append(target)
    return out


class TestReadmeFileLinks:
    def test_it_has_a_relative_file_link_to_check(self):
        """Otherwise the tests below pass vacuously."""
        assert relative_file_links()

    @pytest.mark.parametrize("target", sorted(set(relative_file_links())))
    def test_file_exists(self, target):
        path = (README.parent / target.split("#", 1)[0]).resolve()
        assert path.is_file(), f"{target} does not exist"


class TestReadmeMentionsItsLicence:
    """The one licence concern that belongs here: the README points at it.

    Whether the licence *file* exists, and whether it agrees with pyproject, is
    `tests/test_packaging.py` -- packaging metadata is not a README concern, and
    burying it in this file meant nobody auditing packaging would think to look
    here for it.
    """

    def test_readme_has_a_licence_section(self):
        text = README.read_text(encoding="utf-8")
        assert re.search(r"(?im)^#+ .*licen[sc]e", text), "README has no licence section"


def status_block() -> str:
    """The `## Status` section, which is where the project's claims live."""
    text = README.read_text(encoding="utf-8")
    start = text.index("## Status")
    rest = text[start + len("## Status"):]
    end = rest.find("\n## ")
    return rest[:end] if end != -1 else rest


class TestStatusBlockCountsAreDerivedNotTyped:
    """The counts in `## Status` are claims, and typed claims rot.

    This file once said 551 tests passing. By the time it was read it was 837,
    because every test added since went unrecorded. An external review quoted
    the stale number back, which is how it was noticed.

    So the numbers are derived here instead. The README is the only place a
    reader looks for them, and a test is the only thing that cannot forget.
    """

    def test_the_stated_test_count_is_current(self):
        collected = _collected_test_count()
        stated = _stated_count(status_block(), r"(\d[\d,]*)\s+tests collected")
        assert stated is not None, "Status block states no test count"
        assert int(stated) == collected, (
            f"Status block claims {stated} tests; the suite actually collects "
            f"{collected}. Update the README or the suite."
        )

    def test_the_stated_eval_case_count_is_current(self):
        from factory.eval.runner import load_all

        root = README.parent
        cases = sum(len(s.cases) for s in load_all(root / "evals"))
        stated = _stated_count(status_block(), r"(\d+)\s+eval cases")
        assert stated is not None, "Status block states no eval case count"
        assert int(stated) == cases, f"Status block claims {stated} eval cases; there are {cases}"

    def test_the_stated_sabotage_count_is_current(self):
        from factory.eval.sabotage import SABOTAGES

        stated = _stated_count(status_block(), r"sabotage`?\*\*?:?\s*(\d+) mitigations")
        assert stated is not None, "Status block states no sabotage count"
        assert int(stated) == len(SABOTAGES), (
            f"Status block claims {stated} mitigations; the catalogue has "
            f"{len(SABOTAGES)}"
        )

    def test_the_eval_suite_count_is_current(self):
        from factory.eval.runner import load_all

        suites = len(load_all(README.parent / "evals"))
        stated = _stated_count(status_block(), r"across\s+(\d+)\s+suites")
        assert stated is not None, "Status block states no suite count"
        assert int(stated) == suites, f"Status block claims {stated} suites; there are {suites}"


class TestFindingsAccountingIsExact:
    """The defect an external review actually caught: 'twelve' over a table of
    thirteen. Audit accounting has to be right to be worth anything."""

    def _table_rows(self) -> int:
        text = README.read_text(encoding="utf-8")
        start = text.index("## Audit findings")
        rows = re.findall(r"(?m)^\|\s*\d+\s*\|", text[start:])
        return len(rows)

    def test_the_table_is_not_empty(self):
        assert self._table_rows() >= 10

    def test_the_prose_count_matches_the_table(self):
        text = README.read_text(encoding="utf-8")
        start = text.index("## Audit findings")
        section = text[start:]
        prose = re.search(r"produced (twelve|thirteen|\d+) findings", section)
        assert prose, "Audit findings section states no count in prose"

        word = prose.group(1)
        table = self._table_rows()

        if word.isdigit():
            assert int(word) == table
        else:
            numbers = {
                "twelve": 12, "thirteen": 13, "eleven": 11, "fourteen": 14,
            }
            assert numbers[word] == table, (
                f"prose says '{word}' ({numbers[word]}) but the table lists "
                f"{table} findings"
            )

    def test_the_status_summary_agrees(self):
        """`## Status` also says how many are resolved."""
        block = status_block()
        stated = _stated_count(block, r"all (twelve|thirteen|\d+) are")
        assert stated is not None, "Status block states no resolved count"
        word = stated
        table = self._table_rows()
        if not word.isdigit():
            word = {"twelve": 12, "thirteen": 13}[word]
        assert int(word) == table


class TestStatusBlockDoesNotClaimWhatIsUnverified:
    """The worst defect found in the Status block, which the review did not spot.

    It claimed `Langfuse verified end-to-end against cloud.langfuse.com` while
    the Audit findings section said the live round trip was untested, and no
    credentials exist in the environment to have made it true.

    This is a necessary condition, not a sufficient one: having credentials
    present does not mean `verify_langfuse_live.py` was ever run. What it does
    stop is the specific failure of a claim standing with nothing behind it.
    """

    LANGFUSE_LINE = re.compile(
        r"(?im)^.*langfuse verified.*$", re.M
    )

    def test_no_live_langfuse_claim_without_credentials(self):
        claim = self.LANGFUSE_LINE.search(status_block())
        if claim is None:
            return  # nothing claimed; nothing to check

        have_keys = bool(
            os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
        )
        assert have_keys, (
            f"Status block claims Langfuse was verified live, but "
            f"LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are not set, so it cannot "
            f"have been: {claim.group(0).strip()}"
        )

    def test_the_status_block_points_at_the_unverified_section(self):
        """The honest counterpart: the block must say what is not verified."""
        block = status_block().lower()
        assert "not** verified" in block or "not verified" in block, (
            "Status block lists what works but never says what does not"
        )

    def test_python_execute_is_not_described_as_a_sandbox(self):
        """
        `python.execute` has subprocess hardening and no containment. Calling it
        a sandbox in the summary contradicts the rest of the project, which
        asserts in tests and eval cases that it is not one.
        """
        block = status_block().lower()
        assert "not* a sandbox" in block or "not a sandbox" in block, (
            "Status block describes python.execute without saying it is not a sandbox"
        )
        assert "subprocess sandbox" not in block, (
            "Status block calls python.execute a sandbox, which it is not"
        )


class TestTheDeploymentPostureIsStated:
    """An external review's central objection: this must not be *marketed* as a
    sandbox. The honesty was scattered across the README, so a reader had to
    assemble it. It is now one table, and it has to stay a table.

    The failure these guard against is a document that lists only what works.
    That was the actual defect in the Status block: a list of strengths with the
    qualifications forty lines below and not in the list at all.
    """

    def _section(self) -> str:
        text = README.read_text(encoding="utf-8")
        start = text.index("## Where this is safe to use")
        rest = text[start + len("## Where this is safe to use"):]
        end = rest.find("\n## ")
        return rest[:end] if end != -1 else rest

    def test_the_section_exists(self):
        assert "## Where this is safe to use" in README.read_text(encoding="utf-8")

    def test_it_has_a_verdict_table(self):
        assert "| Deployment | Verdict |" in self._section()

    def test_it_refuses_at_least_one_deployment(self):
        """A table where everything is supported is marketing."""
        section = self._section()
        assert "Not supported" in section, (
            "the deployment table lists no unsupported case, so it reads as an "
            "endorsement rather than a boundary"
        )

    def test_it_names_untrusted_code_execution(self):
        section = self._section().lower()
        assert "untrusted" in section and "user-supplied" in section

    def test_it_says_supported_is_not_production_readiness(self):
        section = self._section().lower()
        assert "production" in section and "supported" in section

    def test_it_does_not_call_the_project_a_sandbox(self):
        section = self._section().lower()
        assert "not isolation" in section or "not a sandbox" in section


def _stated_count(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return match.group(1).replace(",", "")


_COLLECTED: int | None = None


def _collected_test_count() -> int:
    """How many tests the suite collects, asked of pytest rather than remembered.

    `--collect-only` does not execute anything, so this cannot recurse into the
    run it is called from.
    """
    global _COLLECTED
    if _COLLECTED is not None:
        return _COLLECTED

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=README.parent, capture_output=True, text=True,
    )
    match = re.search(r"(\d+)\s+tests? collected", proc.stdout)
    assert match, f"could not read the collected count:\n{proc.stdout[-2000:]}"
    _COLLECTED = int(match.group(1))
    return _COLLECTED