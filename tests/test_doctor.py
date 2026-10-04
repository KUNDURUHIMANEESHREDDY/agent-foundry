"""The preflight must report the truth, and must not overstate it.

`doctor.py` exists because "P2 is blocked" was a claim repeated in conversation
for a long time without anyone being able to check it. A command settles it.

The dangerous failure mode for a tool like this is overstating: reporting
something as available because a port is open, or exiting non-zero on a machine
that is merely not equipped. Both are asserted against here.
"""

from __future__ import annotations

import contextlib
import io
import json
import socket
from pathlib import Path

import pytest

import doctor
from factory.eval import runner


class TestItReportsEveryBackend:
    def test_it_covers_the_shipped_specs(self, monkeypatch):
        monkeypatch.setattr(socket, "create_connection", lambda *a, **k: contextlib.nullcontext())
        rows = runner.live_model_status(Path("agents"))

        providers = {r["provider"] for r in rows}
        assert "openai_compat" in providers
        assert "ollama" in providers

    def test_it_deduplicates_shared_backends(self, monkeypatch):
        """Three specs pointing at ollama is one answer, not three."""
        monkeypatch.setattr(socket, "create_connection", lambda *a, **k: contextlib.nullcontext())
        rows = runner.live_model_status(Path("agents"))

        bases = [r["base_url"] for r in rows]
        assert len(bases) == len(set(bases)), f"duplicate backends reported: {bases}"

    def test_an_unreachable_backend_is_reported_as_such(self, monkeypatch):
        def refuse(*a, **k):
            raise TimeoutError("timed out")

        monkeypatch.setattr(socket, "create_connection", refuse)
        rows = runner.live_model_status(Path("agents"))

        assert rows
        assert all(r["reachable"] is False for r in rows)
        assert all(r["detail"] for r in rows), "an unreachable row must say why"


class TestItDoesNotOverstate:
    def test_reachable_does_not_mean_verified(self, monkeypatch):
        """A socket accepting is not a model answering.

        The caveat has to survive the case where a backend *is* reachable -- that
        is the moment a reader would take this as "P2 is done".
        """
        monkeypatch.setattr(socket, "create_connection",
                            lambda *a, **k: contextlib.nullcontext())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor.report(as_json=True)
        payload = json.loads(buf.getvalue())

        assert payload["live_model_qualification"]["possible"] is True
        assert "does not prove" in payload["live_model_qualification"]["note"]

    def test_a_spec_that_will_not_parse_is_reported_not_hidden(self, tmp_path, monkeypatch):
        (tmp_path / "broken.yaml").write_text("name: has space\n", encoding="utf-8")
        (tmp_path / "fine.yaml").write_text(
            "name: fine\nversion: 0.1.0\nsystem_prompt: p\n"
            "model:\n  provider: fake\n  name: none\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(socket, "create_connection", lambda *a, **k: contextlib.nullcontext())

        rows = runner.live_model_status(tmp_path)

        assert any("spec invalid" in str(r["detail"]) for r in rows), (
            "a spec that does not parse must appear in the report rather than "
            "being skipped -- a silently skipped spec is a hole in the preflight"
        )


class TestItIsUsableAsAPreflight:
    def test_it_exits_zero_on_an_unequipped_machine(self, monkeypatch):
        """A machine with no model is normal, not broken.

        Exiting non-zero would make people pipe it to `||` and ignore it, which
        defeats the point of having it.
        """
        monkeypatch.setattr(socket, "create_connection", lambda *a, **k: contextlib.nullcontext())
        assert doctor.report() == 0

    def test_json_output_parses(self, monkeypatch, capsys):
        monkeypatch.setattr(socket, "create_connection", lambda *a, **k: contextlib.nullcontext())
        doctor.report(as_json=True)
        payload = json.loads(capsys.readouterr().out)

        for key in ("live_model_qualification", "container_isolation", "langfuse_live"):
            assert key in payload

    def test_it_names_the_command_that_actually_proves_it(self, monkeypatch, capsys):
        monkeypatch.setattr(socket, "create_connection", lambda *a, **k: contextlib.nullcontext())
        doctor.report()
        out = capsys.readouterr().out

        assert "factory eval --live" in out, (
            "the preflight must point at the command that does the proving, not "
            "just report that something is missing"
        )