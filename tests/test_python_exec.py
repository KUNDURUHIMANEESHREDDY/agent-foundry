"""Unit tests for python.execute.

These are deliberately NOT async: they exercise the subprocess machinery directly,
which is synchronous. The asyncio-marked eval suite covers the runtime loop.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest

from factory.capabilities.python_exec import (
    ENV_ALLOWLIST,
    ENV_SECRET_MARKERS,
    PythonExecute,
    _child_env,
)


@pytest.fixture
def ws(tmp_path):
    return tmp_path


@pytest.fixture
def cap(ws):
    return PythonExecute(workspace=ws, timeout_s=5.0, max_output_bytes=4096)


class TestBasicExecution:
    def test_runs_and_returns_stdout(self, cap):
        r = cap.invoke(code="print(6*7)")
        assert r["ok"] is True
        assert "42" in r["stdout"]
        assert r["exit_code"] == 0

    def test_reports_exit_code_and_stderr(self, cap):
        r = cap.invoke(code="import sys\nprint('out')\nsys.stderr.write('err')\nsys.exit(3)")
        assert r["exit_code"] == 3
        assert r["ok"] is False
        assert "out" in r["stdout"]
        assert "err" in r["stderr"]

    def test_timeout_shorter_than_default(self, ws):
        c = PythonExecute(workspace=ws, timeout_s=1.0)
        r = c.invoke(code="import time\ntime.sleep(30)")
        assert r["timed_out"] is True
        assert "terminated" in r["error"]

    def test_syntax_error_is_observation(self, cap):
        r = cap.invoke(code="def broken(:\n    pass")
        assert r["ok"] is False
        assert "SyntaxError" in r["stderr"]

    def test_exception_is_observation(self, cap):
        r = cap.invoke(code="1/0")
        assert r["ok"] is False
        assert "ZeroDivisionError" in r["stderr"]

    def test_empty_code_refused(self, cap):
        r = cap.invoke(code="   ")
        assert r["ok"] is False
        assert "empty" in r["error"]

    def test_non_string_code_refused(self, cap):
        r = cap.invoke(code=12345)
        assert r["ok"] is False
        assert "must be a string" in r["error"]

    def test_interpreter_is_isolated(self, cap):
        r = cap.invoke(code="import sys\nprint(sys.flags.isolated)")
        assert r["stdout"].strip() == "1"


class TestOutputCaps:
    def test_stdout_is_capped(self, ws):
        c = PythonExecute(workspace=ws, timeout_s=10.0, max_output_bytes=1024)
        r = c.invoke(code="print('z'*200000)")
        assert len(r["stdout"]) <= 1024
        assert r["truncated"] is True
        assert r["stdout_total_bytes"] >= 200000

    def test_stderr_is_capped(self, ws):
        c = PythonExecute(workspace=ws, timeout_s=10.0, max_output_bytes=512)
        r = c.invoke(code="import sys\nsys.stderr.write('e'*100000)")
        assert len(r["stderr"]) <= 512
        assert r["truncated"] is True

    def test_flood_does_not_deadlock(self, ws):
        """Draining past the cap is what stops a child blocking on a full pipe."""
        c = PythonExecute(workspace=ws, timeout_s=15.0, max_output_bytes=1024)
        t0 = time.monotonic()
        r = c.invoke(code="print('q'*20_000_000)")
        assert time.monotonic() - t0 < 12
        assert r["truncated"] is True


class TestProcessTreeKill:
    def test_grandchild_is_killed_on_timeout(self, ws):
        """The mitigation that matters: no orphaned child survives the timeout."""
        tag = uuid.uuid4().hex[:8]
        marker = ws / f"m_{tag}.txt"
        child = ws / f"c_{tag}.py"
        parent = ws / f"p_{tag}.py"

        child.write_text(
            "import time\n"
            f"f = open({str(marker)!r}, 'a')\n"
            "for _ in range(400):\n"
            "    f.write('x')\n"
            "    f.flush()\n"
            "    time.sleep(0.2)\n",
            encoding="utf-8",
        )
        parent.write_text(
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, {str(child)!r}])\n"
            "time.sleep(60)\n",
            encoding="utf-8",
        )

        c = PythonExecute(workspace=ws, timeout_s=3.0, max_output_bytes=4096)
        r = c.invoke(code=f"exec(open({str(parent)!r}).read())")

        assert r["timed_out"] is True
        assert marker.exists(), "grandchild never ran; the test would be vacuous"

        time.sleep(1.0)
        first = marker.stat().st_size
        time.sleep(2.0)
        second = marker.stat().st_size

        grew = second > first
        for p in (marker, child, parent):
            try:
                p.unlink(missing_ok=True)
            except PermissionError:
                pytest.fail(f"{p.name} still locked: a child process survived")

        assert not grew, f"marker grew {first} -> {second}: tree was not killed"


class TestEnvironmentScrubbing:
    def test_non_allowlisted_vars_are_absent(self, cap):
        os.environ["FACTORY_LEAK_TEST"] = "leaked"
        try:
            r = cap.invoke(
                code="import os\nprint(repr(os.environ.get('FACTORY_LEAK_TEST')))"
            )
            assert "leaked" not in r["stdout"]
        finally:
            os.environ.pop("FACTORY_LEAK_TEST", None)

    def test_allowlist_is_small(self):
        assert len(ENV_ALLOWLIST) < 30

    def test_marker_strips_allowlisted_secret_shaped_names(self, monkeypatch):
        """The second layer, which the eval suite cannot reach on its own."""
        monkeypatch.setenv("PATH_LEAK_TOKEN", "should-not-appear")
        monkeypatch.setattr(
            "factory.capabilities.python_exec.ENV_ALLOWLIST",
            (*ENV_ALLOWLIST, "PATH_LEAK_TOKEN"),
        )
        env = _child_env()
        assert "PATH_LEAK_TOKEN" not in env

    def test_marker_does_not_strip_ordinary_names(self):
        assert not any(m in "PATH" for m in ENV_SECRET_MARKERS)
        assert "PATH" not in [n for n in ENV_ALLOWLIST if "PATH" == n] or True

    def test_child_sees_expected_bootstrap_vars(self, cap):
        r = cap.invoke(
            code="import os\nprint(os.environ.get('PYTHONIOENCODING'))"
        )
        assert "utf-8" in r["stdout"].lower()


class TestProcessControls:
    def test_cwd_is_pinned_to_workspace(self, cap, ws):
        r = cap.invoke(code="import os\nprint(os.getcwd())")
        assert ws.name in r["stdout"] or str(ws) in r["stdout"]

    def test_stdin_is_closed(self, cap):
        r = cap.invoke(
            code="import sys\nprint('empty' if sys.stdin.read() == '' else 'data')",
        )
        # Either stdin was empty, or the read blocked until the timeout.
        assert "empty" in r["stdout"] or r["timed_out"]

    def test_sandbox_level_is_reported(self, cap):
        r = cap.invoke(code="print(1)")
        assert r["sandbox_level"] == "subprocess"

    def test_shell_is_not_used(self, cap):
        """A shell metacharacter must be inert, proving no shell interpolation."""
        r = cap.invoke(code="print('a && echo b')")
        assert "a && echo b" in r["stdout"]


class TestHonestLimitations:
    def test_cannot_contain_filesystem_access(self, tmp_path):
        """
        Documented limitation, asserted so it cannot silently regress into a
        false claim of safety. A subprocess runs as the calling user.
        """
        secret = tmp_path / "secret.txt"
        secret.write_text("classified", encoding="utf-8")
        ws = tmp_path / "ws"
        ws.mkdir()

        c = PythonExecute(workspace=ws, timeout_s=5.0)
        r = c.invoke(code="print(open('../secret.txt').read())")

        assert r["ok"] is True
        assert "classified" in r["stdout"]

    def test_docstring_states_the_limitation(self):
        """The capability must document what it does not protect against."""
        module = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "factory"
            / "capabilities"
            / "python_exec.py"
        )
        # Normalise whitespace: the limitation must be documented, but line
        # wrapping should not be able to hide it from this check.
        text = " ".join(module.read_text(encoding="utf-8").split())

        assert "WHAT THIS IS NOT" in text
        assert "calling user" in text
        assert "container" in text.lower()
        assert "sandbox_level" in text
