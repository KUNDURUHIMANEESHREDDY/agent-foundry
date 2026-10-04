"""python.execute — a hardened subprocess runner.

WHAT THIS IS
    Isolation from crashes, hangs, output floods, environment leakage, and
    accidental writes. Execution is bounded and observable.

WHAT THIS IS NOT
    A security boundary against hostile code. `subprocess` runs as the calling
    user, so arbitrary Python can read any file that user can read, open sockets,
    and spawn children. No amount of argument validation fixes that — it is
    inherent to executing code.

    If the code is genuinely untrusted, this must be replaced with OS-level
    isolation: a container, a VM, or a Windows Job Object with network
    restrictions. See `sandbox_level` on the capability, which records the
    assumption the caller is making.

Mitigations actually enforced here:
    - wall-clock timeout, with the whole process tree killed
    - incremental output reading with a hard byte cap (drain-and-discard past
      the cap, so a flood cannot block the child on a full pipe)
    - environment scrubbed to an allowlist, with secret-shaped names denied
    - cwd pinned to the workspace
    - stdin closed, so a script cannot wait on input
    - POSIX rlimits for CPU, address space, and file size
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from factory.capabilities.registry import Capability, object_schema

# Variables a child legitimately needs to run Python at all.
ENV_ALLOWLIST = (
    "PATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "COMSPEC",
    "PATHEXT",
    "WINDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
    "HOME",
    "LANG",
    "LC_ALL",
    "PYTHONIOENCODING",
    "PYTHONHASHSEED",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUNBUFFERED",
    "PYTHONPATH",
    "PYTHONHOME",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
)

# Names that must never reach a child, allowlisted or not.
ENV_SECRET_MARKERS = ("SECRET", "TOKEN", "KEY", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "SESSION")

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024
DEFAULT_MAX_WALL_S = 30.0


class _BoundedReader:
    """Drains a stream continuously, keeping only the first `cap` bytes.

    Continuing to read after the cap matters: if we stopped, a chatty child would
    block on a full pipe and we'd report a hang instead of an output flood.
    """

    def __init__(self, stream, cap: int) -> None:
        self._stream = stream
        self._cap = cap
        self.buf = bytearray()
        self.total = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                chunk = self._stream.read(8192)
                if not chunk:
                    return
                self.total += len(chunk)
                if len(self.buf) < self._cap:
                    self.buf.extend(chunk[: self._cap - len(self.buf)])
        except (ValueError, OSError):
            return
        finally:
            try:
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass

    def join(self, timeout: float = 2.0) -> None:
        self._thread.join(timeout)

    @property
    def truncated(self) -> bool:
        return self.total > self._cap

    def text(self) -> str:
        return self.buf.decode("utf-8", errors="replace")


def _child_env() -> dict[str, str]:
    """Allowlist the environment, and refuse anything secret-shaped."""
    env: dict[str, str] = {}
    for name in ENV_ALLOWLIST:
        value = os.environ.get(name)
        if value is not None:
            env[name] = value

    # An allowlisted name that looks like a secret is still dropped.
    for name in list(env):
        upper = name.upper()
        if any(marker in upper for marker in ENV_SECRET_MARKERS):
            env.pop(name, None)

    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # Deterministic hashing, so runs are comparable.
    env.setdefault("PYTHONHASHSEED", "0")
    return env


def _posix_limits(cpu_s: int, mem_bytes: int, file_bytes: int):
    """Build a preexec_fn applying rlimits. POSIX only; None elsewhere."""
    if platform.system() == "Windows":
        return None
    try:
        import resource
    except ImportError:
        return None

    def apply() -> None:  # pragma: no cover - POSIX only
        for res, soft, hard in (
            (resource.RLIMIT_CPU, cpu_s, cpu_s + 2),
            (resource.RLIMIT_FSIZE, file_bytes, file_bytes),
            (resource.RLIMIT_AS, mem_bytes, mem_bytes),
        ):
            try:
                resource.setrlimit(res, (soft, hard))
            except (ValueError, OSError):
                pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (ValueError, OSError):
            pass

    return apply


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process and anything it spawned.

    Without this, a script that forks a child leaves that child running after the
    timeout — the most common way a "sandbox" quietly leaks.
    """
    try:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=15,
                check=False,
            )
        else:
            import signal

            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass

    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass


class PythonExecute(Capability):
    """Run Python source in a bounded, scrubbed subprocess."""

    name = "python.execute"
    description = (
        "Execute a Python snippet and return its stdout, stderr and exit code. "
        "The snippet runs as the calling user in a separate process: it is "
        "bounded in time and output, NOT contained against hostile code."
    )
    pinned_params = ("workspace", "timeout_s", "max_output_bytes", "sandbox_level")
    required_params = ("workspace",)

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return object_schema(
            {
                "code": {
                    "type": "string",
                    "description": (
                        "Python source to execute. Prefer the standard library; "
                        "network access and installing packages are not "
                        "available."
                    ),
                }
            },
            required=["code"],
        )

    def __init__(
        self,
        workspace: Path,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_wall_s: float = DEFAULT_MAX_WALL_S,
        sandbox_level: str = "subprocess",
    ) -> None:
        self._workspace = Path(workspace).resolve()
        self._timeout_s = float(timeout_s)
        self._max_output_bytes = int(max_output_bytes)
        self._max_wall_s = float(max_wall_s)
        self._sandbox_level = sandbox_level

    def describe(self) -> str:
        return (
            f"python.execute(workspace={self._workspace}, timeout_s={self._timeout_s}, "
            f"max_output_bytes={self._max_output_bytes}, sandbox_level={self._sandbox_level})"
        )

    @property
    def sandbox_level(self) -> str:
        return self._sandbox_level

    def invoke(self, code: Any = "", **_: Any) -> dict[str, Any]:
        if not isinstance(code, str):
            return {
                "ok": False,
                "error": f"code must be a string, got {type(code).__name__}",
                "capability": self.name,
            }
        if not code.strip():
            return {
                "ok": False,
                "error": "code must not be empty",
                "capability": self.name,
            }

        started = time.monotonic()
        argv = [sys.executable, "-I", "-c", code]
        popen_kwargs: dict[str, Any] = {
            "cwd": str(self._workspace),
            "env": _child_env(),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "shell": False,
            "close_fds": True,
        }

        if platform.system() == "Windows":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
            popen_kwargs["preexec_fn"] = _posix_limits(
                cpu_s=int(self._timeout_s) + 2,
                mem_bytes=1024 * 1024 * 1024,
                file_bytes=self._max_output_bytes * 4,
            )

        try:
            proc = subprocess.Popen(argv, **popen_kwargs)
        except OSError as exc:
            return {
                "ok": False,
                "error": f"failed to start interpreter: {exc}",
                "capability": self.name,
            }

        out = _BoundedReader(proc.stdout, self._max_output_bytes)
        err = _BoundedReader(proc.stderr, self._max_output_bytes)
        out.start()
        err.start()

        timed_out = False
        exit_code: int | None = None
        try:
            exit_code = proc.wait(timeout=self._timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            try:
                exit_code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        finally:
            out.join()
            err.join()

        elapsed = time.monotonic() - started
        if elapsed > self._max_wall_s and proc.poll() is None:
            _kill_tree(proc)

        result: dict[str, Any] = {
            "ok": not timed_out and exit_code == 0,
            "capability": self.name,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_ms": int(elapsed * 1000),
            "stdout": out.text(),
            "stderr": err.text(),
            "truncated": out.truncated or err.truncated,
            "stdout_total_bytes": out.total,
            "stderr_total_bytes": err.total,
            "sandbox_level": self._sandbox_level,
        }

        if timed_out:
            result["error"] = (
                f"execution exceeded {self._timeout_s}s timeout and was terminated "
                f"(process tree killed)"
            )
        elif exit_code != 0:
            head = result["stderr"].strip().splitlines()
            result["error"] = (
                f"exit code {exit_code}"
                + (f": {head[-1][:200]}" if head else "")
            )

        return result
