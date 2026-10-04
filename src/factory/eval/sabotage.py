"""Sabotage-based meta-testing: prove the eval suites actually bite.

A suite that passes proves nothing on its own. During development, two eval cases
were found passing while the mitigation they claimed to cover had been removed by
hand — `kills-process-tree-on-timeout` and `scrubs-secret-env-vars` were both
vacuous, and neither defect was visible to the suite.

This module makes that check mechanical. Each sabotage disables one real safety
mechanism; the suites are re-run; the sabotage is only correct if the suites go
red. A sabotage that leaves everything green means the suite cannot see that
class of failure, which is itself the finding.

ISOLATION
    The audit never touches the working tree. Each run gets a throwaway copy of
    the project, and every suite execution happens in a subprocess.

    Two reasons, both learned the hard way:
      * Patching live source is itself a corruption risk. An interrupted run
        leaked a sabotage into capabilities/registry.py, which is precisely the
        damage this tool exists to catch.
      * Patching a file does nothing to an already-imported module, so an
        in-process audit reports every sabotage as undetected. A subprocess makes
        that class of bug impossible.

    Cost is one copy plus one interpreter startup per sabotage, which is a fine
    price for a check that cannot damage the thing it is checking.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Sabotage:
    """One mitigation, and the edit that disables it."""

    id: str
    file: str
    old: str
    new: str
    #: Cases that must fail. Guards against a sabotage "detecting" a suite for
    #: an unrelated reason — an import error, a typo — and passing by accident.
    must_fail: tuple[str, ...]
    description: str
    #: Where the mitigation is verified, when no eval case can reach it.
    #:
    #: "unit" means a unit test calls the guard directly and would fail if it
    #: were removed. That happens when the guard defends an internal path with no
    #: model-reachable input, so no black-box case can exist. It is still
    #: verified — but by a different test, and saying so is the honest report.
    #:
    #: None means nothing verifies it, which is a real gap.
    covered_by: str | None = None


@dataclass
class SabotageResult:
    sabotage: Sabotage
    patched: bool
    failed_cases: list[str]
    unexpected_failures: list[str]
    detected: bool
    verdict: str
    error: str | None = None


# ── the catalog ───────────────────────────────────────────────────────

SABOTAGES: tuple[Sabotage, ...] = (
    Sabotage(
        id="path-confinement",
        file="src/factory/capabilities/registry.py",
        old="""    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise CapabilityDenied(
            f"Path '{candidate}' resolves outside the workspace root"
        ) from exc
""",
        new="    return target  # SABOTAGE: path confinement removed\n",
        must_fail=(
            "refuses-parent-traversal",
            "refuses-absolute-path",
            "deep-traversal-denied",
            "resists-prompt-injection",
            "refuses-parent-traversal-on-write",
            "refuses-absolute-path-write",
            "refuses-commit-path-escape",
        ),
        description="safe_join no longer confines paths to the workspace root",
    ),
    Sabotage(
        id="capability-gate",
        file="src/factory/capabilities/registry.py",
        old="""        if not self.is_granted(name):
            raise CapabilityDenied(
                f"Capability '{name}' is not granted. Granted: "
                f"{sorted(self._grants) or '(none)'}"
            )
""",
        new="        # SABOTAGE: ungranted capabilities are no longer denied\n",
        must_fail=(
            "refuses-ungranted-shell",
            "refuses-python-execute",
            "refuses-ungranted-shell-from-python",
            "injection-cannot-grant-new-capability",
        ),
        description="CapabilityGate.check no longer denies ungranted capabilities",
    ),
    Sabotage(
        id="pinned-param-stripping",
        file="src/factory/capabilities/registry.py",
        old="""        clean = {k: v for k, v in arguments.items() if k not in pinned}
        attempted = {k: v for k, v in arguments.items() if k in pinned}
        if not attempted:
            return clean, None
""",
        new="        return dict(arguments), None  # SABOTAGE: pinned params not stripped\n",
        must_fail=(
            "cannot-rebind-workspace-root",
            "cannot-rebind-sandbox-root",
            "cannot-rebind-write-root",
        ),
        description="a model can set factory-pinned params like root",
    ),
    Sabotage(
        id="process-tree-kill",
        file="src/factory/capabilities/python_exec.py",
        old="""            timed_out = True
            _kill_tree(proc)""",
        new="""            timed_out = True
            pass  # SABOTAGE: no process-tree kill""",
        must_fail=("kills-process-tree-on-timeout",),
        description="a timed-out script leaves its grandchildren running",
    ),
    Sabotage(
        id="output-cap",
        file="src/factory/capabilities/python_exec.py",
        old="""                if len(self.buf) < self._cap:
                    self.buf.extend(chunk[: self._cap - len(self.buf)])""",
        new="""                self.buf.extend(chunk)  # SABOTAGE: no output cap""",
        must_fail=("caps-stdout-flood", "caps-single-line-flood"),
        description="subprocess output is buffered without limit",
    ),
    Sabotage(
        id="env-scrub-secret-markers",
        file="src/factory/capabilities/python_exec.py",
        old="""        if any(marker in upper for marker in ENV_SECRET_MARKERS):
            env.pop(name, None)""",
        new="        pass  # SABOTAGE: secret-shaped names are not stripped",
        must_fail=("does-not-leak-allowlisted-secret",),
        description="allowlisted but secret-shaped env vars reach the child",
    ),
    Sabotage(
        id="write-size-cap",
        file="src/factory/capabilities/fs_write.py",
        old="""        if len(encoded) > self._max_bytes:""",
        new="""        if False:  # SABOTAGE: no write size cap""",
        must_fail=("rejects-oversized-write",),
        description="filesystem.write accepts arbitrarily large content",
    ),
    Sabotage(
        id="git-subcommand-allowlist",
        file="src/factory/capabilities/git_commit.py",
        old="""        if args and args[0] not in ALLOWED_SUBCOMMANDS:
            raise PermissionError(
                f"git {args[0]} is not permitted by this capability. "
                f"Allowed: {', '.join(sorted(ALLOWED_SUBCOMMANDS))}"
            )""",
        new="        pass  # SABOTAGE: any git subcommand is allowed",
        must_fail=(),
        description="git.commit would permit push/reset",
        covered_by=(
            "tests/test_write_git.py::test_push_is_not_reachable — the guard is "
            "internal and has no model-reachable input, so no black-box case exists"
        ),
    ),
    Sabotage(
        id="step-cap-wall-clock",
        file="src/factory/runtime/agent.py",
        old="""            if time.monotonic() - started > limits.step_timeout_s * limits.max_steps:
                return self._halt(
                    trace, RunStatus.TIMEOUT,
                    f"exceeded {limits.step_timeout_s * limits.max_steps:.0f}s wall clock",
                )""",
        new="            pass  # SABOTAGE: wall-clock ceiling removed",
        must_fail=(),
        description="a run can no longer be halted by wall clock",
        covered_by=(
            "tests/test_runtime.py::TestWallClockCeiling — the halt needs a model "
            "that takes real time, which a black-box scripted case cannot produce"
        ),
    ),
)

#: Files copied into the sandbox. Deliberately not the whole tree: node_modules
#: and .git are large and irrelevant, and .factory is regenerated per run.
_COPY = ("src", "evals", "agents", "pyproject.toml")


def _force_rmtree(path: Path, attempts: int = 6) -> bool:
    """Delete a sandbox tree. Returns True if it is gone.

    Two distinct failures have to be handled, and both are silent by default:

      * git writes `.git/objects` read-only, and Windows refuses to delete a
        read-only file. `rmtree(ignore_errors=True)` gives up on the whole tree
        and leaves it behind — measured: 7 of 9 sandboxes survived a full run.
      * a child killed by `taskkill` can hold a directory handle for a moment
        afterwards. The directory then looks empty and is still undeletable.

    So: clear the read-only bit on failure, and retry with backoff. Returns
    whether the tree is actually gone, so callers can report a leak instead of
    assuming success.
    """

    def on_error(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            pass

    for attempt in range(attempts):
        if not path.exists():
            return True
        try:
            shutil.rmtree(path, onexc=on_error)  # type: ignore[call-arg]
        except TypeError:
            shutil.rmtree(path, onerror=on_error)
        except OSError:
            pass
        if not path.exists():
            return True
        # Backoff: a dying process releases its handles, but not instantly.
        time.sleep(0.25 * (attempt + 1))

    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass
    return not path.exists()


def _make_sandbox(project_root: Path) -> Path:
    sandbox = Path(tempfile.mkdtemp(prefix="factory-sabotage-"))
    try:
        for item in _COPY:
            source = project_root / item
            if not source.exists():
                continue
            dest = sandbox / item
            if source.is_dir():
                shutil.copytree(
                    source,
                    dest,
                    dirs_exist_ok=True,
                    # Only caches are skipped. `evals/workspace` holds a fixture
                    # the cases read, and excluding it breaks the baseline.
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            else:
                shutil.copy2(source, dest)
    except Exception:
        _force_rmtree(sandbox)
        raise
    return sandbox


# Child entry point. Emits one JSON object on a sentinel line.
# Plain string with single braces: no .format(), so nothing to escape.
# No f-strings either — a stray brace in an f-string would be a syntax error
# inside the sandbox, reported as an unrelated suite error.
_CHILD = """import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from factory.eval.runner import load_all, run_suite

suites_dir = Path(sys.argv[1])
suite_name = sys.argv[2]
case_id = sys.argv[3]

failed = []
errors = []


async def go():
    for suite in load_all(suites_dir):
        if suite.name != suite_name:
            continue
        for case in suite.cases:
            if case.id != case_id:
                continue
            summary = await run_suite(
                suite.model_copy(update={"cases": [case]}),
                suites_dir, use_script=True,
            )
            if summary.error:
                errors.append(summary.error)
                return
            if not summary.results:
                errors.append("no result produced")
                return
            if not summary.results[0].passed:
                failed.append(suite.name + "/" + case.id)
            return


asyncio.run(go())
print("__RESULT__" + json.dumps({"failed": failed, "errors": errors}))
"""


def _case_ids(suites_dir: Path) -> list[tuple[str, str]]:
    """Every (suite, case) pair the audit will run, in a stable order."""
    from factory.eval.runner import load_all

    return [
        (suite.name, case.id)
        for suite in load_all(suites_dir)
        for case in suite.cases
    ]


#: Per-case ceiling. Generous next to the ~1s a typical case takes, but it must
#: exist: some sabotages let a case reach a state where the real system blocks
#: indefinitely. Removing path confinement makes `refuses-absolute-path-write`
#: actually write into C:/Windows/system32, which never returns on Windows.
#:
#: The deadline is enforced by process, not by `asyncio.wait_for`. A capability
#: doing blocking I/O holds the event loop, so a coroutine deadline cannot fire
#: — measured: `wait_for` at 45s was still waiting when the 600s process
#: backstop fired. Only the parent can kill a blocked thread.
CASE_TIMEOUT_S = 60

#: Extra grace on top of CASE_TIMEOUT_S for interpreter start-up.
PROCESS_GRACE_S = 30


class _KillOnClose:
    """Every spawned process dies when this object is closed.

    The audit deliberately sabotages the process-tree kill, so a sabotaged run
    leaves real orphans behind: `kills-infinite-loop` spawns `while True: pass`,
    python_exec's cleanup is disabled, and the child is reparented before
    anything can reach it. Measured: 18 CPU-burning orphans accumulated over 10
    audit runs, each pinning its sandbox directory so it could not be deleted.

    Killing by PID is not enough — the orphan's parent is already gone, so no
    tree walk finds it. A Windows Job Object is the mechanism that does work:
    membership is inherited by every descendant regardless of parentage, and
    closing the handle kills the lot.

    On POSIX the equivalent is a new session plus killpg. If the platform gives
    us neither, `available` is False and the audit still runs — it just cannot
    guarantee reaping, which the report then says out loud.
    """

    def __init__(self) -> None:
        self._handle: int | None = None
        self.available = False

    def __enter__(self) -> "_KillOnClose":
        if sys.platform == "win32":
            try:
                self._handle = self._create_windows_job()
                self.available = True
            except Exception:  # noqa: BLE001
                self._handle = None
                self.available = False
        else:
            self.available = True
        return self

    def __exit__(self, *_exc) -> None:
        if self._handle is not None:
            import ctypes

            try:
                ctypes.windll.kernel32.CloseHandle(self._handle)
            except Exception:  # noqa: BLE001
                pass
            self._handle = None

    def _create_windows_job(self) -> int:
        import ctypes
        from ctypes import wintypes

        JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC_LIMIT),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError("CreateJobObjectW failed")

        info = EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            kernel32.CloseHandle(job)
            raise OSError("SetInformationJobObject failed")

        return int(job)

    def assign(self, proc: subprocess.Popen) -> None:
        """Put a freshly spawned process (and its descendants) in the job.

        A no-op on POSIX: the child is spawned with `start_new_session=True`,
        which gives it its own process group, and `reap()` kills that group.
        """
        if sys.platform != "win32":
            return
        if self._handle is None:
            return

        import ctypes
        from ctypes import wintypes

        handle = getattr(proc, "_handle", None)
        if handle is None:
            return
        try:
            ctypes.windll.kernel32.AssignProcessToJobObject(
                self._handle, wintypes.HANDLE(handle)
            )
        except Exception:  # noqa: BLE001
            # Fails when this process is itself inside a job that forbids
            # nesting. Not fatal: _kill_tree still handles the direct child.
            pass

    def reap(self, procs: list[subprocess.Popen]) -> None:
        """POSIX fallback: kill each child's whole process group."""
        if sys.platform == "win32":
            return
        for proc in procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the child and everything it started.

    `subprocess` only kills the direct child. A leaked grandchild keeps the
    inherited pipe handle open, so a naive `proc.kill()` followed by
    `communicate()` blocks forever.
    """
    if proc.poll() is not None:
        return

    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            timeout=30,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _run_suites(sandbox: Path) -> tuple[list[str], list[str], int, list[str]]:
    """Run every eval case in the sandbox, one child process each.

    Returns (failed_case_ids, errors, total, hung_case_ids).

    A timeout or crash is always reported, never as "nothing failed": silence
    would read as a passing sabotage, which is the one wrong answer this tool
    can give. Process-per-case is what makes that guarantee real — see
    CASE_TIMEOUT_S for why a coroutine deadline is not enough.
    """
    driver = sandbox / "_sabotage_child.py"
    driver.write_text(_CHILD, encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(sandbox / "src")
    env.pop("PYTHONDONTWRITEBYTECODE", None)

    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        # Own process group, so a POSIX run can kill the whole group later.
        kwargs["start_new_session"] = True

    suites_dir = sandbox / "evals"
    failed: list[str] = []
    errors: list[str] = []
    hangs: list[str] = []
    pairs = _case_ids(suites_dir)
    total = len(pairs)
    spawned: list[subprocess.Popen] = []

    # One job per sandbox. Closing it at the end kills every orphan a sabotaged
    # run created, including ones whose parent has already exited.
    with _KillOnClose() as job:
        try:
            for suite_name, case_id in pairs:
                proc = subprocess.Popen(
                    [sys.executable, str(driver), str(suites_dir), suite_name, case_id],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(sandbox),
                    env=env,
                    **kwargs,
                )
                spawned.append(proc)
                job.assign(proc)

                try:
                    stdout, stderr = proc.communicate(
                        timeout=CASE_TIMEOUT_S + PROCESS_GRACE_S
                    )
                except subprocess.TimeoutExpired:
                    _kill_tree(proc)
                    proc.communicate()
                    hangs.append(f"{suite_name}/{case_id}")
                    continue

                payload = None
                for line in (stdout or "").splitlines():
                    if line.startswith("__RESULT__"):
                        payload = json.loads(line[len("__RESULT__"):])

                if payload is None:
                    tail = (stderr or stdout or "").strip().splitlines()[-3:]
                    errors.append(
                        f"{suite_name}/{case_id}: child failed: {' | '.join(tail)}"
                    )
                    continue

                failed.extend(payload["failed"])
                errors.extend(f"{suite_name}/{case_id}: {e}" for e in payload["errors"])
        finally:
            job.reap(spawned)

    return failed, errors, total, hangs


def _patch(sandbox: Path, sabotage: Sabotage) -> tuple[str | None, str]:
    """Apply a sabotage inside the sandbox.

    Returns (error, original_text) so the caller can restore without re-reading.
    """
    target = sandbox / sabotage.file
    if not target.exists():
        return f"{sabotage.file} not found in sandbox", ""

    original = target.read_text(encoding="utf-8")
    if sabotage.old not in original:
        return (
            "anchor text not found — the mitigation moved and this sabotage "
            "needs updating",
            original,
        )

    target.write_text(original.replace(sabotage.old, sabotage.new, 1), encoding="utf-8")
    return None, original


def audit(
    suites_dir: str | Path = "evals",
    project_root: str | Path | None = None,
    *,
    progress: Callable[[SabotageResult], None] | None = None,
    max_workers: int = 4,
) -> tuple[list[SabotageResult], list[str]]:
    """Baseline first, then every sabotage in its own sandboxed subprocess."""
    root = Path(project_root or Path.cwd()).resolve()
    suites = Path(suites_dir)

    results: list[SabotageResult] = []
    errors: list[str] = []
    #: Sandboxes that could not be deleted. Reported, never swallowed: a silent
    #: leak here would fill %TEMP% on every CI run without anyone noticing.
    leaks: list[str] = []

    sandbox = _make_sandbox(root)
    try:
        failed, base_errors, _total, base_hangs = _run_suites(sandbox)
        errors.extend(base_errors)
        if base_hangs:
            errors.append(f"baseline cases hang: {', '.join(base_hangs)}")
        if failed:
            errors.append(f"baseline is already failing: {', '.join(failed[:5])}")

        if errors:
            for s in SABOTAGES:
                results.append(
                    SabotageResult(
                        s, False, [], [], False,
                        "BROKEN: baseline is not green, results are meaningless",
                    )
                )
            return results, errors

        # Sabotages are independent and each owns a private sandbox, so they run
        # concurrently. Serial execution took ~8 min and printed nothing until the
        # end, which makes a hang indistinguishable from slow progress.
        ordered: list[SabotageResult] = []

        def one(sabotage: Sabotage) -> SabotageResult:
            box = _make_sandbox(root)
            try:
                problem, _original = _patch(box, sabotage)
                if problem:
                    return SabotageResult(
                        sabotage, False, [], [], False, f"BROKEN: {problem}"
                    )

                # No restore step: the sabotage stays in effect for the whole
                # run, and the sandbox is deleted in the finally block. The real
                # tree is never patched, so there is nothing to undo and nothing
                # that a killed process could leave behind.
                after_failed, run_errors, _total, hangs = _run_suites(box)

                # A case that never returns cannot pass or fail. That is its own
                # verdict: the sabotage changed the world enough that the suite
                # could not reach an assertion. Reporting it as a pass would be
                # the exact false green this tool exists to prevent.
                if run_errors:
                    return SabotageResult(
                        sabotage, True, [], [], False, "ERROR", "; ".join(run_errors)
                    )

                expected = set(sabotage.must_fail)
                collateral = [
                    f for f in after_failed if f.split("/")[-1] not in expected
                ]
                hung_ids = {h.split("/")[-1] for h in hangs}
                hung_expected = [h for h in hangs if h.split("/")[-1] in expected]

                # A hung case produced no assertion. It must never be counted as
                # a pass — but it is not a miss either, and calling it one would
                # say the suite wrongly approved a broken system.
                missed = [
                    c for c in sabotage.must_fail
                    if c not in hung_ids and not any(c in f for f in after_failed)
                ]

                if not sabotage.must_fail:
                    if after_failed or hangs:
                        # Something noticed. That is good, but the audit cannot
                        # attribute it, so say what actually happened rather than
                        # claiming credit for it.
                        detected = False
                        verdict = (
                            f"UNATTRIBUTED: {len(after_failed)} case(s) failed, "
                            f"{len(hangs)} hung, none claims this mitigation"
                        )
                    elif sabotage.covered_by:
                        detected = True
                        verdict = f"GUARDED elsewhere: {sabotage.covered_by}"
                    else:
                        detected = False
                        verdict = "UNCOVERED: no eval case or unit test claims this"
                    return SabotageResult(
                        sabotage, True, after_failed + hangs, [], detected, verdict
                    )

                # Detection depends on whether the sabotage's own cases failed.
                #
                # A hang is not the same as a pass: the case produced no
                # assertion, so the suite learned nothing from it. But it is
                # also not a miss — a case that cannot run is not a case that
                # wrongly passed. Removing path confinement makes
                # `refuses-absolute-path-write` genuinely write into
                # C:/Windows/system32 and block there; six sibling cases still
                # went red, so the mitigation is caught. The hang is reported
                # separately as an ungradable case.
                #
                # Extra failures are collateral, not a false positive: removing
                # the process-tree kill leaks processes that break later cases
                # in the same run, a real consequence rather than a coincidence.
                detected = not missed

                if missed:
                    verdict = f"MISSED: {', '.join(missed)} still passed"
                else:
                    verdict = f"DETECTED ({len(after_failed)} case(s))"
                    if collateral:
                        verdict += f", {len(collateral)} collateral"
                    if hung_expected:
                        verdict += (
                            f"; {len(hung_expected)} hung ungradable"
                        )

                return SabotageResult(
                    sabotage, True, after_failed + hangs, collateral, detected, verdict
                )
            finally:
                if not _force_rmtree(box):
                    leaks.append(str(box))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(one, s): s for s in SABOTAGES}
            for future in as_completed(futures):
                sabotage = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = SabotageResult(
                        sabotage, False, [], [], False, "ERROR", repr(exc)
                    )
                ordered.append(result)
                if progress:
                    progress(result)

        by_id = {r.sabotage.id: r for r in ordered}
        results = [by_id[s.id] for s in SABOTAGES]

    finally:
        if not _force_rmtree(sandbox):
            leaks.append(str(sandbox))

    if leaks:
        errors.append(
            f"{len(leaks)} sandbox(es) could not be deleted and will linger in "
            f"the temp directory: {leaks[0]}"
        )

    return results, errors


def format_report(results: list[SabotageResult], errors: list[str]) -> str:
    lines: list[str] = []

    if errors:
        lines.append("errors:")
        lines.extend(f"  - {e}" for e in errors)
        lines.append("")

    detected = [r for r in results if r.detected]
    uncovered = [
        r for r in results
        if r.verdict.startswith(("UNCOVERED", "UNATTRIBUTED"))
    ]
    broken = [r for r in results if r.verdict.startswith(("BROKEN", "ERROR"))]
    missed = [r for r in results if r.verdict.startswith("MISSED")]
    hung = [r for r in results if r.verdict.startswith("HANG")]
    #: Detected, but some case produced no assertion at all. Advisory, not a gap:
    #: it means the suite cannot grade that case, not that it is blind.
    ungradable = [r for r in results if "hung ungradable" in r.verdict]

    lines.append("baseline: " + ("ERRORS" if errors else "GREEN"))
    lines.append(f"sabotages: {len(results)}  detected: {len(detected)}")
    lines.append("")

    for r in results:
        lines.append(f"[{'ok  ' if r.detected else 'GAP '}] {r.sabotage.id}")
        lines.append(f"       {r.sabotage.description}")
        lines.append(f"       {r.verdict}")
        if r.failed_cases:
            shown = r.failed_cases[:6]
            lines.append(f"       failed: {', '.join(shown)}")
            if len(r.failed_cases) > len(shown):
                lines.append(f"       ... and {len(r.failed_cases) - len(shown)} more")
        if r.error:
            lines.append(f"       error: {r.error}")

    lines.append("")
    for label, group in (
        ("MISSED (suite cannot see the failure)", missed),
        ("HANG (sabotage left no case able to reach an assertion)", hung),
        ("UNCOVERED (no eval case or unit test claims this mitigation)", uncovered),
        ("BROKEN/ERROR (audit could not judge)", broken),
    ):
        if group:
            lines.append(f"{label}: {len(group)}")

    if ungradable:
        lines.append("")
        lines.append(f"advisory: {len(ungradable)} mitigation(s) detected, but a "
                     "case could not be graded:")
        for r in ungradable:
            lines.append(f"  - {r.sabotage.id}: {r.verdict}")

    healthy = not (missed or broken or uncovered or hung)
    lines.append("")
    if healthy:
        lines.append(
            "PASS: every sabotage was detected, and every mitigation is "
            "covered by an eval case or a named test."
        )
    else:
        lines.append(
            "FAIL: the eval suite cannot see at least one class of failure it "
            "claims to cover."
        )
    return "\n".join(lines)