"""Suite loading and the runner that executes cases against a spec."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os
import re
import yaml
from pydantic import ValidationError

from factory.capabilities.registry import registry_for
from factory.compiler import CompileError, compile_agent
from factory.eval.case import CaseResult, EvalCase, Suite, evaluate
from factory.models.base import ModelResponse
from factory.models.fake import ScriptedAdapter
from factory.runtime.trace import MemoryTraceStore, RunTrace
from factory.spec.loader import SpecError, load_spec


class SuiteError(ValueError):
    pass


def load_suite(path: str | Path) -> Suite:
    p = Path(path)
    if not p.exists():
        raise SuiteError(f"Suite not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SuiteError(f"Invalid YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise SuiteError(f"Suite {p} must be a mapping")
    try:
        return Suite.model_validate(data)
    except ValidationError as exc:
        detail = "\n".join(
            f"  - {'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise SuiteError(f"Invalid suite {p}:\n{detail}") from exc


def load_all(directory: str | Path) -> list[Suite]:
    d = Path(directory)
    if not d.exists():
        raise SuiteError(f"No suite directory: {d}")
    suites = [load_suite(f) for f in sorted(d.glob("*.yaml"))]
    if not suites:
        raise SuiteError(f"No .yaml suites in {d}")
    return suites


GEN_RE = re.compile(r"\$\{gen:(\d+)\}")


def _expand(value: Any) -> Any:
    """Expand `${gen:N}` into N filler bytes.

    Some limits are only meaningfully tested above their threshold: a 3-byte
    write does not exercise a 512KB cap. Pasting half a megabyte of filler into
    a YAML file would be unreadable, so the case declares the size and the
    runner materialises it deterministically.
    """
    if isinstance(value, str):
        return GEN_RE.sub(lambda m: "x" * int(m.group(1)), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def scripted_adapter(case: EvalCase) -> ScriptedAdapter:
    """Turn a case's scripted steps into a deterministic model.

    Cases without a `scripted` block still run: the model returns a single plain
    answer, which is enough to exercise assertions that do not depend on tools.
    """
    if not case.scripted:
        return ScriptedAdapter([ModelResponse(text="(no script)")])

    responses = [
        ModelResponse(text=s.text, tool_calls=[
            __import__("factory.models.base", fromlist=["ToolCall"]).ToolCall(
                name=tc.name, arguments=_expand(tc.arguments)
            )
            for tc in s.tool_calls
        ])
        for s in case.scripted
    ]
    return ScriptedAdapter(responses)


@dataclass
class RunSummary:
    suite: str
    spec_name: str
    spec_version: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    #: Cases not run because the environment lacked a declared requirement.
    #: Reported separately from `failed`, because a skip is not a pass and is
    #: not a failure either -- it is an absence of evidence.
    skipped: int = 0
    tokens: int = 0
    duration_ms: int = 0
    violations: int = 0
    results: list[CaseResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    error: str | None = None

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "spec": f"{self.spec_name}@{self.spec_version}",
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "pass_rate": round(self.pass_rate, 3),
            "tokens": self.tokens,
            "duration_ms": self.duration_ms,
            "violations": self.violations,
            "error": self.error,
            "cases": [
                {
                    "id": r.case_id,
                    "passed": r.passed,
                    "steps": r.steps,
                    "tokens": r.tokens,
                    "violations": r.violations,
                    "status": r.status,
                    "halt_reason": r.halt_reason,
                    "failures": r.failures,
                }
                for r in self.results
            ],
        }


async def run_case(
    case: EvalCase, spec_path: str, workspace: Path, use_script: bool = True
) -> CaseResult:
    try:
        spec = load_spec(spec_path)
    except SpecError as exc:
        return CaseResult(case.id, False, [f"spec load failed: {exc}"])

    model = scripted_adapter(case) if use_script else None

    if case.limits is not None:
        # Narrow or widen this case's bounds without touching the spec, so an
        # orthogonal runtime limit cannot mask the property under test.
        spec = spec.model_copy(update={"limits": case.limits})

    try:
        compiled = compile_agent(
            spec,
            registry_for(workspace, spec.isolation),
            workspace=workspace,
            model=model,
        )
    except CompileError as exc:
        return CaseResult(case.id, False, [f"compile failed: {exc}"])

    # Preconditions for cases that assert on environment scrubbing. The
    # secret-shaped name is also added to the allowlist so the second layer
    # (the marker check) is what rejects it, not merely the allowlist.
    restore = {k: os.environ.get(k) for k in case.inject_env}
    os.environ.update(case.inject_env)

    from factory.capabilities import python_exec as pe

    original_allowlist = pe.ENV_ALLOWLIST
    injected_names = [n for n in case.inject_env if n not in original_allowlist]
    if injected_names:
        pe.ENV_ALLOWLIST = (*original_allowlist, *injected_names)

    try:
        result = await compiled.runtime.run(case.task)
    except Exception as exc:  # noqa: BLE001
        return CaseResult(case.id, False, [f"runtime raised: {exc}"])
    finally:
        pe.ENV_ALLOWLIST = original_allowlist
        for key, value in restore.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # Some checks read the filesystem, so tell the case where it is.
    case.workspace = str(workspace)
    evaluated = evaluate(case, result.trace)

    # Clean up filesystem side-effects so cases stay independent. A file still
    # locked by a surviving child is itself evidence the tree was not killed, so
    # surface it as a failure rather than crashing the whole run.
    for leftover in ("tree_marker.txt", "tree_child.py"):
        target = workspace / leftover
        try:
            target.unlink(missing_ok=True)
        except PermissionError:
            evaluated.passed = False
            evaluated.failures.append(
                f"cleanup: {leftover} is still locked by a surviving child process "
                f"(tree was not killed)"
            )

    return evaluated


def unmet_requirements(suite: Suite) -> list[str]:
    """External requirements this suite declares that the environment lacks.

    Kept separate from the run so it can be checked without running anything,
    and so a test can assert the declaration exists at all.

    A requirement this function does not recognise is reported as unmet. Treating
    it as satisfied would disable the gate exactly when someone has mistyped
    `contaienr` -- the case where skipping quietly becomes the default.
    """
    unmet: list[str] = []

    for requirement in suite.requires:
        if requirement == "container":
            from factory.capabilities import container

            try:
                container.probe()
            except container.ContainerUnavailable:
                unmet.append(requirement)
        else:
            unmet.append(requirement)

    return unmet


def live_model_status(specs_dir: Path) -> list[dict[str, object]]:
    """Which model backends the shipped specs need, and whether they answer.

    P2 -- live-model qualification -- is the one thing this project cannot prove
    about itself, because it needs a reachable model. That has been reported as a
    claim; this makes it a command anyone can run and disagree with.

    Reachability is a TCP connect, not a completion: enough to distinguish
    "nothing is listening" from "something is there", which is the question that
    gets asked, and cheap enough to never be a reason the answer goes stale.
    """
    import socket
    from urllib.parse import urlparse

    from factory.spec.loader import SpecError, load_spec

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str | None]] = set()

    for path in sorted(Path(specs_dir).glob("*.yaml")):
        try:
            spec = load_spec(path)
        except SpecError as exc:
            rows.append({"spec": path.name, "provider": "?", "base_url": None,
                         "reachable": False, "detail": f"spec invalid: {exc}"})
            continue

        base = spec.model.base_url
        if spec.model.provider == "ollama" and not base:
            base = "http://localhost:11434"

        key = (spec.model.provider, base)
        if key in seen:
            continue
        seen.add(key)

        detail = ""
        reachable = False
        if base:
            parsed = urlparse(base)
            host, port = parsed.hostname or "localhost", parsed.port or (
                443 if parsed.scheme == "https" else 80
            )
            try:
                with socket.create_connection((host, port), timeout=2.0):
                    reachable = True
                    detail = f"{host}:{port} accepted a connection"
            except OSError as exc:
                detail = f"{host}:{port} unreachable ({exc.__class__.__name__})"
        else:
            detail = "no base_url configured"

        rows.append({"spec": path.name, "provider": spec.model.provider,
                     "base_url": base, "reachable": reachable, "detail": detail})

    return rows


async def run_suite(
    suite: Suite, base_dir: Path, use_script: bool = True
) -> RunSummary:
    spec_path = base_dir / suite.spec
    workspace = base_dir / "workspace"
    temp_workspace: Path | None = None

    unmet = unmet_requirements(suite)
    if unmet:
        # Reported, not silently dropped. A suite whose runtime is missing has
        # proved nothing; the caller decides whether that is acceptable, and CI
        # asserts it is not.
        return RunSummary(
            suite.name, "?", "?",
            skipped=len(suite.cases),
            error=f"requirements not met: {', '.join(unmet)}",
        )

    try:
        spec = load_spec(spec_path)
    except SpecError as exc:
        return RunSummary(suite.name, "?", "?", error=f"spec load failed: {exc}")

    # Suites whose capabilities have real-world side effects need their own
    # workspace. The shared read-only fixture is not a git repo, so
    # git.commit could not be granted at all — a capability that silently
    # disappears is worse than an obvious setup failure.
    if "git.commit" in spec.capabilities:
        import shutil
        import subprocess
        import tempfile

        temp_workspace = Path(tempfile.mkdtemp(prefix="factory-eval-"))
        seeded = base_dir / "workspace"
        if seeded.exists():
            shutil.copytree(seeded, temp_workspace, dirs_exist_ok=True)

        subprocess.run(
            ["git", "init", "-q", str(temp_workspace)], check=False, capture_output=True
        )
        subprocess.run(
            ["git", "add", "-A"], cwd=str(temp_workspace), check=False, capture_output=True
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=agent-factory",
                "-c",
                "user.email=agent-factory@localhost",
                "commit",
                "-q",
                "-m",
                "eval fixture",
                "--allow-empty",
            ],
            cwd=str(temp_workspace),
            check=False,
            capture_output=True,
        )
        workspace = temp_workspace

    summary = RunSummary(suite.name, spec.name, spec.version)

    try:
        for case in suite.cases:
            result = await run_case(case, str(spec_path), workspace, use_script)
            summary.results.append(result)
            summary.total += 1
            summary.passed += int(result.passed)
            summary.failed += int(not result.passed)
            summary.tokens += result.tokens
            summary.duration_ms += result.duration_ms
            summary.violations += result.violations
    finally:
        if temp_workspace is not None:
            import shutil

            shutil.rmtree(temp_workspace, ignore_errors=True)

    return summary
