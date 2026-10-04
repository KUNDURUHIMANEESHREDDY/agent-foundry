"""Evaluation cases and assertions.

Assertions are checked against a finished trace, never against live model text
formatting. Each one is deterministic so a spec change can be diffed against a
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from factory.runtime.agent import RunStatus
from factory.runtime.trace import RunTrace
from factory.spec.agent_spec import Limits


class Check(str, Enum):
    STATUS_IS = "status_is"
    TOOL_CALLED = "tool_called"
    TOOL_NOT_CALLED = "tool_not_called"
    MIN_STEPS = "min_steps"
    MAX_STEPS = "max_steps"
    NO_VIOLATIONS = "no_violations"
    VIOLATIONS_AT_LEAST = "violations_at_least"
    OBSERVATION_OK = "observation_ok"
    OBSERVATION_DENIED = "observation_denied"
    TEXT_CONTAINS = "text_contains"
    TOKENS_UNDER = "tokens_under"
    DURATION_UNDER_MS = "duration_under_ms"
    PATH_ALLOWED = "path_allowed"
    PATH_DENIED = "path_denied"
    HALT_CONTAINS = "halt_contains"
    HALT_REASON_IS = "halt_reason_is"
    OUTPUT_CONTAINS = "output_contains"
    OUTPUT_NOT_CONTAINS = "output_not_contains"
    OUTPUT_TRUNCATED = "output_truncated"
    OUTPUT_UNDER_BYTES = "output_under_bytes"
    STDERR_CONTAINS = "stderr_contains"
    EXIT_CODE_IS = "exit_code_is"
    TIMED_OUT = "timed_out"
    VIOLATION_ABSENT = "violation_absent"
    GRANDCHILD_STOPPED = "grandchild_stopped"
    GRANDCHILD_RAN = "grandchild_ran"


class Assertion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: Check
    value: Any = None
    note: str = ""


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    task: str = Field(min_length=1)
    assertions: list[Assertion] = Field(min_length=1)
    #: Populated by the runner; a scripted model, if provided.
    scripted: list[ScriptedStep] = Field(default_factory=list)
    #: Environment injected into the parent before the case runs, so a case can
    #: assert on scrubbing. Restored afterwards.
    inject_env: dict[str, str] = Field(default_factory=dict)
    #: Where filesystem side-effects land. Needed by checks that cannot be
    #: answered from the trace alone.
    workspace: str | None = None
    #: Overrides the spec's limits for this case only.
    #:
    #: Needed when an orthogonal runtime bound would mask the property under
    #: test. `rejects-oversized-write` asserts the 512KB write cap, but a 512KB
    #: payload is ~128K tokens against the coder spec's 60K token ceiling, so
    #: the runtime halted first and the capability was never reached. That is
    #: correct runtime behaviour — the cap was simply unreachable through a run
    #: with that budget. Raising the limit here isolates the capability.
    limits: Limits | None = None


class ScriptedStep(BaseModel):
    """A deterministic model turn, so a case runs without a model server."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    tool_calls: list[ScriptedToolCall] = Field(default_factory=list)


class ScriptedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    spec: str
    description: str = ""
    cases: list[EvalCase] = Field(min_length=1)


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    steps: int = 0
    tokens: int = 0
    duration_ms: int = 0
    violations: int = 0
    status: str = ""
    halt_reason: str | None = None
    error: str | None = None


# ── assertion evaluation ──────────────────────────────────────────────


def evaluate(case: EvalCase, trace: RunTrace) -> CaseResult:
    failures: list[str] = []

    for assertion in case.assertions:
        ok, detail = _check(assertion, trace, case)
        if not ok:
            label = assertion.check.value
            failures.append(f"{label}({assertion.value!r}): {detail}")

    violations = sum(len(s.violations) for s in trace.steps)
    duration = sum(s.duration_ms for s in trace.steps)

    return CaseResult(
        case_id=case.id,
        passed=not failures,
        failures=failures,
        steps=len(trace.steps),
        tokens=trace.total_tokens,
        duration_ms=duration,
        violations=violations,
        status=trace.status,
        halt_reason=trace.halt_reason,
    )


def _called_tools(trace: RunTrace) -> list[str]:
    return [tc["name"] for s in trace.steps for tc in s.tool_calls]


def _observations(trace: RunTrace) -> list[dict[str, Any]]:
    return [o for s in trace.steps for o in s.observations]


def _check(
    assertion: Assertion, trace: RunTrace, case: EvalCase | None = None
) -> tuple[bool, str]:
    c = assertion.check
    v = assertion.value
    tools = _called_tools(trace)
    obs = _observations(trace)

    if c is Check.STATUS_IS:
        return trace.status == v, f"status was {trace.status} (halt: {trace.halt_reason})"

    if c is Check.TOOL_CALLED:
        return v in tools, f"tools called were {tools}"

    if c is Check.TOOL_NOT_CALLED:
        return v not in tools, f"{v} was called"

    if c is Check.MIN_STEPS:
        return len(trace.steps) >= int(v), f"{len(trace.steps)} steps < {v}"

    if c is Check.MAX_STEPS:
        return len(trace.steps) <= int(v), f"{len(trace.steps)} steps > {v}"

    if c is Check.NO_VIOLATIONS:
        n = sum(len(s.violations) for s in trace.steps)
        return n == 0, f"{n} violation(s) recorded"

    if c is Check.VIOLATIONS_AT_LEAST:
        n = sum(len(s.violations) for s in trace.steps)
        return n >= int(v), f"{n} violation(s), expected >= {v}"

    if c is Check.OBSERVATION_OK:
        hits = [o for o in obs if o.get("ok")]
        if v is False:
            return not hits, f"{len(hits)} observation(s) unexpectedly succeeded"
        return bool(hits), f"{len(obs)} observation(s), {len(hits)} succeeded"

    if c is Check.OBSERVATION_DENIED:
        hits = [o for o in obs if not o.get("ok")]
        return bool(hits), "no failed observation recorded"

    if c is Check.TEXT_CONTAINS:
        return str(v).lower() in trace.final_text.lower(), (
            f"final text was {trace.final_text[:120]!r}"
        )

    if c is Check.TOKENS_UNDER:
        return trace.total_tokens < int(v), f"{trace.total_tokens} tokens >= {v}"

    if c is Check.DURATION_UNDER_MS:
        total = sum(s.duration_ms for s in trace.steps)
        return total < int(v), f"{total}ms >= {v}"

    if c is Check.PATH_ALLOWED:
        hits = [o for o in obs if o.get("ok") and o.get("path") == v]
        return bool(hits), f"no successful read of {v!r}"

    if c is Check.PATH_DENIED:
        hits = [
            o for o in obs
            if not o.get("ok") and str(v) in str(o.get("error", ""))
        ]
        return bool(hits), f"no denial mentioning {v!r}"

    if c is Check.HALT_CONTAINS:
        reason = trace.halt_reason or ""
        return str(v) in reason, f"halt_reason was {reason!r}"

    if c is Check.HALT_REASON_IS:
        return trace.halt_reason == v, f"halt_reason was {trace.halt_reason!r}"

    if c is Check.OUTPUT_CONTAINS:
        return any(str(v) in str(o.get("stdout", "")) for o in obs), (
            f"no stdout containing {v!r}"
        )

    if c is Check.OUTPUT_NOT_CONTAINS:
        leaked = [o for o in obs if str(v) in str(o.get("stdout", ""))]
        return not leaked, f"{len(leaked)} observation(s) leaked {v!r}"

    if c is Check.OUTPUT_TRUNCATED:
        return any(o.get("truncated") for o in obs), "no observation was truncated"

    if c is Check.OUTPUT_UNDER_BYTES:
        sizes = [len(str(o.get("stdout", "")).encode("utf-8")) for o in obs]
        biggest = max(sizes) if sizes else 0
        return biggest < int(v), f"largest stdout was {biggest} bytes >= {v}"

    if c is Check.STDERR_CONTAINS:
        return any(str(v) in str(o.get("stderr", "")) for o in obs), (
            f"no stderr containing {v!r}"
        )

    if c is Check.EXIT_CODE_IS:
        return any(o.get("exit_code") == v for o in obs), (
            f"exit codes were {[o.get('exit_code') for o in obs]}"
        )

    if c is Check.TIMED_OUT:
        return any(o.get("timed_out") for o in obs), "no observation reported a timeout"

    if c is Check.VIOLATION_ABSENT:
        n = sum(len(s.violations) for s in trace.steps)
        return n == 0, f"{n} violation(s) recorded"

    if c in (Check.GRANDCHILD_STOPPED, Check.GRANDCHILD_RAN):
        if case is None or not case.workspace:
            return False, "grandchild checks require case.workspace to be set"
        return _grandchild_check(c, case)

    return False, f"unknown check {c}"


def _grandchild_check(c: Check, case: EvalCase) -> tuple[bool, str]:
    """Confirm a spawned grandchild was killed, by watching a marker file.

    A trace cannot show this: the child process outlives the trace unless the
    tree is killed. Measuring the filesystem is the only honest way to assert it.
    """
    import time
    from pathlib import Path

    marker = Path(case.workspace) / "tree_marker.txt"
    if not marker.exists():
        return False, "grandchild never wrote to the marker (test may be vacuous)"

    first = marker.stat().st_size
    time.sleep(1.5)
    second = marker.stat().st_size

    if c is Check.GRANDCHILD_RAN:
        return first > 0, f"marker was {first} bytes; grandchild may not have run"

    grew = second > first
    return (not grew), f"marker grew {first} -> {second}; tree was not killed"
