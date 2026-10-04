"""Report formatting and baseline comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from factory.eval.runner import RunSummary

# Checks that indicate a security boundary was crossed, not merely a quality miss.
SECURITY_CHECKS = {
    "no_violations",
    "violations_at_least",
    "observation_denied",
    "path_denied",
    "tool_not_called",
    "halt_contains",
    "halt_reason_is",
}


def security_failures(summary: RunSummary) -> list[str]:
    """Cases where a security assertion failed.

    Tracked separately from ordinary failures: a quality regression is annoying,
    a boundary failure is a release blocker.
    """
    out = []
    for result in summary.results:
        for failure in result.failures:
            check = failure.split("(", 1)[0]
            if check in SECURITY_CHECKS:
                out.append(f"{result.case_id}: {failure}")
    return out


def to_json(summaries: list[RunSummary]) -> str:
    return json.dumps(
        {
            "suites": [s.to_dict() for s in summaries],
            "totals": {
                "cases": sum(s.total for s in summaries),
                "passed": sum(s.passed for s in summaries),
                "failed": sum(s.failed for s in summaries),
                "tokens": sum(s.tokens for s in summaries),
                "violations": sum(s.violations for s in summaries),
                "security_failures": sum(len(security_failures(s)) for s in summaries),
            },
        },
        indent=2,
    )


def format_text(summaries: list[RunSummary]) -> str:
    lines: list[str] = []

    for s in summaries:
        icon = "PASS" if s.failed == 0 and not s.error else "FAIL"
        lines.append(f"[{icon}] {s.suite}  ({s.spec_name}@{s.spec_version})")
        if s.error:
            lines.append(f"    error: {s.error}")
            continue
        for r in s.results:
            mark = "  ok  " if r.passed else "  FAIL"
            line = f"{mark} {r.case_id}  steps={r.steps} tokens={r.tokens} status={r.status}"
            if r.halt_reason:
                line += f" halt={r.halt_reason}"
            lines.append(line)
            for failure in r.failures:
                lines.append(f"        - {failure}")
        lines.append(
            f"    pass_rate={s.pass_rate:.0%} "
            f"({s.passed}/{s.total}) tokens={s.tokens} violations={s.violations}"
        )

    sec = sum(len(security_failures(s)) for s in summaries)
    total = sum(s.total for s in summaries)
    passed = sum(s.passed for s in summaries)
    lines.append("")
    lines.append(f"TOTAL {passed}/{total} cases, {sec} security assertion failure(s)")
    if sec:
        lines.append("BLOCKED: a security assertion failed. Do not ship this change.")
    return "\n".join(lines)


def save_baseline(summaries: list[RunSummary], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_json(summaries), encoding="utf-8")


def compare(baseline: dict[str, Any], summaries: list[RunSummary]) -> str:
    """Diff a run against a saved baseline, per suite and per case."""
    base_suites = {s["suite"]: s for s in baseline.get("suites", [])}
    lines: list[str] = []

    for s in summaries:
        now = s.to_dict()
        before = base_suites.get(s.suite)

        if before is None:
            lines.append(f"[NEW ] {s.suite}: {now['passed']}/{now['total']}")
            continue

        rate_before = before.get("pass_rate", 0.0)
        delta = now["pass_rate"] - rate_before
        arrow = "same" if delta == 0 else ("better" if delta > 0 else "WORSE")
        spec_before = before.get("spec", "?")
        spec_now = now.get("spec", "?")
        spec_note = "" if spec_before == spec_now else f"  ({spec_before} -> {spec_now})"
        lines.append(
            f"[{arrow:>5}] {s.suite}: {rate_before:.0%} -> {now['pass_rate']:.0%} "
            f"(+{now['tokens'] - before.get('tokens', 0)} tokens){spec_note}"
        )

        before_cases = {c["id"]: c for c in before.get("cases", [])}
        now_cases = {c["id"]: c for c in now["cases"]}
        for cid, case in now_cases.items():
            old = before_cases.get(cid)
            if old is None:
                lines.append(f"         + new case {cid}: {'ok' if case['passed'] else 'FAIL'}")
            elif old["passed"] != case["passed"]:
                state = "fixed" if case["passed"] else "REGRESSED"
                lines.append(f"         ! {cid}: {state}")
        for cid in before_cases:
            if cid not in now_cases:
                lines.append(f"         - removed case {cid}")

    return "\n".join(lines)
