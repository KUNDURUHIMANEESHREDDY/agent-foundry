from __future__ import annotations

import json
from pathlib import Path

import pytest

from factory.eval import (
    Assertion,
    Check,
    EvalCase,
    SuiteError,
    compare,
    format_text,
    load_all,
    load_suite,
    run_suite,
    save_baseline,
    security_failures,
    to_json,
)
from factory.models.base import ModelResponse, ToolCall
from factory.models.fake import ScriptedAdapter
from factory.runtime.agent import AgentRuntime, RunStatus
from factory.runtime.trace import MemoryTraceStore
from factory.spec.agent_spec import AgentSpec, Limits, ModelRef

EVALS = Path(__file__).resolve().parents[1] / "evals"


def trace_from(steps, status="success", final="", halt=None):
    from factory.runtime.trace import RunTrace, StepRecord

    t = RunTrace(id="r1", agent="a", agent_version="0.1.0", task="t", status=status)
    t.final_text = final
    t.halt_reason = halt
    t.steps = steps
    return t


def step(tool_calls=(), observations=(), violations=(), text="", tokens=(0, 0)):
    from factory.runtime.trace import StepRecord

    return StepRecord(
        index=1,
        model="scripted",
        prompt_tokens=tokens[0],
        completion_tokens=tokens[1],
        tool_calls=[{"name": n, "arguments": {}} for n in tool_calls],
        observations=list(observations),
        violations=list(violations),
        text=text,
    )


class TestChecks:
    def test_status_is(self):
        from factory.eval.case import evaluate

        case = EvalCase(id="c", task="t", assertions=[Assertion(check=Check.STATUS_IS, value="success")])
        assert evaluate(case, trace_from([step()])).passed
        assert not evaluate(case, trace_from([step()], status="max_steps_exceeded")).passed

    def test_tool_called_and_not(self):
        from factory.eval.case import evaluate

        called = EvalCase(id="c", task="t", assertions=[Assertion(check=Check.TOOL_CALLED, value="filesystem.read")])
        not_called = EvalCase(id="c", task="t", assertions=[Assertion(check=Check.TOOL_NOT_CALLED, value="filesystem.read")])
        tr = trace_from([step(tool_calls=["filesystem.read"])])

        assert evaluate(called, tr).passed
        assert not evaluate(not_called, tr).passed

    def test_observation_ok_inverts_on_false(self):
        from factory.eval.case import evaluate

        tr = trace_from([step(observations=[{"ok": True, "path": "a"}])])
        want_fail = EvalCase(id="c", task="t", assertions=[Assertion(check=Check.OBSERVATION_OK, value=False)])
        want_pass = EvalCase(id="c", task="t", assertions=[Assertion(check=Check.OBSERVATION_OK, value=True)])

        assert not evaluate(want_fail, tr).passed
        assert evaluate(want_pass, tr).passed

    def test_path_denied_matches_error_text(self):
        from factory.eval.case import evaluate

        denial = {"ok": False, "error": "Path '../x' resolves outside the workspace root"}
        tr = trace_from([step(observations=[denial])])
        case = EvalCase(id="c", task="t", assertions=[Assertion(check=Check.PATH_DENIED, value="outside the workspace root")])
        assert evaluate(case, tr).passed

    def test_path_denied_does_not_match_unrelated_error(self):
        from factory.eval.case import evaluate

        tr = trace_from([step(observations=[{"ok": False, "error": "No such file"}])])
        case = EvalCase(id="c", task="t", assertions=[Assertion(check=Check.PATH_DENIED, value="outside the workspace root")])
        assert not evaluate(case, tr).passed

    def test_halt_contains(self):
        from factory.eval.case import evaluate

        tr = trace_from([step()], status="max_steps_exceeded", halt="exceeded max_steps=8")
        case = EvalCase(id="c", task="t", assertions=[Assertion(check=Check.HALT_CONTAINS, value="max_steps")])
        assert evaluate(case, tr).passed

    def test_no_violations(self):
        from factory.eval.case import evaluate

        clean = trace_from([step()])
        dirty = trace_from([step(violations=[{"capability": "x", "reason": "y"}])])
        case = EvalCase(id="c", task="t", assertions=[Assertion(check=Check.NO_VIOLATIONS)])

        assert evaluate(case, clean).passed
        assert not evaluate(case, dirty).passed

    def test_tokens_under(self):
        from factory.eval.case import evaluate

        tr = trace_from([step(tokens=(100, 50))])
        assert evaluate(EvalCase(id="c", task="t", assertions=[Assertion(check=Check.TOKENS_UNDER, value=200)]), tr).passed
        assert not evaluate(EvalCase(id="c", task="t", assertions=[Assertion(check=Check.TOKENS_UNDER, value=100)]), tr).passed

    def test_text_contains_is_case_insensitive(self):
        from factory.eval.case import evaluate

        tr = trace_from([step()], final="The File Does Not EXIST here")
        case = EvalCase(id="c", task="t", assertions=[Assertion(check=Check.TEXT_CONTAINS, value="does not exist")])
        assert evaluate(case, tr).passed


class TestSuiteLoading:
    def test_loads_shipped_suite(self):
        suite = load_suite(EVALS / "reader-boundary.yaml")
        assert suite.name == "reader-boundary"
        assert len(suite.cases) >= 10

    def test_every_case_has_assertions(self):
        suite = load_suite(EVALS / "reader-boundary.yaml")
        for case in suite.cases:
            assert case.assertions, f"{case.id} has no assertions"

    def test_every_case_has_a_script(self):
        """A case with no script cannot exercise tool behaviour."""
        suite = load_suite(EVALS / "reader-boundary.yaml")
        for case in suite.cases:
            assert case.scripted, f"{case.id} has no scripted turns"

    def test_load_all_finds_suites(self):
        suites = load_all(EVALS)
        assert suites

    def test_missing_suite(self, tmp_path):
        with pytest.raises(SuiteError, match="not found"):
            load_suite(tmp_path / "nope.yaml")

    def test_missing_directory(self, tmp_path):
        with pytest.raises(SuiteError, match="No suite directory"):
            load_all(tmp_path / "nope")

    def test_reports_bad_assertion_field(self, tmp_path):
        p = tmp_path / "s.yaml"
        p.write_text(
            "name: s\nspec: ../agents/reader.yaml\ncases:\n"
            "  - id: c\n    task: t\n    assertions:\n      - check: nonsense\n        value: 1\n",
            encoding="utf-8",
        )
        with pytest.raises(SuiteError, match="check"):
            load_suite(p)


class TestRunner:
    @pytest.mark.asyncio
    async def test_runs_every_case(self):
        suite = load_suite(EVALS / "reader-boundary.yaml")
        summary = await run_suite(suite, EVALS, use_script=True)

        assert summary.total == len(suite.cases)
        assert summary.passed == summary.total
        assert summary.error is None

    @pytest.mark.asyncio
    async def test_reports_missing_spec_instead_of_crashing(self, tmp_path):
        p = tmp_path / "s.yaml"
        p.write_text(
            "name: broken\nspec: ../agents/nope.yaml\ncases:\n"
            "  - id: c\n    task: t\n    assertions:\n      - check: status_is\n        value: success\n",
            encoding="utf-8",
        )
        summary = await run_suite(load_suite(p), tmp_path, use_script=True)
        assert summary.error
        assert "spec load failed" in summary.error

    @pytest.mark.asyncio
    async def test_aggregates_tokens_and_violations(self):
        suite = load_suite(EVALS / "reader-boundary.yaml")
        summary = await run_suite(suite, EVALS, use_script=True)
        # scripted cases legitimately record denials as violations
        assert summary.violations > 0
        assert summary.pass_rate == 1.0


class TestRegistryKeyedRuns:
    """The ratchet: an eval run is only meaningful against an agent+version."""

    @pytest.mark.asyncio
    async def test_suite_reports_which_agent_it_ran(self):
        suite = load_suite(EVALS / "reader-boundary.yaml")
        summary = await run_suite(suite, EVALS, use_script=True)

        assert summary.spec_name == "reader"
        assert summary.spec_version == "0.1.0"

    @pytest.mark.asyncio
    async def test_baseline_distinguishes_agent_versions(self, tmp_path):
        from factory.registry import AgentRegistry
        from factory.spec.loader import load_spec

        suite = load_suite(EVALS / "reader-boundary.yaml")
        reg = AgentRegistry(tmp_path / "agents.db")
        reg.register_file(EVALS.parent / "agents" / "reader.yaml")

        base = await run_suite(suite, EVALS, use_script=True)
        path = tmp_path / "b.json"
        save_baseline([base], path)
        baseline = json.loads(path.read_text(encoding="utf-8"))
        assert baseline["suites"][0]["spec"] == "reader@0.1.0"

        # Same suite, newer agent version -> compare must flag the version move.
        newer = json.loads(json.dumps(baseline))
        newer["suites"][0]["spec"] = "reader@0.2.0"
        newer["suites"][0]["pass_rate"] = 0.75
        newer["suites"][0]["passed"] = 9
        newer["suites"][0]["failed"] = 3

        out = compare(newer, [base])
        assert "reader@0.2.0 -> reader@0.1.0" in out

    @pytest.mark.asyncio
    async def test_regression_on_a_newer_version_is_visible(self, tmp_path):
        suite = load_suite(EVALS / "reader-boundary.yaml")
        good = await run_suite(suite, EVALS, use_script=True)
        path = tmp_path / "b.json"
        save_baseline([good], path)
        baseline = json.loads(path.read_text(encoding="utf-8"))
        baseline["suites"][0]["spec"] = "reader@0.1.0"

        worse = await run_suite(suite, EVALS, use_script=True)
        worse.passed -= 1
        worse.failed += 1
        worse.results[0].passed = False
        worse.results[0].failures = ["path_denied: fabricated"]

        out = compare(baseline, [worse])
        assert "WORSE" in out


class TestReporting:
    @pytest.mark.asyncio
    async def test_security_failures_separated_from_quality(self):
        from factory.eval.runner import RunSummary, CaseResult

        quality = CaseResult("q", passed=False, failures=["text_contains('x'): no"])
        security = CaseResult("s", passed=False, failures=["path_denied('x'): no denial"])
        summary = RunSummary("s", "a", "0.1.0", total=2, passed=0, failed=2, results=[quality, security])

        assert len(security_failures(summary)) == 1

    @pytest.mark.asyncio
    async def test_text_report_flags_block(self):
        from factory.eval.runner import RunSummary, CaseResult

        summary = RunSummary(
            "s", "a", "0.1.0", total=1, passed=0, failed=1,
            results=[CaseResult("c", False, ["path_denied('x'): nope"])],
        )
        out = format_text([summary])
        assert "BLOCKED" in out

    @pytest.mark.asyncio
    async def test_json_roundtrip_and_baseline(self, tmp_path):
        suite = load_suite(EVALS / "reader-boundary.yaml")
        summary = await run_suite(suite, EVALS, use_script=True)

        path = tmp_path / "baseline.json"
        save_baseline([summary], path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["totals"]["cases"] == summary.total

    @pytest.mark.asyncio
    async def test_compare_reports_no_change(self, tmp_path):
        suite = load_suite(EVALS / "reader-boundary.yaml")
        summary = await run_suite(suite, EVALS, use_script=True)
        path = tmp_path / "b.json"
        save_baseline([summary], path)
        baseline = json.loads(path.read_text(encoding="utf-8"))

        out = compare(baseline, [summary])
        assert "same" in out

    @pytest.mark.asyncio
    async def test_compare_flags_regression(self, tmp_path):
        suite = load_suite(EVALS / "reader-boundary.yaml")
        good = await run_suite(suite, EVALS, use_script=True)
        path = tmp_path / "b.json"
        save_baseline([good], path)
        baseline = json.loads(path.read_text(encoding="utf-8"))

        # A run that fails one case must be reported as WORSE.
        broken = await run_suite(suite, EVALS, use_script=True)
        broken.passed -= 1
        broken.failed += 1
        broken.results[0].passed = False
        broken.results[0].failures = ["no_violations: fabricated"]

        out = compare(baseline, [broken])
        assert "WORSE" in out
        assert "REGRESSED" in out
