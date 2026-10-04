"""Live mode must never quietly become a scripted run.

This is the load-bearing assumption underneath live-model qualification, and
nothing asserted it. Every test in the suite calls `run_suite(..., use_script=True)`,
so if `--live` ever fell back to `ScriptedAdapter` on failure -- an easy,
attractive "robustness" change -- every case would pass, the eval report would
read 100%, and the qualification would have proved nothing at all.

A qualification run that silently scripts is worse than no run, because it is
believed. So the fallback is made impossible and asserted from three angles:

    - structurally: the scripted adapter is never even constructed
    - behaviourally: an unreachable model produces model_error, not a pass
    - by trace: no tool call appears, because nothing chose one
"""

from __future__ import annotations

from pathlib import Path

import pytest

from factory.eval import runner
from factory.eval.case import Suite
from factory.eval.runner import run_case, run_suite
from factory.runtime.agent import RunStatus

EVALS = Path(__file__).resolve().parents[1] / "evals"

#: Points at a port nothing listens on, so the model call fails.
DEAD_SPEC = """
name: live-probe
version: 0.1.0
description: a spec whose model cannot be reached
system_prompt: You are a probe.
capabilities: []
model:
  provider: openai_compat
  name: nothing
  base_url: http://127.0.0.1:1/v1
  temperature: 0.0
  context_window: 4096
  max_output_tokens: 64
limits:
  max_steps: 2
  max_tokens: 1000
  max_tool_calls: 2
  step_timeout_s: 5
"""

CASE = {
    "id": "live-probe-case",
    "task": "Say hello.",
    "assertions": [{"check": "status_is", "value": "success"}],
}


@pytest.fixture
def spec_and_case(tmp_path: Path):
    spec = tmp_path / "probe.yaml"
    spec.write_text(DEAD_SPEC, encoding="utf-8")

    case = tmp_path / "suite.yaml"
    case.write_text(
        f"name: probe\nspec: probe.yaml\ncases:\n"
        f"  - id: {CASE['id']}\n"
        f"    task: {CASE['task']!r}\n"
        f"    assertions:\n      - check: status_is\n        value: success\n",
        encoding="utf-8",
    )
    suite = Suite.model_validate(
        {"name": "probe", "spec": "probe.yaml",
         "cases": [{"id": CASE["id"], "task": CASE["task"],
                    "assertions": [{"check": "status_is", "value": "success"}]}]}
    )
    return str(spec), suite, tmp_path


class TestLiveModeNeverScripts:
    async def test_the_scripted_adapter_is_never_constructed(self, spec_and_case, monkeypatch):
        """The direct structural claim.

        `scripted_adapter` is the only thing that can produce a scripted answer.
        Making it explode proves live mode never reaches for one.
        """
        spec_path, suite, workspace = spec_and_case

        def explode(case):
            raise AssertionError(
                "live mode constructed a ScriptedAdapter -- a --live run would "
                "report scripted results as if a model had produced them"
            )

        monkeypatch.setattr(runner, "scripted_adapter", explode)

        result = await run_case(suite.cases[0], spec_path, workspace, use_script=False)
        assert result.status != RunStatus.SUCCESS

    async def test_an_unreachable_model_is_a_model_error(self, spec_and_case):
        """Behavioural: the failure is visible, not absorbed."""
        spec_path, suite, workspace = spec_and_case

        result = await run_case(suite.cases[0], spec_path, workspace, use_script=False)

        assert result.status == RunStatus.MODEL_ERROR, (
            f"expected model_error with no model reachable, got {result.status}"
        )
        assert result.passed is False

    async def test_no_tool_call_appears_in_the_trace(self, spec_and_case):
        """By trace: nothing chose a tool, because nothing chose anything."""
        spec_path, suite, workspace = spec_and_case

        result = await run_case(suite.cases[0], spec_path, workspace, use_script=False)

        assert result.steps <= 1, (
            "a run with no model should stop at the first model call"
        )
        assert result.violations == 0

    async def test_the_halt_reason_names_the_model(self, spec_and_case):
        """The report has to say *why*, or a reader cannot act on it."""
        spec_path, suite, workspace = spec_and_case

        result = await run_case(suite.cases[0], spec_path, workspace, use_script=False)

        assert result.halt_reason and "model" in result.halt_reason.lower()


class TestScriptedModeStillScripts:
    """The other half: live mode must not have broken the normal path."""

    async def test_a_scripted_case_still_passes_offline(self, spec_and_case):
        spec_path, suite, workspace = spec_and_case

        # Same spec, same case, but scripted. It cannot reach the model either --
        # so the only thing that makes this pass is the script.
        scripted = Suite.model_validate(
            {
                "name": "probe",
                "spec": "probe.yaml",
                "cases": [
                    {
                        "id": "scripted-probe",
                        "task": "Say hello.",
                        "scripted": [{"text": "hello"}],
                        "assertions": [{"check": "status_is", "value": "success"}],
                    }
                ],
            }
        )

        result = await run_case(scripted.cases[0], spec_path, workspace, use_script=True)
        assert result.passed is True, (
            "scripted mode regressed: a case with a script must pass without a "
            "reachable model"
        )


class TestASuiteCanDeclareItNeedsALiveModel:
    """So a live suite skips loudly instead of running scripted.

    The same machinery as `requires: [container]`, for the other external
    dependency: a model. Without it, someone adding live cases to a suite that
    also has scripted ones would get a mixed, meaningless report.
    """

    def test_the_requirement_is_recognised(self, monkeypatch):
        suite = Suite.model_validate(
            {"name": "live", "spec": "x.yaml", "requires": ["live-model"],
             "cases": [{"id": "a", "task": "t",
                        "assertions": [{"check": "status_is", "value": "success"}]}]}
        )
        # No model adapter is wired for the live-model requirement, so it reports
        # unmet -- which is the safe direction.
        assert "live-model" in runner.unmet_requirements(suite)

    def test_an_unknown_requirement_is_unmet_too(self):
        """A typo must not silently disable the gate."""
        suite = Suite.model_validate(
            {"name": "live", "spec": "x.yaml", "requires": ["live-modle"],
             "cases": [{"id": "a", "task": "t",
                        "assertions": [{"check": "status_is", "value": "success"}]}]}
        )
        assert runner.unmet_requirements(suite) == ["live-modle"]


class TestTheRunnerDoesNotMixModes:
    async def test_use_script_reaches_every_case(self, spec_and_case):
        """A suite run live must not script any case in it."""
        spec_path, suite, workspace = spec_and_case

        async def run_all(*_a, **_k):  # pragma: no cover - not reached
            raise AssertionError

        summary = await run_suite(suite, workspace, use_script=False)
        assert summary.failed == 1, (
            "running a suite live with no model must fail its cases, not pass them"
        )
        assert summary.passed == 0