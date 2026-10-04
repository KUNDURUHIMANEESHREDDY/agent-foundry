"""A suite whose runtime is missing must be skipped, reported, and gated.

Adding container isolation made the eval suite environment-dependent for the
first time: `container-isolation` cannot mean anything without a container
runtime. That produced a choice between three bad behaviours, and the project
has to pick the least bad one deliberately:

    fail    The capability refuses, so every containment assertion fails for the
            wrong reason. The baseline goes red on a machine that is fine, and
            "everything is broken" stops being a useful signal.
    pass    Silently skip. CI goes green having tested no containment at all --
            the exact false green this project exists to prevent.
    skip    Report the gap, and gate it in CI where the runtime is known to
            exist. This is what is implemented.

The gate is the part that matters. `tests/test_isolation.py` and the CI workflow
both assert the container suite reported zero skips, so a runtime that quietly
disappears fails the build instead of emptying it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from factory.capabilities import container
from factory.eval.case import Suite
from factory.eval.runner import unmet_requirements

EVALS = Path(__file__).resolve().parents[1] / "evals"
README = EVALS.parent / "README.md"


def _suites() -> list[Suite]:
    from factory.eval.runner import load_all

    return load_all(EVALS)


class TestTheContainerSuiteDeclaresItsRequirement:
    """If this fails, the suite will fail confusingly instead of skipping."""

    def _container_suite(self) -> Suite:
        found = [s for s in _suites() if "container" in s.name]
        assert found, "no container isolation suite exists"
        return found[0]

    def test_it_declares_the_requirement(self):
        assert "container" in self._container_suite().requires

    def test_it_declares_it_on_the_file(self):
        text = (EVALS / "container-isolation.yaml").read_text(encoding="utf-8")
        assert "requires:" in text

    def test_it_has_cases_that_would_notice(self):
        """A suite with no cases is not a suite."""
        assert len(self._container_suite().cases) >= 5

    def test_only_that_suite_requires_a_runtime(self):
        """Every other suite must stay runnable anywhere."""
        needing = [s.name for s in _suites() if s.requires]
        assert needing == ["container-isolation"], (
            f"unexpected suites now need a runtime: {needing}"
        )


class TestUnmetRequirementsAreDetected:
    def _suite(self, requires: list[str]) -> Suite:
        return Suite(
            name="probe",
            spec="../agents/python-worker.yaml",
            requires=requires,
            cases=[
                {
                    "id": "x",
                    "task": "t",
                    "assertions": [{"check": "status_is", "value": "success"}],
                }
            ],
        )

    def test_no_requirements_means_nothing_unmet(self):
        assert unmet_requirements(self._suite([])) == []

    def test_a_met_requirement_is_not_unmet(self, monkeypatch):
        monkeypatch.setattr(container, "probe", lambda *a, **k: "29.8.1")
        assert unmet_requirements(self._suite(["container"])) == []

    def test_a_stopped_runtime_is_unmet(self, monkeypatch):
        def boom(*a, **k):
            raise container.ContainerUnavailable("not answering")

        monkeypatch.setattr(container, "probe", boom)
        assert unmet_requirements(self._suite(["container"])) == ["container"]

    def test_an_unknown_requirement_is_ignored_not_silently_ok(self):
        """A typo in `requires:` must not silently disable the gate.

        It cannot be satisfied and cannot be checked, so it is reported rather
        than treated as met.
        """
        unmet = unmet_requirements(self._suite(["contaienr"]))
        assert unmet == ["contaienr"], (
            "an unrecognised requirement was treated as satisfied, so the suite "
            "would run and fail confusingly instead of being reported"
        )


class TestSkipsAreVisibleAndGated:
    def test_run_summary_records_skips_separately(self):
        """A skip is not a pass and not a failure; it is recorded on its own."""
        from factory.eval.runner import RunSummary

        s = RunSummary("container-isolation", "a", "0.1.0", total=9, skipped=9)
        payload = s.to_dict()

        assert payload["skipped"] == 9
        assert payload["failed"] == 0
        assert payload["passed"] == 0

    def test_a_skipped_suite_does_not_report_100_percent(self):
        from factory.eval.runner import RunSummary

        assert RunSummary("x", "a", "1", total=9, skipped=9).pass_rate == 0.0

    def test_the_text_report_names_the_skip(self):
        from factory.eval.report import format_text
        from factory.eval.runner import RunSummary

        s = RunSummary(
            "container-isolation", "a", "0.1.0",
            total=9, skipped=9, error="requirements not met: container",
        )
        out = format_text([s])
        assert "SKIPPED 9" in out
        assert "container" in out

    def test_the_cli_warns_about_skips(self):
        import inspect

        from factory.cli import cmd_eval

        assert "WARNING" in inspect.getsource(cmd_eval)

    def test_a_skip_makes_the_cli_exit_non_zero(self):
        """Decided deliberately: a suite that proved nothing must not exit 0.

        The tempting alternative is exit 0 with a warning, on the reasoning that
        a laptop without Docker should still be able to run the harness. But this
        tool exists to prove things, and an exit code is how a build reads it.

        `--expect-skips` is the escape hatch, not the default: a runner that
        cannot run containers names the suite rather than having every skip
        forgiven, and a named suite that runs anyway is itself a failure, so the
        declaration cannot quietly widen.
        """
        import argparse

        import factory.eval as eval_pkg
        from factory.cli import cmd_eval
        from factory.eval.case import Suite
        from factory.eval.runner import RunSummary

        suite = Suite(
            name="container-isolation",
            spec="../agents/python-worker.yaml",
            requires=["container"],
            cases=[
                {
                    "id": "x",
                    "task": "t",
                    "assertions": [{"check": "status_is", "value": "success"}],
                }
            ],
        )

        async def skipped(s, base, use_script=True):
            return RunSummary(s.name, "spec", "0.1.0", total=1, skipped=1,
                              error="requirements not met: container")

        async def ran(s, base, use_script=True):
            return RunSummary(s.name, "spec", "0.1.0", total=1, passed=1)

        def _run(expect_skips, runner=skipped):
            originals = (eval_pkg.load_all, eval_pkg.run_suite)
            eval_pkg.load_all = lambda _p: [suite]
            eval_pkg.run_suite = runner
            try:
                return cmd_eval(
                    argparse.Namespace(suites="evals", json=False, live=False,
                                       baseline=None, compare=None,
                                       expect_skips=expect_skips)
                )
            finally:
                eval_pkg.load_all, eval_pkg.run_suite = originals

        assert _run("") == 1, "an undeclared skip must fail the run"
        assert _run("container-isolation") == 0, (
            "a declared skip must not fail the run"
        )
        assert _run("some-other-suite") == 1, (
            "naming a different suite must not excuse this one"
        )
        assert _run("container-isolation", runner=ran) == 1, (
            "a suite expected to skip that ran anyway means the declaration is "
            "stale, and must not pass silently"
        )


class TestTheCiGateExists:
    """The gate is the part that makes skipping safe."""

    WORKFLOW = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    )

    def test_the_workflow_exists(self):
        assert self.WORKFLOW.is_file()

    def test_ci_asserts_nothing_unexpected_was_skipped(self):
        """The gate's original wording; generalised when windows was accounted for.

        It no longer says "containment was not tested" because containment *is*
        legitimately untested on windows-latest. What it still refuses is a skip
        the runner was not supposed to have.
        """
        text = self.WORKFLOW.read_text(encoding="utf-8")
        assert "Containment cases actually ran" in text
        assert "unexpected skip" in text

    def test_the_check_unwraps_the_report_envelope(self):
        """The gate must read `report["suites"]`, not iterate the top level.

        Iterating the envelope yields the key strings, and the first version of
        this gate died on `'str' object has no attribute 'get'` -- in the gate,
        so every build failed for a reason that had nothing to do with the code.
        """
        text = self.WORKFLOW.read_text(encoding="utf-8")
        assert 'report["suites"]' in text

    def test_the_check_requires_the_suite_to_have_run(self):
        """Zero skips is not enough if the suite vanished entirely."""
        text = self.WORKFLOW.read_text(encoding="utf-8")
        assert "container-isolation" in text

    def test_the_gate_uses_the_json_report(self):
        text = self.WORKFLOW.read_text(encoding="utf-8")
        assert "--json" in text
        assert '"skipped"' in text

    def test_both_eval_steps_carry_the_declaration(self):
        """The gate re-runs eval, so it needs the same flag the eval step used.

        Without it the gate blocks on the very skip the matrix permits, and the
        windows job fails while ubuntu passes -- which is exactly what happened.
        """
        text = self.WORKFLOW.read_text(encoding="utf-8")
        assert text.count("--expect-skips") >= 2, (
            "both `factory eval` invocations must pass --expect-skips"
        )
        assert 'EXPECTED_SKIPPED_SUITES" --json' in text, (
            "the gate must pass the matrix value into its own eval run"
        )

    def test_each_runner_declares_what_may_skip(self):
        """Containment is proven on ubuntu; windows has no usable runtime.

        windows-latest cannot run a Linux container image, so the suite skips
        there. Naming it is the point: an *unexpected* skip still fails, so the
        declaration cannot quietly widen, and containment is proven somewhere in
        the matrix, so the build cannot pass without it.
        """
        text = self.WORKFLOW.read_text(encoding="utf-8")
        assert "expect_skips" in text
        assert "container-isolation" in text

    def test_the_gate_checks_both_directions(self):
        """A stale declaration fails too.

        If windows-latest ever grows a usable runtime, `expect_skips` would be
        out of date, and a gate that only checked for unexpected skips would
        keep allowing a suite to stop being tested.
        """
        text = self.WORKFLOW.read_text(encoding="utf-8")
        assert "unexpected skip" in text
        assert "expected to skip but ran" in text

    def test_no_ci_debug_logs_are_tracked(self):
        """They get committed by accident when a failing run is inspected."""
        import subprocess

        tracked = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, cwd=README.parent
        ).stdout.splitlines()
        logs = [p for p in tracked if p.endswith(".log")]
        assert not logs, f"CI logs are tracked: {logs}"

    def test_json_stdout_is_pure_json_even_when_warnings_fire(self):
        """`--json` must be parseable on its own.

        The BLOCKED banner and the skip warning used to go to stdout, so the gate
        hit "Extra data: line 548" the moment anything had something to say.
        They go to stderr now; this asserts the contract, not one warning.
        """
        import inspect

        from factory.cli import cmd_eval

        source = inspect.getsource(cmd_eval)
        assert "sys.stderr if args.json else sys.stdout" in source, (
            "--json output must be machine-readable on its own"
        )


@pytest.mark.parametrize(
    "name", ["container-isolation", "subprocess-vs-container", "python-sandbox"]
)
def test_named_suites_exist(name: str):
    """Named here so a rename shows up as a failure rather than a silent drop."""
    assert name in {s.name for s in _suites()}


def test_the_json_report_round_trips():
    """CI parses this output, so it has to be real JSON with the field present."""
    from factory.eval.runner import RunSummary

    payload = json.loads(json.dumps(RunSummary("x", "a", "1", skipped=2).to_dict()))
    assert payload["skipped"] == 2


def _cli_json(*summaries) -> dict:
    """The real CLI report, not a summary in isolation."""
    from factory.eval.report import to_json

    return json.loads(to_json(list(summaries)))


class TestTheCiGateCanReadTheCliReport:
    """The gate is Python embedded in a workflow, which nothing type-checks.

    Its first version iterated the top-level JSON object as if it were the list
    of suites and died with `'str' object has no attribute 'get'`. That made
    every build fail, and it failed in the gate rather than in the code -- which
    is the shape of bug a test has to cover, because CI would otherwise be the
    only place it could ever appear.

    These run the gate's own expressions against the real report envelope.
    """

    def _summary(self, name: str, skipped: int = 0, error: str | None = None):
        from factory.eval.runner import RunSummary

        return RunSummary(name, "spec", "0.1.0", total=9, skipped=skipped, error=error)

    def test_the_envelope_is_not_a_bare_list(self):
        report = _cli_json(self._summary("container-isolation"))
        assert isinstance(report, dict), (
            "the report is an object with 'suites' and 'totals', not a list"
        )
        assert "suites" in report and "totals" in report

    def test_the_gates_expression_works_on_the_real_envelope(self):
        report = _cli_json(
            self._summary("reader-boundary"),
            self._summary("container-isolation", skipped=9,
                          error="requirements not met: container"),
        )
        suites = report["suites"]
        skipped = sum(s.get("skipped", 0) for s in suites)
        assert skipped == 9

    def test_iterating_the_envelope_directly_is_the_bug(self):
        """Pins why the gate must unwrap: the top level yields strings."""
        report = _cli_json(self._summary("container-isolation"))
        with pytest.raises(AttributeError):
            [s.get("skipped") for s in report]

    def test_a_clean_report_reports_zero_skips(self):
        report = _cli_json(
            self._summary("reader-boundary"), self._summary("container-isolation")
        )
        suites = report["suites"]
        assert sum(s.get("skipped", 0) for s in suites) == 0

    def test_every_summary_carries_the_skipped_field(self):
        report = _cli_json(self._summary("reader-boundary"))
        for suite in report["suites"]:
            assert "skipped" in suite, (
                "a summary without 'skipped' makes the gate's sum raise or "
                "silently ignore that suite"
            )

    def test_totals_are_reported(self):
        report = _cli_json(self._summary("reader-boundary", skipped=2))
        assert report["totals"]["cases"] == 9