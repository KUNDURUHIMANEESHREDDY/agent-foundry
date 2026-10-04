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
        """The exit code stays 0 -- the environment may legitimately lack Docker.

        The warning is what stops it being silent, and the CI step is what stops
        it being invisible.
        """
        import inspect

        from factory.cli import cmd_eval

        assert "WARNING" in inspect.getsource(cmd_eval)


class TestTheCiGateExists:
    """The gate is the part that makes skipping safe."""

    WORKFLOW = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    )

    def test_the_workflow_exists(self):
        assert self.WORKFLOW.is_file()

    def test_ci_asserts_nothing_was_skipped(self):
        text = self.WORKFLOW.read_text(encoding="utf-8")
        assert "Containment cases actually ran" in text
        assert "containment was not tested" in text

    def test_the_check_reads_the_json_report(self):
        text = self.WORKFLOW.read_text(encoding="utf-8")
        assert "--json" in text
        assert '"skipped"' in text


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