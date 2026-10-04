"""The audit's coverage claims must be checkable, not asserted.

`covered_by` is a string naming a unit test. It is how the audit says "no
black-box case can reach this guard, but a test does" -- for a mitigation that
would otherwise read as a hole. Which makes it the easiest claim in the project
to let rot: it is prose in a dataclass, nothing verifies it, and a stale string
turns an admitted gap into a green line in the report.

These tests also pin the flake that made the audit's own result a coin flip.
For a mitigation with no `must_fail`, any unrelated failure used to force an
UNATTRIBUTED verdict, so whether the mitigation counted as covered depended on
whether a timing-sensitive case happened to fail during its run. Observed in CI:
`sabotages: 9 detected: 8`, on `python-sandbox/kills-process-tree-on-timeout`,
which no mitigation needed to fail.
"""

from __future__ import annotations

import pytest

from factory.eval import sabotage as sb


class TestEveryNamedGuardResolves:
    """The live catalogue, checked rather than trusted."""

    def test_there_are_claims_to_check(self):
        assert [s for s in sb.SABOTAGES if s.covered_by], "no mitigation uses covered_by"

    @pytest.mark.parametrize(
        "sabotage", [s for s in sb.SABOTAGES if s.covered_by], ids=lambda s: s.id
    )
    def test_the_named_test_exists(self, sabotage):
        problem = sb._named_guard(sabotage.covered_by)
        assert problem is None, f"{sabotage.id}: {problem}"


class TestGuardVerificationRejectsFalseClaims:
    """The point of the check: a claim that is not true must be refused."""

    def test_a_function_that_does_not_exist(self):
        problem = sb._named_guard(
            "tests/test_write_git.py::test_definitely_not_a_real_test because"
        )
        assert problem is not None
        assert "neither a function nor a class" in problem

    def test_a_file_that_does_not_exist(self):
        problem = sb._named_guard(
            "tests/test_no_such_file.py::test_anything because"
        )
        assert problem is not None
        assert "does not exist" in problem

    def test_prose_with_no_test_id(self):
        problem = sb._named_guard("a unit test covers this somehow")
        assert problem is not None
        assert "does not begin with a test id" in problem

    def test_an_empty_claim(self):
        assert sb._named_guard("") is not None

    def test_a_class_counts_as_a_guard(self, tmp_path):
        """`step-cap-wall-clock` names a class, not a function.

        Requiring a `test_` prefix here would have failed a mitigation that is
        genuinely covered -- a false negative introduced while fixing a flake.
        """
        mod = tmp_path / "tests" / "test_thing.py"
        mod.parent.mkdir(parents=True)
        mod.write_text("class TestSomething:\n    def test_inner(self):\n        pass\n")

        assert sb._named_guard("tests/test_thing.py::TestSomething because", root=tmp_path) is None


class TestCollateralDoesNotDecideCoverage:
    """The flake: unrelated failures must not overrule a verified guard.

    These call the audit's own `_verdict_for_unclaimed`, not a copy of it. An
    earlier version of this file reimplemented the rule inline and passed while
    the real code was reverted -- a test that cannot fail is worse than none.
    """

    GOOD = "tests/test_write_git.py::test_push_is_not_reachable because"

    def _sabotage(self, covered_by):
        return sb.Sabotage(
            id="probe",
            file="src/factory/capabilities/fs_write.py",
            old="x",
            new="y",
            must_fail=(),
            description="probe",
            covered_by=covered_by,
        )

    def _result(self, covered_by, failed=None, hangs=None):
        return sb._verdict_for_unclaimed(
            self._sabotage(covered_by), list(failed or []), list(hangs or [])
        )

    def test_clean_run_is_guarded_elsewhere(self):
        detected, verdict = self._result(self.GOOD)
        assert detected is True
        assert verdict.startswith("GUARDED elsewhere")

    def test_collateral_does_not_flip_it_to_unattributed(self):
        """The regression CI hit: 9 sabotages, 8 detected."""
        detected, verdict = self._result(
            self.GOOD, ["python-sandbox/kills-process-tree-on-timeout"]
        )
        assert detected is True, "collateral wrongly overruled a verified guard"
        assert "UNATTRIBUTED" not in verdict
        assert "attributed elsewhere" in verdict, "collateral must still be reported"

    def test_a_hang_also_does_not_overrule_the_guard(self):
        detected, verdict = self._result(self.GOOD, [], ["reader-boundary/hangs"])
        assert detected is True
        assert "UNATTRIBUTED" not in verdict

    def test_a_broken_claim_is_uncovered_even_with_no_collateral(self):
        detected, verdict = self._result(
            "tests/test_write_git.py::test_gone_missing because"
        )
        assert detected is False
        assert verdict.startswith("UNCOVERED")

    def test_no_claim_at_all_is_uncovered(self):
        detected, _ = self._result(None)
        assert detected is False

    def test_no_claim_and_collateral_is_still_uncovered(self):
        detected, verdict = self._result(None, ["some/case"])
        assert detected is False
        assert verdict.startswith("UNCOVERED")


class TestTheCatalogueStillLinesUp:
    def test_every_sabotage_is_detectable_somehow(self):
        """Either an eval case must fail, or a guard must be named and real."""
        for s in sb.SABOTAGES:
            assert s.must_fail or s.covered_by, f"{s.id} has no way to be detected"
            if not s.must_fail:
                assert sb._named_guard(s.covered_by) is None, s.id

    def test_must_fail_cases_name_real_cases(self):
        """A `must_fail` naming a case that does not exist can never be detected."""
        from factory.eval.runner import load_all
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        known = {c.id for suite in load_all(root / "evals") for c in suite.cases}

        for s in sb.SABOTAGES:
            for case in s.must_fail:
                assert case in known, f"{s.id} must_fail names unknown case {case!r}"