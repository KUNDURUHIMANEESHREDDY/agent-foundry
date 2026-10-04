"""Tests for the sabotage audit itself.

An audit that cannot fail is worse than no audit, because it reports confidence
it has not earned. These tests pin the properties that make the audit's output
mean something:

  - it never touches the working tree
  - a sabotage whose anchor has drifted is reported BROKEN, never as a pass
  - verdicts distinguish detected / missed / uncovered / broken
  - the report format is legible and its exit status matches its content

The audit is slow (it runs whole eval suites in subprocesses), so these tests
exercise the judgement logic directly with a stubbed runner. End-to-end
behaviour is covered by running `factory sabotage` itself.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from factory.eval import sabotage as sb


def make_project(root: Path) -> Path:
    """A minimal project the sandbox builder can copy."""
    (root / "src" / "factory").mkdir(parents=True)
    (root / "src" / "factory" / "__init__.py").write_text("", encoding="utf-8")
    (root / "evals").mkdir()
    (root / "evals" / "s.yaml").write_text("name: s\ncases: []\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return root


class TestSandboxIsolation:
    def test_sandbox_does_not_alias_the_real_tree(self, tmp_path):
        project = make_project(tmp_path / "project")
        box = sb._make_sandbox(project)
        try:
            target = box / "src" / "factory" / "__init__.py"
            target.write_text("# sabotaged\n", encoding="utf-8")

            original = project / "src" / "factory" / "__init__.py"
            assert original.read_text(encoding="utf-8") == ""
        finally:
            shutil.rmtree(box, ignore_errors=True)

    def test_sandbox_is_outside_the_project(self, tmp_path):
        project = make_project(tmp_path / "project")
        box = sb._make_sandbox(project)
        try:
            assert project not in box.parents
            assert box != project
        finally:
            shutil.rmtree(box, ignore_errors=True)

    def test_pycache_is_not_copied(self, tmp_path):
        project = make_project(tmp_path / "project")
        cache = project / "src" / "factory" / "__pycache__"
        cache.mkdir()
        (cache / "x.pyc").write_bytes(b"\x00")

        box = sb._make_sandbox(project)
        try:
            assert not (box / "src" / "factory" / "__pycache__").exists()
        finally:
            shutil.rmtree(box, ignore_errors=True)

    def test_eval_fixtures_are_copied(self):
        """`evals/workspace` holds a fixture the cases read.

        Excluding it to make cleanup easier silently broke five python-sandbox
        cases and turned the whole audit into "baseline is not green". Cleanup
        is handled by `_force_rmtree` instead.
        """
        project = Path(__file__).resolve().parents[1]
        box = sb._make_sandbox(project)
        try:
            assert (box / "evals" / "workspace" / "config.yaml").exists()
        finally:
            sb._force_rmtree(box)

    def test_every_declared_item_is_copied(self):
        project = Path(__file__).resolve().parents[1]
        box = sb._make_sandbox(project)
        try:
            missing = [
                item for item in sb._COPY
                if (project / item).exists() and not (box / item).exists()
            ]
            assert not missing, f"not copied into sandbox: {missing}"
        finally:
            sb._force_rmtree(box)


class TestCleanup:
    """Sandboxes must actually go away.

    `rmtree(ignore_errors=True)` looks like cleanup and is not: git writes
    `.git/objects` read-only, Windows refuses, and the directory survives. The
    failure is silent, so it needs a test.
    """

    def test_removes_read_only_tree(self, tmp_path):
        import os
        import stat as stat_mod

        tree = tmp_path / "sandbox"
        deep = tree / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "object").write_bytes(b"\x00")
        os.chmod(deep / "object", stat_mod.S_IREAD)

        sb._force_rmtree(tree)

        assert not tree.exists()

    def test_removing_a_missing_tree_is_not_an_error(self, tmp_path):
        sb._force_rmtree(tmp_path / "never-existed")

    def test_cleanup_is_idempotent(self, tmp_path):
        project = make_project(tmp_path / "project")
        box = sb._make_sandbox(project)
        sb._force_rmtree(box)
        sb._force_rmtree(box)  # must not raise

    def test_reports_whether_it_succeeded(self, tmp_path):
        project = make_project(tmp_path / "project")
        box = sb._make_sandbox(project)
        assert sb._force_rmtree(box) is True
        assert sb._force_rmtree(tmp_path / "gone") is True


class TestOrphanReaping:
    """A sabotaged run must not leave processes behind.

    The `process-tree-kill` sabotage removes the cleanup, so `kills-infinite-loop`
    orphans a `while True: pass`. Its parent is gone, so no PID walk finds it —
    18 CPU-burning orphans accumulated over 10 audit runs before the Job Object.
    """

    def test_grandchildren_die_when_the_job_closes(self, tmp_path):
        import subprocess
        import sys

        marker = tmp_path / "still-alive"
        code = (
            "import subprocess, sys, pathlib\n"
            f"p = pathlib.Path({str(marker)!r})\n"
            # A grandchild whose parent exits immediately: nothing can reach it
            # by walking a process tree, because its parent is already gone.
            "subprocess.Popen([sys.executable, '-c',\n"
            f"  'import time, pathlib; time.sleep(3); pathlib.Path({str(marker)!r}).write_text(\"x\")'])\n"
        )

        with sb._KillOnClose() as job:
            parent = subprocess.Popen([sys.executable, "-c", code])
            job.assign(parent)
            parent.wait(timeout=30)

        # Give the orphan a chance to write the marker if it survived.
        import time

        for _ in range(40):
            if marker.exists():
                break
            time.sleep(0.25)

        assert not marker.exists(), (
            "a grandchild outlived the job; the audit would leak processes"
        )

    def test_job_reports_availability(self):
        with sb._KillOnClose() as job:
            assert isinstance(job.available, bool)
        # Closing twice must not raise.
        job.__exit__()


class TestPatchDetection:
    def test_patch_reports_missing_anchor(self, tmp_path):
        project = make_project(tmp_path / "project")
        box = sb._make_sandbox(project)
        try:
            s = sb.Sabotage(
                id="x", file="src/factory/__init__.py",
                old="text that is not present", new="replacement",
                must_fail=(), description="",
            )
            problem, _original = sb._patch(box, s)
            assert problem is not None
            assert "anchor text not found" in problem
        finally:
            shutil.rmtree(box, ignore_errors=True)

    def test_patch_reports_missing_file(self, tmp_path):
        project = make_project(tmp_path / "project")
        box = sb._make_sandbox(project)
        try:
            s = sb.Sabotage(
                id="x", file="src/nope.py", old="a", new="b",
                must_fail=(), description="",
            )
            problem, _ = sb._patch(box, s)
            assert problem is not None and "not found" in problem
        finally:
            shutil.rmtree(box, ignore_errors=True)

    def test_patch_applies_once(self, tmp_path):
        project = make_project(tmp_path / "project")
        init = project / "src" / "factory" / "__init__.py"
        init.write_text("a = 1\nb = 2\na = 3\n", encoding="utf-8")

        box = sb._make_sandbox(project)
        try:
            s = sb.Sabotage(
                id="x", file="src/factory/__init__.py",
                old="a = 1", new="a = 99", must_fail=(), description="",
            )
            problem, original = sb._patch(box, s)
            assert problem is None
            patched = (box / "src" / "factory" / "__init__.py").read_text(
                encoding="utf-8"
            )
            assert patched == "a = 99\nb = 2\na = 3\n"
            assert original == "a = 1\nb = 2\na = 3\n"
        finally:
            shutil.rmtree(box, ignore_errors=True)


class TestCatalogIntegrity:
    def test_ids_are_unique(self):
        ids = [s.id for s in sb.SABOTAGES]
        assert len(ids) == len(set(ids))

    def test_every_sabotage_has_a_description(self):
        for s in sb.SABOTAGES:
            assert s.description.strip()

    def test_covered_by_is_only_used_without_must_fail(self):
        """`covered_by` excuses a missing eval case; it must not mask a broken one.

        If a sabotage lists cases that must fail, the eval suite is the thing
        under test and a unit-test pointer is not evidence.
        """
        for s in sb.SABOTAGES:
            if s.covered_by:
                assert not s.must_fail, f"{s.id} claims both eval and unit coverage"

    def test_anchor_files_exist_in_this_project(self):
        root = Path(__file__).resolve().parents[1]
        for s in sb.SABOTAGES:
            assert (root / s.file).exists(), f"{s.id} targets a missing file"
            source = (root / s.file).read_text(encoding="utf-8")
            assert s.old in source, (
                f"{s.id}: anchor text has drifted from {s.file}. "
                "The audit cannot judge this mitigation until it is updated."
            )

    def test_must_fail_cases_exist_in_some_suite(self):
        root = Path(__file__).resolve().parents[1]
        from factory.eval.runner import load_all

        known = {
            case.id
            for suite in load_all(root / "evals")
            for case in suite.cases
        }
        for s in sb.SABOTAGES:
            for case_id in s.must_fail:
                assert case_id in known, (
                    f"{s.id} expects case '{case_id}', which no suite defines"
                )


class TestReportFormat:
    def _result(self, sabotage, verdict, detected):
        return sb.SabotageResult(
            sabotage=sabotage, patched=True, failed_cases=[],
            unexpected_failures=[], detected=detected, verdict=verdict,
        )

    def _fake(self):
        return sb.Sabotage(
            id="demo", file="a.py", old="x", new="y",
            must_fail=(), description="demo mitigation",
        )

    def test_all_good_reports_pass(self):
        results = [
            self._result(self._fake(), "DETECTED (2 case(s))", True),
            self._result(self._fake(), "GUARDED elsewhere: tests/x.py", True),
        ]
        out = sb.format_report(results, [])
        assert "PASS" in out
        assert "FAIL" not in out

    def test_any_gap_reports_fail(self):
        results = [
            self._result(self._fake(), "DETECTED (2 case(s))", True),
            self._result(self._fake(), "MISSED: refuses-x still passed", False),
        ]
        out = sb.format_report(results, [])
        assert "FAIL" in out
        assert "MISSED" in out

    def test_uncovered_is_a_gap(self):
        results = [self._result(self._fake(), "UNCOVERED: nothing claims this", False)]
        assert "FAIL" in sb.format_report(results, [])

    def test_unattributed_is_a_gap(self):
        results = [
            self._result(self._fake(), "UNATTRIBUTED: 1 case(s) failed", False)
        ]
        assert "FAIL" in sb.format_report(results, [])

    def test_hang_is_a_gap(self):
        results = [
            self._result(self._fake(), "HANG: suite/case never returned", False)
        ]
        out = sb.format_report(results, [])
        assert "FAIL" in out
        assert "HANG" in out

    def test_ungradable_case_is_advisory_not_a_gap(self):
        """Detected, but one case produced no assertion.

        The suite cannot grade that case. That is worth saying, but it is not
        the same as being blind to the mitigation, so it must not fail the run.
        """
        results = [
            self._result(
                self._fake(),
                "DETECTED (6 case(s)); 1 hung ungradable",
                True,
            )
        ]
        out = sb.format_report(results, [])
        assert "PASS" in out
        assert "advisory" in out

    def test_guarded_elsewhere_is_not_a_gap(self):
        results = [
            self._result(self._fake(), "GUARDED elsewhere: tests/x.py", True)
        ]
        assert "PASS" in sb.format_report(results, [])

    def test_baseline_errors_are_surfaced(self):
        out = sb.format_report([], ["baseline is already failing: a, b"])
        assert "errors" in out
        assert "a, b" in out

    def test_leaked_sandboxes_are_reported(self):
        out = sb.format_report([], ["2 sandbox(es) could not be deleted"])
        assert "could not be deleted" in out

    def test_counts_are_reported(self):
        results = [
            self._result(self._fake(), "DETECTED (1 case(s))", True),
            self._result(self._fake(), "MISSED: x still passed", False),
        ]
        out = sb.format_report(results, [])
        assert "sabotages: 2" in out
        assert "detected: 1" in out


class TestCliExitCode:
    """`factory sabotage` must fail the build when the suite has a blind spot."""

    def test_exit_code_matches_health(self, monkeypatch, capsys):
        from factory import cli

        good = sb.SabotageResult(
            sabotage=sb.SABOTAGES[0], patched=True, failed_cases=[],
            unexpected_failures=[], detected=True, verdict="DETECTED (1 case(s))",
        )
        monkeypatch.setattr(cli, "_sabotage_audit", lambda *a, **k: ([good], []), raising=False)
        args = type("A", (), {"suites": "evals", "jobs": 1})()
        assert cli.cmd_sabotage(args) == 0

        bad = sb.SabotageResult(
            sabotage=sb.SABOTAGES[0], patched=True, failed_cases=[],
            unexpected_failures=[], detected=False,
            verdict="UNCOVERED: no eval case claims this",
        )
        monkeypatch.setattr(cli, "_sabotage_audit", lambda *a, **k: ([bad], []), raising=False)
        assert cli.cmd_sabotage(args) == 1