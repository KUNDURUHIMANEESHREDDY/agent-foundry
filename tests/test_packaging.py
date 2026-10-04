"""Packaging metadata, and the files a clean checkout actually needs.

The rule this file encodes was learned twice, expensively:

    tests/__init__.py            absent, so `from tests.langfuse_helpers import ...`
                                 resolved locally only because `python -m pytest`
                                 puts the CWD on sys.path. CI runs bare `pytest`
                                 and could not import it at all.
    evals/workspace/config.yaml  excluded by a .gitignore rule added while tidying.
                                 Looked like scratch space. 14 eval cases and a unit
                                 test read it; on a clean checkout reader-boundary
                                 scored 10/12.

Both were fully green locally and red in CI, which is the worst combination
available: the local suite was reporting on a machine the project had never been
tested on. Presence on disk is not evidence that a file is part of the project,
so these tests ask git rather than the filesystem.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
LICENSE = ROOT / "LICENSE"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def _in_git_work_tree() -> bool:
    if shutil.which("git") is None:
        return False
    return _git("rev-parse", "--is-inside-work-tree").stdout.strip() == "true"


requires_git = pytest.mark.skipif(
    not _in_git_work_tree(),
    reason="not a git checkout, so tracked-file state cannot be asserted",
)


#: Individual files that must survive a clean checkout, each with what breaks
#: without it. These stay named because the diagnosis is worth writing down:
#: `tests/__init__.py` and `evals/workspace/config.yaml` each cost a red CI run
#: before anything checked them, and "untracked file" is not the first thing
#: anyone would suspect.
NAMED_FILES = {
    "LICENSE": "the repo is public; without it nobody may legally reuse anything",
    "README.md": "the project has no other description of itself",
    "pyproject.toml": "nothing is installable, including the `factory` CLI",
    ".gitattributes": "LF endings; without it every file churns between OSes",
    "tests/__init__.py": (
        "makes `tests` a package so `from tests.langfuse_helpers import ...` "
        "resolves under bare `pytest`; see the module docstring"
    ),
    "evals/workspace/config.yaml": (
        "read by the reader-boundary eval cases and test_sabotage.py; it looks "
        "like scratch space and is not"
    ),
}

#: Trees whose every file must be tracked. Derived, not enumerated: a hand list
#: of load-bearing paths covered 2 of the 73 that matter and missed `src`
#: entirely, which is the part a clean checkout cannot survive losing.
#:
#: Deriving means a new test file or agent spec is covered the moment it is
#: written, with no list to remember to extend.
SHIPPED_TREES = {
    "src": "the package itself; nothing imports if a module is missing",
    "tests": "a missing test module means that coverage silently vanishes",
    "agents": "eval suites load these specs by path",
    "evals": "a missing suite or fixture turns a check into a no-op",
    ".github/workflows": "the sabotage audit is the point; this runs it",
}

#: Build artefacts that live inside those trees and must NOT be tracked.
_NOT_SHIPPED = ("__pycache__", ".pyc", ".pyo", ".egg-info")


def load_bearing_files() -> dict[str, str]:
    """Every file that must ship, derived from the layout plus the named ones."""
    found: dict[str, str] = dict(NAMED_FILES)

    for tree, reason in SHIPPED_TREES.items():
        root = ROOT / tree
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if any(marker in rel for marker in _NOT_SHIPPED):
                continue
            found.setdefault(rel, reason)

    return found


@requires_git
class TestRequiredFilesSurviveACleanCheckout:
    @pytest.mark.parametrize("relpath", sorted(load_bearing_files()))
    def test_it_exists_on_disk(self, relpath):
        assert (ROOT / relpath).is_file(), f"{relpath} is missing entirely"

    @pytest.mark.parametrize("relpath", sorted(load_bearing_files()))
    def test_git_actually_tracks_it(self, relpath):
        """The check that was missing when both incidents happened.

        `is_file()` passed while the file was untracked, which is exactly the
        state that made the local suite lie.
        """
        result = _git("ls-files", "--error-unmatch", "--", relpath)
        assert result.returncode == 0, (
            f"{relpath} exists but git does not track it, so it is absent from "
            f"every clean checkout. {load_bearing_files()[relpath]}"
        )

    @pytest.mark.parametrize("relpath", sorted(load_bearing_files()))
    def test_no_ignore_rule_would_hide_it(self, relpath):
        """Catches the failure one step earlier than the commit does.

        `--no-index` asks whether an ignore pattern matches the path as though it
        were untracked, so this goes red while the rule is being written rather
        than in CI after it is committed.
        """
        result = _git("check-ignore", "--no-index", "--", relpath)
        assert result.returncode != 0, (
            f"{relpath} is matched by a .gitignore rule, so it will vanish from "
            f"any fresh clone. {load_bearing_files()[relpath]}"
        )

    def test_no_tracked_file_is_hidden_by_an_ignore_rule(self):
        """A tracked-but-ignored file is a trap: present now, gone on re-clone."""
        hidden = _git("ls-files", "-i", "-c", "--exclude-standard").stdout.strip()
        assert not hidden, (
            "these files are tracked but also matched by a .gitignore rule, so "
            f"they would disappear from a fresh clone:\n{hidden}"
        )


class TestTheDerivationCoversMoreThanTheNamedList:
    """Guards the derivation itself.

    The first version of this file hand-listed seven paths and covered 2 of the
    73 that matter -- `src` was absent entirely. If the glob ever stops matching
    anything, the tests above go quietly green and the guarantee is gone. So the
    size of the derived set is itself asserted.
    """

    def test_it_finds_the_package(self):
        derived = load_bearing_files()
        assert any(p.startswith("src/factory/") for p in derived), (
            "derivation missed src/ -- the hand list did too"
        )

    def test_it_finds_every_test_module(self):
        derived = load_bearing_files()
        on_disk = {
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / "tests").rglob("*.py")
            if "__pycache__" not in p.as_posix()
        }
        missing = on_disk - set(derived)
        assert not missing, f"test modules not covered: {sorted(missing)}"

    def test_it_is_much_wider_than_the_named_list(self):
        derived = load_bearing_files()
        assert len(derived) >= 3 * len(NAMED_FILES), (
            f"derived only {len(derived)} paths from {len(NAMED_FILES)} named "
            f"ones -- the trees are not being walked"
        )

    def test_build_artifacts_are_excluded(self):
        """Caches live in those trees and must not be demanded as load-bearing."""
        derived = load_bearing_files()
        junk = [p for p in derived if "__pycache__" in p or p.endswith(".pyc")]
        assert not junk, f"caches treated as load-bearing: {junk}"


class TestLicenseIsDeclaredConsistently:
    """A licence has to be discoverable by a human and by tooling."""

    def test_license_file_is_present_and_not_empty(self):
        assert LICENSE.is_file()
        assert LICENSE.read_text(encoding="utf-8").strip()

    def test_pyproject_declares_a_bare_spdx_identifier(self):
        """
        A bare LICENSE file is invisible to tooling: PyPI, `pip show` and GitHub
        all read package metadata. The identifier must be a plain SPDX expression
        rather than free text, which is what PEP 639's `license = "MIT"` gives.
        """
        declared = re.search(
            r'(?m)^\s*license\s*=\s*"([^"]+)"', PYPROJECT.read_text(encoding="utf-8")
        )
        assert declared, "pyproject declares no license"

        spdx = declared.group(1).strip()
        assert re.fullmatch(r"[A-Za-z0-9.+-]+", spdx), (
            f"{spdx!r} is not a bare SPDX identifier. If this is deliberately "
            f"free text, it is invisible to every tool that reads metadata."
        )

    def test_the_identifier_matches_the_license_text(self):
        """The metadata and the document must agree on which licence this is.

        This is the test that earns its keep: pyproject saying Apache-2.0 while
        LICENSE says MIT produces a package that installs cleanly and lies.
        """
        declared = re.search(
            r'(?m)^\s*license\s*=\s*"([^"]+)"', PYPROJECT.read_text(encoding="utf-8")
        )
        assert declared, "pyproject declares no license"

        spdx = declared.group(1).strip()
        assert spdx in LICENSE.read_text(encoding="utf-8"), (
            f"pyproject declares {spdx!r} but LICENSE does not mention it"
        )

    def test_pyproject_ships_the_license_text_in_the_wheel(self):
        """
        `license = "MIT"` alone embeds no text, so `pip show` reports a licence
        nobody can read. PEP 639 pairs it with `license-files`.
        """
        assert re.search(
            r'(?m)^\s*license-files\s*=\s*\[[^\]]*LICENSE', 
            PYPROJECT.read_text(encoding="utf-8"),
        ), "pyproject declares a licence but does not ship its text"

    def test_the_build_backend_can_understand_the_declaration(self):
        """
        `license = "MIT"` needs setuptools 77+. A lower floor makes the build fail
        for anyone who installs from source, which is a packaging bug that only
        shows up on a machine that is not this one.
        """
        floor = re.search(
            r'(?m)^\s*requires\s*=\s*\[\s*"setuptools>=(\d+)"',
            PYPROJECT.read_text(encoding="utf-8"),
        )
        assert floor, "build-system does not pin a setuptools floor"

        assert int(floor.group(1)) >= 77, (
            f"setuptools>={floor.group(1)} predates PEP 639, so the SPDX "
            f"`license` field will not build. 77 is where it landed."
        )