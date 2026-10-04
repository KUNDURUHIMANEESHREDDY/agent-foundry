"""An eval run must not modify the committed fixture workspace.

`evals/workspace/` looks like scratch space and is not: `config.yaml` is a
committed fixture that the reader-boundary cases and `test_sabotage.py` read.
Treating it as disposable is exactly how a case came to leave
`subprocess-wrote-this.txt` in it, which then made `test_packaging.py` report an
untracked artefact on every subsequent run -- a test failing for a reason that
had nothing to do with the code.

The catch was purely reactive. Nothing asserted the property, so nothing caught
it at the point where it was introduced. These do, by running the suites that
touch the workspace and comparing before and after.

Slower than a unit test on purpose: it is an end-to-end property, and a cheap
proxy for it would be a proxy that does not hold.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from factory.eval.runner import load_all

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "evals" / "workspace"

#: Suites whose cases write into the workspace. `subprocess-vs-container` is the
#: one that regressed, and it costs 0.3s.
#:
#: `python-sandbox` is deliberately absent. Running it here costs 45s -- most of
#: the unit job's runtime -- and none of its 19 cases write to the workspace. The
#: static check below already catches a leftover from any suite, immediately, at
#: a tenth of a second; this test is the proactive complement for the suite whose
#: whole purpose is writing.
WRITING_SUITES = ("subprocess-vs-container",)


def _snapshot() -> set[str]:
    return {p.name for p in FIXTURE.iterdir()} if FIXTURE.is_dir() else set()


@pytest.mark.parametrize("suite_name", WRITING_SUITES)
def test_a_suite_that_writes_leaves_the_fixture_unchanged(suite_name):
    import asyncio

    from factory.eval.runner import run_suite

    suite = next((s for s in load_all(ROOT / "evals") if s.name == suite_name), None)
    if suite is None:
        pytest.skip(f"no suite named {suite_name}")

    before = _snapshot()

    asyncio.run(run_suite(suite, ROOT / "evals"))

    after = _snapshot()
    assert after == before, (
        f"{suite_name} changed the committed fixture workspace: "
        f"added {sorted(after - before)}, removed {sorted(before - after)}. "
        f"Either clean up inside the case, or give the suite its own workspace."
    )


def test_the_fixture_is_not_being_used_as_scratch():
    """Direct statement of the property, independent of any suite's behaviour.

    The assertion above can be satisfied by a suite that simply stops writing.
    This one says the directory itself is not scratch space, which is the rule
    someone editing a case needs to know.
    """
    leftovers = [
        p.name
        for p in FIXTURE.iterdir()
        if p.is_file() and p.name != "config.yaml"
    ]
    assert not leftovers, (
        f"evals/workspace/ is a committed fixture, not scratch space, and holds "
        f"unexpected files: {leftovers}"
    )


def test_the_fixture_is_actually_tracked():
    """If this fails, the directory is untracked and the tests above prove nothing."""
    import subprocess as sp

    tracked = sp.run(
        ["git", "ls-files", "evals/workspace"],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout.split()
    assert tracked, (
        "evals/workspace is not tracked, so nothing here can hold a fixture"
    )


def test_an_untracked_leftover_is_caught_by_packaging():
    """The reactive catch is kept deliberately, and now asserted to exist.

    `test_packaging.py` is what noticed the stray file in the first place. If that
    check were removed, a case could regress silently, so its presence is a test.
    """
    src = (ROOT / "tests" / "test_packaging.py").read_text(encoding="utf-8")
    assert "test_git_actually_tracks_it" in src
    assert "SHIPPED_TREES" in src, (
        "packaging derives what must ship from the tree layout; a derived list "
        "is what caught the stray file"
    )