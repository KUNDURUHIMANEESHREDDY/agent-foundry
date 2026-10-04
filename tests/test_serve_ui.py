"""`factory serve --ui` must refuse the unsafe combinations, loudly.

The refusals are the feature. Serving a frontend from an API that executes code
has three ways to go wrong, and all three are checked here rather than left to
the operator:

  - a non-loopback bind, which reaches the network
  - a UI directory that does not exist, which serves a bare API while the
    operator believes a UI is there
  - no CORS policy, which must stay absent even once a UI is mounted
"""

from __future__ import annotations

import argparse

import pytest

from factory.cli import cmd_serve


def _args(**kw):
    base = {"host": "127.0.0.1", "port": 8000, "ui": None}
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def ui_dir(tmp_path):
    d = tmp_path / "dist"
    d.mkdir()
    (d / "index.html").write_text("<!doctype html>", encoding="utf-8")
    return d


@pytest.fixture
def no_uvicorn(monkeypatch):
    """Refuse to actually start a server from a test."""
    import sys
    import types

    def boom(*a, **k):
        raise AssertionError("a test tried to start a real server")

    fake = types.ModuleType("uvicorn")
    fake.run = boom
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    return fake


class TestUnsafeBindsAreRefused:
    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.50"])
    def test_a_ui_on_a_network_bind_exits_non_zero(self, ui_dir, no_uvicorn, host, monkeypatch, capsys):
        monkeypatch.setenv("FACTORY_API_INSECURE", "1")

        rc = cmd_serve(_args(host=host, ui=str(ui_dir)))

        assert rc == 2
        assert "refusing" in capsys.readouterr().err.lower()


class TestBadUiDirectoriesAreRefused:
    def test_a_missing_directory_exits_before_serving(self, no_uvicorn, tmp_path, capsys):
        rc = cmd_serve(_args(ui=str(tmp_path / "nope")))

        assert rc == 2
        assert "--ui" in capsys.readouterr().err

    def test_a_directory_without_index_is_refused(self, no_uvicorn, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()

        rc = cmd_serve(_args(ui=str(empty)))

        assert rc == 2
        assert "index.html" in capsys.readouterr().err


class TestTheUiIsAnnounced:
    def test_it_says_same_origin_and_no_cors(self, ui_dir, no_uvicorn, capsys):
        # Reaches the announce path, then fails on the stubbed uvicorn run.
        with pytest.raises(AssertionError):
            cmd_serve(_args(ui=str(ui_dir)))

        err = capsys.readouterr().err
        assert "Same origin" in err
        assert "no CORS is configured" in err
        assert "rebound DNS" in err, (
            "the announcement must mention the DNS-rebinding guard, because "
            "turning auth off is only safe while that guard is what stands there"
        )


class TestWithoutAUiNothingChanges:
    def test_the_plain_path_does_not_mount_anything(self, no_uvicorn, monkeypatch):
        from factory.api import ui as ui_mod

        called = []
        monkeypatch.setattr(ui_mod, "mount_ui", lambda *a, **k: called.append(a))

        with pytest.raises(AssertionError):
            cmd_serve(_args(ui=None))

        assert called == [], "a UI mount happened when --ui was not given"