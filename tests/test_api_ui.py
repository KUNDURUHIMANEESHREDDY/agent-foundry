"""The local-app boundary: same-origin UI, loopback host guard, no wildcard CORS.

These tests exist because the obvious way to connect a frontend to this API is
also the dangerous one. Opening CORS would let any page in the user's browser
`POST /run` and execute code on their machine. Serving the UI from the API's own
origin avoids that, and this file asserts the avoidance rather than trusting it.

Nothing here needs a browser or a network: the host guard is a header check, and
the traversal cases are ordinary requests through the ASGI app.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from factory.api import ui


@pytest.fixture
def ui_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dist"
    d.mkdir()
    (d / "index.html").write_text("<!doctype html><title>app</title>", encoding="utf-8")
    (d / "app.js").write_text("console.log('hi')", encoding="utf-8")
    return d


@pytest.fixture
def secret_beside_ui(tmp_path: Path) -> Path:
    """A file outside the UI directory that must never be reachable."""
    (tmp_path / "secret.txt").write_text("do not serve me", encoding="utf-8")
    return tmp_path / "secret.txt"


class TestLoopbackBinding:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]", "LOCALHOST"])
    def test_loopback_binds_are_recognised(self, host):
        assert ui.is_loopback_bind(host) is True

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com", "", "  "])
    def test_everything_else_is_not_loopback(self, host):
        assert ui.is_loopback_bind(host) is False


class TestRefusingUnsafeBinds:
    def test_a_ui_on_a_public_bind_is_refused(self, ui_dir):
        """The bind is checked, not trusted.

        A UI means auth can be off. An unauthenticated API that executes code,
        bound to `0.0.0.0`, hands that to the network.
        """
        from fastapi import FastAPI

        with pytest.raises(ui.UnsafeBindError) as exc:
            ui.mount_ui(FastAPI(), ui_dir, bind_host="0.0.0.0")
        assert "127.0.0.1" in str(exc.value)

    def test_a_missing_directory_is_refused(self, tmp_path):
        from fastapi import FastAPI

        with pytest.raises(ui.MissingUiError):
            ui.mount_ui(FastAPI(), tmp_path / "nope", bind_host="127.0.0.1")

    def test_a_directory_without_index_is_refused(self, tmp_path):
        from fastapi import FastAPI

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ui.MissingUiError) as exc:
            ui.mount_ui(FastAPI(), empty, bind_host="127.0.0.1")
        assert "index.html" in str(exc.value)


class TestHostGuard:
    @pytest.mark.parametrize("value", ["127.0.0.1", "127.0.0.1:8000", "localhost:5173",
                                       "localhost", "[::1]:8000", "::1"])
    def test_loopback_hosts_are_allowed(self, value):
        assert ui.host_is_allowed(value) is True

    @pytest.mark.parametrize("value", ["evil.com", "evil.com:8000", "127.0.0.1.evil.com",
                                       "localhost.evil.com", "", None])
    def test_everything_else_is_refused(self, value):
        """DNS rebinding is the attack: the name resolves to us, the header does not."""
        assert ui.host_is_allowed(value) is False

    def test_a_lookalike_host_is_not_mistaken_for_loopback(self):
        """`localhost.evil.com` resolves elsewhere and must not pass a substring test."""
        assert ui.host_is_allowed("localhost.evil.com") is False

    def test_host_parsing_handles_the_three_forms(self):
        assert ui.host_name("example.com:8080") == "example.com"
        assert ui.host_name("[::1]:8080") == "::1"
        assert ui.host_name("example.com") == "example.com"


class TestTheGuardIsActuallyWired:
    def test_a_foreign_host_gets_403(self, ui_dir):
        from fastapi import FastAPI

        app = FastAPI()
        ui.mount_ui(app, ui_dir, bind_host="127.0.0.1")

        with TestClient(app, base_url="http://evil.example") as c:
            r = c.get("/index.html")

        assert r.status_code == 403
        assert "localhost" in r.text

    def test_a_loopback_host_is_served(self, ui_dir):
        from fastapi import FastAPI

        app = FastAPI()
        ui.mount_ui(app, ui_dir, bind_host="127.0.0.1")

        with TestClient(app, base_url="http://127.0.0.1:8000") as c:
            r = c.get("/")

        assert r.status_code == 200
        assert "app" in r.text


class TestNoWildcardCors:
    """The trap this whole design exists to avoid.

    A wildcard CORS policy on an API that executes code means any page the user
    opens can drive it. Asserted by inspecting the app's middleware, because a
    test that merely calls the API would pass whether or not the header was set.
    """

    def _origins(self, app) -> list[str]:
        from starlette.middleware.cors import CORSMiddleware

        found = []
        for middleware in app.user_middleware:
            cls = getattr(middleware, "cls", None)
            if cls is CORSMiddleware:
                found.extend(middleware.kwargs.get("allow_origins", []) or [])
        return found

    def test_mounting_a_ui_adds_no_cors(self, ui_dir):
        from fastapi import FastAPI

        app = FastAPI()
        before = self._origins(app)
        ui.mount_ui(app, ui_dir, bind_host="127.0.0.1")

        assert self._origins(app) == before == []

    def test_the_served_app_sends_no_cors_headers(self, ui_dir):
        from fastapi import FastAPI

        app = FastAPI()
        ui.mount_ui(app, ui_dir, bind_host="127.0.0.1")

        with TestClient(app, base_url="http://127.0.0.1:8000") as c:
            r = c.get("/", headers={"Origin": "https://evil.example"})

        assert "access-control-allow-origin" not in {
            k.lower() for k in r.headers
        }


class TestStaticServingDoesNotEscape:
    """The UI directory is a boundary, not a starting point."""

    def _app(self, ui_dir):
        from fastapi import FastAPI

        app = FastAPI()
        ui.mount_ui(app, ui_dir, bind_host="127.0.0.1")
        return app

    @pytest.mark.parametrize(
        "target",
        [
            "/../secret.txt",
            "/../../secret.txt",
            "/%2e%2e/secret.txt",
            "/%2e%2e%2fsecret.txt",
            "/..%2fsecret.txt",
            "/assets/../../secret.txt",
        ],
    )
    def test_traversal_does_not_reach_a_file_beside_the_ui(self, ui_dir, secret_beside_ui, target):
        with TestClient(self._app(ui_dir), base_url="http://127.0.0.1:8000") as c:
            r = c.get(target)

        assert r.status_code in (403, 404), (
            f"{target} returned {r.status_code}; a static mount that answers 200 "
            f"here is serving files outside the UI directory"
        )
        assert "do not serve me" not in r.text

    def test_the_apps_own_files_are_served(self, ui_dir):
        with TestClient(self._app(ui_dir), base_url="http://127.0.0.1:8000") as c:
            assert c.get("/index.html").status_code == 200
            assert c.get("/app.js").status_code == 200


class TestApiRoutesStillWin:
    """The UI mounts at "/", which is a catch-all. It must not shadow the API."""

    def test_a_registered_route_is_not_swallowed(self, ui_dir):
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/health")
        def health():
            return {"ok": True}

        ui.mount_ui(app, ui_dir, bind_host="127.0.0.1")

        with TestClient(app, base_url="http://127.0.0.1:8000") as c:
            r = c.get("/health")

        assert r.status_code == 200
        assert r.json() == {"ok": True}