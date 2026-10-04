"""End-to-end: a real server, a real frontend, a real browser-shaped request.

The unit tests use TestClient. This runs an actual uvicorn on loopback and
checks the three things a frontend author will hit first:

  - the built app is served at /
  - the API answers from that same origin without a key in insecure mode
  - a request naming a foreign Host is refused

The last one cannot be checked through TestClient alone, because it changes the
Host header on the wire rather than a keyword argument.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """Run `factory serve --ui` for real, on a free loopback port."""
    dist = tmp_path_factory.mktemp("dist")
    (dist / "index.html").write_text(
        "<!doctype html><title>Agent Factory</title><h1>app</h1>", encoding="utf-8"
    )
    (dist / "app.js").write_text("console.log('x')", encoding="utf-8")

    port = _free_port()
    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": str(ROOT / "src"),
        "FACTORY_API_INSECURE": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "factory.cli", "serve",
         "--host", "127.0.0.1", "--port", str(port), "--ui", str(dist)],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"server exited: {proc.stdout.read()[:2000]}")
        try:
            urllib.request.urlopen(base + "/health", timeout=1).read()
            break
        except Exception:
            time.sleep(0.4)
    else:
        proc.kill()
        pytest.fail("server did not come up")

    yield base

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _get(url: str, host: str | None = None):
    request = urllib.request.Request(url)
    if host is not None:
        request.add_header("Host", host)
    try:
        with urllib.request.urlopen(request, timeout=5) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


class TestTheUiIsServed:
    def test_index_is_returned_at_root(self, live_server):
        status, body = _get(live_server + "/")
        assert status == 200
        assert "Agent Factory" in body

    def test_assets_are_served(self, live_server):
        status, _ = _get(live_server + "/app.js")
        assert status == 200


class TestTheApiAnswersWithoutAKey:
    def test_health_is_public(self, live_server):
        status, body = _get(live_server + "/health")
        assert status == 200
        assert json.loads(body)

    def test_a_secured_endpoint_works_with_no_key_in_insecure_mode(self, live_server):
        """This is the mode a local app runs in: no key pasted into the browser."""
        status, body = _get(live_server + "/capabilities")
        assert status == 200, body
        assert "capabilities" in json.loads(body) or isinstance(json.loads(body), (list, dict))


class TestTheHostGuardHoldsOnTheWire:
    def test_a_foreign_host_is_refused(self, live_server):
        port = live_server.rsplit(":", 1)[1]
        status, body = _get(f"http://127.0.0.1:{port}/health", host="evil.example")
        assert status == 403
        assert "localhost" in body

    def test_a_rebinding_lookalike_is_refused(self, live_server):
        port = live_server.rsplit(":", 1)[1]
        status, _ = _get(
            f"http://127.0.0.1:{port}/capabilities", host="localhost.evil.example"
        )
        assert status == 403


class TestNoCorsHeadersReachTheBrowser:
    def test_an_evil_origin_gets_no_allow_header(self, live_server):
        request = urllib.request.Request(live_server + "/")
        request.add_header("Origin", "https://evil.example")
        with urllib.request.urlopen(request, timeout=5) as r:
            names = {k.lower() for k in r.headers}

        assert "access-control-allow-origin" not in names, (
            "a browser will treat any allow-origin header as permission to drive "
            "an API that executes code"
        )