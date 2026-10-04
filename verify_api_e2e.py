"""End-to-end: a run through a real HTTP server is readable from /traces.

Drives uvicorn over a socket rather than TestClient, because the whole defect was
a disagreement between two requests and TestClient short-circuits some of the
stack. Uses a throwaway trace database so it never touches the repo's own.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request(url: str, key: str | None = None, body: dict | None = None):
    data = json_bytes(body) if body is not None else None
    req = urllib.request.Request(url, data=data)
    if data:
        req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            import json

            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        import json

        return exc.code, json.loads(exc.read())


def json_bytes(value: dict) -> bytes:
    import json

    return json.dumps(value).encode()


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="factory-e2e-"))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["FACTORY_API_KEY"] = "e2e-test-key"
    env["FACTORY_TRACE_DB"] = str(workdir / "traces.db")
    env["FACTORY_WORKSPACES"] = f"default={ROOT}"
    # Two tenants: alice may only read, bob may do anything.
    env["FACTORY_TENANTS"] = "alice=filesystem.read;bob=*"
    env["FACTORY_TENANT_KEYS"] = "k-alice=alice,k-bob=bob"

    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "factory.cli", "serve", "--port", str(port)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    base = f"http://127.0.0.1:{port}"
    key = "k-bob"

    try:
        for _ in range(80):
            try:
                status, _body = request(f"{base}/health")
                if status == 200:
                    break
            except Exception:
                time.sleep(0.25)
        else:
            print("server never came up")
            return 1

        failures = []

        def check(label, ok, detail=""):
            print(f"  [{'ok  ' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
            if not ok:
                failures.append(label)

        print("--- unauthenticated ---")
        status, _ = request(f"{base}/traces")
        check("GET /traces without a key is refused", status == 401, f"got {status}")

        print("\n--- authenticated ---")
        status, body = request(f"{base}/workspaces", key)
        check("GET /workspaces", status == 200, str(body)[:80])

        status, body = request(f"{base}/traces", key)
        check("GET /traces is reachable with a key", status == 200)

        spec = (
            "name: e2e\nversion: '0.1.0'\n"
            "system_prompt: 'You read files.'\n"
            "capabilities: ['filesystem.read']\n"
            "model:\n  provider: fake\n  name: none\n"
        )

        status, body = request(
            f"{base}/run",
            key,
            {"yaml": spec, "task": "read README.md", "dry_run": True},
        )
        check("POST /run", status == 200, str(body)[:120])
        trace_id = body.get("trace_id") if status == 200 else None
        check("POST /run returns a trace_id", bool(trace_id), str(trace_id))

        status, listed = request(f"{base}/traces", key)
        ids = [t["id"] for t in listed.get("traces", [])]
        check("the run appears in /traces", bool(trace_id) and trace_id in ids, str(ids))

        if trace_id:
            status, detail = request(f"{base}/traces/{trace_id}", key)
            check("GET /traces/{id}", status == 200, str(detail)[:100])
            check(
                "detail agrees with /run",
                status == 200 and detail.get("task") == "read README.md",
            )

        print("\n--- persistence is not client-controlled ---")
        planted = workdir / "planted" / "evil.db"
        hostile = spec + f"\nmemory:\n  type: sqlite\n  path: {planted}\n"
        status, body = request(
            f"{base}/run", key, {"yaml": hostile, "task": "t", "dry_run": True}
        )
        check("POST /run with a hostile memory block succeeds", status == 200)
        check("no database created where the caller asked", not planted.exists())
        check(
            "no directories created where the caller asked",
            not planted.parent.exists(),
        )
        check(
            "the override is reported",
            any("ignored" in w for w in body.get("warnings", [])),
            str(body.get("warnings", []))[:100],
        )

        print("\n--- workspace cannot be chosen ---")
        status, _ = request(
            f"{base}/compile", key, {"yaml": spec, "task": "t", "workspace": "/"}
        )
        check("legacy workspace field refused", status == 422, f"got {status}")

        status, _ = request(
            f"{base}/compile", key, {"yaml": spec, "task": "t", "workspace_id": "/etc"}
        )
        check("workspace_id=/etc refused", status == 403, f"got {status}")

        print("\n--- tenant policy and scoping ---")
        # `k-alice` may only read; `k-bob` may do anything.
        status, _ = request(
            f"{base}/compile",
            "k-alice",
            {"yaml": spec.replace("filesystem.read", "python.execute"), "task": "t"},
        )
        check(
            "restricted tenant refused a forbidden capability",
            status == 403,
            f"got {status}",
        )

        status, body = request(
            f"{base}/run",
            key,
            {"yaml": spec, "task": "bob's private run", "dry_run": True},
        )
        bob_trace = body.get("trace_id")
        check("bob's run succeeded", status == 200, f"got {status}")

        status, body = request(f"{base}/traces", "k-alice")
        alice_ids = [t["id"] for t in body.get("traces", [])]
        check(
            "alice cannot see bob's trace",
            status == 200 and bob_trace not in alice_ids,
            f"alice sees {alice_ids}",
        )

        status, _ = request(f"{base}/traces/{bob_trace}", "k-alice")
        check("bob's trace is 404 for alice", status == 404, f"got {status}")

        print()
        if failures:
            print(f"FAIL: {len(failures)} check(s) failed: {failures}")
            return 1
        print("PASS: every end-to-end check held")
        return 0

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())