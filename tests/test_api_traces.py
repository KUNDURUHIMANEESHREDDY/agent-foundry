"""Trace persistence must be server-owned and actually observable.

Two defects lived here, both invisible to unit tests because the divergence only
appears across two HTTP calls:

  1. `/run` built its own trace store from the submitted spec, whose default is
     `memory`. Every API run was traced into a throwaway `MemoryTraceStore`, so
     `/traces` never saw a trace the API itself produced.
  2. `memory.path` is reachable from the spec YAML, and `SqliteTraceStore` does
     `Path(path).parent.mkdir(parents=True)`. A caller could choose where the
     server creates directories and writes a database — the same class of bug as
     accepting a workspace path.
"""

from __future__ import annotations

import uuid

import pytest

from factory.api import server as srv

READER_SPEC = """
name: {name}
version: "0.1.0"
system_prompt: "You read files."
capabilities: ["filesystem.read"]
model:
  provider: fake
  name: none
"""


def spec(name: str = "traced", extra: str = "") -> str:
    return READER_SPEC.format(name=name) + extra


# `api_client`, `auth_headers` and `api_deps` come from conftest. Each test gets its
# own app with its own in-memory store, so nothing global is patched or restored
# and no test can pass on another's leftovers.


class TestRunTracesAreVisibleToTraces:
    def test_a_run_appears_in_traces(self, api_client, auth_headers, api_deps):
        """The core defect: /run wrote to memory, /traces read sqlite."""
        name = "run-" + uuid.uuid4().hex[:8]
        before = len(api_deps.store.list(limit=100))

        resp = api_client.post(
            "/run",
            json={"yaml": spec(name), "task": "hello", "dry_run": True},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text

        listed = api_client.get("/traces", headers=auth_headers).json()["traces"]
        assert len(listed) == before + 1, "the run did not reach the trace store"

        trace_id = resp.json()["trace_id"]
        assert any(t["id"] == trace_id for t in listed)

    def test_run_returns_the_trace_id(self, api_client, auth_headers, api_deps):
        resp = api_client.post(
            "/run",
            json={"yaml": spec("traced"), "task": "hello", "dry_run": True},
            headers=auth_headers,
        )
        assert resp.json().get("trace_id")

    def test_trace_detail_is_fetchable_by_that_id(self, api_client, auth_headers, api_deps):
        run = api_client.post(
            "/run",
            json={"yaml": spec("traced"), "task": "a task", "dry_run": True},
            headers=auth_headers,
        ).json()

        detail = api_client.get(f"/traces/{run['trace_id']}", headers=auth_headers)
        assert detail.status_code == 200, detail.text
        body = detail.json()
        assert body["task"] == "a task"
        assert body["status"] == "success"

    def test_consecutive_runs_are_all_recorded(self, api_client, auth_headers, api_deps):
        ids = set()
        for _ in range(3):
            resp = api_client.post(
                "/run",
                json={"yaml": spec("traced"), "task": "t", "dry_run": True},
                headers=auth_headers,
            )
            ids.add(resp.json()["trace_id"])
        assert len(ids) == 3, "trace ids collided; runs are overwriting each other"

        listed = api_client.get("/traces", headers=auth_headers).json()["traces"]
        assert len(ids - {t["id"] for t in listed}) == 0

    def test_stored_trace_agrees_with_the_run_response(self, api_client, auth_headers, api_deps):
        """The two endpoints must not disagree about what happened.

        A trace that says `success` while `/run` said otherwise would be worse
        than an empty store: it would be confidently wrong.
        """
        run = api_client.post(
            "/run",
            json={"yaml": spec("traced"), "task": "a task", "dry_run": True},
            headers=auth_headers,
        ).json()

        detail = api_client.get(f"/traces/{run['trace_id']}", headers=auth_headers).json()

        assert detail["status"] == run["status"]
        assert len(detail["steps"]) == len(run["steps"])
        assert detail["task"] == "a task"

    def test_a_failed_compile_leaves_no_trace(self, api_client, auth_headers, api_deps):
        """A rejected spec must not produce a half-written trace."""
        before = len(api_deps.store.list(limit=100))

        resp = api_client.post(
            "/run",
            json={
                "yaml": spec("broken", extra='\ncapabilities: ["no.such.thing"]\n'),
                "task": "t",
                "dry_run": True,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 422
        assert len(api_deps.store.list(limit=100)) == before


class TestPersistenceIsNotClientControlled:
    """`memory:` in a submitted spec must not choose where the server writes."""

    def test_memory_sqlite_path_is_ignored(self, api_client, auth_headers, api_deps, tmp_path):
        target = tmp_path / "attacker" / "planted.db"
        payload = spec(
            "traced",
            extra=f"\nmemory:\n  type: sqlite\n  path: {target}\n",
        )

        resp = api_client.post(
            "/run",
            json={"yaml": payload, "task": "t", "dry_run": True},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        assert not target.exists(), (
            "the caller's memory.path was honoured; the server wrote where it "
            "was told to"
        )
        assert not target.parent.exists(), (
            "the server created directories chosen by the caller"
        )

    def test_ignored_memory_is_reported_not_silently_dropped(self, api_client, auth_headers, api_deps):
        payload = spec("traced", extra="\nmemory:\n  type: sqlite\n  path: /tmp/x.db\n")
        resp = api_client.post(
            "/compile",
            json={"yaml": payload, "task": "t"},
            headers=auth_headers,
        )
        warnings = " ".join(resp.json()["warnings"])
        assert "server configuration" in warnings
        assert "memory" in warnings

    def test_memory_block_on_compile_is_still_reported(self, api_client, auth_headers):
        payload = spec("traced", extra="\nmemory:\n  type: sqlite\n  path: /tmp/y.db\n")
        resp = api_client.post("/compile", json={"yaml": payload, "task": "t"}, headers=auth_headers)
        assert resp.status_code == 200
        assert any("ignored" in w for w in resp.json()["warnings"])

    def test_plain_run_reports_no_memory_warning(self, api_client, auth_headers):
        resp = api_client.post(
            "/compile", json={"yaml": spec("traced"), "task": "t"}, headers=auth_headers
        )
        assert not any("ignored" in w for w in resp.json()["warnings"])

    def test_the_server_store_is_the_one_used(self, api_client, auth_headers, api_deps):
        """Whatever the spec says, the runtime must hold the server's store."""
        before = len(api_deps.store.list(limit=100))
        api_client.post(
            "/run",
            json={
                "yaml": spec("traced", extra="\nmemory:\n  type: memory\n"),
                "task": "t",
                "dry_run": True,
            },
            headers=auth_headers,
        )
        assert len(api_deps.store.list(limit=100)) == before + 1

    def test_compile_does_not_write_a_trace(self, api_client, auth_headers, api_deps):
        """`/compile` reports grants; it must not fabricate run history."""
        before = len(api_deps.store.list(limit=100))
        api_client.post("/compile", json={"yaml": spec("traced"), "task": "t"}, headers=auth_headers)
        assert len(api_deps.store.list(limit=100)) == before


class TestStoreIsConfiguredServerSide:
    def test_trace_db_is_configurable_by_environment(self):
        assert srv.TRACE_DB_ENV == "FACTORY_TRACE_DB"

    def test_default_store_lives_under_the_data_dir(self):
        """Not the process cwd, and not anywhere a request can name."""
        assert srv.DATA_DIR.name == ".factory"

    def test_store_path_is_not_derived_from_any_request(self, api_client, auth_headers, tmp_path):
        planted = tmp_path / "planted.db"
        for payload in (
            spec("traced", extra=f"\nmemory:\n  type: sqlite\n  path: {planted}\n"),
            spec("traced", extra=f"\nmemory:\n  type: sqlite\n  path: {tmp_path}\n"),
        ):
            api_client.post(
                "/run", json={"yaml": payload, "task": "t", "dry_run": True}, headers=auth_headers
            )
        assert not planted.exists()

    def test_traces_endpoint_and_run_share_one_store(self, api_client, auth_headers, api_deps):
        """The property the whole finding was about, stated directly."""
        run = api_client.post(
            "/run",
            json={"yaml": spec("traced"), "task": "t", "dry_run": True},
            headers=auth_headers,
        ).json()

        # Same object, not merely a similar one.
        assert api_deps.store.get(run["trace_id"]) is not None

    def test_unknown_trace_id_is_404(self, api_client, auth_headers, api_deps):
        resp = api_client.get("/traces/definitely-not-a-real-id", headers=auth_headers)
        assert resp.status_code == 404