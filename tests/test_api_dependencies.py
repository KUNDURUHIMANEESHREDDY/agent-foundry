"""The API's server-side state must be per-app, not per-process.

`STORE`, `AGENTS`, `WORKSPACES` and `AUTH` were module globals, so there was one
database per process and no way to serve two tenants. Isolation in tests came
from monkeypatching module attributes and restoring them — the pattern that most
often leaks state or makes a suite order-dependent.

These tests assert the property the injection was for: two apps, two independent
sets of state, no globals involved.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from factory.api.auth import AuthSettings
from factory.api.server import ApiDependencies, create_app
from factory.api.workspaces import WorkspaceRegistry
from factory.registry import AgentRegistry
from factory.runtime.trace import MemoryTraceStore, SqliteTraceStore

SPEC = """
name: {name}
version: "0.1.0"
system_prompt: "You read files."
capabilities: ["filesystem.read"]
model:
  provider: fake
  name: none
"""


def deps_for(workspace, key="k", store=None):
    workspaces = WorkspaceRegistry()
    workspaces.add("default", workspace)

    return ApiDependencies(
        store=store or MemoryTraceStore(),
        agents=AgentRegistry(":memory:"),
        workspaces=workspaces,
        auth=AuthSettings(key),
        data_dir=workspace / "data",
    )


class TestAppsAreIndependent:
    def test_two_apps_do_not_share_a_trace_store(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()

        store_a, store_b = MemoryTraceStore(), MemoryTraceStore()
        client_a = TestClient(create_app(deps_for(a, store=store_a)))
        client_b = TestClient(create_app(deps_for(b, store=store_b)))

        resp = client_a.post(
            "/run",
            json={"yaml": SPEC.format(name="probe"), "task": "t", "dry_run": True},
            headers={"Authorization": "Bearer k"},
        )
        assert resp.status_code == 200
        assert len(store_a.list()) == 1

        # The second app must see nothing: no shared global.
        assert client_b.get("/traces", headers={"Authorization": "Bearer k"}).json()[
            "traces"
        ] == []

    def test_two_apps_have_different_workspaces(self, tmp_path):
        a, b = tmp_path / "alpha", tmp_path / "beta"
        a.mkdir()
        b.mkdir()

        client_a = TestClient(create_app(deps_for(a)))
        client_b = TestClient(create_app(deps_for(b)))

        listed_a = client_a.get("/workspaces", headers={"Authorization": "Bearer k"}).json()
        listed_b = client_b.get("/workspaces", headers={"Authorization": "Bearer k"}).json()

        assert listed_a["workspaces"]["default"] == str(a.resolve())
        assert listed_b["workspaces"]["default"] == str(b.resolve())

    def test_a_workspace_id_from_one_app_is_unknown_to_the_other(self, tmp_path):
        a, b = tmp_path / "alpha", tmp_path / "beta"
        a.mkdir()
        b.mkdir()

        client_a = TestClient(create_app(deps_for(a)))
        client_b = TestClient(create_app(deps_for(b)))

        client_a.post(
            "/compile",
            json={"yaml": SPEC.format(name="p"), "task": "t", "workspace_id": "default"},
            headers={"Authorization": "Bearer k"},
        )
        # `other` is approved by neither, so this is refused.
        for client in (client_a, client_b):
            resp = client.post(
                "/compile",
                json={
                    "yaml": SPEC.format(name="p"),
                    "task": "t",
                    "workspace_id": "other",
                },
                headers={"Authorization": "Bearer k"},
            )
            assert resp.status_code == 403

    def test_two_apps_do_not_share_the_registry(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()

        client_a = TestClient(create_app(deps_for(a)))
        client_b = TestClient(create_app(deps_for(b)))
        headers = {"Authorization": "Bearer k"}

        client_a.post(
            "/agents", json={"yaml": SPEC.format(name="only-in-a")}, headers=headers
        )

        assert len(client_a.get("/agents", headers=headers).json()["agents"]) == 1
        assert client_b.get("/agents", headers=headers).json()["agents"] == []

    def test_two_apps_have_different_keys(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()

        client_a = TestClient(create_app(deps_for(ws, key="key-a")))
        client_b = TestClient(create_app(deps_for(ws, key="key-b")))

        assert client_a.get("/traces", headers={"Authorization": "Bearer key-a"}).status_code == 200
        assert client_b.get("/traces", headers={"Authorization": "Bearer key-a"}).status_code == 401
        assert client_b.get("/traces", headers={"Authorization": "Bearer key-b"}).status_code == 200


class TestDependenciesAreReachable:
    def test_dependencies_live_on_app_state(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        deps = deps_for(ws)
        app = create_app(deps)

        assert app.state.deps is deps

    def test_module_globals_are_derived_from_the_default_app(self, tmp_path):
        """`STORE` etc. must be the default app's state, not a second copy.

        They exist for `uvicorn factory.api.server:app` and for `factory serve`.
        If they were built separately they could drift from what handlers
        actually use, which is the failure the old layout invited.
        """
        from factory.api import server as srv

        assert srv.STORE is srv.app.state.deps.store
        assert srv.AGENTS is srv.app.state.deps.agents
        assert srv.WORKSPACES is srv.app.state.deps.workspaces
        assert srv.AUTH is srv.app.state.deps.auth


class TestDependenciesFromEnv:
    def test_builds_without_touching_the_cwd(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FACTORY_API_KEY", "env-key")
        monkeypatch.delenv("FACTORY_API_INSECURE", raising=False)
        monkeypatch.delenv("FACTORY_TRACE_DB", raising=False)

        deps = ApiDependencies.from_env(data_dir=tmp_path / "data")

        assert deps.auth.key == "env-key"
        assert deps.data_dir == (tmp_path / "data").resolve()
        assert deps.data_dir.is_dir()

    def test_honours_the_trace_db_override(self, tmp_path, monkeypatch):
        target = tmp_path / "custom" / "traces.db"
        monkeypatch.setenv("FACTORY_API_KEY", "k")
        monkeypatch.setenv("FACTORY_TRACE_DB", str(target))

        deps = ApiDependencies.from_env(data_dir=tmp_path / "data")

        assert isinstance(deps.store, SqliteTraceStore)
        assert target.parent.is_dir()

    def test_default_store_is_sqlite(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FACTORY_API_KEY", "k")
        monkeypatch.delenv("FACTORY_TRACE_DB", raising=False)

        deps = ApiDependencies.from_env(data_dir=tmp_path / "data")

        assert isinstance(deps.store, SqliteTraceStore)

    def test_default_workspace_resolves(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FACTORY_API_KEY", "k")
        monkeypatch.delenv("FACTORY_WORKSPACES", raising=False)

        deps = ApiDependencies.from_env(data_dir=tmp_path / "data")

        assert deps.default_workspace.is_dir()
        assert deps.workspaces.ids() == ["default"]


class TestEveryRouteGetsItsState:
    """No handler may read a module global; they all go through `get_deps`."""

    ROUTES = [
        "/capabilities",
        "/workspaces",
        "/traces",
        "/agents",
    ]

    @pytest.mark.parametrize("path", ROUTES)
    def test_routes_answer_from_the_injected_app(self, path, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        client = TestClient(create_app(deps_for(ws, key="k")))

        resp = client.get(path, headers={"Authorization": "Bearer k"})

        assert resp.status_code == 200, path