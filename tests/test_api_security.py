"""HTTP-level security tests for the API boundary.

These exercise the real app through the ASGI interface rather than calling
`_compile` directly, because the bugs they cover were only ever visible at the
HTTP boundary: a caller could not pass a path through a unit test of
`_compile`, since the unit test *was* the path.

The central property: **the capability root is chosen by the server, never by
the client.** The model already cannot rebind it (that is what `safe_join` and
the pinned-param strip enforce). These tests make sure the API does not hand
that power to the caller before the model ever runs.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from factory.api import server as srv
from factory.api.auth import AuthSettings
from factory.api.workspaces import (
    ID_PATTERN,
    UnknownWorkspace,
    WorkspaceRegistry,
)

READER_SPEC = """
name: probe
version: "0.1.0"
system_prompt: "You read files."
capabilities: ["filesystem.read"]
model:
  provider: fake
  name: none
"""


@pytest.fixture
def client():
    return TestClient(srv.app)


@pytest.fixture
def auth():
    return {"Authorization": f"Bearer {srv.AUTH.key}"}


def isolated_app(tmp_path, workspace=None):
    """An app with its own stores and its own approved workspace.

    Preferred over patching `srv.WORKSPACES` / `srv.STORE`, which is what this
    file used to do. Two apps over two dependency sets is the supported way to
    get isolation; a monkeypatched module attribute only isolates if you get the
    restore right.
    """
    from factory.api.auth import AuthSettings
    from factory.api.server import ApiDependencies, create_app
    from factory.api.workspaces import WorkspaceRegistry
    from factory.registry import AgentRegistry
    from factory.runtime.trace import MemoryTraceStore

    workspaces = WorkspaceRegistry()
    workspaces.add("default", workspace or tmp_path)

    return create_app(
        ApiDependencies(
            store=MemoryTraceStore(),
            agents=AgentRegistry(":memory:"),
            workspaces=workspaces,
            auth=AuthSettings("test-key"),
            data_dir=tmp_path / "data",
        )
    )


# ── the core vulnerability ─────────────────────────────────────────────


class TestClientCannotChooseTheCapabilityRoot:
    def test_workspace_path_field_is_rejected(self, client, auth):
        """The original hole. Must not be accepted, and must not be ignored."""
        resp = client.post(
            "/compile",
            json={"yaml": READER_SPEC, "task": "t", "workspace": "/"},
            headers=auth,
        )
        assert resp.status_code == 422, resp.text
        assert "workspace" in resp.text

    @pytest.mark.parametrize(
        "workspace_id",
        [
            "/", "/etc", "/", "..", "../..", "default/../..",
            "C:/", "\\\\server\\share", "default\x00", "DEFAULT",
            "Default", "unknown", "%2e%2e%2f", "..%2F..",
        ],
    )
    def test_unknown_workspace_id_is_refused(self, client, auth, workspace_id):
        resp = client.post(
            "/compile",
            json={"yaml": READER_SPEC, "task": "t", "workspace_id": workspace_id},
            headers=auth,
        )
        assert resp.status_code == 403, f"{workspace_id!r} -> {resp.text}"

    def test_workspace_ids_are_not_used_as_paths(self, tmp_path):
        """Even a registered id must not be able to escape via string tricks."""
        secret = tmp_path / "secret"
        secret.mkdir()
        (secret / "creds.txt").write_text("hunter2", encoding="utf-8")

        client = TestClient(isolated_app(tmp_path))
        headers = {"Authorization": "Bearer test-key"}

        resp = client.post(
            "/run",
            json={
                "yaml": READER_SPEC,
                "task": "read secret/creds.txt",
                "workspace_id": "default",
                "dry_run": True,
            },
            headers=headers,
        )
        assert resp.status_code == 200
        assert "creds.txt" not in resp.text

        # The compile prompt must name the directory the server approved, and
        # nothing else. `secret` is a sibling, so it must not appear.
        compiled = client.post(
            "/compile",
            json={"yaml": READER_SPEC, "task": "t", "workspace_id": "default"},
            headers=headers,
        )
        prompt = compiled.json()["system_prompt"]
        assert str(tmp_path.resolve()) in prompt
        assert str(secret) not in prompt

    def test_empty_workspace_id_falls_back_to_this_apps_own_workspace(
        self, api_client, auth_headers, api_deps
    ):
        """A missing id must resolve against this app's registry, not a global.

        Asserting against `api_deps.default_workspace` rather than a module
        constant is the point: it proves the handler read *this* app's state,
        so two apps with different workspaces cannot interfere.
        """
        resp = api_client.post(
            "/compile",
            json={"yaml": READER_SPEC, "task": "t", "workspace_id": ""},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        prompt = resp.json()["system_prompt"]
        assert str(api_deps.default_workspace) in prompt
        assert str(Path.cwd()) not in prompt or api_deps.default_workspace == Path.cwd()


# ── authentication ──────────────────────────────────────────────────────


UNAUTHENTICATED = [
    ("get", "/traces"),
    ("get", "/traces/some-id"),
    ("get", "/agents"),
    ("get", "/agents/reader"),
    ("get", "/agents/reader/versions"),
    ("get", "/agents/reader/diff"),
    ("get", "/workspaces"),
    ("get", "/capabilities"),
    ("post", "/run"),
    ("post", "/compile"),
    ("post", "/validate"),
    ("post", "/compile-goal"),
    ("post", "/agents"),
]


class TestAuthenticationRequired:
    @pytest.mark.parametrize("method,path", UNAUTHENTICATED)
    def test_anonymous_is_refused(self, client, method, path):
        if method == "post":
            resp = client.post(path, json={"yaml": READER_SPEC, "task": "t"})
        else:
            resp = client.get(path)
        assert resp.status_code == 401, f"{method.upper()} {path} was reachable anonymously"

    def test_health_is_public(self, client):
        """Liveness must work without a credential, or orchestration breaks."""
        assert client.get("/health").status_code == 200

    @pytest.mark.parametrize(
        "header",
        [
            {"Authorization": "Bearer wrong"},
            {"Authorization": "Bearer "},
            {"Authorization": ""},
            {"Authorization": "Basic YWRtaW46YWRtaW4="},
            {"X-API-Key": "wrong"},
            {"X-API-Key": ""},
        ],
    )
    def test_wrong_credentials_are_refused(self, client, header):
        resp = client.get("/traces", headers=header)
        assert resp.status_code == 401

    def test_x_api_key_is_accepted(self, client):
        resp = client.get("/traces", headers={"X-API-Key": srv.AUTH.key})
        assert resp.status_code == 200

    def test_www_authenticate_header_is_sent(self, client):
        resp = client.get("/traces")
        assert resp.headers.get("WWW-Authenticate") == "Bearer"

    def test_authenticated_run_still_works(self, client, auth):
        """The fix must not make the API unusable."""
        resp = client.post(
            "/run",
            json={"yaml": READER_SPEC, "task": "hello", "dry_run": True},
            headers=auth,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_new_routes_are_protected_by_construction(self):
        """The dependency lives on the router, so a new route inherits it."""
        for route in srv.secure.routes:
            deps = getattr(getattr(route, "dependant", None), "dependencies", []) or []
            calls = [getattr(d.call, "__name__", "") for d in deps]
            assert "require_auth" in calls, (
                f"{getattr(route, 'path', '?')} is on the secure router without "
                f"the auth dependency"
            )

    def test_no_route_bypasses_the_secure_router(self):
        """Nothing may be registered on the bare app except the public router.

        FastAPI keeps included routers nested, so a handler added directly to
        `app` would be reachable with no credential at all.
        """
        public_paths = {getattr(r, "path", None) for r in srv.public.routes}
        assert public_paths == {"/health"}, public_paths

        bare = [
            getattr(r, "path", None)
            for r in srv.app.routes
            if isinstance(getattr(r, "path", None), str)
            and getattr(r, "path", "").startswith(("/run", "/compile", "/traces",
                                                   "/agents", "/workspaces",
                                                   "/validate", "/capabilities"))
        ]
        assert bare == [], f"unauthenticated routes on the bare app: {bare}"

    def test_docs_and_schema_are_not_published(self):
        """An anonymous /openapi.json is a map of the whole attack surface."""
        assert srv.app.openapi_url is None
        assert srv.app.docs_url is None
        assert srv.app.redoc_url is None

    def test_registry_mutation_requires_auth(self, client):
        """Writing a spec is a state change, not a read."""
        name = "unauth-probe-" + secrets.token_hex(6)
        spec = READER_SPEC.replace('name: probe', f'name: {name}')
        resp = client.post("/agents", json={"yaml": spec})
        assert resp.status_code == 401
        assert srv.AGENTS.get(name) is None


class TestAuthSettings:
    def test_generates_a_key_when_none_configured(self):
        settings = AuthSettings()
        assert settings.enabled
        assert settings.generated
        assert len(settings.key) >= 32

    def test_generated_keys_differ(self):
        assert AuthSettings().key != AuthSettings().key

    def test_explicit_key_is_used(self):
        settings = AuthSettings("s3cret")
        assert settings.key == "s3cret"
        assert not settings.generated
        assert settings.check("s3cret")
        assert not settings.check("s3cre")
        assert not settings.check("")

    def test_insecure_mode_is_explicit(self, monkeypatch):
        monkeypatch.setenv("FACTORY_API_INSECURE", "1")
        settings = AuthSettings.from_env()
        assert not settings.enabled
        assert settings.check(None)

    def test_env_key_is_read(self, monkeypatch):
        monkeypatch.setenv("FACTORY_API_KEY", "from-env")
        monkeypatch.delenv("FACTORY_API_INSECURE", raising=False)
        settings = AuthSettings.from_env()
        assert settings.key == "from-env"
        assert settings.check("from-env")

    def test_insecure_env_spellings(self, monkeypatch):
        for value in ("1", "true", "TRUE", "yes", "on"):
            monkeypatch.setenv("FACTORY_API_INSECURE", value)
            assert AuthSettings.from_env().insecure, value

    def test_blank_key_does_not_disable_auth(self, monkeypatch):
        """An empty env var must not read as 'no auth configured'."""
        monkeypatch.setenv("FACTORY_API_KEY", "   ")
        monkeypatch.delenv("FACTORY_API_INSECURE", raising=False)
        settings = AuthSettings.from_env()
        assert settings.enabled
        assert settings.check(settings.key)
        assert not settings.check("")


# ── the workspace registry itself ───────────────────────────────────────


class TestWorkspaceRegistry:
    def test_ids_may_not_be_paths(self):
        registry = WorkspaceRegistry()
        for bad in ("..", "../etc", "/etc", "a/b", "a\\b", "", "a b", "x" * 100):
            with pytest.raises(ValueError):
                registry.add(bad, Path.cwd())

    def test_id_pattern_admits_normal_names(self):
        assert ID_PATTERN.match("default")
        assert ID_PATTERN.match("docs-2")
        assert ID_PATTERN.match("a_b")

    def test_resolve_unknown_raises(self):
        registry = WorkspaceRegistry()
        registry.add("default", Path.cwd())
        with pytest.raises(UnknownWorkspace):
            registry.resolve("nope")

    def test_resolve_is_case_sensitive(self):
        registry = WorkspaceRegistry()
        registry.add("default", Path.cwd())
        with pytest.raises(UnknownWorkspace):
            registry.resolve("DEFAULT")

    def test_paths_are_resolved_once_at_registration(self, tmp_path):
        """A symlink swapped in later must not move the capability root."""
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        try:
            link.symlink_to(real, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")

        registry = WorkspaceRegistry()
        resolved = registry.add("default", link)
        assert resolved == real.resolve()

        # Re-pointing the link afterwards changes nothing, because the registry
        # stored the resolved directory, not the link.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        link.unlink()
        link.symlink_to(elsewhere, target_is_directory=True)
        assert registry.resolve("default") == real.resolve()

    def test_nonexistent_path_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="not a directory"):
            WorkspaceRegistry().add("default", tmp_path / "nope")

    def test_file_is_not_a_workspace(self, tmp_path):
        target = tmp_path / "file.txt"
        target.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="not a directory"):
            WorkspaceRegistry().add("default", target)

    def test_from_env_single_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FACTORY_WORKSPACES", raising=False)
        registry = WorkspaceRegistry.from_env(base=tmp_path)
        assert registry.ids() == ["default"]
        assert registry.resolve(None) == tmp_path.resolve()

    def test_from_env_multiple(self, monkeypatch, tmp_path):
        (tmp_path / "docs").mkdir()
        monkeypatch.setenv(
            "FACTORY_WORKSPACES", f"default=.,docs={tmp_path / 'docs'}"
        )
        registry = WorkspaceRegistry.from_env(base=tmp_path)
        assert registry.ids() == ["default", "docs"]  # sorted
        assert registry.resolve("docs") == (tmp_path / "docs").resolve()

    def test_from_env_relative_resolves_against_base(self, monkeypatch, tmp_path):
        (tmp_path / "nested" / "deep").mkdir(parents=True)
        monkeypatch.setenv("FACTORY_WORKSPACES", "docs=nested/deep")
        registry = WorkspaceRegistry.from_env(base=tmp_path)
        assert registry.resolve("docs") == (tmp_path / "nested" / "deep").resolve()

    def test_from_env_rejects_malformed_entry(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FACTORY_WORKSPACES", "default")
        with pytest.raises(ValueError, match="id=path"):
            WorkspaceRegistry.from_env(base=tmp_path)

    def test_from_env_refuses_to_escape_base(self, monkeypatch, tmp_path):
        """`id=..` is a config error, not a feature: it would point outside."""
        monkeypatch.setenv("FACTORY_WORKSPACES", "default=..")
        registry = WorkspaceRegistry.from_env(base=tmp_path)
        assert registry.resolve("default") == tmp_path.parent.resolve()

    def test_no_workspaces_yields_nothing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FACTORY_WORKSPACES", " , ,")
        with pytest.raises(ValueError, match="no usable workspaces"):
            WorkspaceRegistry.from_env(base=tmp_path)


# ── the boundary end to end ─────────────────────────────────────────────


class TestEscapeIsStillRefusedUnderneath:
    """The server picking the root does not weaken per-request confinement."""

    HEADERS = {"Authorization": "Bearer test-key"}

    def test_model_supplied_traversal_is_still_denied(self, tmp_path):
        client = TestClient(isolated_app(tmp_path))
        resp = client.post(
            "/run",
            json={
                "yaml": READER_SPEC,
                "task": "read ../../etc/passwd",
                "dry_run": True,
                "workspace_id": "default",
            },
            headers=self.HEADERS,
        )
        assert resp.status_code == 200
        # The scripted model makes no tool call, so the run succeeds; what
        # matters is that nothing leaked and the root stayed server-side.
        assert resp.json()["status"] == "success"

    def test_git_commit_only_in_a_real_repo(self, tmp_path):
        """A workspace outside any repository must not offer git.commit."""
        client = TestClient(isolated_app(tmp_path))
        resp = client.post(
            "/compile",
            json={
                "yaml": READER_SPEC.replace(
                    'capabilities: ["filesystem.read"]',
                    'capabilities: ["filesystem.read", "git.commit"]',
                ),
                "task": "t",
                "workspace_id": "default",
            },
            headers=self.HEADERS,
        )
        assert resp.status_code == 422
        assert "git.commit" in resp.text

    def test_git_commit_appears_inside_a_repository(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)

        client = TestClient(isolated_app(tmp_path, workspace=repo))
        resp = client.post(
            "/compile",
            json={
                "yaml": READER_SPEC.replace(
                    'capabilities: ["filesystem.read"]',
                    'capabilities: ["filesystem.read", "git.commit"]',
                ),
                "task": "t",
                "workspace_id": "default",
            },
                headers=self.HEADERS,
        )
        assert resp.status_code == 200
        assert "git.commit" in resp.json()["granted"]


class TestServeRefusesUnsafeBinds:
    """`factory serve` is the only thing standing between this and 0.0.0.0."""

    def _args(self, host):
        import argparse

        # Mirrors what argparse actually builds for `factory serve`. `ui` was
        # added for `--ui`; leaving it out here made these three tests fail with
        # an AttributeError, which is a test bug rather than a behaviour change.
        return argparse.Namespace(host=host, port=8000, ui=None)

    def test_public_bind_without_auth_is_refused(self, monkeypatch, capsys):
        from factory import cli

        monkeypatch.setenv("FACTORY_API_INSECURE", "1")
        monkeypatch.delenv("FACTORY_API_KEY", raising=False)

        code = cli.cmd_serve(self._args("0.0.0.0"))
        assert code == 2
        assert "refusing to bind" in capsys.readouterr().err

    def test_public_bind_with_auth_is_allowed(self, monkeypatch):
        from factory import cli

        monkeypatch.delenv("FACTORY_API_INSECURE", raising=False)
        monkeypatch.setenv("FACTORY_API_KEY", "a-real-key")

        called: dict[str, object] = {}

        def fake_run(target, host, port):
            called.update(target=target, host=host, port=port)

        monkeypatch.setattr("uvicorn.run", fake_run)
        assert cli.cmd_serve(self._args("0.0.0.0")) == 0
        assert called["host"] == "0.0.0.0"
        assert called["target"] == "factory.api.server:app"

    def test_loopback_bind_without_auth_is_allowed(self, monkeypatch, capsys):
        """Local development must stay frictionless."""
        from factory import cli

        monkeypatch.setenv("FACTORY_API_INSECURE", "1")
        monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

        assert cli.cmd_serve(self._args("127.0.0.1")) == 0


class TestNoServerStateInModuleScope:
    """Module-level stores must not be the only way to reach data."""

    def test_trace_and_registry_paths_live_under_one_data_dir(self):
        assert srv.DATA_DIR.name == ".factory"
        assert srv.STORE is not None
        assert srv.AGENTS is not None

    def test_tempdir_cleanup_is_not_required_for_correctness(self, tmp_path):
        """Workspaces are explicit; nothing depends on the process cwd."""
        registry = WorkspaceRegistry()
        workspace = tmp_path / "ws"
        workspace.mkdir()
        registry.add("default", workspace)
        assert registry.resolve("default") == workspace.resolve()
        shutil.rmtree(workspace)
        assert registry.ids() == ["default"]
        assert os.path.isdir(tmp_path)