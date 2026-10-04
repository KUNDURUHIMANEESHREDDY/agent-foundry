"""Callers must be isolated: capabilities, traces, and registered specs.

Two findings were the same gap. An authenticated caller could request any
capability including `python.execute`, so the API key was a grant of code
execution with nothing narrower to hand out; and every caller could read every
trace and every registered spec.

A tenant fixes both: a named principal with a capability allowlist and its own
data scope.

Two behaviours are load-bearing and easy to get wrong:

  * A capability outside the tenant's policy is REFUSED, not dropped. Silently
    running a weaker agent while reporting success is the same lie as the old
    `memory.path` behaviour.
  * Another tenant's run is 404, not 403. Confirming it exists would leak that
    the id is real, which is the information the scope withholds.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from factory.api.auth import AuthSettings
from factory.api.server import ApiDependencies, create_app
from factory.api.tenants import (
    ALL,
    Tenant,
    TenantRegistry,
    UnknownTenant,
    key_map_from_env,
)
from factory.api.workspaces import WorkspaceRegistry
from factory.registry import AgentRegistry
from factory.runtime.trace import MemoryTraceStore

READER = """
name: reader
version: "0.1.0"
system_prompt: "You read files."
capabilities: ["filesystem.read"]
model:
  provider: fake
  name: none
"""

CODER = """
name: coder
version: "0.1.0"
system_prompt: "You write files."
capabilities: ["filesystem.write", "python.execute"]
model:
  provider: fake
  name: none
"""


def app_for(workspace, keys: dict[str, str], tenants: TenantRegistry):
    workspaces = WorkspaceRegistry()
    workspaces.add("default", workspace)

    return create_app(
        ApiDependencies(
            store=MemoryTraceStore(),
            agents=AgentRegistry(":memory:"),
            workspaces=workspaces,
            auth=AuthSettings(key_tenants=keys),
            data_dir=workspace / "data",
            tenants=tenants,
        )
    )


def two_tenant_app(tmp_path):
    """`alice` may only read; `bob` may do everything."""
    tenants = TenantRegistry(
        {
            "alice": Tenant(id="alice", capabilities=frozenset({"filesystem.read"})),
            "bob": Tenant(id="bob", capabilities=frozenset({ALL})),
        }
    )
    client = TestClient(app_for(tmp_path, {"k-alice": "alice", "k-bob": "bob"}, tenants))
    return client, {"alice": {"Authorization": "Bearer k-alice"}, "bob": {"Authorization": "Bearer k-bob"}}


def run(client, headers, yaml, task="t"):
    return client.post(
        "/run", json={"yaml": yaml, "task": task, "dry_run": True}, headers=headers
    )


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    return tmp_path


# ── capability policy ──────────────────────────────────────────────────


class TestCapabilityPolicy:
    def test_a_restricted_tenant_cannot_run_a_forbidden_capability(self, workspace):
        client, keys = two_tenant_app(workspace)

        resp = run(client, keys["alice"], CODER)

        assert resp.status_code == 403
        assert "python.execute" in resp.json()["detail"]

    def test_the_refusal_names_the_tenant(self, workspace):
        client, keys = two_tenant_app(workspace)
        detail = run(client, keys["alice"], CODER).json()["detail"]
        assert "alice" in detail

    def test_the_refusal_lists_what_is_allowed(self, workspace):
        client, keys = two_tenant_app(workspace)
        detail = run(client, keys["alice"], CODER).json()["detail"]
        assert "filesystem.read" in detail

    def test_the_capability_is_refused_not_silently_dropped(self, workspace):
        """A refusal, because a downgrade would report success falsely."""
        client, keys = two_tenant_app(workspace)

        resp = run(client, keys["alice"], CODER)

        assert resp.status_code == 403
        # Crucially: nothing ran, and nothing claimed to have.
        assert "status" not in resp.json()

    def test_an_allowed_capability_still_works(self, workspace):
        client, keys = two_tenant_app(workspace)
        assert run(client, keys["alice"], READER).status_code == 200

    def test_an_unrestricted_tenant_is_unaffected(self, workspace):
        client, keys = two_tenant_app(workspace)
        assert run(client, keys["bob"], CODER).status_code == 200

    def test_policy_applies_to_compile_as_well_as_run(self, workspace):
        client, keys = two_tenant_app(workspace)
        resp = client.post(
            "/compile", json={"yaml": CODER, "task": "t"}, headers=keys["alice"]
        )
        assert resp.status_code == 403

    def test_policy_applies_to_registration(self, workspace):
        client, keys = two_tenant_app(workspace)
        resp = client.post("/agents", json={"yaml": CODER}, headers=keys["alice"])
        assert resp.status_code == 403

    def test_one_forbidden_capability_refuses_the_whole_spec(self, workspace):
        """Not a partial grant — the caller asked for a specific agent."""
        client, keys = two_tenant_app(workspace)
        mixed = CODER.replace(
            'capabilities: ["filesystem.write", "python.execute"]',
            'capabilities: ["filesystem.read", "python.execute"]',
        )
        resp = run(client, keys["alice"], mixed)
        assert resp.status_code == 403


# ── trace visibility ───────────────────────────────────────────────────


class TestTraceVisibility:
    def test_a_tenant_sees_only_its_own_traces(self, workspace):
        client, keys = two_tenant_app(workspace)

        alice_run = run(client, keys["alice"], READER).json()["trace_id"]
        bob_run = run(client, keys["bob"], READER).json()["trace_id"]

        alice_sees = [
            t["id"] for t in client.get("/traces", headers=keys["alice"]).json()["traces"]
        ]
        bob_sees = [
            t["id"] for t in client.get("/traces", headers=keys["bob"]).json()["traces"]
        ]

        assert alice_sees == [alice_run]
        assert bob_sees == [bob_run]
        assert bob_run not in alice_sees

    def test_another_tenants_trace_is_404_not_403(self, workspace):
        client, keys = two_tenant_app(workspace)

        alice_run = run(client, keys["alice"], READER).json()["trace_id"]
        resp = client.get(f"/traces/{alice_run}", headers=keys["bob"])

        assert resp.status_code == 404

    def test_a_404_does_not_confirm_the_run_exists(self, workspace):
        """403 would leak that the id is real; the message must be generic."""
        client, keys = two_tenant_app(workspace)
        alice_run = run(client, keys["alice"], READER).json()["trace_id"]

        detail = client.get(f"/traces/{alice_run}", headers=keys["bob"]).json()["detail"]

        assert "not found" in detail.lower()
        assert alice_run not in detail

    def test_a_tenant_reads_its_own_trace(self, workspace):
        client, keys = two_tenant_app(workspace)
        alice_run = run(client, keys["alice"], READER).json()["trace_id"]

        resp = client.get(f"/traces/{alice_run}", headers=keys["alice"])

        assert resp.status_code == 200
        assert resp.json()["tenant"] == "alice"


# ── registry visibility ────────────────────────────────────────────────


class TestRegistryVisibility:
    def test_two_tenants_may_register_the_same_name(self, workspace):
        client, keys = two_tenant_app(workspace)

        assert client.post("/agents", json={"yaml": READER}, headers=keys["alice"]).status_code == 200
        assert client.post("/agents", json={"yaml": READER}, headers=keys["bob"]).status_code == 200

        # Same name, two tenants, no collision.
        for who in ("alice", "bob"):
            listed = client.get("/agents", headers=keys[who]).json()["agents"]
            assert [a["name"] for a in listed] == ["reader"]

    def test_a_tenant_does_not_see_another_tenants_agents(self, workspace):
        client, keys = two_tenant_app(workspace)

        client.post("/agents", json={"yaml": READER.replace("reader", "alice-only")}, headers=keys["alice"])

        assert client.get("/agents", headers=keys["bob"]).json()["agents"] == []
        assert len(client.get("/agents", headers=keys["alice"]).json()["agents"]) == 1

    def test_another_tenants_agent_is_404(self, workspace):
        client, keys = two_tenant_app(workspace)
        client.post("/agents", json={"yaml": READER}, headers=keys["alice"])

        assert client.get("/agents/reader", headers=keys["bob"]).status_code == 404
        assert client.get("/agents/reader", headers=keys["alice"]).status_code == 200

    def test_another_tenants_versions_are_404(self, workspace):
        client, keys = two_tenant_app(workspace)
        client.post("/agents", json={"yaml": READER}, headers=keys["alice"])

        assert client.get("/agents/reader/versions", headers=keys["bob"]).status_code == 404

    def test_diff_is_scoped(self, workspace):
        client, keys = two_tenant_app(workspace)

        # alice has 0.1.0, bob has 0.2.0 — the same agent name, different
        # versions, per tenant.
        client.post("/agents", json={"yaml": READER}, headers=keys["alice"])
        client.post(
            "/agents",
            json={"yaml": READER.replace('version: "0.1.0"', 'version: "0.2.0"')},
            headers=keys["bob"],
        )

        assert client.get(
            "/agents/reader/diff?from_version=0.1.0&to=0.2.0", headers=keys["bob"]
        ).status_code == 404, "bob has no 0.1.0 to diff from"
        assert client.get(
            "/agents/reader/versions", headers=keys["alice"]
        ).json()["versions"][0]["version"] == "0.1.0"

    def test_two_tenants_keep_separate_versions_of_one_name(self, workspace):
        client, keys = two_tenant_app(workspace)

        client.post("/agents", json={"yaml": READER}, headers=keys["alice"])
        bumped = READER.replace('version: "0.1.0"', 'version: "0.2.0"')
        client.post("/agents", json={"yaml": bumped}, headers=keys["bob"])

        assert [
            v["version"]
            for v in client.get("/agents/reader/versions", headers=keys["alice"]).json()["versions"]
        ] == ["0.1.0"]
        assert [
            v["version"]
            for v in client.get("/agents/reader/versions", headers=keys["bob"]).json()["versions"]
        ] == ["0.2.0"]


# ── configuration ──────────────────────────────────────────────────────


class TestTenantConfiguration:
    def test_default_is_one_unrestricted_tenant(self, monkeypatch):
        from factory.api.tenants import ENV_TENANTS

        monkeypatch.delenv(ENV_TENANTS, raising=False)
        registry = TenantRegistry.from_env()

        assert registry.ids() == ["default"]
        assert registry.require("default").unrestricted

    def test_policy_is_parsed(self, monkeypatch):
        from factory.api.tenants import ENV_TENANTS

        monkeypatch.setenv(
            ENV_TENANTS, "alice=filesystem.read,filesystem.write;bob=*"
        )
        registry = TenantRegistry.from_env()

        assert registry.ids() == ["alice", "bob"]
        assert registry.require("alice").allows("filesystem.read")
        assert not registry.require("alice").allows("python.execute")
        assert registry.require("bob").unrestricted

    def test_malformed_policy_raises_rather_than_granting_all(self, monkeypatch):
        """A typo must not silently widen access."""
        from factory.api.tenants import ENV_TENANTS

        monkeypatch.setenv(ENV_TENANTS, "alice")
        with pytest.raises(ValueError, match="tenant=cap"):
            TenantRegistry.from_env()

    def test_a_capability_list_is_not_mistaken_for_more_tenants(self, monkeypatch):
        """The reason tenants are separated by `;` and not `,`.

        A comma-only format read `alice=filesystem.read,filesystem.write` as
        tenant `alice` plus a nonsense tenant `filesystem.write` — which, in a
        version where the unknown tenant defaulted to unrestricted, would have
        handed out everything.
        """
        from factory.api.tenants import ENV_TENANTS

        monkeypatch.setenv(ENV_TENANTS, "alice=filesystem.read,filesystem.write")

        registry = TenantRegistry.from_env()

        assert registry.ids() == ["alice"]
        assert registry.require("alice").allows("filesystem.write")
        assert not registry.require("alice").allows("python.execute")

    def test_empty_policy_value_means_unrestricted(self, monkeypatch):
        from factory.api.tenants import ENV_TENANTS

        monkeypatch.setenv(ENV_TENANTS, "alice=;bob=filesystem.read")
        registry = TenantRegistry.from_env()
        assert registry.require("alice").unrestricted
        assert not registry.require("bob").unrestricted

    def test_unknown_tenant_raises(self):
        with pytest.raises(UnknownTenant):
            TenantRegistry.single().require("nobody")

    def test_single_key_maps_to_the_only_tenant(self, monkeypatch):
        from factory.api.tenants import ENV_KEYS, ENV_TENANTS

        monkeypatch.setenv(ENV_TENANTS, "alice=filesystem.read")
        monkeypatch.setenv("FACTORY_API_KEY", "solo")
        monkeypatch.delenv(ENV_KEYS, raising=False)

        pairs = key_map_from_env(TenantRegistry.from_env())
        assert pairs == {"solo": "alice"}

    def test_one_key_and_many_tenants_is_refused(self, monkeypatch):
        """Otherwise every key would silently grant the first tenant."""
        from factory.api.tenants import ENV_KEYS, ENV_TENANTS

        monkeypatch.setenv(ENV_TENANTS, "alice=filesystem.read;bob=filesystem.read")
        monkeypatch.setenv("FACTORY_API_KEY", "solo")
        monkeypatch.delenv(ENV_KEYS, raising=False)

        with pytest.raises(ValueError, match="FACTORY_TENANT_KEYS"):
            key_map_from_env(TenantRegistry.from_env())

    def test_explicit_key_map_is_used(self, monkeypatch):
        from factory.api.tenants import ENV_KEYS, ENV_TENANTS

        monkeypatch.setenv(ENV_TENANTS, "alice=filesystem.read;bob=filesystem.read")
        monkeypatch.setenv(ENV_KEYS, "ka=alice,kb=bob")
        monkeypatch.delenv("FACTORY_API_KEY", raising=False)

        assert key_map_from_env(TenantRegistry.from_env()) == {
            "ka": "alice",
            "kb": "bob",
        }

    def test_a_key_naming_an_undefined_tenant_is_refused(self, monkeypatch):
        from factory.api.tenants import ENV_KEYS, ENV_TENANTS

        monkeypatch.setenv(ENV_TENANTS, "alice=filesystem.read")
        monkeypatch.setenv(ENV_KEYS, "ka=ghost")

        with pytest.raises(ValueError, match="does not define"):
            key_map_from_env(TenantRegistry.from_env())


# ── key resolution ─────────────────────────────────────────────────────


class TestKeyResolution:
    def test_a_key_resolves_to_its_tenant(self):
        settings = AuthSettings(key_tenants={"k1": "alice", "k2": "bob"})
        assert settings.tenant_for("k1") == "alice"
        assert settings.tenant_for("k2") == "bob"

    def test_an_unknown_key_resolves_to_nothing(self):
        settings = AuthSettings(key_tenants={"k1": "alice"})
        assert settings.tenant_for("k2") is None
        assert settings.tenant_for("") is None
        assert settings.tenant_for(None) is None

    def test_check_and_tenant_for_agree(self):
        settings = AuthSettings(key_tenants={"k1": "alice"})
        assert settings.check("k1") is True
        assert settings.check("k2") is False

    def test_a_near_miss_key_is_refused(self):
        settings = AuthSettings(key_tenants={"k1": "alice"})
        assert settings.tenant_for("k1x") is None

    def test_a_key_for_an_undefined_tenant_is_refused_at_the_boundary(self, tmp_path):
        """A key that resolves to nothing configured must not get a tenant."""
        workspaces = WorkspaceRegistry()
        workspaces.add("default", tmp_path)
        client = TestClient(
            create_app(
                ApiDependencies(
                    store=MemoryTraceStore(),
                    agents=AgentRegistry(":memory:"),
                    workspaces=workspaces,
                    auth=AuthSettings(key_tenants={"k": "ghost"}),
                    data_dir=tmp_path / "data",
                    tenants=TenantRegistry.single(),
                )
            )
        )
        resp = client.get("/traces", headers={"Authorization": "Bearer k"})
        assert resp.status_code == 403
        assert "ghost" in resp.json()["detail"]