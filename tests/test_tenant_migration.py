"""The tenant migration must preserve data, not just avoid crashing.

Both stores gained a `tenant` column and the registry's primary keys changed.
`CREATE TABLE IF NOT EXISTS` is a no-op against an existing table, so without a
migration an old database keeps the old shape — which is not a clean failure:
the registry's reads would be unscoped while the code believed they were scoped.

The registry migration is a rebuild, so these tests check the rows survive.
"""

from __future__ import annotations

import sqlite3

import pytest

from factory.registry import AgentRegistry
from factory.runtime.trace import MemoryTraceStore, RunTrace, SqliteTraceStore
from factory.spec.agent_spec import AgentSpec, ModelRef

LEGACY_SCHEMA = """
CREATE TABLE agents (
    name         TEXT PRIMARY KEY,
    description  TEXT NOT NULL DEFAULT '',
    priority     TEXT NOT NULL DEFAULT 'medium',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE versions (
    agent_name   TEXT NOT NULL,
    version      TEXT NOT NULL,
    spec_json    TEXT NOT NULL,
    notes        TEXT NOT NULL DEFAULT '',
    tags_json    TEXT NOT NULL DEFAULT '[]',
    source       TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    PRIMARY KEY (agent_name, version)
);
"""

LEGACY_TRACE_SCHEMA = """
CREATE TABLE traces (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    agent_version TEXT NOT NULL,
    task TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    completed_at REAL,
    total_prompt_tokens INTEGER NOT NULL,
    total_completion_tokens INTEGER NOT NULL,
    halt_reason TEXT,
    final_text TEXT,
    steps_json TEXT NOT NULL
);
"""

SPEC = AgentSpec(
    name="legacy",
    version="0.1.0",
    system_prompt="old",
    capabilities=[],
    model=ModelRef(provider="fake", name="none", context_window=4096, max_output_tokens=256),
)


def build_legacy_registry(path):
    """A registry written by the pre-tenant code, with two rows."""
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA)
    conn.execute(
        "INSERT INTO agents (name, description, priority, created_at, updated_at) "
        "VALUES ('legacy','an old agent','medium',100.0,100.0)"
    )
    conn.execute(
        "INSERT INTO versions "
        "(agent_name, version, spec_json, notes, tags_json, source, created_at) "
        "VALUES ('legacy','0.1.0',?,'n','[]','file',100.0)",
        (SPEC.model_dump_json(),),
    )
    conn.commit()
    conn.close()


def build_legacy_traces(path):
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_TRACE_SCHEMA)
    conn.execute(
        "INSERT INTO traces (id, agent, agent_version, task, status, started_at, "
        "completed_at, total_prompt_tokens, total_completion_tokens, halt_reason, "
        "final_text, steps_json) "
        "VALUES ('run-old','legacy','0.1.0','old task','success',1.0,2.0,5,6,NULL,'x','[]')"
    )
    conn.commit()
    conn.close()


class TestRegistryMigration:
    def test_a_pre_tenant_database_opens(self, tmp_path):
        path = tmp_path / "agents.db"
        build_legacy_registry(path)
        AgentRegistry(path)  # must not raise

    def test_existing_agents_survive(self, tmp_path):
        path = tmp_path / "agents.db"
        build_legacy_registry(path)

        registry = AgentRegistry(path)

        assert [a.name for a in registry.list()] == ["legacy"]
        assert registry.get("legacy", "0.1.0") is not None

    def test_existing_versions_survive(self, tmp_path):
        path = tmp_path / "agents.db"
        build_legacy_registry(path)

        versions = AgentRegistry(path).versions("legacy")

        assert [v.version for v in versions] == ["0.1.0"]
        assert versions[0].spec.system_prompt == "old"

    def test_migrated_rows_belong_to_default(self, tmp_path):
        """They were written by a single shared identity, so `default` is honest.

        Reattributing them to a named tenant would be a guess about who ran
        them before tenants existed.
        """
        path = tmp_path / "agents.db"
        build_legacy_registry(path)

        registry = AgentRegistry(path)

        # `get()` returns the AgentSpec; the tenant lives on the record.
        assert registry.get_version("legacy", "0.1.0").tenant == "default"
        assert registry.list(tenant="default")
        assert registry.list(tenant="nobody") == []

    def test_no_temporary_tables_are_left_behind(self, tmp_path):
        path = tmp_path / "agents.db"
        build_legacy_registry(path)

        AgentRegistry(path)

        conn = sqlite3.connect(path)
        tables = [
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        conn.close()

        assert not any(t.endswith("_pre_tenant") for t in tables), tables

    def test_migration_is_idempotent(self, tmp_path):
        """Opening twice must not rebuild twice or duplicate rows."""
        path = tmp_path / "agents.db"
        build_legacy_registry(path)

        first = AgentRegistry(path)
        second = AgentRegistry(path)

        assert len(first.list()) == len(second.list()) == 1
        assert len(second.versions("legacy")) == 1

    def test_tenants_can_register_the_same_name_afterwards(self, tmp_path):
        """The rebuild is what makes the new primary key take effect."""
        path = tmp_path / "agents.db"
        build_legacy_registry(path)

        registry = AgentRegistry(path)
        other = SPEC.model_copy(update={"name": "legacy"})
        registry.register(other, tenant="bob")

        assert len(registry.list(tenant="default")) == 1
        assert len(registry.list(tenant="bob")) == 1


class TestTraceMigration:
    def test_a_pre_tenant_database_opens(self, tmp_path):
        path = tmp_path / "traces.db"
        build_legacy_traces(path)
        SqliteTraceStore(path)

    def test_existing_traces_survive(self, tmp_path):
        path = tmp_path / "traces.db"
        build_legacy_traces(path)

        store = SqliteTraceStore(path)

        assert store.get("run-old") is not None
        assert store.get("run-old").task == "old task"

    def test_migrated_traces_belong_to_default(self, tmp_path):
        path = tmp_path / "traces.db"
        build_legacy_traces(path)

        store = SqliteTraceStore(path)

        assert store.get("run-old", tenant="default") is not None
        assert store.get("run-old", tenant="nobody") is None

    def test_migration_is_idempotent(self, tmp_path):
        path = tmp_path / "traces.db"
        build_legacy_traces(path)

        SqliteTraceStore(path)
        store = SqliteTraceStore(path)

        assert len(store.list()) == 1

    def test_a_fresh_database_has_the_column(self, tmp_path):
        store = SqliteTraceStore(tmp_path / "new.db")
        columns = {
            row[1]
            for row in store._conn().execute("PRAGMA table_info(traces)").fetchall()
        }
        assert "tenant" in columns


class TestTenantScoping:
    def _trace(self, trace_id, tenant):
        return RunTrace(
            id=trace_id, agent="a", agent_version="0.1.0", task="t", tenant=tenant
        )

    def test_memory_store_scopes_list(self):
        store = MemoryTraceStore()
        store.save(self._trace("run-a", "alice"))
        store.save(self._trace("run-b", "bob"))

        assert [t.id for t in store.list(tenant="alice")] == ["run-a"]
        assert [t.id for t in store.list(tenant="bob")] == ["run-b"]

    def test_memory_store_hides_another_tenants_trace(self):
        store = MemoryTraceStore()
        store.save(self._trace("run-a", "alice"))
        assert store.get("run-a", tenant="bob") is None

    def test_memory_store_unscoped_still_works(self):
        store = MemoryTraceStore()
        store.save(self._trace("run-a", "alice"))
        store.save(self._trace("run-b", "bob"))
        assert len(store.list()) == 2

    def test_sqlite_store_scopes_list(self, tmp_path):
        store = SqliteTraceStore(tmp_path / "t.db")
        store.save(self._trace("run-a", "alice"))
        store.save(self._trace("run-b", "bob"))

        assert [t.id for t in store.list(tenant="alice")] == ["run-a"]
        assert [t.id for t in store.list(tenant="bob")] == ["run-b"]
        assert len(store.list()) == 2

    def test_sqlite_store_hides_another_tenants_trace(self, tmp_path):
        store = SqliteTraceStore(tmp_path / "t.db")
        store.save(self._trace("run-a", "alice"))
        assert store.get("run-a", tenant="bob") is None

    def test_a_trace_with_no_tenant_is_visible_to_default_only(self, tmp_path):
        store = SqliteTraceStore(tmp_path / "t.db")
        store.save(self._trace("run-a", None))

        assert store.get("run-a", tenant="default") is not None
        assert store.get("run-a", tenant="alice") is None

    def test_tenant_survives_a_resave(self, tmp_path):
        store = SqliteTraceStore(tmp_path / "t.db")
        trace = self._trace("run-a", "alice")
        store.save(trace)
        store.save(trace, final=True)
        assert store.get("run-a").tenant == "alice"