"""Agent Registry — the missing piece between "specs in files" and a factory.

A registry makes specs addressable. Without it, `agents/reader.yaml` is a file
you happen to know the path of; with it, `reader@0.1.0` is a thing you can list,
version, diff, and attach a measured pass rate to.

That last part is the point. An eval baseline is only meaningful if it is keyed
by agent *and version* — "reader v0.1.0 passed 100%, v0.2.0 passed 87%, here is
the diff" is what turns the eval harness from a tripwire into a ratchet.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from factory.spec.agent_spec import AgentSpec

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    tenant       TEXT NOT NULL DEFAULT 'default',
    name         TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    priority     TEXT NOT NULL DEFAULT 'medium',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    PRIMARY KEY (tenant, name)
);

CREATE TABLE IF NOT EXISTS versions (
    tenant       TEXT NOT NULL DEFAULT 'default',
    agent_name   TEXT NOT NULL,
    version      TEXT NOT NULL,
    spec_json    TEXT NOT NULL,
    notes        TEXT NOT NULL DEFAULT '',
    tags_json    TEXT NOT NULL DEFAULT '[]',
    source       TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    PRIMARY KEY (tenant, agent_name, version)
);

CREATE INDEX IF NOT EXISTS idx_versions_agent ON versions(agent_name);
"""

#: Indexes over `tenant`, created *after* migration rather than in SCHEMA.
#:
#: They cannot live in SCHEMA: `executescript` runs the whole script, and on a
#: pre-tenant database `CREATE TABLE IF NOT EXISTS agents` is a no-op against the
#: old shape — so a following `CREATE INDEX ... ON agents(tenant)` would fail
#: with "no such column", before the migration ever ran. Which is exactly what
#: happened the first time.
TENANT_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_versions_tenant ON versions(tenant);
CREATE INDEX IF NOT EXISTS idx_agents_tenant ON agents(tenant);
"""

#: Rows created before tenants existed belong to whoever could reach the API,
#: which was a single shared identity. `default` preserves them.
LEGACY_TENANT = "default"


class RegistryError(RuntimeError):
    pass


@dataclass
class AgentVersion:
    agent: str
    version: str
    spec: AgentSpec
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = ""
    created_at: float = 0.0
    #: Which tenant owns this agent. Two tenants may each register `reader`.
    tenant: str = LEGACY_TENANT

    def summary(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "version": self.version,
            "description": self.spec.description,
            "priority": self.spec.priority.value,
            "capabilities": list(self.spec.capabilities),
            "model": f"{self.spec.model.provider}/{self.spec.model.name}",
            "max_steps": self.spec.limits.max_steps,
            "notes": self.notes,
            "tags": list(self.tags),
            "source": self.source,
            "created_at": self.created_at,
        }


@dataclass
class AgentSummary:
    name: str
    description: str
    priority: str
    versions: int
    latest: str | None
    created_at: float
    updated_at: float


def version_key(version: str) -> tuple[int, ...]:
    """Order semver-ish strings numerically, ignoring any pre-release suffix.

    '1.2.3-rc1' must sort as 1.2.3, not as 1.2.31 or 1.2.0. Leading digits win;
    a chunk with no digits sorts as 0 so it never outranks a real version.
    """
    parts: list[int] = []
    for chunk in version.split(".")[:3]:
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


class AgentRegistry:
    """SQLite-backed registry of agent specs and their versions."""

    def __init__(self, path: str | Path = ".factory/agents.db") -> None:
        self._path = str(path)
        self._memory = self._path == ":memory:"
        if not self._memory:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        if self._memory:
            # Each new connection to ":memory:" gets a fresh, empty database, so
            # schema creation would be discarded immediately. Keep one open
            # connection for the lifetime of the registry instead.
            self._shared = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared.executescript(SCHEMA)
            self._shared.executescript(TENANT_INDEXES)
            self._shared.commit()
        else:
            self._shared = None
            with self._conn() as c:
                c.executescript(SCHEMA)
                self._migrate(c)
                # Only now does `tenant` exist on both tables.
                c.executescript(TENANT_INDEXES)
                c.commit()

    def _conn(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        # check_same_thread=False: FastAPI's sync endpoints may open the
        # registry from a worker thread. WAL keeps concurrent readers happy.
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Rebuild tables that predate the `tenant` column.

        `CREATE TABLE IF NOT EXISTS` is a no-op against an existing table, so a
        registry created before tenants existed would keep its old primary key
        — agents named identically in different tenants would collide, and reads
        could not be scoped at all. Detected by inspecting the primary key rather
        than a version number, so it works on any pre-tenant database.

        Rows are copied across as `default`: they were written by a single
        shared identity, and reattributing them to a specific tenant would be a
        guess.
        """
        for table in ("agents", "versions"):
            columns = {
                row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if not columns or "tenant" in columns:
                continue

            conn.execute(f"ALTER TABLE {table} RENAME TO {table}_pre_tenant")

        if not any(
            r[0].endswith("_pre_tenant")
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ):
            return

        if "agents_pre_tenant" in {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }:
            conn.execute(
                """
                CREATE TABLE agents (
                    tenant TEXT NOT NULL DEFAULT 'default',
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    priority TEXT NOT NULL DEFAULT 'medium',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant, name)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO agents
                    (tenant, name, description, priority, created_at, updated_at)
                SELECT ?, name, description, priority, created_at, updated_at
                FROM agents_pre_tenant
                """,
                (LEGACY_TENANT,),
            )
            conn.execute("DROP TABLE agents_pre_tenant")

        if "versions_pre_tenant" in {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }:
            conn.execute(
                """
                CREATE TABLE versions (
                    tenant TEXT NOT NULL DEFAULT 'default',
                    agent_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    PRIMARY KEY (tenant, agent_name, version)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO versions
                    (tenant, agent_name, version, spec_json, notes, tags_json,
                     source, created_at)
                SELECT ?, agent_name, version, spec_json, notes, tags_json,
                       source, created_at
                FROM versions_pre_tenant
                """,
                (LEGACY_TENANT,),
            )
            conn.execute("DROP TABLE versions_pre_tenant")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_versions_agent ON versions(agent_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_versions_tenant ON versions(tenant)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agents_tenant ON agents(tenant)")

    def close(self) -> None:
        if self._shared is not None:
            self._shared.close()
            self._shared = None

    # ── writes ──────────────────────────────────────────────────

    def register(
        self,
        spec: AgentSpec,
        notes: str = "",
        tags: list[str] | None = None,
        source: str = "",
        replace: bool = False,
        tenant: str = LEGACY_TENANT,
    ) -> AgentVersion:
        if spec.name != spec.name.strip():
            raise RegistryError("agent name has surrounding whitespace")

        tags = tags or []
        now = time.time()

        with self._conn() as c:
            row = c.execute(
                "SELECT version FROM versions "
                "WHERE tenant=? AND agent_name=? AND version=?",
                (tenant, spec.name, spec.version),
            ).fetchone()

            if row and not replace:
                raise RegistryError(
                    f"{spec.name}@{spec.version} already exists. "
                    f"Bump the version, or pass replace=True."
                )

            created = row[0] if row is not None else now
            c.execute(
                """INSERT OR REPLACE INTO versions
                   (tenant, agent_name, version, spec_json, notes, tags_json,
                    source, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    tenant,
                    spec.name,
                    spec.version,
                    spec.model_dump_json(),
                    notes,
                    json.dumps(tags),
                    source,
                    created,
                ),
            )
            c.execute(
                """INSERT INTO agents
                   (tenant, name, description, priority, created_at, updated_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(tenant, name) DO UPDATE SET
                     description=excluded.description,
                     priority=excluded.priority,
                     updated_at=excluded.updated_at""",
                (tenant, spec.name, spec.description, spec.priority.value, now, now),
            )

        return AgentVersion(
            agent=spec.name,
            version=spec.version,
            spec=spec,
            notes=notes,
            tags=tags,
            source=source,
            created_at=now,
            tenant=tenant,
        )

    def register_file(self, path: str | Path, **kwargs) -> AgentVersion:
        from factory.spec.loader import load_spec

        spec = load_spec(path)
        kwargs.setdefault("source", str(path))
        return self.register(spec, **kwargs)

    def delete(
        self, name: str, version: str | None = None, tenant: str = LEGACY_TENANT
    ) -> int:
        """Delete one version, or every version. Returns rows removed."""
        with self._conn() as c:
            if version:
                cur = c.execute(
                    "DELETE FROM versions WHERE tenant=? AND agent_name=? AND version=?",
                    (tenant, name, version),
                )
                removed = cur.rowcount
            else:
                cur = c.execute(
                    "DELETE FROM versions WHERE tenant=? AND agent_name=?",
                    (tenant, name),
                )
                removed = cur.rowcount
                c.execute(
                    "DELETE FROM agents WHERE tenant=? AND name=?", (tenant, name)
                )
        return removed

    # ── reads ───────────────────────────────────────────────────
    #
    # Every read is tenant-scoped by default and omitted-tenant means
    # "unscoped", which only a caller that already knows the tenant should use.
    # The API always passes one.

    def exists(
        self, name: str, version: str | None = None, tenant: str | None = LEGACY_TENANT
    ) -> bool:
        if version:
            return self.get_version(name, version, tenant=tenant) is not None
        return any(v.agent == name for v in self.versions(name, tenant=tenant))

    def get(
        self,
        name: str,
        version: str | None = None,
        tenant: str | None = LEGACY_TENANT,
    ) -> AgentSpec | None:
        """Latest version if `version` is omitted."""
        rec = (
            self.get_version(name, version, tenant=tenant)
            if version
            else self.latest(name, tenant=tenant)
        )
        return rec.spec if rec else None

    def get_version(
        self, name: str, version: str, tenant: str | None = LEGACY_TENANT
    ) -> AgentVersion | None:
        with self._conn() as c:
            if tenant is None:
                row = c.execute(
                    "SELECT spec_json, notes, tags_json, source, created_at "
                    "FROM versions WHERE agent_name=? AND version=?",
                    (name, version),
                ).fetchone()
            else:
                row = c.execute(
                    "SELECT spec_json, notes, tags_json, source, created_at, tenant "
                    "FROM versions WHERE tenant=? AND agent_name=? AND version=?",
                    (tenant, name, version),
                ).fetchone()
        return self._to_version(name, version, row) if row else None

    def latest(
        self, name: str, tenant: str | None = LEGACY_TENANT
    ) -> AgentVersion | None:
        recs = self.versions(name, tenant=tenant)
        return recs[-1] if recs else None

    def versions(
        self, name: str, tenant: str | None = LEGACY_TENANT
    ) -> list[AgentVersion]:
        with self._conn() as c:
            if tenant is None:
                rows = c.execute(
                    "SELECT version, spec_json, notes, tags_json, source, created_at "
                    "FROM versions WHERE agent_name=?",
                    (name,),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT version, spec_json, notes, tags_json, source, created_at, tenant "
                    "FROM versions WHERE tenant=? AND agent_name=?",
                    (tenant, name),
                ).fetchall()
        out = [self._to_version(name, r[0], r[1:]) for r in rows]
        return sorted(out, key=lambda v: version_key(v.version))

    def list(self, tenant: str | None = LEGACY_TENANT) -> list[AgentSummary]:
        with self._conn() as c:
            if tenant is None:
                rows = c.execute(
                    "SELECT a.name, a.description, a.priority, a.created_at, "
                    "a.updated_at, (SELECT COUNT(*) FROM versions v "
                    "WHERE v.agent_name=a.name AND v.tenant=a.tenant) "
                    "FROM agents a ORDER BY a.name"
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT a.name, a.description, a.priority, a.created_at, "
                    "a.updated_at, (SELECT COUNT(*) FROM versions v "
                    "WHERE v.agent_name=a.name AND v.tenant=a.tenant) "
                    "FROM agents a WHERE a.tenant=? ORDER BY a.name",
                    (tenant,),
                ).fetchall()

        out: list[AgentSummary] = []
        for name, desc, priority, created, updated, count in rows:
            recs = self.versions(name, tenant=tenant)
            out.append(
                AgentSummary(
                    name=name,
                    description=desc,
                    priority=priority,
                    versions=int(count),
                    latest=recs[-1].version if recs else None,
                    created_at=created,
                    updated_at=updated,
                )
            )
        return out

    def _to_version(self, name: str, version: str, row: tuple) -> AgentVersion:
        spec_json, notes, tags_json, source, created = row[:5]
        # Rows read through an unscoped query do not select `tenant`, so only
        # take it when the caller asked for a specific one.
        tenant = row[5] if len(row) > 5 else LEGACY_TENANT
        return AgentVersion(
            agent=name,
            version=version,
            spec=AgentSpec.model_validate_json(spec_json),
            notes=notes,
            tags=json.loads(tags_json),
            source=source,
            created_at=created,
            tenant=tenant,
        )

    # ── diff ────────────────────────────────────────────────────

    def diff(
        self,
        name: str,
        from_version: str,
        to_version: str,
        tenant: str | None = LEGACY_TENANT,
    ) -> dict[str, Any]:
        """Field-level change between two versions. What a ratchet needs."""
        a = self.get_version(name, from_version, tenant=tenant)
        b = self.get_version(name, to_version, tenant=tenant)
        if a is None:
            raise RegistryError(f"{name}@{from_version} not found")
        if b is None:
            raise RegistryError(f"{name}@{to_version} not found")

        changes: list[dict[str, Any]] = []
        before, after = a.spec.model_dump(mode="json"), b.spec.model_dump(mode="json")

        # `version` is the row key, not agent behaviour. Comparing it would make
        # every version bump look like a change.
        for field in sorted((set(before) | set(after)) - {"version"}):
            old, new = before.get(field), after.get(field)
            if old != new:
                changes.append({"field": field, "from": old, "to": new})

        return {
            "agent": name,
            "from": from_version,
            "to": to_version,
            "changed": bool(changes),
            "changes": changes,
            # Capability changes are the ones that can move a security boundary.
            "capabilities_added": sorted(set(b.spec.capabilities) - set(a.spec.capabilities)),
            "capabilities_removed": sorted(set(a.spec.capabilities) - set(b.spec.capabilities)),
        }


def open_registry(path: str | Path | None = None) -> AgentRegistry:
    return AgentRegistry(path or ".factory/agents.db")