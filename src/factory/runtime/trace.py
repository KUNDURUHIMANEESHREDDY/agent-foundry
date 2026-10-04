"""Run traces. SQLite by default; in-memory for tests.

Every step is recorded: what the model was given, what it asked for, what the
gate decided, and what came back. A denied capability is recorded as a
violation, not silently dropped.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class StepRecord:
    index: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    violations: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    duration_ms: int = 0
    truncated: bool = False
    error: str | None = None


@dataclass
class RunTrace:
    id: str
    agent: str
    agent_version: str
    task: str
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    steps: list[StepRecord] = field(default_factory=list)
    final_text: str = ""
    halt_reason: str | None = None
    #: Which caller this run belongs to. `None` means the single-tenant case,
    #: which is also what rows written before this field existed carry.
    tenant: str | None = None

    @property
    def total_tokens(self) -> int:
        """Authoritative token count.

        The runtime accumulates as it goes so the `max_tokens` check can fire
        mid-run, before the current step is appended. A trace rebuilt from
        persisted steps has no accumulator. Taking the max of both means the
        check is never under-counted and a reconstructed trace is never
        under-reported — an eval that under-reports tokens hides cost
        regressions.
        """
        return max(
            self.total_prompt_tokens + self.total_completion_tokens,
            sum(s.prompt_tokens + s.completion_tokens for s in self.steps),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class TraceStore(Protocol):
    def begin_run(self, trace_id: str) -> None:
        """Called by the runtime when a run starts, before any model call.

        Observability backends use this to attach later observations to the
        right trace. The default is a no-op, so existing stores are unaffected.
        """
        ...

    def save(self, trace: RunTrace, *, final: bool = False) -> None:
        """Persist the run's state so far.

        The runtime saves after every step, not only at the end, so that a run
        killed mid-flight still leaves recoverable state. That makes `save` an
        upsert, and it is the right behaviour for a local store.

        It is the wrong behaviour for an *external* exporter. Langfuse used to
        export on every call, so one 5-step run produced six `factory.run`
        observations, each a cumulative snapshot with more steps than the last.
        Any aggregate over "steps per run" then saw a sawtooth rather than the
        run's shape.

        `final=True` marks the last save of a run. Stores that only accumulate
        state can ignore it; stores that export elsewhere must not.
        """
        ...

    def get(self, run_id: str) -> RunTrace | None: ...
    def list(self, limit: int = 50) -> list[RunTrace]: ...


class MemoryTraceStore:
    def __init__(self) -> None:
        self._traces: dict[str, RunTrace] = {}

    def begin_run(self, trace_id: str) -> None:
        return None

    def save(self, trace: RunTrace, *, final: bool = False) -> None:
        # Upsert, so `final` changes nothing: the newest snapshot wins.
        self._traces[trace.id] = trace

    def get(self, run_id: str, tenant: str | None = None) -> RunTrace | None:
        found = self._traces.get(run_id)
        if found is None:
            return None
        if tenant is not None and (found.tenant or "default") != tenant:
            return None
        return found

    def list(self, limit: int = 50, tenant: str | None = None) -> list[RunTrace]:
        rows = self._traces.values()
        if tenant is not None:
            rows = [t for t in rows if (t.tenant or "default") == tenant]
        return sorted(rows, key=lambda t: t.started_at, reverse=True)[:limit]


class SqliteTraceStore:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def begin_run(self, trace_id: str) -> None:
        return None

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _init(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
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
                )
                """
            )
            self._migrate(c)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add `tenant` to a database created before tenants existed.

        `CREATE TABLE IF NOT EXISTS` silently does nothing when the table is
        already there with the old shape, so without this every insert would
        fail on an existing `.factory/traces.db` — or worse, succeed into a table
        with no way to scope reads. Existing rows are attributed to `default`,
        which is the tenant a single-key deployment always was.
        """
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(traces)").fetchall()
        }
        if "tenant" in columns:
            return

        conn.execute(
            "ALTER TABLE traces ADD COLUMN tenant TEXT NOT NULL DEFAULT 'default'"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_tenant ON traces(tenant)")

    def save(self, trace: RunTrace, *, final: bool = False) -> None:
        # INSERT OR REPLACE, so a partial save followed by a final one leaves
        # exactly one row: this store is what makes dropping Langfuse's
        # mid-run exports acceptable, because a killed run is still recoverable.
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO traces
                   (id, agent, agent_version, task, status, started_at, completed_at,
                    total_prompt_tokens, total_completion_tokens, halt_reason,
                    final_text, steps_json, tenant)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trace.id,
                    trace.agent,
                    trace.agent_version,
                    trace.task,
                    trace.status,
                    trace.started_at,
                    trace.completed_at,
                    trace.total_prompt_tokens,
                    trace.total_completion_tokens,
                    trace.halt_reason,
                    trace.final_text,
                    json.dumps([asdict(s) for s in trace.steps]),
                    trace.tenant or "default",
                ),
            )

    def get(self, run_id: str, tenant: str | None = None) -> RunTrace | None:
        """Fetch one trace.

        With `tenant` given, a trace belonging to another caller is reported as
        absent rather than returned. 404 for someone else's run is deliberate:
        confirming it exists would leak that the id is real.
        """
        with self._conn() as c:
            if tenant is None:
                row = c.execute(
                    "SELECT * FROM traces WHERE id = ?", (run_id,)
                ).fetchone()
            else:
                row = c.execute(
                    "SELECT * FROM traces WHERE id = ? AND tenant = ?",
                    (run_id, tenant),
                ).fetchone()
        return self._row_to_trace(row) if row else None

    def list(self, limit: int = 50, tenant: str | None = None) -> list[RunTrace]:
        with self._conn() as c:
            if tenant is None:
                rows = c.execute(
                    "SELECT * FROM traces ORDER BY started_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM traces WHERE tenant = ? "
                    "ORDER BY started_at DESC LIMIT ?",
                    (tenant, limit),
                ).fetchall()
        return [self._row_to_trace(r) for r in rows]

    def _row_to_trace(self, row: tuple) -> RunTrace:
        # Column order must match `save`. Positional indexing is brittle once a
        # column is added: an earlier version saved `tenant` and never read it
        # back, so every trace came out unattributed while the scope checks
        # appeared to work — they were just matching "default" against "default".
        return RunTrace(
            id=row[0],
            agent=row[1],
            agent_version=row[2],
            task=row[3],
            status=row[4],
            started_at=row[5],
            completed_at=row[6],
            total_prompt_tokens=row[7],
            total_completion_tokens=row[8],
            halt_reason=row[9],
            final_text=row[10] or "",
            steps=[StepRecord(**s) for s in json.loads(row[11])],
            tenant=row[12] if len(row) > 12 else None,
        )


def build_store(memory_type: str, path: str | None) -> TraceStore:
    if memory_type == "memory" or not path:
        return MemoryTraceStore()
    return SqliteTraceStore(path)
