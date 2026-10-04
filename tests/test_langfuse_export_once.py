"""A run must reach Langfuse exactly once.

Every test here would have passed while the store exported on every save. They
could not catch it because each looked up "the first `factory.run` span" and
asserted a field on it — which is exactly as true of the first of six duplicates
as of the only one.

So these assert cardinality: one run in, one `factory.run` observation out,
carrying every step. The duplicates were cumulative, so the step-count sequence
is asserted too. A suite that cannot count is a suite that cannot see
duplication.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from factory.capabilities.builtin import FilesystemRead
from factory.capabilities.registry import CapabilityGate, CapabilityRegistry
from factory.models.base import ModelResponse, ToolCall
from factory.models.fake import ScriptedAdapter
from factory.runtime.agent import AgentRuntime
from factory.runtime.trace import RunTrace
from factory.spec.agent_spec import AgentSpec, Limits, ModelRef
from factory.tracing.langfuse import LangfuseObservability, LangfuseTraceStore

RUN_PREFIX = "factory.run:"

# `lf_exporter`, `lf_client` and the span-clearing fixture are session-scoped in
# conftest.py. Langfuse registers a global OTel tracer provider, so a second
# lf_client in this file would re-register it and the two files would steal spans
# from each other — an order-dependent failure that only shows up in a full run.


def runs(lf_exporter) -> list:
    return lf_exporter.find_all(RUN_PREFIX)


def looping_adapter():
    """Asks for a tool every turn, so the run always reaches max_steps."""
    return ScriptedAdapter(
        [
            ModelResponse(
                text="", tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})]
            )
        ]
    )


def build_runtime(workspace: Path, store, adapter, max_steps: int = 5) -> AgentRuntime:
    reg = CapabilityRegistry(binder=lambda _n: {"root": workspace})
    reg.register(FilesystemRead)
    gate = CapabilityGate.of(reg, [reg.issue("filesystem.read")])

    spec = AgentSpec(
        name="counter",
        system_prompt="read files",
        capabilities=["filesystem.read"],
        model=ModelRef(
            provider="fake", name="none", context_window=4096, max_output_tokens=256
        ),
        limits=Limits(
            max_steps=max_steps, max_tokens=100_000, max_tool_calls=max_steps + 5
        ),
    )
    return AgentRuntime(
        spec=spec,
        model=adapter,
        gate=gate,
        store=store,
        tool_schemas=[
            {
                "name": "filesystem.read",
                "description": "read",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    return tmp_path


class TestOneExportPerRun:
    def test_a_five_step_run_exports_once(self, lf_exporter, lf_client, workspace):
        store = LangfuseTraceStore(lf_client)
        runtime = build_runtime(workspace, store, looping_adapter(), max_steps=5)

        result = asyncio.run(runtime.run("read a.txt"))
        assert len(result.trace.steps) == 5, "test setup: expected a 5-step run"
        lf_client.flush()

        found = runs(lf_exporter)
        assert len(found) == 1, f"expected 1 run observation, got {len(found)}"

    def test_no_duplicate_is_left_behind(self, lf_exporter, lf_client, workspace):
        """The duplicates were cumulative: 1, 2, 3, 4, 5, 5 steps.

        One export of a 5-step run reads as exactly [5].
        """
        store = LangfuseTraceStore(lf_client)
        runtime = build_runtime(workspace, store, looping_adapter(), max_steps=5)
        asyncio.run(runtime.run("read a.txt"))
        lf_client.flush()

        counts = [
            lf_exporter.meta(found.name, "steps") for found in runs(lf_exporter)
        ]
        assert counts == [5], f"expected a single export of 5 steps, saw {counts}"

    def test_the_single_observation_carries_every_step(self, lf_exporter, lf_client, workspace):
        store = LangfuseTraceStore(lf_client)
        runtime = build_runtime(workspace, store, looping_adapter(), max_steps=5)
        asyncio.run(runtime.run("read a.txt"))
        lf_client.flush()

        assert lf_exporter.find_all("step-") != []
        assert len(lf_exporter.find_all("step-")) == 5

    def test_repeated_final_saves_export_once(self, lf_exporter, lf_client):
        """Insurance: a caller marking a final save twice must not double-export."""
        store = LangfuseTraceStore(lf_client)
        trace = RunTrace(
            id="run-once", agent="reader", agent_version="0.1.0", task="t"
        )
        trace.status = "success"

        for _ in range(5):
            store.save(trace, final=True)
        lf_client.flush()

        assert len(runs(lf_exporter)) == 1
        assert store.exported_runs == ["run-once"]

    def test_two_runs_export_twice(self, lf_exporter, lf_client):
        store = LangfuseTraceStore(lf_client)
        for run_id in ("run-a", "run-b"):
            trace = RunTrace(
                id=run_id, agent="reader", agent_version="0.1.0", task="t"
            )
            trace.status = "success"
            store.save(trace, final=True)
        lf_client.flush()

        assert len(runs(lf_exporter)) == 2
        assert store.exported_runs == ["run-a", "run-b"]

    def test_success_run_exports_once(self, lf_exporter, lf_client, workspace):
        store = LangfuseTraceStore(lf_client)
        runtime = build_runtime(
            workspace, store, ScriptedAdapter.text_then_final("all done"), max_steps=5
        )
        result = asyncio.run(runtime.run("read a.txt"))

        assert result.status == "success"
        lf_client.flush()
        assert len(runs(lf_exporter)) == 1

    def test_halted_run_exports_once_at_error_level(self, lf_exporter, lf_client, workspace):
        store = LangfuseTraceStore(lf_client)
        runtime = build_runtime(workspace, store, looping_adapter(), max_steps=3)
        result = asyncio.run(runtime.run("read a.txt"))

        assert result.status == "max_steps_exceeded"
        lf_client.flush()

        found = runs(lf_exporter)
        assert len(found) == 1
        assert lf_exporter.attr(found[0].name, "langfuse.observation.level") == "ERROR"

    def test_model_error_run_exports_once(self, lf_exporter, lf_client, workspace):
        """This path saves twice: once partial, once in `_halt`."""
        from factory.models.fake import FailingAdapter

        store = LangfuseTraceStore(lf_client)
        runtime = build_runtime(workspace, store, FailingAdapter(), max_steps=5)
        result = asyncio.run(runtime.run("read a.txt"))

        assert result.status == "model_error"
        lf_client.flush()
        assert len(runs(lf_exporter)) == 1


class TestPartialSavesAreNotExported:
    def test_a_non_final_save_sends_nothing(self, lf_exporter, lf_client):
        store = LangfuseTraceStore(lf_client)
        trace = RunTrace(
            id="run-partial", agent="reader", agent_version="0.1.0", task="t"
        )
        trace.status = "running"

        store.save(trace)
        lf_client.flush()

        assert runs(lf_exporter) == []
        assert store.exported_runs == []

    def test_intermediate_saves_then_one_final(self, lf_exporter, lf_client):
        """The realistic sequence: several partial saves, then one final."""
        store = LangfuseTraceStore(lf_client)
        trace = RunTrace(
            id="run-seq", agent="reader", agent_version="0.1.0", task="t"
        )
        trace.status = "running"

        for _ in range(3):
            store.save(trace)
        lf_client.flush()
        assert runs(lf_exporter) == []

        trace.status = "success"
        store.save(trace, final=True)
        lf_client.flush()

        assert len(runs(lf_exporter)) == 1


class TestLocalStateIsUnaffected:
    """Dropping partial exports must not cost us the local record."""

    def test_partial_saves_are_still_readable_locally(self, lf_client):
        store = LangfuseTraceStore(lf_client)
        trace = RunTrace(
            id="run-local", agent="reader", agent_version="0.1.0", task="t"
        )
        trace.status = "running"

        store.save(trace)
        assert store.get("run-local") is trace
        assert store.list() == [trace]

    def test_a_run_that_never_completes_is_retained_locally(self, lf_exporter, lf_client):
        """The regression risk of not exporting partials.

        Not exported — correct, it never finished — but the local store keeps it,
        which is why dropping partial exports is acceptable.
        """
        store = LangfuseTraceStore(lf_client)
        trace = RunTrace(
            id="run-crashed", agent="reader", agent_version="0.1.0", task="t"
        )
        trace.status = "running"

        store.save(trace)
        lf_client.flush()

        assert runs(lf_exporter) == []
        assert store.get("run-crashed") is trace

    def test_runtime_sends_partials_and_exactly_one_final(self, lf_client, workspace):
        """The runtime must mark its last save, or nothing would ever export."""

        class Recorder:
            def __init__(self):
                self.finals: list[str] = []
                self.partials = 0

            def begin_run(self, trace_id: str) -> None:
                pass

            def save(self, trace, *, final: bool = False) -> None:
                if final:
                    self.finals.append(trace.id)
                else:
                    self.partials += 1

            def get(self, run_id):
                return None

            def list(self, limit=50):
                return []

        recorder = Recorder()
        runtime = build_runtime(workspace, recorder, looping_adapter(), max_steps=5)
        asyncio.run(runtime.run("read a.txt"))

        assert recorder.partials == 5, "one partial save per step"
        assert len(recorder.finals) == 1, "exactly one final save per run"


class TestObservabilityWiring:
    def test_observability_store_also_exports_once(self, lf_exporter, lf_client, workspace):
        obs = LangfuseObservability(lf_client)
        runtime = build_runtime(workspace, obs.store, looping_adapter(), max_steps=5)
        asyncio.run(runtime.run("read a.txt"))
        lf_client.flush()

        assert len(runs(lf_exporter)) == 1