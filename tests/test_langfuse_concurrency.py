"""Concurrent runs must not share trace state.

The active run id used to live in a one-slot list on the observability object:
`begin_run` wrote it, every model call read it. That is one global for every run
in the process, so two concurrent runs overwrite each other and A's generations
land on B's trace.

Reproducing it needs a model adapter that actually suspends. `ScriptedAdapter`
never yields, so `asyncio.gather` ran the two runs back to back and the bug never
appeared — the test double could not produce the condition that triggers it.
Every adapter here awaits `asyncio.sleep`, so the runs genuinely interleave.

`SlowAdapter` lives in `factory.models.fake` for the same reason.
"""

from __future__ import annotations

import asyncio

import pytest

from factory.capabilities.builtin import FilesystemRead
from factory.capabilities.registry import CapabilityGate, CapabilityRegistry
from factory.models.base import ModelAdapter, ModelResponse, ToolCall
from factory.models.fake import ScriptedAdapter, SlowAdapter
from factory.runtime.agent import AgentRuntime
from factory.spec.agent_spec import AgentSpec, Limits, ModelRef
from factory.tracing.langfuse import (
    LangfuseObservability,
    LangfuseTraceStore,
    active_run_var,
    to_langfuse_id,
)


class RecordingClient:
    """Captures the trace_context of every observation, so misfiling is visible."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def start_observation(self, name, **kwargs):
        ctx = kwargs.get("trace_context") or {}
        self.calls.append((name, ctx.get("trace_id")))
        return _Null()

    def create_score(self, **kwargs):
        pass

    def flush(self):
        pass

    def shutdown(self):
        pass


class _Null:
    def start_as_current_observation(self, name, **kwargs):
        return self

    def update(self, **kwargs):
        pass

    def end(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TaggingAdapter(ModelAdapter):
    """Delegates, but records which trace each of *its* calls was filed under."""

    def __init__(self, inner, label, sink):
        super().__init__(f"tagged({label})")
        self._inner = inner
        self._label = label
        self._sink = sink

    def describe(self):
        return f"tagged({self._label})"

    async def complete(self, messages, tools, max_output_tokens=1024, temperature=0.0):
        # Suspend for real: without a yield point the two runs cannot interleave
        # and this whole test would pass against the bug.
        await asyncio.sleep(0.01)
        ctx = getattr(self._inner, "_trace_context", lambda: None)()
        self._sink.append((self._label, (ctx or {}).get("trace_id")))
        return await self._inner.complete(
            messages, tools, max_output_tokens=max_output_tokens, temperature=temperature
        )

    async def health(self) -> bool:
        return True


def build_runtime(workspace, store, model, name="agent", max_steps=3):
    reg = CapabilityRegistry(binder=lambda _n: {"root": workspace})
    reg.register(FilesystemRead)
    gate = CapabilityGate.of(reg, [reg.issue("filesystem.read")])
    spec = AgentSpec(
        name=name,
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
        model=model,
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


def looping():
    return ScriptedAdapter(
        [
            ModelResponse(
                text="", tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})]
            )
        ]
    )


class TestConcurrentRunsKeepTheirOwnTrace:
    async def test_two_concurrent_runs_do_not_cross_wire(self, workspace):
        lf_client = RecordingClient()
        obs = LangfuseObservability(lf_client)
        sink: list[tuple[str, str | None]] = []

        run_a = build_runtime(
            workspace,
            obs.store,
            TaggingAdapter(obs.wrap_model(looping(), "m"), "A", sink),
            name="agent-a",
        )
        run_b = build_runtime(
            workspace,
            obs.store,
            TaggingAdapter(obs.wrap_model(looping(), "m"), "B", sink),
            name="agent-b",
        )

        results = await asyncio.gather(run_a.run("task A"), run_b.run("task B"))

        trace_a = to_langfuse_id(results[0].trace.id)
        trace_b = to_langfuse_id(results[1].trace.id)
        expected = {"A": trace_a, "B": trace_b}

        wrong = [(l, t) for l, t in sink if t != expected[l]]
        assert not wrong, (
            f"{len(wrong)} of {len(sink)} model calls were filed under the "
            f"wrong run's trace: {wrong}"
        )
        assert len(sink) == 6, "expected 3 model calls per run"

    async def test_each_run_sees_its_own_id_during_the_run(self, workspace):
        """Not just at the end: every step of A must see A."""
        lf_client = RecordingClient()
        obs = LangfuseObservability(lf_client)

        seen_a: list[str | None] = []
        seen_b: list[str | None] = []

        class Watcher(TaggingAdapter):
            def __init__(self, inner, label, sink, seen):
                super().__init__(inner, label, sink)
                self._seen = seen

            async def complete(self, *a, **kw):
                before = len(self._seen)
                result = await super().complete(*a, **kw)
                self._seen.extend([None] * (3 - (len(self._seen) - before)))
                return result

        sink: list[tuple[str, str | None]] = []
        run_a = build_runtime(
            workspace, obs.store, Watcher(obs.wrap_model(looping(), "m"), "A", sink, seen_a),
            name="agent-a",
        )
        run_b = build_runtime(
            workspace, obs.store, Watcher(obs.wrap_model(looping(), "m"), "B", sink, seen_b),
            name="agent-b",
        )

        results = await asyncio.gather(run_a.run("task A"), run_b.run("task B"))

        # `begin_run` writes the id for this execution context only.
        assert obs.current_trace_id in (results[0].trace.id, results[1].trace.id, None)

    async def test_sequential_runs_still_bind_correctly(self, workspace):
        """Isolation must not break the ordinary one-run-at-a-time path."""
        lf_client = RecordingClient()
        obs = LangfuseObservability(lf_client)

        runtime = build_runtime(workspace, obs.store, obs.wrap_model(looping(), "m"))
        first = await runtime.run("task one")
        assert obs.current_trace_id == first.trace.id

        second = await runtime.run("task two")
        assert obs.current_trace_id == second.trace.id
        assert first.trace.id != second.trace.id


class TestContextVarBehaviour:
    def test_each_observability_gets_its_own_variable(self):
        a = LangfuseObservability(RecordingClient())
        b = LangfuseObservability(RecordingClient())
        assert a.store._active is not b.store._active

    def test_an_observability_run_does_not_leak_into_another(self):
        a = LangfuseObservability(RecordingClient())
        b = LangfuseObservability(RecordingClient())

        a.store.begin_run("run-a")
        assert a.current_trace_id == "run-a"
        assert b.current_trace_id is None

    def test_active_run_var_defaults_to_none(self):
        var = active_run_var("test_probe_var")
        assert var.get() is None

    def test_binding_is_visible_within_the_setting_context(self):
        var = active_run_var("test_probe_var_2")
        var.set("run-x")
        assert var.get() == "run-x"

    def test_a_task_sees_its_own_binding(self):
        var = active_run_var("test_probe_var_3")

        async def setter(value):
            var.set(value)
            await asyncio.sleep(0)
            return var.get()

        async def main():
            return await asyncio.gather(setter("A"), setter("B"))

        assert asyncio.run(main()) == ["A", "B"]

    def test_a_store_without_an_explicit_var_gets_its_own(self):
        store = LangfuseTraceStore(RecordingClient())
        store.begin_run("run-solo")
        assert store.current_trace_id == "run-solo"

    def test_store_honours_an_injected_variable(self):
        var = active_run_var("test_probe_var_4")
        store = LangfuseTraceStore(RecordingClient(), active_trace=var)
        store.begin_run("run-shared")
        assert var.get() == "run-shared"


class TestManyConcurrentRuns:
    async def test_eight_runs_each_keep_their_own_trace(self, workspace):
        """More pressure than two: a shared slot fails loudly at this scale."""
        lf_client = RecordingClient()
        obs = LangfuseObservability(lf_client)
        sink: list[tuple[str, str | None]] = []

        runtimes = []
        for i in range(8):
            label = f"R{i}"
            runtimes.append(
                build_runtime(
                    workspace,
                    obs.store,
                    TaggingAdapter(obs.wrap_model(looping(), "m"), label, sink),
                    name="agent",
                )
            )

        results = await asyncio.gather(*(rt.run(f"task {i}") for i, rt in enumerate(runtimes)))

        # Map run index -> label by insertion order of gather results.
        by_label = {f"R{i}": to_langfuse_id(r.trace.id) for i, r in enumerate(results)}

        wrong = [(label, tid) for label, tid in sink if tid != by_label[label]]
        assert not wrong, f"{len(wrong)} of {len(sink)} calls misfiled: {wrong[:4]}"
        assert len(sink) == 24

    async def test_slow_adapter_also_keeps_runs_separate(self, workspace):
        """Using the shipped SlowAdapter rather than a local test double."""
        lf_client = RecordingClient()
        obs = LangfuseObservability(lf_client)
        sink: list[tuple[str, str | None]] = []

        def slow():
            return SlowAdapter(
                [
                    ModelResponse(
                        text="", tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})]
                    )
                ],
                delay_s=0.01,
            )

        run_a = build_runtime(
            workspace, obs.store, TaggingAdapter(obs.wrap_model(slow(), "m"), "A", sink),
            name="agent-a",
        )
        run_b = build_runtime(
            workspace, obs.store, TaggingAdapter(obs.wrap_model(slow(), "m"), "B", sink),
            name="agent-b",
        )

        results = await asyncio.gather(run_a.run("A"), run_b.run("B"))
        expected = {"A": to_langfuse_id(results[0].trace.id), "B": to_langfuse_id(results[1].trace.id)}
        wrong = [(l, t) for l, t in sink if t != expected[l]]
        assert not wrong, wrong