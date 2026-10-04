from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from factory.capabilities.registry import registry_for
from factory.compiler import compile_agent
from factory.models.base import Message, ModelResponse, ToolCall
from factory.models.fake import FailingAdapter, ScriptedAdapter
from factory.runtime.trace import RunTrace, StepRecord
from factory.spec.loader import load_spec
from factory.tracing.langfuse import (
    LangfuseModelAdapter,
    LangfuseNotConfigured,
    LangfuseObservability,
    LangfuseSettings,
    LangfuseTraceStore,
    to_langfuse_id,
)
from tests.langfuse_helpers import CapturingExporter, make_test_client

AGENTS = Path(__file__).resolve().parents[1] / "agents"

# `lf_exporter`, `lf_client` and the span-clearing fixture live in conftest.py: one
# session-scoped Langfuse lf_client for the whole run, or the tests interfere.


def sample_trace(trace_id: str = "run-abc123") -> RunTrace:
    trace = RunTrace(id=trace_id, agent="reader", agent_version="0.1.0", task="read a.txt")
    trace.status = "success"
    trace.final_text = "the file says hello"
    trace.steps = [
        StepRecord(
            index=1,
            model="qwen2.5:7b",
            prompt_tokens=100,
            completion_tokens=20,
            tool_calls=[{"name": "filesystem.read", "arguments": {"path": "a.txt"}}],
            observations=[{"ok": True, "path": "a.txt", "content": "hello"}],
            duration_ms=12,
        ),
        StepRecord(
            index=2,
            model="qwen2.5:7b",
            prompt_tokens=20,
            completion_tokens=8,
            text="the file says hello",
        ),
    ]
    return trace


class TestIdMapping:
    def test_produces_32_hex_chars(self):
        """Langfuse raises ValueError on int(trace_id, 16) for non-hex ids."""
        mapped = to_langfuse_id("run-abc123")
        assert len(mapped) == 32
        assert mapped == mapped.lower()
        int(mapped, 16)

    def test_is_deterministic(self):
        assert to_langfuse_id("run-abc123") == to_langfuse_id("run-abc123")

    def test_distinguishes_different_runs(self):
        assert to_langfuse_id("run-a") != to_langfuse_id("run-b")


class TestSettings:
    def test_missing_credentials_raise(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        with pytest.raises(LangfuseNotConfigured, match="LANGFUSE_PUBLIC_KEY"):
            LangfuseSettings.from_env()

    def test_reads_credentials_from_env(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-x")
        monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")

        settings = LangfuseSettings.from_env()
        assert settings.public_key == "pk-x"
        assert settings.host == "http://localhost:3000"

    def test_default_host_is_cloud(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-x")
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        assert LangfuseSettings.from_env().host == "https://cloud.langfuse.com"

    def test_partial_credentials_rejected(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-x")
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        with pytest.raises(LangfuseNotConfigured):
            LangfuseSettings.from_env()


class TestModelAdapterTracing:
    @pytest.mark.asyncio
    async def test_emits_generation_with_usage(self, lf_exporter, lf_client):
        inner = ScriptedAdapter([
            ModelResponse(text="done", prompt_tokens=120, completion_tokens=45)
        ])
        traced = LangfuseModelAdapter(inner, lf_client, model_name="qwen2.5:7b")
        await traced.complete([Message(role="user", content="hi")], [])
        lf_client.flush()

        assert lf_exporter.attr("model.complete", "langfuse.observation.type") == "generation"
        assert lf_exporter.attr("model.complete", "langfuse.observation.model.name") == "qwen2.5:7b"
        usage = lf_exporter.json_attr("model.complete", "langfuse.observation.usage_details")
        assert usage == {"input": 120, "output": 45, "total": 165}

    @pytest.mark.asyncio
    async def test_records_tool_calls_in_output(self, lf_exporter, lf_client):
        inner = ScriptedAdapter([
            ModelResponse(tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})])
        ])
        traced = LangfuseModelAdapter(inner, lf_client)
        await traced.complete([Message(role="user", content="read")], [])
        lf_client.flush()

        out = lf_exporter.json_attr("model.complete", "langfuse.observation.output")
        assert out["tool_calls"][0]["name"] == "filesystem.read"
        assert out["tool_calls"][0]["arguments"] == {"path": "a.txt"}

    @pytest.mark.asyncio
    async def test_records_messages_and_tool_names_as_input(self, lf_exporter, lf_client):
        traced = LangfuseModelAdapter(ScriptedAdapter([ModelResponse(text="x")]), lf_client)
        await traced.complete(
            [Message(role="system", content="S"), Message(role="user", content="U")],
            [{"name": "filesystem.read"}, {"name": "python.execute"}],
        )
        lf_client.flush()

        inp = lf_exporter.json_attr("model.complete", "langfuse.observation.input")
        assert inp["messages"] == [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ]
        assert inp["tools"] == ["filesystem.read", "python.execute"]

    @pytest.mark.asyncio
    async def test_model_error_is_recorded_as_error_span(self, lf_exporter, lf_client):
        traced = LangfuseModelAdapter(FailingAdapter("boom"), lf_client)
        with pytest.raises(RuntimeError, match="boom"):
            await traced.complete([Message(role="user", content="x")], [])
        lf_client.flush()

        assert lf_exporter.attr("model.complete", "langfuse.observation.level") == "ERROR"
        assert lf_exporter.attr("model.complete", "langfuse.observation.status_message") == "boom"
        assert lf_exporter.meta("model.complete", "error") == "RuntimeError"

    @pytest.mark.asyncio
    async def test_binds_to_a_factory_trace(self, lf_exporter, lf_client):
        inner = ScriptedAdapter([ModelResponse(text="x")])
        traced = LangfuseModelAdapter(inner, lf_client)
        traced.bind_trace("run-abc123")
        assert traced.bound_trace == "run-abc123"

        await traced.complete([Message(role="user", content="x")], [])
        lf_client.flush()
        assert lf_exporter.find("model.complete") is not None

    @pytest.mark.asyncio
    async def test_health_passes_through(self, lf_client):
        traced = LangfuseModelAdapter(ScriptedAdapter([ModelResponse(text="x")]), lf_client)
        assert await traced.health() is True

    def test_describe_shows_wrapping(self, lf_client):
        traced = LangfuseModelAdapter(ScriptedAdapter([ModelResponse(text="x")]), lf_client)
        assert traced.describe().startswith("langfuse(")


class TestTraceStoreExport:
    def test_exports_agent_observation(self, lf_exporter, lf_client):
        LangfuseTraceStore(lf_client).save(sample_trace(), final=True)
        lf_client.flush()

        assert lf_exporter.attr("factory.run:reader", "langfuse.observation.type") == "agent"
        assert lf_exporter.meta("factory.run:reader", "status") == "success"
        assert lf_exporter.meta("factory.run:reader", "steps") == 2
        assert lf_exporter.meta("factory.run:reader", "violations") == 0

    def test_factory_run_id_recorded_for_correlation(self, lf_exporter, lf_client):
        LangfuseTraceStore(lf_client).save(sample_trace("run-xyz"), final=True)
        lf_client.flush()
        assert lf_exporter.meta("factory.run:reader", "factory_run_id") == "run-xyz"

    def test_exports_step_spans_with_usage(self, lf_exporter, lf_client):
        LangfuseTraceStore(lf_client).save(sample_trace(), final=True)
        lf_client.flush()

        assert lf_exporter.find("step-1") is not None
        usage = lf_exporter.json_attr("step-1", "langfuse.observation.usage_details")
        assert usage == {"input": 100, "output": 20, "total": 120}

    def test_exports_tool_observation(self, lf_exporter, lf_client):
        LangfuseTraceStore(lf_client).save(sample_trace(), final=True)
        lf_client.flush()

        assert lf_exporter.attr("filesystem.read", "langfuse.observation.type") == "tool"
        out = lf_exporter.json_attr("filesystem.read", "langfuse.observation.output")
        assert out["content"] == "hello"

    def test_nesting_uses_otel_ids(self, lf_exporter, lf_client):
        """Langfuse display ids and OTel ids are different spaces; nest on OTel."""
        LangfuseTraceStore(lf_client).save(sample_trace(), final=True)
        lf_client.flush()

        agent = lf_exporter.find("factory.run:reader")
        child_names = {c.name for c in lf_exporter.children_of(agent)}
        assert {"run.result", "step-1", "step-2", "filesystem.read"} <= child_names

    def test_all_observations_share_one_trace(self, lf_exporter, lf_client):
        LangfuseTraceStore(lf_client).save(sample_trace(), final=True)
        lf_client.flush()
        assert len(lf_exporter.otel_trace_ids()) == 1

    def test_halted_run_exported_as_error_level(self, lf_exporter, lf_client):
        trace = RunTrace(id="run-halt", agent="reader", agent_version="0.1.0", task="loop")
        trace.status = "max_steps_exceeded"
        trace.halt_reason = "exceeded max_steps=8"
        trace.steps = [StepRecord(index=1, model="m", prompt_tokens=1, completion_tokens=1)]

        LangfuseTraceStore(lf_client).save(trace, final=True)
        lf_client.flush()

        assert lf_exporter.attr("factory.run:reader", "langfuse.observation.level") == "ERROR"
        assert lf_exporter.meta("factory.run:reader", "halt_reason") == "exceeded max_steps=8"

    def test_violations_surface_and_flag_the_tool(self, lf_exporter, lf_client):
        trace = RunTrace(id="run-viol", agent="reader", agent_version="0.1.0", task="escape")
        trace.status = "success"
        trace.steps = [
            StepRecord(
                index=1,
                model="m",
                prompt_tokens=1,
                completion_tokens=1,
                tool_calls=[{"name": "shell.execute", "arguments": {"cmd": "rm"}}],
                observations=[{"ok": False, "error": "denied"}],
                violations=[{"capability": "shell.execute", "reason": "not granted"}],
            )
        ]

        LangfuseTraceStore(lf_client).save(trace, final=True)
        lf_client.flush()

        assert lf_exporter.meta("factory.run:reader", "violations") == 1
        # A denied tool must never look like a plain success.
        assert lf_exporter.attr("shell.execute", "langfuse.observation.level") == "WARNING"

    def test_local_readback_still_serves_the_eval_harness(self, lf_exporter, lf_client):
        store = LangfuseTraceStore(lf_client)
        trace = sample_trace()
        store.save(trace, final=True)
        assert store.get("run-abc123") is trace
        assert store.list() == [trace]

    def test_local_retention_can_be_disabled(self, lf_exporter, lf_client):
        store = LangfuseTraceStore(lf_client, keep_local=False)
        store.save(sample_trace(), final=True)
        assert store.get("run-abc123") is None


class TestScoring:
    def test_score_maps_run_id_to_a_valid_langfuse_id(self):
        """create_score is HTTP-only, so assert the mapping we own, not the call."""
        recorded = {}

        class StubClient:
            def create_score(self, **kwargs):
                recorded.update(kwargs)

        store = LangfuseTraceStore(StubClient())
        store.score("run-abc123", "eval_pass_rate", 1.0, "12/12")

        assert recorded["trace_id"] == to_langfuse_id("run-abc123")
        int(recorded["trace_id"], 16)  # must be valid hex for Langfuse
        assert recorded["name"] == "eval_pass_rate"
        assert recorded["value"] == 1.0
        assert recorded["comment"] == "12/12"


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_generations_nest_under_the_run_trace(self, lf_exporter, lf_client, tmp_path):
        """A whole run: two generations plus a tool, all on ONE Langfuse trace."""
        (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")

        inner = ScriptedAdapter([
            ModelResponse(
                prompt_tokens=90,
                completion_tokens=15,
                tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})],
            ),
            ModelResponse(prompt_tokens=25, completion_tokens=10, text="the file says hello"),
        ])

        obs = LangfuseObservability(lf_client)
        compiled = compile_agent(
            load_spec(AGENTS / "reader.yaml"),
            registry_for(tmp_path),
            workspace=tmp_path,
            model=obs.wrap_model(inner, model_name="qwen2.5:7b"),
            store=obs.store,
        )

        result = await compiled.runtime.run("what does a.txt say?")
        lf_client.flush()

        assert result.status == "success"
        assert result.text == "the file says hello"

        # The run id reached the model adapter via the store.
        assert obs.current_trace_id == result.trace.id

        # One generation per model call.
        assert len(lf_exporter.find_all("model.complete")) == 2

        # Everything on a single trace — the whole point of the facade.
        assert len(lf_exporter.otel_trace_ids()) == 1

        # The run, with the tool nested beneath it.
        agent = lf_exporter.find("factory.run:reader")
        assert agent is not None
        child_names = {c.name for c in lf_exporter.children_of(agent)}
        assert "filesystem.read" in child_names
        assert "step-1" in child_names

        # The run id is recoverable for later scoring.
        assert lf_exporter.meta("factory.run:reader", "factory_run_id") == result.trace.id

        # Local store still answers, so the eval harness is unaffected.
        assert obs.store.get(result.trace.id) is not None

    @pytest.mark.asyncio
    async def test_unbound_adapter_still_traces(self, lf_exporter, lf_client, tmp_path):
        """Without the facade, generations export as their own traces, not lost."""
        inner = ScriptedAdapter([ModelResponse(text="standalone", prompt_tokens=5, completion_tokens=2)])
        traced = LangfuseModelAdapter(inner, lf_client)
        await traced.complete([Message(role="user", content="x")], [])
        lf_client.flush()

        assert lf_exporter.find("model.complete") is not None
        assert len(lf_exporter.otel_trace_ids()) == 1

    def test_stores_without_begin_run_still_work(self):
        """begin_run is duck-typed, so a bare store is unaffected."""

        class Bare:
            def __init__(self):
                self.saved = []
                self.final_saves = 0

            def save(self, trace, *, final=False):
                self.saved.append(trace)
                if final:
                    self.final_saves += 1

        store = Bare()
        result = asyncio.run(_run_with_bare_store(store))
        assert result.status == "success"
        assert len(store.saved) == 1
        # A bare store must still receive the final marker, or a store that
        # wants it has no way to learn the run is over.
        assert store.final_saves == 1


async def _run_with_bare_store(store):
    compiled = compile_agent(
        load_spec(AGENTS / "locked-down.yaml"),
        registry_for(Path(".")),
        workspace=Path("."),
        model=ScriptedAdapter([ModelResponse(text="no tools needed")]),
        store=store,
    )
    return await compiled.runtime.run("say something")
