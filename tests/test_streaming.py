"""Streaming a run, and the progress hook that makes it possible.

Two things are being checked, and the second matters more than the first:

  1. the SSE framing is right, so a browser's EventSource parses it
  2. a broken `on_step` cannot fail a run

(2) is the one worth having tests for. A progress callback is a presentation
channel bolted onto the execution loop, and the temptation in any loop like this
is to let an exception in the callback propagate -- which would mean a frontend
with a bug in it can abort an agent run whose trace is already being written.
That is backwards: the trace is the record, the callback is decoration.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from factory.api.server import _sse
from factory.models.base import ModelResponse, ToolCall
from factory.models.fake import ScriptedAdapter
from factory.runtime.agent import AgentRuntime, RunStatus
from factory.runtime.trace import MemoryTraceStore
from factory.spec.agent_spec import AgentSpec, Limits


def _spec(**kw) -> AgentSpec:
    base = dict(
        name="streamer",
        system_prompt="You are a streamer.",
        capabilities=[],
        model={"provider": "fake", "name": "none", "context_window": 4096},
        limits=Limits(max_steps=4, max_tokens=5000, max_tool_calls=4, step_timeout_s=10),
    )
    base.update(kw)
    return AgentSpec.model_validate(base)


def _runtime(responses, workspace, **kw):
    from factory.capabilities.builtin import FilesystemRead
    from factory.capabilities.registry import CapabilityGate, CapabilityRegistry

    reg = CapabilityRegistry(binder=lambda _n: {"root": workspace})
    reg.register(FilesystemRead)
    gate = CapabilityGate.of(reg, [reg.issue("filesystem.read")])

    return AgentRuntime(
        spec=_spec(),
        model=ScriptedAdapter(responses),
        gate=gate,
        store=MemoryTraceStore(),
        tool_schemas=[
            {"name": "filesystem.read", "description": "read", "parameters": {}}
        ],
        **kw,
    )


class TestSseFraming:
    def test_an_event_has_a_name_and_data_and_a_blank_line(self):
        raw = _sse("step", {"index": 1})

        assert raw.startswith("event: step\n")
        assert "data: " in raw
        assert raw.endswith("\n\n"), "SSE frames need a blank line or clients buffer forever"

    def test_the_data_is_one_line_of_json(self):
        raw = _sse("done", {"status": "success", "text": "hello\nworld"})

        line = next(l for l in raw.splitlines() if l.startswith("data: "))
        payload = json.loads(line[len("data: "):])

        assert payload["status"] == "success"
        assert payload["text"] == "hello\nworld", (
            "a newline inside a value must not break the frame"
        )

    def test_data_is_a_single_line(self):
        """An embedded newline would split the frame and corrupt the stream."""
        raw = _sse("x", {"text": "a\nb\nc"})
        assert len([l for l in raw.splitlines() if l.startswith("data: ")]) == 1


class TestProgressHook:
    async def test_it_fires_for_intermediate_steps(self, tmp_path):
        """One callback per intermediate step, in order.

        Not one per trace step: the terminal step is appended by `_halt` and
        carries the final text, which the caller gets from the run result instead.
        Asserting `len(steps) == len(trace.steps)` would be asserting something
        the hook does not promise.
        """
        steps = []
        rt = _runtime(
            [
                ModelResponse(tool_calls=[ToolCall("filesystem.read", {"path": "a"})]),
                ModelResponse(text="done"),
            ],
            tmp_path,
        )

        result = await rt.run("go", on_step=lambda rec, tr: steps.append(rec.index))

        assert result.ok
        assert steps == sorted(steps), "steps must arrive in order"
        assert len(steps) == len(result.trace.steps) - 1
        assert result.text == "done", "the terminal text comes from the result"

    async def test_it_is_optional(self, tmp_path):
        rt = _runtime([ModelResponse(text="done")], tmp_path)
        result = await rt.run("go")
        assert result.status == RunStatus.SUCCESS

    async def test_a_raising_callback_does_not_fail_the_run(self, tmp_path):
        """The whole reason the call is wrapped.

        A frontend bug must not be able to abort an agent whose trace is being
        recorded. The run still completes and still reports its real status.
        """
        def explode(rec, tr):
            raise RuntimeError("the UI is broken")

        rt = _runtime([ModelResponse(text="done")], tmp_path)

        result = await rt.run("go", on_step=explode)

        assert result.status == RunStatus.SUCCESS
        assert result.text == "done"
        assert result.trace.steps, "the trace must still be recorded"

    async def test_a_callback_that_raises_does_not_hide_the_halt(self, tmp_path):
        """Same protection when the run ends badly.

        A silent failure mode here would be: run halts, callback explodes, and the
        caller sees success because the exception was swallowed in the wrong
        place. The status has to come from the run, not the callback.
        """
        calls = []

        def explode(rec, tr):
            calls.append(1)
            raise ValueError("boom")

        rt = _runtime(
            [
                ModelResponse(tool_calls=[ToolCall("filesystem.read", {"path": "a"})])
                for _ in range(10)
            ],
            tmp_path,
        )

        result = await rt.run("go", on_step=explode)

        assert result.status == RunStatus.MAX_STEPS
        assert calls, "the callback should have been called before the halt"
        assert "max_steps" in (result.trace.halt_reason or "")

    async def test_the_trace_is_saved_before_the_callback_runs(self, tmp_path):
        """Progress must reflect durable state, not something a client cannot verify.

        A client that reconnects and reads `/traces/{id}` must see the steps the
        stream already showed. If the callback ran first, the stream could report
        a step the trace does not contain yet.
        """
        rt = _runtime(
            [
                ModelResponse(tool_calls=[ToolCall("filesystem.read", {"path": "a"})]),
                ModelResponse(text="done"),
            ],
            tmp_path,
        )

        order: list[str] = []
        original = rt._store.save

        def spy(trace, **kw):
            order.append(f"save:{len(trace.steps)}")
            return original(trace, **kw)

        rt._store.save = spy  # type: ignore[assignment]

        await rt.run("go", on_step=lambda rec, tr: order.append(f"cb:{rec.index}"))

        callbacks = [i for i, v in enumerate(order) if v.startswith("cb:")]
        saves = [i for i, v in enumerate(order) if v.startswith("save:")]

        assert callbacks, "the callback never ran"
        assert saves, "the store was never called"
        assert min(callbacks) > min(saves), (
            "the callback ran before the step was durable"
        )