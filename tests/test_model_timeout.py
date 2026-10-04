"""A hanging model call must not hang the run.

`await model.complete(...)` had no deadline. The loop checked elapsed time at
the top of each iteration, so an adapter that never returned blocked forever and
neither `max_steps` nor `step_timeout_s` was reachable: the limits were
decorative. Measured with a 1.0s budget and a hanging adapter, the run never
halted.

The deadline is `min(step_timeout_s, remaining run budget)`, and a model timeout
is reported as `model_timeout` rather than `timeout` — one means a slow provider,
the other a long run, and collapsing them would make the trace unable to say
which bound was hit.
"""

from __future__ import annotations

import asyncio

import pytest

from factory.capabilities.builtin import FilesystemRead
from factory.capabilities.registry import CapabilityGate, CapabilityRegistry
from factory.models.base import ModelAdapter, ModelResponse, ToolCall
from factory.models.fake import FailingAdapter, ScriptedAdapter
from factory.runtime.agent import AgentRuntime, RunStatus
from factory.runtime.trace import MemoryTraceStore
from factory.spec.agent_spec import AgentSpec, Limits, ModelRef

pytestmark = pytest.mark.asyncio


class HangingAdapter(ModelAdapter):
    """Never returns, like a provider stuck on a socket that never closes.

    Records whether the runtime cancelled it: a timeout that leaks the coroutine
    would leave the connection open, which is worse than the hang.
    """

    def __init__(self):
        super().__init__("hanging")
        self.cancelled = False
        self.calls = 0

    async def complete(self, messages, tools, max_output_tokens=1024, temperature=0.0):
        self.calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")

    async def health(self) -> bool:
        return True


class SlowButFinishingAdapter(ModelAdapter):
    """Finishes, but later than the step allowance."""

    def __init__(self, delay_s: float):
        super().__init__("slow-but-finite")
        self._delay = delay_s

    async def complete(self, messages, tools, max_output_tokens=1024, temperature=0.0):
        await asyncio.sleep(self._delay)
        return ModelResponse(text="eventually")

    async def health(self) -> bool:
        return True


def runtime_for(model, tmp_path, step_timeout_s=0.2, max_steps=5):
    reg = CapabilityRegistry(binder=lambda _n: {"root": tmp_path})
    reg.register(FilesystemRead)
    gate = CapabilityGate.of(reg, [reg.issue("filesystem.read")])
    spec = AgentSpec(
        name="probe",
        system_prompt="you are a test agent",
        capabilities=["filesystem.read"],
        model=ModelRef(
            provider="fake", name="none", context_window=4096, max_output_tokens=256
        ),
        limits=Limits(
            max_steps=max_steps,
            max_tokens=100_000,
            max_tool_calls=max_steps,
            step_timeout_s=step_timeout_s,
        ),
    )
    return AgentRuntime(
        spec=spec,
        model=model,
        gate=gate,
        store=MemoryTraceStore(),
        tool_schemas=[
            {
                "name": "filesystem.read",
                "description": "read",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ],
    )


def looping():
    return ScriptedAdapter(
        [ModelResponse(tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})])]
    )


class TestHangingModelCallIsBounded:
    async def test_a_never_returning_adapter_still_halts(self, tmp_path):
        adapter = HangingAdapter()
        rt = runtime_for(adapter, tmp_path, step_timeout_s=0.1, max_steps=5)

        # An outer guard, so a regression fails the test instead of hanging CI.
        result = await asyncio.wait_for(rt.run("t"), timeout=10.0)

        assert result.status == RunStatus.MODEL_TIMEOUT

    async def test_the_coroutine_is_cancelled_not_leaked(self, tmp_path):
        adapter = HangingAdapter()
        rt = runtime_for(adapter, tmp_path, step_timeout_s=0.1, max_steps=5)

        result = await asyncio.wait_for(rt.run("t"), timeout=10.0)

        assert adapter.cancelled, "the hung coroutine was abandoned, not cancelled"

    async def test_it_stops_on_the_first_call(self, tmp_path):
        adapter = HangingAdapter()
        rt = runtime_for(adapter, tmp_path, step_timeout_s=0.1, max_steps=50)

        result = await asyncio.wait_for(rt.run("t"), timeout=10.0)

        assert adapter.calls == 1, "a hung call must not be retried into a stall"
        assert len(result.trace.steps) == 1

    async def test_the_halt_names_the_step(self, tmp_path):
        rt = runtime_for(HangingAdapter(), tmp_path, step_timeout_s=0.1, max_steps=5)
        result = await asyncio.wait_for(rt.run("t"), timeout=10.0)

        assert "step 1" in result.trace.halt_reason
        assert "0.1" in result.trace.halt_reason

    async def test_the_failing_step_is_recorded(self, tmp_path):
        rt = runtime_for(HangingAdapter(), tmp_path, step_timeout_s=0.1, max_steps=5)
        result = await asyncio.wait_for(rt.run("t"), timeout=10.0)

        assert result.trace.steps[0].error
        assert "time budget" in result.trace.steps[0].error

    async def test_the_run_is_saved_so_the_hang_is_visible_afterwards(self, tmp_path):
        store = MemoryTraceStore()
        reg = CapabilityRegistry(binder=lambda _n: {"root": tmp_path})
        reg.register(FilesystemRead)
        gate = CapabilityGate.of(reg, [reg.issue("filesystem.read")])
        spec = AgentSpec(
            name="probe",
            system_prompt="p",
            capabilities=["filesystem.read"],
            model=ModelRef(
                provider="fake", name="none", context_window=4096, max_output_tokens=256
            ),
            limits=Limits(
                max_steps=5, max_tokens=100_000, max_tool_calls=5, step_timeout_s=0.1
            ),
        )
        rt = AgentRuntime(
            spec=spec, model=HangingAdapter(), gate=gate, store=store, tool_schemas=[]
        )

        await asyncio.wait_for(rt.run("t"), timeout=10.0)

        assert store.list(), "a hung run must leave a trace, or it vanishes"
        assert store.list()[0].status == RunStatus.MODEL_TIMEOUT


class TestTimeoutIsDistinctFromOtherFailures:
    async def test_model_timeout_is_not_model_error(self, tmp_path):
        """A timeout is not a provider failure; conflating them hides the cause."""
        rt = runtime_for(HangingAdapter(), tmp_path, step_timeout_s=0.1)
        result = await asyncio.wait_for(rt.run("t"), timeout=10.0)

        assert result.status != RunStatus.MODEL_ERROR
        assert result.status == RunStatus.MODEL_TIMEOUT

    async def test_model_timeout_is_not_the_run_clock(self, tmp_path):
        rt = runtime_for(HangingAdapter(), tmp_path, step_timeout_s=0.1, max_steps=5)
        result = await asyncio.wait_for(rt.run("t"), timeout=10.0)

        assert result.status != RunStatus.TIMEOUT

    async def test_a_genuine_error_is_still_model_error(self, tmp_path):
        rt = runtime_for(FailingAdapter(), tmp_path, step_timeout_s=5.0)
        result = await rt.run("t")

        assert result.status == RunStatus.MODEL_ERROR

    async def test_a_finite_slow_call_within_allowance_succeeds(self, tmp_path):
        rt = runtime_for(SlowButFinishingAdapter(0.02), tmp_path, step_timeout_s=2.0)
        result = await rt.run("t")

        assert result.status == RunStatus.SUCCESS
        assert result.text == "eventually"

    async def test_a_finite_slow_call_beyond_allowance_is_cut(self, tmp_path):
        rt = runtime_for(SlowButFinishingAdapter(5.0), tmp_path, step_timeout_s=0.1)
        result = await asyncio.wait_for(rt.run("t"), timeout=10.0)

        assert result.status == RunStatus.MODEL_TIMEOUT

    async def test_an_instant_run_is_untouched(self, tmp_path):
        rt = runtime_for(ScriptedAdapter.text_then_final("hi"), tmp_path, step_timeout_s=5.0)
        result = await rt.run("t")

        assert result.status == RunStatus.SUCCESS


class TestDeadlineSelection:
    """`min(step_timeout_s, remaining run budget)` — not just one or the other."""

    def _deadline(self, step_timeout_s, max_steps, elapsed):
        import time as _time

        from factory.runtime.agent import AgentRuntime

        limits = Limits(
            max_steps=max_steps,
            max_tokens=1000,
            max_tool_calls=10,
            step_timeout_s=step_timeout_s,
        )
        # `started` is an absolute monotonic reading; elapsed is how far past it
        # we are.
        return AgentRuntime._step_deadline(
            started=_time.monotonic() - elapsed, limits=limits, index=1
        )

    async def test_uses_the_step_allowance_when_the_budget_is_plenty(self):
        assert self._deadline(step_timeout_s=2.0, max_steps=10, elapsed=0.0) == 2.0

    async def test_shrinks_as_the_run_budget_is_consumed(self):
        # Budget = 0.1 * 10 = 1.0s; 0.95s spent leaves 0.05s.
        assert self._deadline(step_timeout_s=0.1, max_steps=10, elapsed=0.95) < 0.1

    async def test_never_returns_zero_or_negative(self):
        # Budget exhausted: must still be a usable positive number.
        assert self._deadline(step_timeout_s=0.1, max_steps=5, elapsed=99.0) > 0