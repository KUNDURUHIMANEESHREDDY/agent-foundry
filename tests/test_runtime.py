from __future__ import annotations

import time

import pytest

from factory.capabilities.builtin import FilesystemRead
from factory.capabilities.registry import CapabilityDenied, CapabilityGate, CapabilityRegistry
from factory.models.base import ToolCall
from factory.models.fake import FailingAdapter, ScriptedAdapter, SlowAdapter
from factory.runtime.agent import AgentRuntime, RunStatus
from factory.runtime.trace import MemoryTraceStore
from factory.spec.agent_spec import AgentSpec, Limits, ModelRef

pytestmark = pytest.mark.asyncio


def spec(**limits) -> AgentSpec:
    return AgentSpec(
        name="t",
        system_prompt="you are a test agent",
        capabilities=["filesystem.read"],
        model=ModelRef(provider="fake", name="none", context_window=4096, max_output_tokens=256),
        limits=Limits(**{"max_steps": 5, "max_tokens": 100_000, "max_tool_calls": 5, **limits}),
    )


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    return tmp_path


def runtime_for(ws, adapter, **limits):
    reg = CapabilityRegistry(binder=lambda n: {"root": ws})
    reg.register(FilesystemRead)
    gate = CapabilityGate.of(reg, [reg.issue("filesystem.read")])
    return AgentRuntime(
        spec=spec(**limits),
        model=adapter,
        gate=gate,
        store=MemoryTraceStore(),
        tool_schemas=[{"name": "filesystem.read", "description": "read", "parameters": {}}],
    )


class TestHappyPath:
    async def test_uses_tool_then_returns_final(self, workspace):
        adapter = ScriptedAdapter.tool_then_final(
            ToolCall("filesystem.read", {"path": "a.txt"}), final="the file says hello"
        )
        rt = runtime_for(workspace, adapter)
        result = await rt.run("read a.txt")

        assert result.status == RunStatus.SUCCESS
        assert result.text == "the file says hello"
        assert len(result.trace.steps) == 2
        assert result.trace.steps[0].observations[0]["content"] == "hello world"

    async def test_records_observation_in_context(self, workspace):
        adapter = ScriptedAdapter.tool_then_final(
            ToolCall("filesystem.read", {"path": "a.txt"}), final="ok"
        )
        rt = runtime_for(workspace, adapter)
        await rt.run("read a.txt")

        second_prompt = adapter.calls[1]
        assert any(m.role == "observation" for m in second_prompt)
        assert any("hello world" in m.content for m in second_prompt)

    async def test_answers_directly_when_no_tool_needed(self, workspace):
        adapter = ScriptedAdapter.text_then_final("just an answer")
        rt = runtime_for(workspace, adapter)
        result = await rt.run("no file needed")

        assert result.status == RunStatus.SUCCESS
        assert len(adapter.calls) == 1
        assert len(result.trace.steps) == 1


class TestStepCapEnforcedInsideLoop:
    async def test_stops_at_max_steps(self, workspace):
        # Model asks for a tool forever and never finishes.
        adapter = ScriptedAdapter([__import__("factory.models.base", fromlist=["ModelResponse"]).ModelResponse(
            tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})]
        )])
        rt = runtime_for(workspace, adapter, max_steps=3)
        result = await rt.run("loop forever")

        assert result.status == RunStatus.MAX_STEPS
        assert "max_steps=3" in result.trace.halt_reason
        assert len(result.trace.steps) == 3
        assert len(adapter.calls) == 3

    async def test_max_steps_one_still_makes_one_call(self, workspace):
        from factory.models.base import ModelResponse

        adapter = ScriptedAdapter([ModelResponse(tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})])])
        rt = runtime_for(workspace, adapter, max_steps=1)
        result = await rt.run("one step")

        assert result.status == RunStatus.MAX_STEPS
        assert len(adapter.calls) == 1


class TestWallClockCeiling:
    """The run-level time budget, independent of the step count.

    A scripted adapter answers instantly, so no number of steps can exceed a
    wall-clock budget. These tests use `SlowAdapter` to make elapsed time real;
    without it the halt path could be deleted and no test would notice. The
    sabotage audit reports this mitigation as uncovered without them.

    The budget is `step_timeout_s * max_steps`. Tool-call limits are set high
    throughout so the clock is what halts the run, never the call cap.
    """

    #: Model asks for a tool forever; each step costs `DELAY` seconds.
    DELAY = 0.02

    def looping_adapter(self):
        from factory.models.base import ModelResponse

        return SlowAdapter(
            [ModelResponse(tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})])],
            delay_s=self.DELAY,
        )

    async def test_a_slow_model_call_halts_on_the_step_deadline(self, workspace):
        """A single slow call must not consume the whole run.

        `step_timeout_s=0.01` against a 0.02s model step: the per-step deadline
        is tighter than the run budget, so this halts as `model_timeout`, not
        `timeout`. Before per-call deadlines existed the run looped until the
        *total* budget expired and reported only `timeout`.
        """
        rt = runtime_for(
            workspace,
            self.looping_adapter(),
            max_steps=60,
            max_tool_calls=1000,
            step_timeout_s=0.01,
        )
        result = await rt.run("loop forever, slowly")

        assert result.status == RunStatus.MODEL_TIMEOUT
        assert "exceeded" in result.trace.halt_reason
        # It stopped on the first call, so no accumulation happened.
        assert len(result.trace.steps) == 1

    async def test_run_level_wall_clock_still_applies(self, workspace):
        from factory.models.base import ModelResponse

        """`timeout` remains reachable — via time the model call does not own.

        The per-step deadline wraps only `model.complete`, so a slow capability
        accumulates into elapsed time until the loop-top check fires. Without
        this the run-level budget would be dead code: if every step finished
        inside `step_timeout_s`, then `max_steps * step_timeout_s` — the whole
        budget — could not be reached by model time at all.
        """

        class SlowCapability:
            name = "filesystem.read"
            pinned_params = ("root",)
            required_params = ("root",)

            @classmethod
            def parameter_schema(cls):
                from factory.capabilities.registry import object_schema

                return object_schema({"path": {"type": "string"}}, required=["path"])

            def __init__(self, root):
                self._root = root

            def invoke(self, path: str = "", **_):
                time.sleep(0.05)
                return {"ok": True, "kind": "text", "content": "hi"}

        reg = CapabilityRegistry(binder=lambda _n: {"root": workspace})
        reg.register(SlowCapability)
        gate = CapabilityGate.of(reg, [reg.issue("filesystem.read")])
        rt = AgentRuntime(
            spec=spec(max_steps=50, max_tool_calls=1000, step_timeout_s=0.02),
            # An instant model: all the elapsed time is the capability's, so the
            # per-step deadline (which wraps only the model call) never trips.
            model=ScriptedAdapter(
                [
                    ModelResponse(
                        tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})]
                    )
                ]
            ),
            gate=gate,
            store=MemoryTraceStore(),
            tool_schemas=[
                {
                    "name": "filesystem.read",
                    "description": "read",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )

        result = await rt.run("loop forever, slowly")

        # Budget = 0.02 * 50 = 1.0s, spent 0.05s per step inside the capability.
        assert result.status == RunStatus.TIMEOUT
        assert "wall clock" in result.trace.halt_reason
        assert len(result.trace.steps) < 50

    async def test_budget_scales_with_max_steps(self, workspace):
        """Same slow workload, two budgets: only the tight one halts on time.

        Margins are deliberately wide. An earlier version had the loose case at
        a 50ms deadline against a 20ms operation — 2.5x — which passed alone and
        failed under the load of a full suite, because Windows scheduling and GC
        can stretch a 20ms sleep well past 50ms. The ratio the test is about is
        preserved; the absolute headroom is not what it is testing, so it is
        cheap to widen.
        """
        # A 10ms model step: `tight` gives it a 2ms deadline (5x under, so the
        # deadline always wins); `loose` gives it 200ms (20x over).
        def adapter():
            from factory.models.base import ModelResponse

            return SlowAdapter(
                [ModelResponse(tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})])],
                delay_s=0.01,
            )

        tight = runtime_for(
            workspace, adapter(), max_steps=3, max_tool_calls=1000, step_timeout_s=0.002
        )
        # Budget = 0.2 * 20 = 4.0s against a workload of 20 * 0.01 = 0.2s, so
        # every step fits its allowance and the step cap ends the run.
        loose = runtime_for(
            workspace, adapter(), max_steps=20, max_tool_calls=1000, step_timeout_s=0.2
        )

        assert (await tight.run("loop")).status == RunStatus.MODEL_TIMEOUT
        assert (await loose.run("loop")).status == RunStatus.MAX_STEPS

    async def test_fast_run_is_not_halted_by_the_clock(self, workspace):
        adapter = ScriptedAdapter.text_then_final("answered")
        rt = runtime_for(
            workspace, adapter, max_steps=5, max_tool_calls=5, step_timeout_s=5.0
        )
        result = await rt.run("quick question")

        assert result.status == RunStatus.SUCCESS


class TestToolCallCap:
    async def test_stops_at_max_tool_calls(self, workspace):
        from factory.models.base import ModelResponse

        adapter = ScriptedAdapter([ModelResponse(tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})])])
        rt = runtime_for(workspace, adapter, max_steps=50, max_tool_calls=2)
        result = await rt.run("many tools")

        assert result.status == RunStatus.MAX_TOOL_CALLS
        assert "max_tool_calls=2" in result.trace.halt_reason


class TestTokenCap:
    async def test_halts_when_cumulative_tokens_exceeded(self, workspace):
        from factory.models.base import ModelResponse

        adapter = ScriptedAdapter([ModelResponse(
            text="x", prompt_tokens=100, completion_tokens=100
        )])
        rt = runtime_for(workspace, adapter, max_steps=10, max_tokens=150)
        result = await rt.run("spend tokens")

        assert result.status == RunStatus.MAX_TOKENS
        assert "max_tokens=150" in result.trace.halt_reason


class TestToolErrorsAreObservationsNotCrashes:
    async def test_missing_file_feeds_back_and_continues(self, workspace):
        adapter = ScriptedAdapter([
            __import__("factory.models.base", fromlist=["ModelResponse"]).ModelResponse(
                tool_calls=[ToolCall("filesystem.read", {"path": "missing.txt"})]
            ),
            __import__("factory.models.base", fromlist=["ModelResponse"]).ModelResponse(
                text="that file does not exist"
            ),
        ])
        rt = runtime_for(workspace, adapter)
        result = await rt.run("read missing.txt")

        assert result.status == RunStatus.SUCCESS
        assert result.trace.steps[0].observations[0]["ok"] is False
        assert "No such file" in result.trace.steps[0].observations[0]["error"]

    async def test_tool_exception_is_captured(self, workspace):
        class Exploding(FilesystemRead):
            def invoke(self, **kwargs):
                raise RuntimeError("disk on fire")

        reg = CapabilityRegistry(binder=lambda n: {"root": workspace})
        reg.register(Exploding)
        gate = CapabilityGate.of(reg, [reg.issue("filesystem.read")])
        rt = AgentRuntime(
            spec=spec(),
            model=ScriptedAdapter.tool_then_final(ToolCall("filesystem.read", {"path": "a.txt"}), "recovered"),
            gate=gate,
            store=MemoryTraceStore(),
            tool_schemas=[],
        )
        result = await rt.run("read a.txt")

        assert result.status == RunStatus.SUCCESS
        obs = result.trace.steps[0].observations[0]
        assert obs["ok"] is False
        assert "disk on fire" in obs["error"]


class TestCapabilityEnforcementAtBoundary:
    async def test_ungranted_tool_call_is_denied_and_recorded(self, workspace):
        adapter = ScriptedAdapter([
            __import__("factory.models.base", fromlist=["ModelResponse"]).ModelResponse(
                tool_calls=[ToolCall("shell.execute", {"cmd": "rm -rf /"})]
            ),
            __import__("factory.models.base", fromlist=["ModelResponse"]).ModelResponse(
                text="I cannot do that"
            ),
        ])
        rt = runtime_for(workspace, adapter)
        result = await rt.run("delete everything")

        first = result.trace.steps[0]
        assert first.violations, "denial must be recorded as a violation"
        assert first.violations[0]["capability"] == "shell.execute"
        assert "denied" in first.observations[0]["error"]
        assert first.observations[0]["ok"] is False

    async def test_agent_cannot_self_grant_by_mentioning_tool(self, workspace):
        """The prompt is irrelevant to enforcement; only grants are."""
        adapter = ScriptedAdapter([
            __import__("factory.models.base", fromlist=["ModelResponse"]).ModelResponse(
                tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})]
            ),
            __import__("factory.models.base", fromlist=["ModelResponse"]).ModelResponse(text="done"),
        ])
        reg = CapabilityRegistry(binder=lambda n: {"root": workspace})
        reg.register(FilesystemRead)
        gate = CapabilityGate.of(reg, [reg.issue("filesystem.read")])
        rt = AgentRuntime(spec=spec(), model=adapter, gate=gate, store=MemoryTraceStore(), tool_schemas=[])
        result = await rt.run("read")

        assert not result.trace.steps[0].violations
        assert result.trace.steps[0].observations[0]["ok"] is True


class TestModelFailure:
    async def test_model_exception_halts_cleanly(self, workspace):
        rt = runtime_for(workspace, FailingAdapter("ollama down"))
        result = await rt.run("anything")

        assert result.status == RunStatus.MODEL_ERROR
        assert "ollama down" in result.trace.halt_reason
        assert result.trace.steps[0].error == "ollama down"


class TestContextTruncation:
    async def test_truncates_when_window_tiny(self, workspace):
        from factory.models.base import ModelResponse

        adapter = ScriptedAdapter([
            ModelResponse(
                tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})] * 3
            ),
            ModelResponse(text="done"),
        ])
        rt = runtime_for(workspace, adapter, max_steps=4)
        # shrink window so truncation must occur
        rt._spec = rt._spec.model_copy(update={
            "model": rt._spec.model.model_copy(update={"context_window": 400, "max_output_tokens": 100})
        })
        result = await rt.run("x" * 4000)

        assert result.status == RunStatus.SUCCESS
        assert any(s.truncated for s in result.trace.steps)

    async def test_truncation_keeps_system_prompt(self):
        from factory.models.base import Message
        from factory.runtime.context import ContextBudget, truncate

        msgs = [
            Message(role="system", content="SYSTEM"),
            Message(role="user", content="TASK"),
            *[Message(role="observation", content="x" * 2000) for _ in range(5)],
        ]
        out, report = truncate(msgs, ContextBudget(window=600, reserve=100))

        assert report.truncated
        assert out[0].content == "SYSTEM"
        assert out[1].content == "TASK"
        assert len(out) < len(msgs)

    async def test_no_truncation_when_it_fits(self):
        from factory.models.base import Message
        from factory.runtime.context import ContextBudget, truncate

        msgs = [Message(role="system", content="S"), Message(role="user", content="U")]
        out, report = truncate(msgs, ContextBudget(window=8000, reserve=100))

        assert not report.truncated
        assert out == msgs


class TestTrace:
    async def test_trace_persisted_and_loadable(self, workspace):
        adapter = ScriptedAdapter.text_then_final("answer")
        rt = runtime_for(workspace, adapter)
        result = await rt.run("q")

        stored = rt._store.get(result.trace.id)
        assert stored is not None
        assert stored.final_text == "answer"
        assert stored.status == "success"
