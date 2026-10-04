"""Token accounting must not depend on the provider's self-reported usage.

Two bugs, one root cause: size was measured only where it was easiest.

  * `Message.estimated_tokens()` counted `content` alone. An assistant turn
    carrying a 5KB Python program measured as 2 tokens, and `truncate()` used
    that figure to decide what still fit in the context window.

  * `max_tokens` was enforced purely on `response.prompt_tokens +
    response.completion_tokens`. A provider reporting 0 — or omitting `usage`,
    or counting completion only — made the ceiling unreachable. Local inference
    servers do all three.

The fix charges `max(reported, locally estimated)` per half, so an honest
provider is unaffected and an under-reporting one cannot under-charge.
"""

from __future__ import annotations

import asyncio

import pytest

from factory.capabilities.builtin import FilesystemRead
from factory.capabilities.registry import CapabilityGate, CapabilityRegistry
from factory.models.base import (
    Message,
    ModelResponse,
    ToolCall,
    estimate_json_tokens,
    estimate_text_tokens,
    estimate_tool_call_tokens,
)
from factory.models.fake import ScriptedAdapter
from factory.runtime.agent import AgentRuntime
from factory.runtime.context import (
    estimate_completion_tokens,
    estimate_request_tokens,
    tools_tokens,
    total_tokens,
    truncate,
    ContextBudget,
)
from factory.runtime.trace import MemoryTraceStore
from factory.spec.agent_spec import AgentSpec, Limits, ModelRef

BIG_CODE = "x = 1\n" * 800
BIG_ANSWER = "result line\n" * 400


# ── estimators ─────────────────────────────────────────────────────────


class TestToolCallPayloadsAreCounted:
    def test_a_large_tool_argument_is_not_free(self):
        call = ToolCall("python.execute", {"code": BIG_CODE})
        assert estimate_tool_call_tokens(call) > 1000

    def test_message_with_a_big_tool_call_costs_more_than_its_content(self):
        bare = Message(role="assistant", content="running it")
        with_call = Message(role="assistant", content="running it", tool_calls=[
            ToolCall("python.execute", {"code": BIG_CODE})
        ])
        assert with_call.estimated_tokens() > bare.estimated_tokens() * 100

    def test_an_empty_message_is_still_at_least_one_token(self):
        assert Message(role="user", content="").estimated_tokens() >= 1

    def test_empty_content_costs_nothing_by_itself(self):
        assert estimate_text_tokens("") == 0

    def test_json_estimate_is_deterministic(self):
        a = estimate_json_tokens({"b": 1, "a": 2})
        b = estimate_json_tokens({"a": 2, "b": 1})
        assert a == b, "key order must not change the cost"

    def test_json_estimate_survives_unserialisable_values(self):
        class Weird:
            def __repr__(self):
                return "<weird>"

        assert estimate_json_tokens({"x": Weird()}) > 0

    def test_tool_name_costs_something(self):
        assert estimate_tool_call_tokens(ToolCall("a", {})) > 0

    def test_tool_schemas_are_counted(self):
        tools = [
            {
                "name": "python.execute",
                "description": "x" * 200,
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
            }
        ]
        assert tools_tokens(tools) > 0

    def test_no_tools_costs_nothing(self):
        assert tools_tokens([]) == 0

    def test_request_estimate_includes_schemas(self):
        messages = [Message(role="user", content="hi")]
        tools = [{"name": "t", "description": "y" * 400, "parameters": {}}]
        assert estimate_request_tokens(messages, tools) > total_tokens(messages)


class TestTruncationSeesToolPayloads:
    def test_a_message_full_of_tool_calls_can_be_dropped(self):
        """The bug in effect: a 1400-token payload measured as 2 tokens.

        With a window that only fits a little, the old estimate kept the fat
        message and dropped something cheap instead.
        """
        fat = Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall("python.execute", {"code": BIG_CODE})],
        )
        cheap = [Message(role="observation", content="ok") for _ in range(20)]

        budget = ContextBudget(window=400, reserve=0)
        _kept, report = truncate([*cheap, fat], budget, protect_first=1)

        assert report.truncated
        assert total_tokens(_kept) <= budget.usable

    def test_budget_is_respected(self):
        messages = [
            Message(role="user", content="x" * 4000),
            *[Message(role="observation", content="y" * 2000) for _ in range(10)],
        ]
        budget = ContextBudget(window=2000, reserve=0)
        kept, report = truncate(messages, budget, protect_first=1)
        assert report.truncated
        assert total_tokens(kept) <= budget.usable


# ── the ceiling ────────────────────────────────────────────────────────


class UnderReportingAdapter(ScriptedAdapter):
    """Reports zero usage, like some local inference servers."""

    async def complete(self, messages, tools, max_output_tokens=1024, temperature=0.0):
        r = await super().complete(messages, tools, max_output_tokens, temperature)
        return ModelResponse(
            text=r.text, tool_calls=r.tool_calls, prompt_tokens=0, completion_tokens=0
        )


class CompletionOnlyAdapter(ScriptedAdapter):
    """Counts completion but not prompt — a partial-usage provider."""

    async def complete(self, messages, tools, max_output_tokens=1024, temperature=0.0):
        r = await super().complete(messages, tools, max_output_tokens, temperature)
        return ModelResponse(
            text=r.text, tool_calls=r.tool_calls, prompt_tokens=0, completion_tokens=10
        )


def build_runtime(model, tmp_path, max_tokens=400, max_steps=50):
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
            max_steps=max_steps, max_tokens=max_tokens, max_tool_calls=max_steps
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
                "description": "read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ],
    )


class TestCeilingHoldsWhenTheProviderUnderReports:
    def test_zero_usage_does_not_disable_max_tokens(self, tmp_path):
        adapter = UnderReportingAdapter([ModelResponse(text=BIG_ANSWER)])
        result = asyncio.run(build_runtime(adapter, tmp_path).run("t"))

        assert result.status == "max_tokens_exceeded"
        assert "exceeds max_tokens" in result.trace.halt_reason

    def test_partial_usage_does_not_disable_max_tokens(self, tmp_path):
        adapter = CompletionOnlyAdapter([ModelResponse(text=BIG_ANSWER)])
        result = asyncio.run(build_runtime(adapter, tmp_path).run("t"))

        assert result.status == "max_tokens_exceeded"

    def test_a_honest_provider_still_trips_it(self, tmp_path):
        adapter = ScriptedAdapter(
            [ModelResponse(text="done", prompt_tokens=500, completion_tokens=10)]
        )
        result = asyncio.run(build_runtime(adapter, tmp_path).run("t"))
        assert result.status == "max_tokens_exceeded"

    def test_honest_usage_is_not_inflated_by_the_estimate(self, tmp_path):
        """max(reported, estimated) must not add the two together."""
        adapter = ScriptedAdapter(
            [ModelResponse(text="done", prompt_tokens=600, completion_tokens=10)]
        )
        result = asyncio.run(build_runtime(adapter, tmp_path, max_tokens=1000).run("t"))

        assert result.status == "success"
        # Reported total, not reported + estimate.
        assert result.trace.total_tokens == 610

    def test_a_small_run_is_not_halted(self, tmp_path):
        adapter = UnderReportingAdapter([ModelResponse(text="short")])
        result = asyncio.run(build_runtime(adapter, tmp_path, max_tokens=100_000).run("t"))
        assert result.status == "success"

    def test_the_charge_is_never_zero(self, tmp_path):
        adapter = UnderReportingAdapter([ModelResponse(text="hello")])
        result = asyncio.run(build_runtime(adapter, tmp_path, max_tokens=100_000).run("t"))
        assert result.trace.total_tokens > 0

    def test_estimate_completion_counts_tool_arguments(self):
        calls = [ToolCall("python.execute", {"code": BIG_CODE})]
        assert estimate_completion_tokens("", calls) > 1000

    def test_estimate_completion_counts_text(self):
        assert estimate_completion_tokens(BIG_ANSWER, []) > 500

    def test_estimate_completion_of_nothing_is_zero(self):
        assert estimate_completion_tokens("", []) == 0


class TestCumulativeChargingAcrossSteps:
    def test_the_ceiling_accumulates_over_steps(self, tmp_path):
        """Each step must be charged, not only the last."""
        adapter = UnderReportingAdapter([ModelResponse(text=BIG_ANSWER)])
        rt = build_runtime(adapter, tmp_path, max_tokens=100_000)
        result = asyncio.run(rt.run("t"))

        assert result.trace.total_tokens > 0
        assert len(result.trace.steps) == 1