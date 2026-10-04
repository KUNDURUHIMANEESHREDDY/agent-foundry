from __future__ import annotations

import json

import httpx
import pytest

from factory.capabilities.registry import registry_for
from factory.compiler import compile_agent
from factory.models.base import Message, ToolCall
from factory.models.openai_compat import OpenAICompatAdapter, OpenAICompatError
from factory.runtime.agent import RunStatus
from factory.runtime.trace import MemoryTraceStore
from factory.spec.agent_spec import AgentSpec, Limits, ModelRef

BASE = "http://127.0.0.1:10100"


def transport_returning(payload: dict, status: int = 200) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(status, json=payload, request=request)
    )


def chat(*, text="", tool_calls=None, prompt_tokens=0, completion_tokens=0):
    message: dict = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": n, "arguments": json.dumps(a)},
            }
            for i, (n, a) in enumerate(tool_calls)
        ]
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def make(**kwargs) -> OpenAICompatAdapter:
    return OpenAICompatAdapter("gpt-6.1-sol", BASE, **kwargs)


class TestRequestShaping:
    @pytest.mark.asyncio
    async def test_parses_text_and_usage(self):
        a = make(transport=transport_returning(
            chat(text="hello", prompt_tokens=11, completion_tokens=3)
        ))
        r = await a.complete([Message(role="user", content="hi")], [])
        assert r.text == "hello"
        assert r.prompt_tokens == 11
        assert r.completion_tokens == 3
        assert not r.wants_tool()

    @pytest.mark.asyncio
    async def test_parses_tool_call(self):
        a = make(transport=transport_returning(
            chat(tool_calls=[("filesystem.read", {"path": "a.txt"})])
        ))
        r = await a.complete([Message(role="user", content="read")], [{"name": "filesystem.read"}])
        assert r.wants_tool()
        assert r.tool_calls[0].name == "filesystem.read"
        assert r.tool_calls[0].arguments == {"path": "a.txt"}

    @pytest.mark.asyncio
    async def test_observation_becomes_tool_role(self):
        """The round trip that makes native tool calling work at all."""
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=chat(text="done"), request=request)

        a = make(transport=httpx.MockTransport(handler))
        await a.complete(
            [
                Message(role="system", content="S"),
                Message(role="user", content="read a.txt"),
                Message(role="assistant", content="", tool_calls=[
                    ToolCall("filesystem.read", {"path": "a.txt"})
                ]),
                Message(role="observation", content="filesystem.read: hello world"),
                Message(role="assistant", content="it says hello"),
            ],
            [],
        )

        roles = [m["role"] for m in seen["messages"]]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]
        assert seen["messages"][3]["tool_call_id"] == "call_0"
        assert "hello world" in seen["messages"][3]["content"]

    @pytest.mark.asyncio
    async def test_orphan_observation_stays_visible(self):
        """After truncation there may be no call to match; don't drop the result."""
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=chat(text="ok"), request=request)

        a = make(transport=httpx.MockTransport(handler))
        await a.complete(
            [Message(role="system", content="S"),
             Message(role="observation", content="orphan result")],
            [],
        )
        assert seen["messages"][-1]["role"] == "user"
        assert "orphan result" in seen["messages"][-1]["content"]

    @pytest.mark.asyncio
    async def test_sends_tool_schemas(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=chat(text="x"), request=request)

        a = make(transport=httpx.MockTransport(handler))
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }
        await a.complete(
            [Message(role="user", content="x")],
            [{"name": "filesystem.read", "description": "read", "parameters": schema}],
        )
        tool = seen["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "filesystem.read"
        assert seen["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_omits_tools_when_none(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=chat(text="x"), request=request)

        await make(transport=httpx.MockTransport(handler)).complete(
            [Message(role="user", content="x")], []
        )
        assert "tools" not in seen

    @pytest.mark.asyncio
    async def test_variant_is_sent_as_model_hash(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=chat(text="x"), request=request)

        await make(variant="high", transport=httpx.MockTransport(handler)).complete(
            [Message(role="user", content="x")], []
        )
        assert seen["model"] == "gpt-6.1-sol#high"

    @pytest.mark.asyncio
    async def test_sends_bearer_auth(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=chat(text="x"), request=request)

        await make(api_key="sk-test", transport=httpx.MockTransport(handler)).complete(
            [Message(role="user", content="x")], []
        )
        assert seen["auth"] == "Bearer sk-test"

    def test_appends_v1_to_bare_base_url(self):
        assert make()._base == f"{BASE}/v1"
        assert OpenAICompatAdapter("m", "http://h:1/v1")._base == "http://h:1/v1"


class TestResponseParsing:
    @pytest.mark.asyncio
    async def test_malformed_arguments_do_not_raise(self):
        a = make(transport=transport_returning({
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "x", "arguments": "{not json"}}]},
                "finish_reason": "tool_calls"}],
            "usage": {},
        }))
        r = await a.complete([Message(role="user", content="x")], [])
        assert r.tool_calls[0].arguments == {"_raw": "{not json"}

    @pytest.mark.asyncio
    async def test_empty_arguments_becomes_empty_dict(self):
        a = make(transport=transport_returning({
            "choices": [{"message": {"tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "x", "arguments": ""}}]}, "finish_reason": "tool_calls"}],
        }))
        r = await a.complete([Message(role="user", content="x")], [])
        assert r.tool_calls[0].arguments == {}

    @pytest.mark.asyncio
    async def test_no_choices_raises(self):
        a = make(transport=transport_returning({"choices": []}))
        with pytest.raises(OpenAICompatError, match="No choices"):
            await a.complete([Message(role="user", content="x")], [])

    @pytest.mark.asyncio
    async def test_http_error_surfaces_body(self):
        a = make(transport=transport_returning({"error": "boom"}, status=500))
        with pytest.raises(OpenAICompatError, match="500"):
            await a.complete([Message(role="user", content="x")], [])


def tool_then_text(tool_name: str, arguments: dict, final: str = "done"):
    """A model that asks once, then answers — as a real one does.

    A transport that always returned the tool call would loop until
    max_steps, which tests the step cap rather than the gate.
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=chat(tool_calls=[(tool_name, arguments)]), request=request)
        return httpx.Response(200, json=chat(text=final), request=request)

    return httpx.MockTransport(handler)


class TestGateStillEnforces:
    """The whole reason for using a chat-completions backend: the gate runs."""

    def _compiled(self, transport, tmp_path, spec):
        (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
        return compile_agent(
            spec,
            registry_for(tmp_path),
            workspace=tmp_path,
            model=OpenAICompatAdapter("gpt-6.1-sol", BASE, transport=transport),
            store=MemoryTraceStore(),
        )

    def _spec(self, caps):
        return AgentSpec(
            name="t",
            system_prompt="you are a test agent",
            capabilities=caps,
            model=ModelRef(provider="openai_compat", name="gpt-6.1-sol"),
            limits=Limits(max_steps=4),
        )

    @pytest.mark.asyncio
    async def test_granted_capability_executes(self, tmp_path):
        transport = tool_then_text("filesystem.read", {"path": "a.txt"})
        compiled = self._compiled(transport, tmp_path, self._spec(["filesystem.read"]))
        result = await compiled.runtime.run("read a.txt")

        assert result.status == RunStatus.SUCCESS
        obs = result.trace.steps[0].observations[0]
        assert obs["ok"] is True
        assert obs["content"] == "hello world"
        assert not result.trace.steps[0].violations

    @pytest.mark.asyncio
    async def test_ungranted_capability_is_denied(self, tmp_path):
        """A model asking for a tool it was never granted is refused here."""
        transport = tool_then_text("shell.execute", {"cmd": "cat /etc/passwd"})
        compiled = self._compiled(transport, tmp_path, self._spec(["filesystem.read"]))
        result = await compiled.runtime.run("run a shell command")

        step = result.trace.steps[0]
        assert step.violations, "the gate must record a denial"
        assert step.violations[0]["capability"] == "shell.execute"
        assert step.observations[0]["ok"] is False
        assert "denied" in step.observations[0]["error"]

    @pytest.mark.asyncio
    async def test_path_traversal_denied_by_the_gate(self, tmp_path):
        (tmp_path.parent / "outside.txt").write_text("classified", encoding="utf-8")
        transport = tool_then_text("filesystem.read", {"path": "../outside.txt"})
        compiled = self._compiled(transport, tmp_path, self._spec(["filesystem.read"]))
        result = await compiled.runtime.run("read the file above")

        obs = result.trace.steps[0].observations[0]
        assert obs["ok"] is False
        assert "outside the workspace" in obs["error"]
        assert "classified" not in json.dumps(obs)