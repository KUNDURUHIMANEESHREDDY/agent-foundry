"""OpenAI-compatible chat-completions adapter.

Talks to any `/v1/chat/completions` endpoint — the shape used by vLLM,
SGLang, Ollama's compat layer, LM Studio, and the local proxy configured in
`~/.config/opencode/opencode.json` (http://127.0.0.1:10100/v1).

WHY THIS EXISTS
    OpenCode's own CLI executes tool calls itself, which bypasses the factory's
    CapabilityGate entirely. A chat-completions endpoint returns tool calls as
    *requests*, so the gate still decides what runs. That is the whole point of
    this project, so this adapter — not a subprocess to the CLI — is the correct
    integration for a real model.

TOOL CALL ROUND TRIP
    assistant  -> tool_calls[]           (the model asking)
    tool       -> role="tool", tool_call_id, content=observation
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from factory.models.base import Message, ModelAdapter, ModelResponse, ToolCall


class OpenAICompatError(RuntimeError):
    pass


class OpenAICompatAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        variant: str | None = None,
        transport: Any | None = None,
    ) -> None:
        super().__init__(f"openai-compat:{model}" + (f"#{variant}" if variant else ""))
        self._model = model
        # OpenCode passes variants as model#variant on the CLI.
        self._variant = variant
        self._api_key = api_key
        self._base = base_url.rstrip("/")
        if not self._base.endswith("/v1") and "/v1" not in self._base:
            self._base = f"{self._base}/v1"
        self._timeout = timeout
        self._transport = transport

    # ── request shaping ─────────────────────────────────────────

    def _wire_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert to the OpenAI wire format.

        The runtime's `observation` role carries a capability result; OpenAI
        requires `role="tool"` with the originating `tool_call_id`, so tool
        results are matched back to their call.
        """
        wire: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []

        for m in messages:
            if m.role == "observation":
                if pending:
                    call = pending.pop(0)
                    wire.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": m.content,
                        }
                    )
                else:
                    # No matching call (e.g. after truncation); keep it visible
                    # to the model rather than silently dropping it.
                    wire.append({"role": "user", "content": m.content})
                continue

            msg: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": json.dumps(c.arguments),
                        },
                    }
                    for i, c in enumerate(m.tool_calls)
                ]
                pending.extend(msg["tool_calls"])
            wire.append(msg)

        return wire

    def _wire_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for t in tools:
            if not isinstance(t, dict) or "name" not in t:
                continue
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters")
                        or {"type": "object", "properties": {}},
                    },
                }
            )
        return out

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _model_id(self) -> str:
        return f"{self._model}#{self._variant}" if self._variant else self._model

    # ── the call ────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self._model_id(),
            "messages": self._wire_messages(messages),
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        wire_tools = self._wire_tools(tools)
        if wire_tools:
            payload["tools"] = wire_tools
            payload["tool_choice"] = "auto"

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                resp = await client.post(
                    f"{self._base}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                )
        except httpx.HTTPError as exc:
            raise OpenAICompatError(
                f"Cannot reach {self._base}: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise OpenAICompatError(
                f"{self._base} returned {resp.status_code}: {resp.text[:300]}"
            )

        return self._parse(resp.json())

    def _parse(self, data: dict[str, Any]) -> ModelResponse:
        choices = data.get("choices") or []
        if not choices:
            raise OpenAICompatError(f"No choices in response: {str(data)[:200]}")

        message = choices[0].get("message") or {}
        text = message.get("content") or ""

        tool_calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            arguments = fn.get("arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            if not isinstance(arguments, dict):
                arguments = {"_value": arguments}
            tool_calls.append(ToolCall(name=fn.get("name", ""), arguments=arguments))

        usage = data.get("usage") or {}
        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            stop_reason=choices[0].get("finish_reason", "stop"),
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(
                timeout=5.0, transport=self._transport
            ) as client:
                resp = await client.get(f"{self._base}/models", headers=self._headers())
            if resp.status_code != 200:
                return False
            models = [m.get("id", "") for m in resp.json().get("data", [])]
            if not models:
                return True
            base = self._model.split("/")[-1]
            return any(
                m == self._model or m.split("/")[-1] == base or base in m
                for m in models
            )
        except httpx.HTTPError:
            return False