"""Ollama adapter. Local models, no API key."""

from __future__ import annotations

import json
from typing import Any

import httpx

from factory.models.base import Message, ModelAdapter, ModelResponse, ToolCall


class OllamaError(RuntimeError):
    pass


class OllamaAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        super().__init__(f"ollama:{model}")
        self._model = model
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def _to_ollama_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "observation":
                # Ollama has no tool role; surface observations as user turns.
                out.append({"role": "user", "content": f"[tool result] {m.content}"})
                continue
            msg: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in m.tool_calls
                ]
            out.append(msg)
        return out

    def _to_ollama_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": t}
            for t in tools
        ]

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_ollama_messages(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
            },
        }
        if tools:
            payload["tools"] = self._to_ollama_tools(tools)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.post(f"{self._base}/api/chat", json=payload)
            except httpx.HTTPError as exc:
                raise OllamaError(f"Cannot reach Ollama at {self._base}: {exc}") from exc

        if resp.status_code != 200:
            raise OllamaError(f"Ollama returned {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        message = data.get("message", {})

        tool_calls: list[ToolCall] = []
        for raw in message.get("tool_calls", []) or []:
            fn = raw.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(ToolCall(name=fn.get("name", ""), arguments=args or {}))

        return ModelResponse(
            text=message.get("content", "") or "",
            tool_calls=tool_calls,
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            stop_reason=data.get("done_reason", "stop"),
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base}/api/tags")
            if resp.status_code != 200:
                return False
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            # Ollama reports names like "qwen3:4b"; match bare name or :latest
            base = self._model.split(":")[0]
            return any(m == self._model or m.split(":")[0] == base for m in models)
        except httpx.HTTPError:
            return False
