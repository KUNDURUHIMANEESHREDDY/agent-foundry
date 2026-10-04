"""Provider-neutral model adapter.

The runtime depends only on `ModelAdapter`. Swapping Qwen for Llama or a hosted
model means writing a new adapter, not touching the loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_reason: str = "stop"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


#: ~4 characters per token is the standard rough heuristic for English text.
#: Deliberately generous so we truncate before the provider rejects us.
CHARS_PER_TOKEN = 4


def estimate_text_tokens(text: str) -> int:
    """Tokens for a plain string. Never returns 0 for non-empty input."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_json_tokens(value: Any) -> int:
    """Tokens for an arbitrary JSON-ish structure, e.g. tool arguments.

    Assembled deterministically (sorted keys) so the same payload always costs
    the same, otherwise the budget would jitter run to run.
    """
    import json

    try:
        encoded = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        encoded = str(value)
    return estimate_text_tokens(encoded)


def estimate_tool_call_tokens(call: ToolCall) -> int:
    """A tool call carries a name and an argument payload, neither in `content`."""
    name_tokens = estimate_text_tokens(call.name)
    argument_tokens = estimate_json_tokens(call.arguments)
    # A few tokens of per-message framing, so an empty payload is not free.
    return name_tokens + argument_tokens + 2


@dataclass(frozen=True)
class Message:
    role: str  # system | user | assistant | observation
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    name: str | None = None

    def estimated_tokens(self) -> int:
        """Rough size of this message on the wire.

        Counting `content` alone under-estimated badly. An assistant turn that
        asks for a tool carries its arguments here and nowhere else, so a
        message with a 4KB Python program in it measured as if it were empty —
        and `truncate()` used that figure to decide what still fit.
        """
        total = estimate_text_tokens(self.content)
        total += sum(estimate_tool_call_tokens(c) for c in self.tool_calls)
        # Role and name cost something in every real tokenizer's framing.
        total += estimate_text_tokens(self.role) + estimate_text_tokens(self.name or "")
        return max(1, total)


class ModelAdapter(ABC):
    """Everything the runtime needs from a model."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Return one completion. Must not raise on ordinary model errors —
        the runtime treats exceptions as a failed step and may retry or halt."""

    @abstractmethod
    async def health(self) -> bool:
        """True if the endpoint is reachable and the model is present."""

    def describe(self) -> str:
        return self.name


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
