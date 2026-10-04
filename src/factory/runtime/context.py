"""Context window management.

Truncation keeps the system prompt and the task, then drops the oldest
observations/turns first, and always reserves headroom for the model's reply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from factory.models.base import (
    Message,
    ToolCall,
    estimate_json_tokens,
    estimate_text_tokens,
)


@dataclass
class ContextBudget:
    window: int
    reserve: int = 512

    @property
    def usable(self) -> int:
        return max(0, self.window - self.reserve)


@dataclass
class TruncationReport:
    dropped: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    dropped_messages: list[str] = field(default_factory=list)

    @property
    def truncated(self) -> bool:
        return self.dropped > 0


def total_tokens(messages: list[Message]) -> int:
    return sum(m.estimated_tokens() for m in messages)


def tools_tokens(tools: list[dict[str, Any]]) -> int:
    """Size of the tool schemas sent with every request.

    These are not in `messages` and so were never counted. A coder spec with four
    capabilities ships a few hundred tokens of schema on every single call,
    which is real prompt cost the truncation budget ignored entirely.
    """
    return estimate_json_tokens(tools) if tools else 0


def estimate_request_tokens(messages: list[Message], tools: list[dict[str, Any]]) -> int:
    """Everything we put on the wire for one model call."""
    return total_tokens(messages) + tools_tokens(tools)


def estimate_completion_tokens(text: str, tool_calls: list[ToolCall]) -> int:
    """What we expect back, estimated locally.

    Used as a floor against provider-reported usage: see
    `AgentRuntime._charge_tokens`.
    """
    return estimate_text_tokens(text) + sum(
        estimate_json_tokens({"name": c.name, "arguments": c.arguments})
        for c in tool_calls
    )


def truncate(
    messages: list[Message],
    budget: ContextBudget,
    protect_first: int = 2,
) -> tuple[list[Message], TruncationReport]:
    """Drop oldest non-protected messages until the window fits.

    `protect_first` keeps the system prompt and the task/goal message. Messages
    dropped are summarized as a single notice so the model knows history exists
    rather than silently seeing a gap.
    """
    report = TruncationReport(tokens_before=total_tokens(messages))
    if protect_first > len(messages):
        protect_first = len(messages)

    head = messages[:protect_first]
    tail = messages[protect_first:]

    if total_tokens(head) > budget.usable:
        # Even the protected head doesn't fit; keep it anyway rather than
        # handing the model an empty prompt, but report it loudly.
        report.tokens_after = total_tokens(head)
        report.dropped = len(tail)
        return head, report

    kept: list[Message] = []
    for msg in reversed(tail):
        candidate = kept + [msg]
        if total_tokens(head + candidate) > budget.usable:
            break
        kept = candidate

    kept.reverse()

    dropped = len(tail) - len(kept)
    report.dropped = dropped
    report.tokens_after = total_tokens(head + kept)

    if dropped:
        notice = Message(
            role="system",
            content=f"[{dropped} earlier tool result(s) omitted to fit the context window]",
        )
        return [*head, notice, *kept], report

    return messages, report
