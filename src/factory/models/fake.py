"""Scripted model adapter for tests and for running the runtime with no model server.

Lets the loop, truncation, capability gating and failure handling be tested
deterministically — which is the only way to prove those paths at all.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from factory.models.base import Message, ModelAdapter, ModelResponse, ToolCall


class ScriptedAdapter(ModelAdapter):
    """Replays a fixed list of responses, then repeats the last one."""

    def __init__(self, responses: Sequence[ModelResponse], name: str = "scripted") -> None:
        super().__init__(name)
        if not responses:
            raise ValueError("ScriptedAdapter needs at least one response")
        self._responses = list(responses)
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[dict]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict],
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        self.calls.append(list(messages))
        self.tools_seen.append(list(tools))
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]

    async def health(self) -> bool:
        return True

    @classmethod
    def text_then_final(cls, *texts: str) -> "ScriptedAdapter":
        return cls([ModelResponse(text=t) for t in texts])

    @classmethod
    def tool_then_final(
        cls, tool: ToolCall, final: str = "done"
    ) -> "ScriptedAdapter":
        return cls(
            [
                ModelResponse(text="", tool_calls=[tool]),
                ModelResponse(text=final),
            ]
        )


class FailingAdapter(ModelAdapter):
    """Always raises. Used to prove the runtime degrades instead of hanging."""

    def __init__(self, message: str = "model exploded") -> None:
        super().__init__("failing")
        self._message = message

    async def complete(self, messages, tools, max_output_tokens=1024, temperature=0.0):
        raise RuntimeError(self._message)

    async def health(self) -> bool:
        return False


class SlowAdapter(ScriptedAdapter):
    """A scripted adapter whose steps take real time.

    Needed to test the wall-clock ceiling. A scripted adapter answers instantly,
    so a run driven by one can never exceed any wall-clock budget no matter how
    the limits are set — the halt path would be untested and could rot.
    """

    def __init__(
        self,
        responses: Sequence[ModelResponse],
        delay_s: float = 0.05,
        name: str = "slow",
    ) -> None:
        super().__init__(responses, name=name)
        self._delay_s = float(delay_s)

    async def complete(self, messages, tools, max_output_tokens=1024, temperature=0.0):
        await asyncio.sleep(self._delay_s)
        return await super().complete(
            messages, tools, max_output_tokens=max_output_tokens, temperature=temperature
        )
