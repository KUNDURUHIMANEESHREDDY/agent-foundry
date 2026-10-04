from factory.models.base import Message, ModelAdapter, ModelResponse, ToolCall, estimate_tokens
from factory.models.fake import FailingAdapter, ScriptedAdapter
from factory.models.ollama import OllamaAdapter, OllamaError
from factory.models.openai_compat import OpenAICompatAdapter, OpenAICompatError

import os

__all__ = [
    "FailingAdapter",
    "Message",
    "ModelAdapter",
    "ModelResponse",
    "OllamaAdapter",
    "OllamaError",
    "OpenAICompatAdapter",
    "OpenAICompatError",
    "ScriptedAdapter",
    "ToolCall",
    "estimate_tokens",
]


def build_adapter(provider: str, name: str, base_url: str | None = None, **kwargs) -> ModelAdapter:
    """Adapter factory. Adding a provider means adding a branch here only."""
    if provider == "ollama":
        return OllamaAdapter(name, base_url or "http://localhost:11434")
    if provider in ("openai_compat", "opencodex", "openai"):
        # Any /v1/chat/completions endpoint: the local OpenCode proxy, vLLM,
        # SGLang, LM Studio, or a hosted provider.
        return OpenAICompatAdapter(
            name,
            base_url or "http://127.0.0.1:10100",
            api_key=kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY"),
        )
    if provider == "fake":
        from factory.models.fake import ScriptedAdapter

        return ScriptedAdapter.text_then_final("no-op")
    raise ValueError(
        f"Unknown provider '{provider}'. Supported: ollama, openai_compat, fake."
    )
