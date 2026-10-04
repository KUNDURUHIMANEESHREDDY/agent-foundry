"""Agent Spec — the declarative source of truth for a runnable agent.

A spec is a *request* for capabilities, not a grant. The CapabilityResolver
turns a request into a grant or a rejection; nothing here is trusted downstream.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ModelRef(BaseModel):
    """Model selection. Kept provider-neutral so adapters are swappable."""

    model_config = ConfigDict(extra="forbid")

    # Defaults point at a local model server. They must be constructible with no
    # arguments: a `default_factory` that raises poisons pydantic's entire error
    # report, hiding unrelated field errors. Model presence is checked at
    # compile/run time by the adapter, not here.
    provider: str = Field(default="ollama", description="Adapter name, e.g. 'ollama', 'openai', 'fake'")
    name: str = Field(default="qwen2.5:7b", description="Provider-specific model identifier")
    base_url: str | None = Field(default=None, description="Override adapter base URL")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    context_window: int = Field(
        default=8192,
        gt=0,
        description="Model context window in tokens. Used for truncation, not enforced by provider.",
    )
    max_output_tokens: int = Field(default=1024, gt=0)


class MemorySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # v0.1 default is the light store. A vector DB earns its dependency later.
    type: str = Field(default="memory", description="sqlite | memory")
    path: str | None = Field(default=None, description="Required when type == 'sqlite'")


class Limits(BaseModel):
    """Every one of these is read *inside* the runtime loop, every iteration."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=20, gt=0, le=500)
    max_tokens: int = Field(default=50_000, gt=0, description="Cumulative token ceiling")
    max_tool_calls: int = Field(default=50, gt=0)
    step_timeout_s: float = Field(default=60.0, gt=0)
    context_reserve_tokens: int = Field(
        default=512,
        ge=0,
        description="Headroom kept free for the model's reply when truncating context.",
    )


class AgentSpec(BaseModel):
    """A declarative agent definition."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: str = Field(default="0.1.0", pattern=r"^\d+\.\d+\.\d+$")
    description: str = Field(default="")
    priority: Priority = Priority.MEDIUM

    system_prompt: str = Field(min_length=1)

    capabilities: Annotated[list[str], Field(default_factory=list)] = Field(
        description="Requested capabilities. Resolved to grants; unknown names are rejected.",
    )

    model: ModelRef = Field(default_factory=ModelRef)
    memory: MemorySpec = Field(default_factory=MemorySpec)
    limits: Limits = Field(default_factory=Limits)

    @field_validator("capabilities")
    @classmethod
    def dedupe(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for name in v:
            key = name.strip()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def wants(self, capability: str) -> bool:
        return capability in self.capabilities


class GoalSpec(BaseModel):
    """High-level intent. The compiler derives capabilities from this."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1)
    requirements: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)


class AgentGoal(BaseModel):
    """Result of compilation: a spec plus the grants that back it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: AgentSpec
    grants: frozenset[str] = frozenset()
    denied: dict[str, str] = Field(default_factory=dict)
    compiled_prompt: str = ""
    workspace: str | None = None
