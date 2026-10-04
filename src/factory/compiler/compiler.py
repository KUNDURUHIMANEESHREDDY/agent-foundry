"""The compiler: Agent Spec -> runnable Agent.

    Validator -> CapabilityResolver -> PromptCompiler -> RuntimeConfig

Each stage fails closed. An unknown capability is a rejection, not a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from factory.capabilities.registry import (
    CapabilityDenied,
    CapabilityGate,
    CapabilityNotFound,
    CapabilityRegistry,
    Grant,
)
from factory.models import build_adapter
from factory.models.base import ModelAdapter
from factory.runtime.agent import AgentRuntime
from factory.runtime.trace import TraceStore, build_store
from factory.spec.agent_spec import AgentGoal, AgentSpec, GoalSpec


class CompileError(ValueError):
    """Raised when a spec cannot become a runnable agent."""


@dataclass
class ValidationReport:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(spec: AgentSpec) -> ValidationReport:
    """Structural checks Pydantic can't express."""
    errors: list[str] = []
    warnings: list[str] = []

    if not spec.system_prompt.strip():
        errors.append("system_prompt is empty; the agent has no instructions")

    if spec.model.context_window <= spec.limits.context_reserve_tokens:
        errors.append(
            f"context_window ({spec.model.context_window}) must exceed "
            f"context_reserve_tokens ({spec.limits.context_reserve_tokens})"
        )

    if spec.model.max_output_tokens >= spec.model.context_window:
        errors.append(
            f"max_output_tokens ({spec.model.max_output_tokens}) must be smaller than "
            f"context_window ({spec.model.context_window})"
        )

    if spec.memory.type == "sqlite" and not spec.memory.path:
        warnings.append("memory.type is sqlite but no path set; falling back to in-memory")

    if spec.priority.value in {"critical"} and not spec.capabilities:
        warnings.append("critical-priority agent declares no capabilities")

    return ValidationReport(errors=errors, warnings=warnings)


def resolve_capabilities(
    spec: AgentSpec, registry: CapabilityRegistry
) -> tuple[frozenset[Grant], dict[str, str]]:
    """Turn requested capabilities into grants. Unknown names are hard errors."""
    grants: list[Grant] = []
    denied: dict[str, str] = {}

    for requested in spec.capabilities:
        try:
            grants.append(registry.issue(requested))
        except (CapabilityNotFound, ValueError) as exc:
            denied[requested] = str(exc)

    return frozenset(grants), denied


def compile_prompt(spec: AgentSpec, granted_names: list[str], workspace: Path | None) -> str:
    """Assemble the system prompt the runtime will actually use."""
    parts = [spec.system_prompt.strip()]

    if granted_names:
        listing = "\n".join(f"  - {n}" for n in sorted(granted_names))
        parts.append(f"You have these capabilities:\n{listing}")
    else:
        parts.append("You have no tool capabilities. Answer from the task directly.")

    if workspace is not None:
        parts.append(f"All relative paths resolve inside: {workspace}")

    parts.append(
        "Call a capability when you need information you do not have. "
        "When you have enough, reply with a final answer and no tool call."
    )

    return "\n\n".join(parts)


@dataclass
class CompiledAgent:
    spec: AgentSpec
    runtime: AgentRuntime
    gate: CapabilityGate
    model: ModelAdapter
    store: TraceStore
    warnings: list[str]
    #: Requested capability -> why it was not granted. Empty in the normal case.
    #:
    #: Under `strict=True` a non-empty value raises CompileError instead, so this
    #: is only populated for `strict=False`, where capabilities are dropped
    #: rather than fatal. It was previously computed by `compile_agent` and
    #: discarded, and `CompiledAgent.denied` was a property that could not work:
    #: it read `self.gate.grants`, which does not exist on `CapabilityGate`.
    denied: dict[str, str] = field(default_factory=dict)

    @property
    def granted(self) -> list[str]:
        return sorted(self.gate.granted_names)


def compile_agent(
    spec: AgentSpec,
    registry: CapabilityRegistry,
    workspace: str | Path | None = None,
    model: ModelAdapter | None = None,
    store: TraceStore | None = None,
    strict: bool = True,
) -> CompiledAgent:
    """Full compile chain. `strict=True` fails on any unresolvable capability."""
    report = validate(spec)
    if not report.ok:
        raise CompileError("Spec failed validation:\n" + "\n".join(f"  - {e}" for e in report.errors))

    warnings = list(report.warnings)

    grants, denied = resolve_capabilities(spec, registry)
    if denied and strict:
        raise CompileError(
            "Spec requested capabilities that cannot be granted:\n"
            + "\n".join(f"  - {name}: {reason}" for name, reason in denied.items())
            + "\nFix the spec, or compile with strict=False to drop them."
        )
    if denied:
        warnings.extend(f"dropped capability '{n}': {r}" for n, r in denied.items())

    ws = Path(workspace).resolve() if workspace else None
    granted_names = [g.name for g in grants]
    prompt = compile_prompt(spec, granted_names, ws)

    effective = spec.model_copy(update={"system_prompt": prompt})
    gate = CapabilityGate.of(registry, grants)
    adapter = model or build_adapter(
        effective.model.provider, effective.model.name, effective.model.base_url
    )
    trace_store = store or build_store(effective.memory.type, effective.memory.path)

    runtime = AgentRuntime(
        spec=effective,
        model=adapter,
        gate=gate,
        store=trace_store,
        tool_schemas=[registry.tool_schema(name) for name in granted_names],
    )

    return CompiledAgent(
        spec=effective,
        runtime=runtime,
        gate=gate,
        model=adapter,
        store=trace_store,
        warnings=warnings,
        denied=dict(denied),
    )


REQUIREMENT_MAP: dict[str, tuple[str, ...]] = {
    "web": ("web.search",),
    "search": ("web.search",),
    "browser": ("browser.navigate",),
    "python": ("python.execute",),
    "code": ("python.execute",),
    "files": ("filesystem.read",),
    "filesystem": ("filesystem.read", "filesystem.write"),
    "read": ("filesystem.read",),
    "write": ("filesystem.write",),
    "citations": ("filesystem.read",),
    "git": ("git.commit",),
    "shell": ("shell.execute",),
}


def plan_goal(
    goal: GoalSpec,
    template: AgentSpec,
    known_capabilities: set[str] | None = None,
) -> tuple[list[str], dict[str, str], list[dict[str, Any]]]:
    """Resolve a goal into capabilities without compiling or executing anything.

    Returns (capabilities, errors, proposal) so a caller can show the operator
    what would happen before anything is built. Requirements are mapped by
    explicit name only — an unrecognised requirement is reported, never guessed
    into a grant.
    """
    capabilities: list[str] = set(template.capabilities)
    errors: dict[str, str] = {}
    proposal: list[dict[str, Any]] = []

    for key in goal.requirements:
        mapped = REQUIREMENT_MAP.get(key)
        if not mapped:
            errors[key] = (
                f"no capability mapping for requirement '{key}'. "
                f"Known: {', '.join(sorted(REQUIREMENT_MAP))}"
            )
            continue

        for capability in mapped:
            if known_capabilities is not None and capability not in known_capabilities:
                errors[key] = (
                    f"requirement '{key}' implies capability '{capability}', "
                    f"which no provider implements"
                )
                continue
            if capability not in capabilities:
                capabilities.add(capability)
                proposal.append(
                    {
                        "requirement": key,
                        "capability": capability,
                        "reason": f"requirement '{key}' maps to '{capability}'",
                    }
                )

    return sorted(capabilities), errors, proposal


def compile_goal(
    goal: GoalSpec,
    template: AgentSpec,
    registry: CapabilityRegistry,
    known_capabilities: set[str] | None = None,
    **kwargs,
) -> CompiledAgent:
    """Derive a spec from intent, then compile it.

    Fails closed: an unrecognised requirement or a requirement whose capability
    does not exist is an error, never a silently dropped grant.
    """
    capabilities, errors, _proposal = plan_goal(goal, template, known_capabilities)
    if errors:
        detail = "\n".join(f"  - {k}: {v}" for k, v in errors.items())
        raise CompileError(f"Goal could not be compiled:\n{detail}")

    spec = template.model_copy(update={"capabilities": capabilities})

    limits = template.limits.model_copy(
        update={
            "max_steps": int(goal.constraints.get("max_steps", template.limits.max_steps))
        }
    )
    spec = spec.model_copy(update={"limits": limits})

    return compile_agent(spec, registry, **kwargs)


def compile_goal_to_spec(
    goal: GoalSpec,
    template: AgentSpec,
    known_capabilities: set[str] | None = None,
    name: str | None = None,
    version: str | None = None,
) -> AgentSpec:
    """Compile a goal into an addressable spec, without touching capabilities.

    This is the registry-facing entry point: produce an agent *definition*
    rather than a running agent. `name`/`version` let the caller stamp the
    result as a new addressable agent instead of inheriting the template's.
    """
    capabilities, errors, _ = plan_goal(goal, template, known_capabilities)
    if errors:
        detail = "\n".join(f"  - {k}: {v}" for k, v in errors.items())
        raise CompileError(f"Goal could not be compiled:\n{detail}")

    updates: dict[str, Any] = {
        "capabilities": capabilities,
        "description": goal.goal,
        "limits": template.limits.model_copy(
            update={
                "max_steps": int(
                    goal.constraints.get("max_steps", template.limits.max_steps)
                )
            }
        ),
    }
    if name:
        updates["name"] = name
    if version:
        updates["version"] = version

    spec = AgentSpec.model_validate(
        {**template.model_dump(mode="json"), **updates}
    )
    report = validate(spec)
    if not report.ok:
        raise CompileError(
            "Compiled spec failed validation:\n"
            + "\n".join(f"  - {e}" for e in report.errors)
        )
    return spec
