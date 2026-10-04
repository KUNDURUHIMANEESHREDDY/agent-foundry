"""Capability registry.

The important property: a spec's `capabilities:` list is a REQUEST. Grants are
issued by the registry, and enforcement happens in `CapabilityGate.check()`,
which is called at the tool-call boundary on EVERY call — not once at
advertisement time. A prompt that mentions a tool it wasn't granted changes
nothing, because the gate is downstream of the model.
"""

from __future__ import annotations

from factory import isolation as isolation_module

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

class CapabilityDenied(PermissionError):
    """Raised when an agent attempts a capability it was not granted."""


class CapabilityNotFound(KeyError):
    """Raised when a spec requests a capability no provider implements."""


@dataclass(frozen=True, eq=False)
class Grant:
    """An issued capability plus the bounds it carries.

    `eq=False` gives grants identity semantics: a grant is a token minted at
    compile time, not a value compared for equality. This also makes it hashable
    despite holding a dict, which frozenset-of-grants requires.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


def object_schema(
    properties: dict[str, Any], required: Iterable[str]
) -> dict[str, Any]:
    """A JSON Schema object for tool arguments.

    `additionalProperties: False` is deliberate. Without it a model may invent
    parameters; with it, anything it invents is a schema violation the provider
    rejects outright, rather than an argument the capability silently ignores.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


class Capability(ABC):
    """Base for a capability. Subclasses declare what they need granted."""

    name: str = ""
    #: Shown to the model. Says what the capability is FOR, in the model's terms.
    description: str = ""
    #: Params an agent may never influence, enforced at the gate. The model can
    #: set these in its arguments all it likes; they are stripped before invoke.
    pinned_params: tuple[str, ...] = ()
    #: Subset of `pinned_params` that has no safe default, so the factory MUST
    #: bind it. Everything else in `pinned_params` may fall back to the
    #: capability's own default.
    required_params: tuple[str, ...] = ()

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        """JSON Schema for the arguments a model is allowed to supply.

        This is the capability's real contract with the model. It must describe
        every argument `invoke` reads, or a real model cannot call the
        capability at all — scripted tests hide this because they manufacture
        `ToolCall` objects directly.

        It must NOT mention `pinned_params`. Advertising a factory-bound
        parameter invites the model to set it; the gate strips it regardless,
        but the attempt is recorded as a violation and teaches the model that
        arguing with the gate is worth trying. `CapabilityRegistry.register`
        enforces this rather than trusting each implementation.
        """
        raise NotImplementedError(
            f"{cls.__name__} must declare parameter_schema(): the model needs a "
            f"real contract for {cls.name!r}, not a generic one"
        )

    @abstractmethod
    def invoke(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the capability. Must validate its own inputs."""

    def describe(self) -> str:
        return f"{self.name}({', '.join(self.pinned_params) if self.pinned_params else 'no bounds'})"


class CapabilityRegistry:
    """Holds capability implementations and mints grants.

    A `binder` supplies factory-bound parameters (e.g. the workspace root) when a
    grant is issued, so callers can request a capability by name alone.
    """

    def __init__(
        self,
        binder: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self._impls: dict[str, type[Capability]] = {}
        self._binder = binder or (lambda _name: {})

    def register(self, impl: type[Capability]) -> None:
        if not impl.name:
            raise ValueError(f"{impl.__name__} must define a non-empty name")
        if impl.name in self._impls:
            raise ValueError(f"Capability '{impl.name}' already registered")

        # Fail at registration, not at the first model call. A capability with no
        # schema cannot be called by a real model, and discovering that from a
        # 400 in production is too late.
        try:
            schema = impl.parameter_schema()
        except NotImplementedError as exc:
            raise ValueError(str(exc)) from exc

        if schema.get("type") != "object":
            raise ValueError(
                f"{impl.__name__}.parameter_schema() must describe an object, "
                f"got {schema.get('type')!r}"
            )

        advertised = set(schema.get("properties", {}))
        leaked = sorted(advertised & set(impl.pinned_params))
        if leaked:
            raise ValueError(
                f"{impl.__name__}.parameter_schema() advertises factory-pinned "
                f"param(s) {leaked}. The model must never be told these exist: "
                f"it can only attempt them, and the attempt is stripped at the "
                f"gate and recorded as a violation. Remove them from the schema."
            )

        self._impls[impl.name] = impl

    def tool_schema(self, name: str) -> dict[str, Any]:
        """The tool definition handed to the model for this capability.

        Assembled from the capability itself, so a capability cannot be granted
        to a model that has no way to call it correctly.
        """
        impl = self._impls.get(name)
        if impl is None:
            raise CapabilityNotFound(f"No implementation for capability '{name}'")
        return {
            "name": impl.name,
            "description": impl.description or f"Use capability {impl.name}",
            "parameters": impl.parameter_schema(),
        }

    def known(self) -> list[str]:
        return sorted(self._impls)

    def pinned_params(self, name: str) -> tuple[str, ...]:
        """Params the factory may bind but the agent may never supply."""
        impl = self._impls.get(name)
        if impl is None:
            raise CapabilityNotFound(f"No implementation for capability '{name}'")
        return impl.pinned_params

    def required_params(self, name: str) -> tuple[str, ...]:
        """Pinned params with no safe default; the factory must bind them."""
        impl = self._impls.get(name)
        if impl is None:
            raise CapabilityNotFound(f"No implementation for capability '{name}'")
        return impl.required_params

    def issue(self, name: str, **params: Any) -> Grant:
        """Mint a grant. Only the factory calls this; unknown names are rejected.

        Pinned params are supplied by the binder, deliberately: the workspace root
        is chosen by the factory, never by the model. Model-supplied attempts to
        set them are stripped later, at the gate.
        """
        impl = self._impls.get(name)
        if impl is None:
            raise CapabilityNotFound(
                f"No implementation for capability '{name}'. Known: {', '.join(self.known()) or '(none)'}"
            )

        bound = {**self._binder(name), **params}
        missing = [p for p in impl.required_params if p not in bound]
        if missing:
            raise ValueError(
                f"Capability '{name}' is missing factory-bound parameter(s): {missing}. "
                f"Configure a binder for this capability."
            )
        return Grant(name=name, params=bound)

    def instantiate(self, grant: Grant) -> Capability:
        impl = self._impls[grant.name]
        return impl(**grant.params)


class CapabilityGate:
    """Enforces grants at the tool-call boundary.

    The runtime holds exactly one of these and consults it before every single
    tool invocation. There is no code path from a model response to a capability
    that does not pass through `check()`.
    """

    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry
        self._grants: dict[str, Grant] = {}
        self._instances: dict[str, Capability] = {}

    @classmethod
    def of(
        cls, registry: CapabilityRegistry, grants: Iterable[Grant]
    ) -> "CapabilityGate":
        return cls(registry).mint(grants)

    def mint(self, grants: Iterable[Grant]) -> "CapabilityGate":
        """Install issued grants as the gate's active set."""
        for g in grants:
            self._grants[g.name] = g
        return self

    @property
    def granted(self) -> frozenset[Grant]:
        return frozenset(self._grants.values())

    @property
    def granted_names(self) -> list[str]:
        return sorted(self._grants)

    def is_granted(self, name: str) -> bool:
        return name in self._grants

    def sanitize_arguments(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Remove factory-pinned params from model-supplied arguments.

        A model that tries to set `root` on filesystem.read does not get a
        different sandbox — it gets the pinned one, and a violation is recorded.
        """
        if not self.is_granted(name):
            return dict(arguments), None
        try:
            pinned = self._registry.pinned_params(name)
        except CapabilityNotFound:
            return dict(arguments), None

        clean = {k: v for k, v in arguments.items() if k not in pinned}
        attempted = {k: v for k, v in arguments.items() if k in pinned}
        if not attempted:
            return clean, None

        return clean, {
            "capability": name,
            "reason": f"model attempted to set factory-pinned params: {sorted(attempted)}",
            "stripped": attempted,
        }

    def check(self, name: str) -> Capability:
        """Return the bound capability, or raise. Called on every tool call."""
        if not self.is_granted(name):
            raise CapabilityDenied(
                f"Capability '{name}' is not granted. Granted: "
                f"{sorted(self._grants) or '(none)'}"
            )
        if name not in self._instances:
            self._instances[name] = self._registry.instantiate(self._grants[name])
        return self._instances[name]

    def describe(self) -> list[str]:
        return [self._registry.instantiate(g).describe() for g in self._grants.values()]


def registry_for(
    workspace: str | Path, isolation: object = None
) -> CapabilityRegistry:
    """Registry with built-in capabilities bound to a workspace root.

    `git.commit` is registered only when the workspace is actually a
    repository, and its constructor raises otherwise. A capability that cannot
    be constructed should not appear available — that would let a spec request
    it and discover the failure at run time instead of compile time.
    """
    from factory.capabilities.builtin import FilesystemRead
    from factory.capabilities.fs_write import FilesystemWrite
    from factory.capabilities.python_exec import PythonExecute

    root = Path(workspace).resolve()
    # The containment a spec REQUIRES. Passed through to python.execute, which
    # refuses at run time if the runtime cannot provide it -- fail closed rather
    # than quietly running unconfined code.
    level = isolation_module.parse(isolation)

    def binder(name: str) -> dict[str, Any]:
        if name in ("filesystem.read", "filesystem.write"):
            return {"root": root}
        if name == "python.execute":
            return {"workspace": root, "sandbox_level": level.value}
        if name == "git.commit":
            return {"root": root}
        return {}

    reg = CapabilityRegistry(binder=binder)
    reg.register(FilesystemRead)
    reg.register(FilesystemWrite)
    reg.register(PythonExecute)

    if (root / ".git").exists():
        from factory.capabilities.git_commit import GitCommit

        reg.register(GitCommit)

    return reg


def safe_join(root: Path, candidate: str) -> Path:
    """Resolve a path inside root, refusing traversal and absolute escapes.

    Used by capabilities that accept a model-supplied path. The check is done
    on the resolved real path, so symlinks cannot escape the root.
    """
    if not candidate or not candidate.strip():
        raise ValueError("Path must be a non-empty string")

    root_resolved = Path(root).resolve()
    target = (root_resolved / candidate).resolve()

    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise CapabilityDenied(
            f"Path '{candidate}' resolves outside the workspace root"
        ) from exc

    return target
