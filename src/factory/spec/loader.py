"""Spec loading from YAML and JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from factory.spec.agent_spec import AgentSpec


class SpecError(ValueError):
    """Raised when a spec file cannot be loaded or fails validation."""


def load_spec(path: str | Path) -> AgentSpec:
    p = Path(path)
    if not p.exists():
        raise SpecError(f"Spec file not found: {p}")

    raw = p.read_text(encoding="utf-8")
    return parse_spec(raw, origin=str(p))


def parse_spec(raw: str, origin: str = "<string>") -> AgentSpec:
    try:
        data: Any = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise SpecError(f"Invalid YAML in {origin}: {exc}") from exc

    if data is None:
        raise SpecError(f"Spec in {origin} is empty")
    if not isinstance(data, dict):
        raise SpecError(f"Spec in {origin} must be a mapping, got {type(data).__name__}")

    try:
        return AgentSpec.model_validate(data)
    except ValidationError as exc:
        raise SpecError(f"Invalid spec in {origin}:\n{_format_errors(exc)}") from exc


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def to_json(spec: AgentSpec) -> str:
    return json.dumps(spec.model_dump(mode="json"), indent=2)
