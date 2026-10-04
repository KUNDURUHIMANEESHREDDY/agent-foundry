from __future__ import annotations

from pathlib import Path

import pytest

from factory.capabilities.builtin import FilesystemRead
from factory.capabilities.registry import Capability, CapabilityRegistry, object_schema
from factory.compiler import (
    CompileError,
    compile_agent,
    compile_goal,
    compile_goal_to_spec,
    plan_goal,
    validate,
)
from factory.models.base import ModelResponse, ToolCall
from factory.models.fake import ScriptedAdapter
from factory.spec.agent_spec import AgentSpec, GoalSpec, Limits, ModelRef


class Shell(Capability):
    name = "shell.execute"

    @classmethod
    def parameter_schema(cls):
        return object_schema({"cmd": {"type": "string"}}, required=["cmd"])

    def invoke(self, **kwargs):
        return {"ok": True, "stdout": "(should never run)"}


def registry() -> CapabilityRegistry:
    r = CapabilityRegistry()
    r.register(FilesystemRead)
    r.register(Shell)
    return r


def bound_registry(workspace) -> CapabilityRegistry:
    r = CapabilityRegistry(
        binder=lambda name: {"root": Path(workspace)} if name == "filesystem.read" else {}
    )
    r.register(FilesystemRead)
    r.register(Shell)
    return r


def spec(**over) -> AgentSpec:
    base = {
        "name": "a",
        "system_prompt": "you are an agent",
        "capabilities": ["filesystem.read"],
        "model": ModelRef(provider="fake", name="none", context_window=4096, max_output_tokens=256),
        "limits": Limits(),
    }
    base.update(over)
    return AgentSpec(**base)


class TestValidation:
    def test_valid_spec_passes(self):
        assert validate(spec()).ok

    def test_window_must_exceed_reserve(self):
        s = spec(model=ModelRef(provider="fake", name="n", context_window=100, max_output_tokens=50),
                 limits=Limits(context_reserve_tokens=200))
        report = validate(s)
        assert not report.ok
        assert any("context_window" in e for e in report.errors)

    def test_output_tokens_must_fit_window(self):
        s = spec(model=ModelRef(provider="fake", name="n", context_window=100, max_output_tokens=500))
        assert not validate(s).ok

    def test_warns_on_sqlite_without_path(self):
        s = spec(memory={"type": "sqlite", "path": None})
        report = validate(s)
        assert any("in-memory" in w for w in report.warnings)


class TestCompileChain:
    def test_grants_only_requested_capabilities(self, tmp_path):
        s = spec(capabilities=["filesystem.read"])
        compiled = compile_agent(s, bound_registry(tmp_path), workspace=tmp_path)

        assert compiled.granted == ["filesystem.read"]
        assert compiled.gate.is_granted("filesystem.read")
        assert not compiled.gate.is_granted("shell.execute")

    def test_empty_capabilities_grants_nothing(self, tmp_path):
        s = spec(capabilities=[])
        compiled = compile_agent(s, bound_registry(tmp_path), workspace=tmp_path)
        assert compiled.granted == []
        assert "no tool capabilities" in compiled.spec.system_prompt

    def test_strict_mode_rejects_unknown_capability(self, tmp_path):
        s = spec(capabilities=["filesystem.read", "quantum.telepathy"])
        with pytest.raises(CompileError, match="cannot be granted"):
            compile_agent(s, bound_registry(tmp_path), workspace=tmp_path)

    def test_lenient_mode_drops_and_warns(self, tmp_path):
        s = spec(capabilities=["filesystem.read", "quantum.telepathy"])
        compiled = compile_agent(s, bound_registry(tmp_path), workspace=tmp_path, strict=False)

        assert compiled.granted == ["filesystem.read"]
        assert any("quantum.telepathy" in w for w in compiled.warnings)

    def test_workspace_pinned_into_capability(self, tmp_path):
        (tmp_path / "in.txt").write_text("inside")
        (tmp_path.parent / "outside.txt").write_text("secret")

        s = spec(capabilities=["filesystem.read"])
        compiled = compile_agent(s, bound_registry(tmp_path), workspace=tmp_path)
        cap = compiled.gate.check("filesystem.read")

        assert cap.invoke(path="in.txt")["content"] == "inside"
        escaped = cap.invoke(path="../outside.txt")
        assert escaped["ok"] is False
        assert "outside the workspace" in escaped["error"]

    def test_prompt_includes_grants_and_workspace(self, tmp_path):
        s = spec(capabilities=["filesystem.read"])
        compiled = compile_agent(s, bound_registry(tmp_path), workspace=tmp_path)

        assert "filesystem.read" in compiled.spec.system_prompt
        assert str(tmp_path.resolve()) in compiled.spec.system_prompt

    def test_injected_model_is_used(self, tmp_path):
        adapter = ScriptedAdapter([ModelResponse(text="ok")])
        compiled = compile_agent(spec(), bound_registry(tmp_path), workspace=tmp_path, model=adapter)
        assert compiled.model is adapter

    def test_validation_failure_blocks_compile(self, tmp_path):
        bad = spec(model=ModelRef(provider="fake", name="n", context_window=10, max_output_tokens=999))
        with pytest.raises(CompileError, match="failed validation"):
            compile_agent(bad, bound_registry(tmp_path), workspace=tmp_path)


class TestCompileGoal:
    def test_maps_requirements_to_capabilities(self, tmp_path):
        goal = GoalSpec(goal="read a file", requirements={"files": True})
        template = spec(capabilities=[])
        compiled = compile_goal(goal, template, bound_registry(tmp_path), workspace=tmp_path)
        assert compiled.granted == ["filesystem.read"]

    def test_merges_template_and_derived(self, tmp_path):
        goal = GoalSpec(goal="do things", requirements={"python": True})
        template = spec(capabilities=["filesystem.read"])
        compiled = compile_goal(goal, template, bound_registry(tmp_path), workspace=tmp_path, strict=False)
        assert "filesystem.read" in compiled.granted

    def test_unknown_requirement_is_rejected_not_guessed(self, tmp_path):
        goal = GoalSpec(goal="teleport", requirements={"quantum": True})
        with pytest.raises(CompileError, match="no capability mapping"):
            compile_goal(goal, spec(), bound_registry(tmp_path), workspace=tmp_path)

    def test_constraint_overrides_limit(self, tmp_path):
        goal = GoalSpec(goal="g", requirements={}, constraints={"max_steps": 3})
        compiled = compile_goal(goal, spec(), bound_registry(tmp_path), workspace=tmp_path)
        assert compiled.spec.limits.max_steps == 3

    def test_requirement_needing_absent_capability_is_rejected(self, tmp_path):
        """A mapped requirement whose capability has no provider must still fail.

        The bare registry has no python.execute, so 'python' cannot be granted
        even though the requirement name is recognised.
        """
        reg = CapabilityRegistry()
        reg.register(FilesystemRead)
        goal = GoalSpec(goal="research", requirements={"python": True})
        with pytest.raises(CompileError, match="no provider implements"):
            compile_goal(
                goal, spec(), reg, known_capabilities=set(reg.known()), workspace=tmp_path
            )

    def test_plan_goal_reports_proposal_without_compiling(self, tmp_path):
        goal = GoalSpec(goal="research", requirements={"python": True, "files": True})
        caps, errors, proposal = plan_goal(goal, spec(capabilities=[]))
        assert caps == ["filesystem.read", "python.execute"]
        assert errors == {}
        assert {p["capability"] for p in proposal} == {"python.execute", "filesystem.read"}

    def test_plan_goal_does_not_dupe_template_capabilities(self):
        goal = GoalSpec(goal="g", requirements={"files": True})
        caps, _, proposal = plan_goal(goal, spec(capabilities=["filesystem.read"]))
        assert caps.count("filesystem.read") == 1
        assert proposal == []
