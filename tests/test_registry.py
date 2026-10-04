from __future__ import annotations

import pytest

from factory.compiler import compile_goal_to_spec, plan_goal
from factory.registry import AgentRegistry, RegistryError, version_key
from factory.spec.agent_spec import AgentSpec, GoalSpec
from factory.spec.loader import load_spec

AGENTS = "agents"


def spec(name="reader", version="0.1.0", caps=("filesystem.read",), steps=8) -> AgentSpec:
    from factory.spec.agent_spec import Limits, ModelRef

    return AgentSpec(
        name=name or "template",
        version=version or "0.0.1",
        description=f"{name or 'template'} agent",
        system_prompt="you are a test agent",
        capabilities=list(caps),
        model=ModelRef(provider="fake", name="none"),
        limits=Limits(max_steps=steps),
    )


@pytest.fixture
def reg() -> AgentRegistry:
    return AgentRegistry(":memory:")


class TestRegister:
    def test_registers_and_reads_back(self, reg):
        rec = reg.register(spec(), notes="first cut", tags=["baseline"])
        assert rec.agent == "reader"
        assert reg.get("reader").version == "0.1.0"
        assert reg.get_version("reader", "0.1.0").notes == "first cut"
        assert reg.get_version("reader", "0.1.0").tags == ["baseline"]

    def test_rejects_duplicate_version(self, reg):
        reg.register(spec())
        with pytest.raises(RegistryError, match="already exists"):
            reg.register(spec())

    def test_replace_overwrites_in_place(self, reg):
        reg.register(spec())
        rec = reg.register(spec(steps=99), replace=True)
        assert rec.spec.limits.max_steps == 99

    def test_second_version_coexists(self, reg):
        reg.register(spec(version="0.1.0"))
        reg.register(spec(version="0.2.0"))
        assert len(reg.versions("reader")) == 2

    def test_register_from_file(self, reg):
        rec = reg.register_file(f"{AGENTS}/reader.yaml")
        assert rec.agent == "reader"
        assert rec.spec.capabilities == ["filesystem.read"]
        assert rec.source.endswith("reader.yaml")

    def test_register_file_rejects_bad_spec(self, reg, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("name: Bad_Name!\n", encoding="utf-8")
        with pytest.raises(Exception):
            reg.register_file(p)

    def test_name_pattern_blocks_whitespace(self, reg):
        """AgentSpec's pattern rejects ' reader ' before the registry sees it."""
        with pytest.raises(Exception, match="pattern"):
            spec(name=" reader ")


class TestListing:
    def test_lists_agents(self, reg):
        reg.register(spec(name="reader"))
        reg.register(spec(name="coder"))
        names = [a.name for a in reg.list()]
        assert names == ["coder", "reader"]

    def test_summary_carries_counts_and_latest(self, reg):
        reg.register(spec(version="0.1.0"))
        reg.register(spec(version="0.3.0"))
        summary = next(a for a in reg.list() if a.name == "reader")
        assert summary.versions == 2
        assert summary.latest == "0.3.0"

    def test_empty_registry(self, reg):
        assert reg.list() == []
        assert reg.versions("nope") == []
        assert reg.get("nope") is None

    def test_exists(self, reg):
        reg.register(spec(version="0.1.0"))
        assert reg.exists("reader")
        assert reg.exists("reader", "0.1.0")
        assert not reg.exists("reader", "9.9.9")
        assert not reg.exists("ghost")

    def test_latest_is_highest_version(self, reg):
        for v in ("0.1.0", "0.10.0", "0.2.0"):
            reg.register(spec(version=v))
        assert reg.latest("reader").version == "0.10.0"

    def test_versions_sorted_numerically(self, reg):
        for v in ("0.10.0", "0.2.0", "0.1.0"):
            reg.register(spec(version=v))
        assert [x.version for x in reg.versions("reader")] == ["0.1.0", "0.2.0", "0.10.0"]


class TestVersionOrdering:
    def test_numeric_not_lexicographic(self):
        assert version_key("0.10.0") > version_key("0.9.0")

    def test_pads_missing_components(self):
        assert version_key("1.2") == (1, 2, 0)

    def test_tolerates_non_numeric(self):
        assert version_key("1.2.3-rc1") == (1, 2, 3)


class TestDiff:
    def test_detects_capability_addition(self, reg):
        reg.register(spec(version="0.1.0", caps=("filesystem.read",)))
        reg.register(spec(version="0.2.0", caps=("filesystem.read", "python.execute")))
        d = reg.diff("reader", "0.1.0", "0.2.0")
        assert d["changed"]
        assert d["capabilities_added"] == ["python.execute"]
        assert d["capabilities_removed"] == []

    def test_detects_capability_removal(self, reg):
        reg.register(spec(version="0.1.0", caps=("filesystem.read", "python.execute")))
        reg.register(spec(version="0.2.0", caps=("filesystem.read",)))
        d = reg.diff("reader", "0.1.0", "0.2.0")
        assert d["capabilities_removed"] == ["python.execute"]

    def test_from_goal_can_derive_write_and_git(self, reg):
        """The 'filesystem' requirement implies write, and 'git' implies commit."""
        goal = GoalSpec(goal="edit and commit", requirements={"filesystem": True, "git": True})
        known = {"filesystem.read", "filesystem.write", "git.commit"}
        out = compile_goal_to_spec(
            goal, spec(caps=()), known, name="author", version="0.1.0"
        )
        assert set(out.capabilities) == {"filesystem.read", "filesystem.write", "git.commit"}
        rec = reg.register(out, tags=["generated"])
        assert reg.get("author").capabilities == out.capabilities

    def test_from_goal_refuses_state_changing_capability_without_provider(self, reg):
        goal = GoalSpec(goal="commit things", requirements={"git": True})
        # No git.commit provider: must fail rather than quietly skip.
        with pytest.raises(Exception, match="no provider implements"):
            compile_goal_to_spec(
                goal, spec(caps=()), {"filesystem.read"}, name="author"
            )

    def test_no_change_when_only_version_differs(self, reg):
        """`version` is the row key, not a diffable field.

        Comparing the stored dicts would always report the version bump as a
        change, so it is excluded.
        """
        reg.register(spec(version="0.1.0"))
        reg.register(spec(version="0.2.0"))
        d = reg.diff("reader", "0.1.0", "0.2.0")
        assert not d["changed"], d["changes"]
        assert d["changes"] == []

    def test_reports_limit_changes(self, reg):
        reg.register(spec(version="0.1.0", steps=8))
        reg.register(spec(version="0.2.0", steps=20))
        d = reg.diff("reader", "0.1.0", "0.2.0")
        assert any(c["field"] == "limits" for c in d["changes"])

    def test_missing_version_raises(self, reg):
        reg.register(spec(version="0.1.0"))
        with pytest.raises(RegistryError, match="not found"):
            reg.diff("reader", "0.1.0", "9.9.9")


class TestDelete:
    def test_deletes_one_version(self, reg):
        reg.register(spec(version="0.1.0"))
        reg.register(spec(version="0.2.0"))
        assert reg.delete("reader", "0.1.0") == 1
        assert [x.version for x in reg.versions("reader")] == ["0.2.0"]

    def test_deletes_agent_and_all_versions(self, reg):
        reg.register(spec(version="0.1.0"))
        reg.register(spec(version="0.2.0"))
        assert reg.delete("reader") == 2
        assert reg.list() == []

    def test_delete_missing_returns_zero(self, reg):
        assert reg.delete("ghost") == 0


class TestPersistence:
    def test_survives_reopen(self, tmp_path):
        db = tmp_path / "agents.db"
        first = AgentRegistry(db)
        first.register(spec(), notes="persisted")

        second = AgentRegistry(db)
        assert second.get_version("reader", "0.1.0").notes == "persisted"

    def test_created_at_preserved_on_replace(self, tmp_path):
        db = tmp_path / "agents.db"
        r = AgentRegistry(db)
        original = r.register(spec(version="0.1.0")).created_at
        replaced = r.register(spec(version="0.1.0", steps=2), replace=True).created_at
        assert replaced >= original


class TestGoalToSpec:
    def test_goal_becomes_addressable_spec(self):
        goal = GoalSpec(goal="read project files", requirements={"files": True})
        out = compile_goal_to_spec(goal, spec(name="", version="", caps=()))
        assert out.capabilities == ["filesystem.read"]
        assert out.description == "read project files"

    def test_applies_step_constraint(self):
        goal = GoalSpec(goal="g", requirements={}, constraints={"max_steps": 3})
        out = compile_goal_to_spec(goal, spec(caps=()))
        assert out.limits.max_steps == 3

    def test_unknown_requirement_fails_closed(self):
        goal = GoalSpec(goal="g", requirements={"quantum": True})
        with pytest.raises(Exception, match="no capability mapping"):
            compile_goal_to_spec(goal, spec(caps=()))

    def test_plan_is_side_effect_free(self):
        goal = GoalSpec(goal="g", requirements={"files": True, "python": True})
        caps, errors, proposal = plan_goal(goal, spec(caps=()))
        assert errors == {}
        assert set(caps) == {"filesystem.read", "python.execute"}
        assert len(proposal) == 2

    def test_end_to_end_goal_into_registry(self, reg):
        """The compiler's payoff: a goal becomes a registered, versioned agent."""
        goal = GoalSpec(goal="research in files", requirements={"files": True})
        generated = compile_goal_to_spec(goal, spec(name="", version="", caps=()))
        generated = generated.model_copy(update={"name": "researcher", "version": "0.1.0"})

        rec = reg.register(generated, tags=["generated"])
        assert rec.agent == "researcher"
        assert reg.get("researcher").capabilities == ["filesystem.read"]
        summary = next(a for a in reg.list() if a.name == "researcher")
        assert summary.latest == "0.1.0"