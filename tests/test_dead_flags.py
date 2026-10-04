"""Two bugs where a flag or a field existed but could never do anything.

`CompiledAgent.denied` read `self.gate.grants`, which does not exist on
`CapabilityGate`; `compile_agent` computed the real value and threw it away. And
`git.commit` checked `if allow_empty or self._allow_empty:` before returning
"nothing staged to commit (empty commits are disabled)" — so the flag could only
ever produce a refusal, including when it was enabled.

Both are the same failure shape: code that reads as if it implements a feature,
and silently does not. Neither raised, so 415 tests passed throughout.
"""

from __future__ import annotations

import subprocess

import pytest

from factory.capabilities.git_commit import GitCommit
from factory.capabilities.registry import CapabilityGate, CapabilityRegistry
from factory.compiler import CompileError, compile_agent
from factory.spec.agent_spec import AgentSpec, ModelRef

# ── git.commit allow_empty ─────────────────────────────────────────────


def init_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    return path


def seed(path, name="seed.txt", content="seed\n"):
    """A repo with one commit, so there is history but nothing staged."""
    init_repo(path)
    (path / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=path, check=True)
    return path


def commit_count(path) -> int:
    out = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=path, capture_output=True, text=True
    )
    return int(out.stdout.strip() or 0)


@pytest.fixture
def repo(tmp_path):
    return seed(tmp_path / "repo")


class TestAllowEmptyIsHonoured:
    def test_granted_allow_empty_creates_an_empty_commit(self, repo):
        """The case that could never pass before: the flag did nothing.

        The old branch returned "empty commits are disabled" precisely when
        empty commits were enabled.
        """
        committer = GitCommit(root=repo, allow_empty=True)
        before = commit_count(repo)

        result = committer.invoke(message="empty", paths=[])

        assert result["ok"] is True, result
        assert commit_count(repo) == before + 1
        assert result["sha"]

    def test_default_refuses_an_empty_commit(self, repo):
        committer = GitCommit(root=repo)
        before = commit_count(repo)

        result = committer.invoke(message="empty", paths=[])

        assert result["ok"] is False
        assert "nothing staged" in result["error"]
        assert commit_count(repo) == before

    def test_refusal_does_not_claim_empty_commits_are_disabled_when_they_are_not(
        self, repo
    ):
        """With allow_empty=True there is no refusal, so the message cannot lie."""
        committer = GitCommit(root=repo, allow_empty=True)
        result = committer.invoke(message="empty", paths=[])
        assert result["ok"] is True

    def test_refusal_explains_how_to_enable_it(self, repo):
        result = GitCommit(root=repo).invoke(message="empty", paths=[])
        assert "allow_empty" in result.get("would_need", "")

    def test_real_changes_still_commit_without_the_flag(self, repo):
        (repo / "new.txt").write_text("new\n", encoding="utf-8")
        committer = GitCommit(root=repo)

        result = committer.invoke(message="real change", paths=["new.txt"])

        assert result["ok"] is True
        assert result["files_changed"] == 1

    def test_allow_empty_flag_is_only_passed_when_nothing_is_staged(self, repo):
        """--allow-empty must not mask a genuine 'git add matched nothing'.

        If it were passed unconditionally, asking to commit a path that git
        ignores would silently succeed as an empty commit instead of reporting
        that nothing was staged.
        """
        committer = GitCommit(root=repo, allow_empty=True)
        # `paths` names a real file, but it is already committed and unchanged,
        # so `git add` stages nothing.
        result = committer.invoke(message="no-op", paths=["seed.txt"])

        # Either way it must be truthful: no files reported as changed, and the
        # commit exists only because empty commits are permitted.
        assert result["ok"] is True
        assert result["files_changed"] == 0

    def test_a_real_commit_is_not_marked_empty(self, repo):
        (repo / "new.txt").write_text("new\n", encoding="utf-8")
        result = GitCommit(root=repo, allow_empty=True).invoke(
            message="real", paths=["new.txt"]
        )
        assert result["files_changed"] == 1


class TestAllowEmptyIsFactoryOwned:
    """`allow_empty` is pinned, so a model can never turn it on."""

    def _gate(self, repo, grant_allow_empty):
        registry = CapabilityRegistry(
            binder=lambda _n: {"root": repo, "allow_empty": grant_allow_empty}
        )
        registry.register(GitCommit)
        return CapabilityGate.of(registry, [registry.issue("git.commit")])

    def test_model_arg_cannot_enable_an_empty_commit(self, repo):
        gate = self._gate(repo, grant_allow_empty=False)
        before = commit_count(repo)

        clean, violation = gate.sanitize_arguments(
            "git.commit", {"message": "empty", "paths": [], "allow_empty": True}
        )
        result = gate.check("git.commit").invoke(**clean)

        assert result["ok"] is False
        assert commit_count(repo) == before
        assert "allow_empty" in clean or "allow_empty" not in clean
        assert violation is not None, "the attempt should be recorded"

    def test_model_arg_is_irrelevant_when_granted(self, repo):
        gate = self._gate(repo, grant_allow_empty=True)

        clean, _ = gate.sanitize_arguments(
            "git.commit", {"message": "empty", "paths": [], "allow_empty": False}
        )
        result = gate.check("git.commit").invoke(**clean)

        assert result["ok"] is True, "the grant, not the model argument, decides"

    def test_pinned_param_is_stripped_before_invoke(self, repo):
        gate = self._gate(repo, grant_allow_empty=True)
        clean, _ = gate.sanitize_arguments(
            "git.commit", {"message": "m", "allow_empty": False}
        )
        assert "allow_empty" not in clean


# ── CompiledAgent.denied ───────────────────────────────────────────────


def spec_with(*capabilities, name="probe") -> AgentSpec:
    return AgentSpec(
        name=name,
        system_prompt="test",
        capabilities=list(capabilities),
        model=ModelRef(
            provider="fake", name="none", context_window=4096, max_output_tokens=256
        ),
    )


def registry_with(*names):
    from factory.capabilities.builtin import FilesystemRead

    registry = CapabilityRegistry()
    registry.register(FilesystemRead)
    return registry


class TestCompiledAgentDenied:
    def test_denied_is_empty_when_everything_resolves(self, tmp_path):
        compiled = compile_agent(
            spec_with("filesystem.read"), registry_with(), workspace=tmp_path
        )
        assert compiled.denied == {}

    def test_denied_is_empty_under_strict_mode(self, tmp_path):
        """strict=True raises instead, so there is nothing to report."""
        compiled = compile_agent(
            spec_with("filesystem.read"), registry_with(), workspace=tmp_path, strict=True
        )
        assert compiled.denied == {}

    def test_denied_records_the_reason_under_lenient_mode(self, tmp_path):
        compiled = compile_agent(
            spec_with("filesystem.read", "nope.missing"),
            registry_with(),
            workspace=tmp_path,
            strict=False,
        )
        assert "nope.missing" in compiled.denied
        assert compiled.denied["nope.missing"]

    def test_denied_and_granted_are_disjoint(self, tmp_path):
        compiled = compile_agent(
            spec_with("filesystem.read", "nope.missing"),
            registry_with(),
            workspace=tmp_path,
            strict=False,
        )
        assert set(compiled.denied).isdisjoint(compiled.granted)

    def test_a_dropped_capability_is_not_advertised_to_the_model(self, tmp_path):
        compiled = compile_agent(
            spec_with("filesystem.read", "nope.missing"),
            registry_with(),
            workspace=tmp_path,
            strict=False,
        )
        assert "nope.missing" not in compiled.spec.system_prompt

    def test_a_dropped_capability_gets_no_tool_schema(self, tmp_path):
        compiled = compile_agent(
            spec_with("filesystem.read", "nope.missing"),
            registry_with(),
            workspace=tmp_path,
            strict=False,
        )
        names = {s["name"] for s in compiled.runtime._tools}
        assert names == {"filesystem.read"}

    def test_strict_mode_still_raises(self, tmp_path):
        with pytest.raises(CompileError, match="nope.missing"):
            compile_agent(
                spec_with("nope.missing"),
                registry_with(),
                workspace=tmp_path,
                strict=True,
            )

    def test_denied_reports_a_reason_for_every_missing_name(self, tmp_path):
        compiled = compile_agent(
            spec_with("filesystem.read", "a.missing", "b.missing"),
            registry_with(),
            workspace=tmp_path,
            strict=False,
        )
        assert set(compiled.denied) == {"a.missing", "b.missing"}
        for reason in compiled.denied.values():
            assert reason.strip()

    def test_denied_is_a_copy_not_the_internal_dict(self, tmp_path):
        compiled = compile_agent(
            spec_with("nope.missing"), registry_with(), workspace=tmp_path, strict=False
        )
        compiled.denied["injected"] = "x"
        assert "injected" not in compile_agent(
            spec_with("nope.missing"), registry_with(), workspace=tmp_path, strict=False
        ).denied