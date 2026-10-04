from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from factory.capabilities.fs_write import FilesystemWrite
from factory.capabilities.git_commit import GitCommit, GitNotARepository
from factory.capabilities.registry import registry_for


@pytest.fixture
def ws(tmp_path):
    return tmp_path


@pytest.fixture
def writer(ws):
    return FilesystemWrite(root=ws)


class TestWriteBasics:
    def test_creates_a_file(self, ws, writer):
        r = writer.invoke(path="a.txt", content="hello")
        assert r["ok"] is True
        assert r["created"] is True
        assert r["bytes_written"] == 5
        assert (ws / "a.txt").read_text(encoding="utf-8") == "hello"

    def test_creates_parent_directories(self, ws, writer):
        r = writer.invoke(path="deep/nested/a.txt", content="x")
        assert r["ok"] is True
        assert (ws / "deep" / "nested" / "a.txt").exists()

    def test_overwrites_when_asked(self, ws, writer):
        writer.invoke(path="a.txt", content="first")
        r = writer.invoke(path="a.txt", content="second", overwrite=True)
        assert r["ok"] is True
        assert r["overwrote"] is True
        assert (ws / "a.txt").read_text(encoding="utf-8") == "second"

    def test_append(self, ws, writer):
        writer.invoke(path="a.txt", content="one\n")
        r = writer.invoke(path="a.txt", content="two\n", append=True)
        assert r["ok"] is True
        assert r["appended"] is True
        assert (ws / "a.txt").read_text(encoding="utf-8") == "one\ntwo\n"

    def test_rejects_overwrite_and_append_together(self, writer):
        r = writer.invoke(path="a.txt", content="x", overwrite=True, append=True)
        assert r["ok"] is False
        assert "both" in r["error"]


class TestWriteValidation:
    def test_rejects_empty_path(self, writer):
        assert writer.invoke(path="", content="x")["ok"] is False

    def test_rejects_non_string_content(self, writer):
        r = writer.invoke(path="a.txt", content=12345)
        assert r["ok"] is False
        assert "string" in r["error"]

    def test_rejects_non_string_path(self, writer):
        assert writer.invoke(path=999, content="x")["ok"] is False

    def test_caps_content_size(self, ws):
        small = FilesystemWrite(root=ws, max_bytes=10)
        r = small.invoke(path="a.txt", content="x" * 100)
        assert r["ok"] is False
        assert "limit" in r["error"]
        assert not (ws / "a.txt").exists(), "oversized write must not create a file"

    def test_refuses_directory_target(self, ws, writer):
        (ws / "adir").mkdir()
        r = writer.invoke(path="adir", content="x")
        assert r["ok"] is False
        assert "directory" in r["error"]


class TestWriteBoundary:
    @pytest.mark.parametrize("bad", ["../escape.txt", "../../etc/passwd", "/etc/passwd"])
    def test_refuses_escape(self, ws, writer, bad):
        r = writer.invoke(path=bad, content="pwned")
        assert r["ok"] is False
        assert "outside the workspace" in r["error"]

    def test_symlink_cannot_escape(self, ws, writer):
        secret = ws.parent / "secret.txt"
        secret.write_text("classified", encoding="utf-8")
        try:
            (ws / "link.txt").symlink_to(secret)
        except OSError:
            pytest.skip("symlinks unavailable")

        r = writer.invoke(path="link.txt", content="pwned")
        assert r["ok"] is False
        assert "outside the workspace" in r["error"]
        assert secret.read_text(encoding="utf-8") == "classified"

    def test_pinned_root_cannot_be_rebound(self, ws, writer):
        """The model asking for root='/' does not move the sandbox."""
        from factory.capabilities.registry import CapabilityGate, CapabilityRegistry

        reg = CapabilityRegistry(binder=lambda n: {"root": ws})
        reg.register(FilesystemWrite)
        gate = CapabilityGate.of(reg, [reg.issue("filesystem.write")])

        clean, violation = gate.sanitize_arguments(
            "filesystem.write", {"path": "a.txt", "root": "/", "allow_overwrite": True}
        )
        assert "root" not in clean
        assert violation is not None
        assert "pinned" in violation["reason"]


class TestWriteOverwritePolicy:
    def test_refuses_by_default_when_disallowed(self, ws):
        strict = FilesystemWrite(root=ws, allow_overwrite=False)
        strict.invoke(path="a.txt", content="one")
        r = strict.invoke(path="a.txt", content="two")
        assert r["ok"] is False
        assert "already exists" in r["error"]
        assert (ws / "a.txt").read_text(encoding="utf-8") == "one"

    def test_allows_when_policy_permits(self, ws):
        lax = FilesystemWrite(root=ws, allow_overwrite=True)
        lax.invoke(path="a.txt", content="one")
        assert lax.invoke(path="a.txt", content="two")["ok"] is True

    def test_explicit_overwrite_wins_even_when_policy_denies(self, ws):
        strict = FilesystemWrite(root=ws, allow_overwrite=False)
        strict.invoke(path="a.txt", content="one")
        r = strict.invoke(path="a.txt", content="two", overwrite=True)
        assert r["ok"] is True


class TestWriteAtomicity:
    def test_no_temp_files_left_behind(self, ws, writer):
        writer.invoke(path="a.txt", content="hello")
        leftovers = [p.name for p in ws.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []

    def test_write_is_atomic_not_truncated_on_failure(self, ws, monkeypatch, writer):
        writer.invoke(path="a.txt", content="original")

        import os as real_os

        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(real_os, "replace", boom)
        r = writer.invoke(path="a.txt", content="replacement", overwrite=True)
        assert r["ok"] is False
        assert (ws / "a.txt").read_text(encoding="utf-8") == "original"
        assert [p.name for p in ws.iterdir() if p.name.endswith(".tmp")] == []


# ── git.commit ────────────────────────────────────────────────────────


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)
    return path


def git(path: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=str(path), capture_output=True, text=True, check=False
    )
    return r.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    p = init_repo(tmp_path / "repo")
    (p / "a.txt").write_text("hello\n", encoding="utf-8")
    (p / "b.txt").write_text("world\n", encoding="utf-8")
    return p


@pytest.fixture
def committer(repo):
    return GitCommit(root=repo)


class TestGitRegistration:
    def test_registered_only_inside_a_repo(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        assert "git.commit" not in registry_for(plain).known()

        repo = init_repo(tmp_path / "repo2")
        assert "git.commit" in registry_for(repo).known()

    def test_constructor_refuses_non_repo(self, tmp_path):
        with pytest.raises(GitNotARepository):
            GitCommit(root=tmp_path)


class TestGitCommit:
    def test_commits_named_paths(self, repo, committer):
        r = committer.invoke(message="add a", paths=["a.txt"])
        assert r["ok"] is True
        assert r["sha"]
        assert "add a" in git(repo, "log", "--oneline")

    def test_reports_changed_files(self, repo, committer):
        r = committer.invoke(message="add both", paths=["a.txt", "b.txt"])
        assert r["ok"] is True
        assert r["files_changed"] == 2

    def test_only_named_paths_are_committed(self, repo, committer):
        r = committer.invoke(message="just a", paths=["a.txt"])
        committed = git(repo, "show", "--name-only", "--format=", r["sha"])
        assert "a.txt" in committed
        assert "b.txt" not in committed

    def test_requires_a_message(self, repo, committer):
        assert committer.invoke(message="", paths=["a.txt"])["ok"] is False
        assert committer.invoke(message="   ", paths=["a.txt"])["ok"] is False

    def test_refuses_empty_commit_by_default(self, repo, committer):
        r = committer.invoke(message="nothing to do", paths=["a.txt"])
        assert r["ok"] is True  # first commit works

        r2 = committer.invoke(message="again", paths=["a.txt"])
        assert r2["ok"] is False
        assert "nothing staged" in r2["error"]

    def test_refuses_unknown_path(self, repo, committer):
        r = committer.invoke(message="m", paths=["ghost.txt"])
        assert r["ok"] is False
        assert "no such path" in r["error"]

    def test_identify_missing_path(self, repo, committer):
        r = committer.invoke(message="m", paths=["../outside.txt"])
        assert r["ok"] is False
        assert "outside the workspace" in r["error"]


class TestGitBoundary:
    def test_refuses_path_escape(self, repo, committer):
        (repo.parent / "outside.txt").write_text("secret", encoding="utf-8")
        r = committer.invoke(message="leak", paths=["../outside.txt"])
        assert r["ok"] is False
        assert "outside the workspace" in r["error"]

    @pytest.mark.parametrize("bad", ["../outside.txt", "/etc/passwd"])
    def test_refuses_absolute_and_traversal(self, repo, committer, bad):
        r = committer.invoke(message="leak", paths=[bad])
        assert r["ok"] is False

    def test_push_is_not_reachable(self, committer):
        for forbidden in ("push", "reset", "amend", "tag", "branch"):
            with pytest.raises(PermissionError, match="not permitted"):
                committer._run_env(forbidden)

    def test_subcommand_allowlist(self):
        from factory.capabilities.git_commit import ALLOWED_SUBCOMMANDS

        assert "push" not in ALLOWED_SUBCOMMANDS
        assert {"add", "commit", "status"} <= ALLOWED_SUBCOMMANDS

    def test_caps_path_count(self, repo, committer):
        r = committer.invoke(message="m", paths=[f"a.txt"] * 51)
        assert r["ok"] is False
        assert "at most" in r["error"]

    def test_caps_message_length(self, repo, committer):
        r = committer.invoke(message="x" * 5000, paths=["a.txt"])
        assert r["ok"] is False
        assert "exceeds" in r["error"]

    def test_author_is_deterministic(self, repo, committer):
        r = committer.invoke(message="authored", paths=["a.txt"])
        assert r["ok"] is True
        name, email = git(repo, "log", "-1", "--format=%an|%ae").split("|")
        assert name == "agent-factory"
        assert email == "agent-factory@localhost"


class TestGitThroughGate:
    @pytest.mark.asyncio
    async def test_git_commit_granted_and_denial_recorded(self, repo):
        from factory.capabilities.git_commit import GitCommit
        from factory.capabilities.registry import CapabilityGate, CapabilityRegistry
        from factory.compiler import compile_agent
        from factory.models.base import ModelResponse, ToolCall
        from factory.models.fake import ScriptedAdapter
        from factory.runtime.trace import MemoryTraceStore
        from factory.spec.agent_spec import AgentSpec, Limits, ModelRef

        def binder(name):
            return {"root": repo} if name == "git.commit" else {}

        reg = CapabilityRegistry(binder=binder)
        reg.register(GitCommit)

        spec = AgentSpec(
            name="committer",
            system_prompt="you commit",
            capabilities=["git.commit"],
            model=ModelRef(provider="fake", name="none"),
            limits=Limits(max_steps=4),
        )

        # Granted: commits for real.
        model = ScriptedAdapter([
            ModelResponse(tool_calls=[ToolCall("git.commit", {
                "message": "add a", "paths": ["a.txt"]})]),
            ModelResponse(text="committed"),
        ])
        compiled = compile_agent(
            spec, reg, workspace=repo, model=model, store=MemoryTraceStore()
        )
        result = await compiled.runtime.run("commit a.txt")
        assert result.status == "success"
        assert result.trace.steps[0].observations[0]["ok"] is True
        assert "add a" in git(repo, "log", "--oneline")

    @pytest.mark.asyncio
    async def test_ungranted_git_commit_is_denied(self, repo):
        """A model asking for git.commit when it was not granted gets nothing."""
        from factory.capabilities.registry import registry_for
        from factory.compiler import compile_agent
        from factory.models.base import ModelResponse, ToolCall
        from factory.models.fake import ScriptedAdapter
        from factory.runtime.trace import MemoryTraceStore
        from factory.spec.agent_spec import AgentSpec, Limits, ModelRef

        spec = AgentSpec(
            name="reader",
            system_prompt="you read",
            capabilities=["filesystem.read"],
            model=ModelRef(provider="fake", name="none"),
            limits=Limits(max_steps=4),
        )
        model = ScriptedAdapter([
            ModelResponse(tool_calls=[ToolCall("git.commit", {
                "message": "sneaky", "paths": ["a.txt"]})]),
            ModelResponse(text="I cannot commit"),
        ])
        compiled = compile_agent(
            spec, registry_for(repo), workspace=repo, model=model, store=MemoryTraceStore()
        )
        before = git(repo, "log", "--oneline")
        result = await compiled.runtime.run("commit a.txt")

        violations = result.trace.steps[0].violations
        assert violations
        assert violations[0]["capability"] == "git.commit"
        assert result.trace.steps[0].observations[0]["ok"] is False
        assert git(repo, "log", "--oneline") == before, "nothing was committed"


class TestRegistryDiffTracksSideEffects:
    @pytest.mark.asyncio
    async def test_capability_diff_flags_a_state_changing_capability(self, repo):
        """git.commit changes real-world state, so the diff must surface it."""
        from factory.registry import AgentRegistry
        from factory.spec.agent_spec import AgentSpec, Limits, ModelRef

        def mk(version, caps):
            return AgentSpec(
                name="worker",
                version=version,
                system_prompt="worker",
                capabilities=caps,
                model=ModelRef(provider="fake", name="none"),
                limits=Limits(max_steps=4),
            )

        reg = AgentRegistry(":memory:")
        reg.register(mk("0.1.0", ["filesystem.read"]))
        reg.register(mk("0.2.0", ["filesystem.read", "filesystem.write", "git.commit"]))

        d = reg.diff("worker", "0.1.0", "0.2.0")
        assert d["changed"] is True
        assert d["capabilities_added"] == ["filesystem.write", "git.commit"]
        assert d["capabilities_removed"] == []

    @pytest.mark.asyncio
    async def test_removing_a_state_changing_capability_is_visible(self, repo):
        from factory.registry import AgentRegistry
        from factory.spec.agent_spec import AgentSpec, Limits, ModelRef

        def mk(version, caps):
            return AgentSpec(
                name="worker",
                version=version,
                system_prompt="worker",
                capabilities=caps,
                model=ModelRef(provider="fake", name="none"),
                limits=Limits(max_steps=4),
            )

        reg = AgentRegistry(":memory:")
        reg.register(mk("0.1.0", ["git.commit", "filesystem.write"]))
        reg.register(mk("0.2.0", ["filesystem.read"]))

        d = reg.diff("worker", "0.1.0", "0.2.0")
        assert d["capabilities_removed"] == ["filesystem.write", "git.commit"]