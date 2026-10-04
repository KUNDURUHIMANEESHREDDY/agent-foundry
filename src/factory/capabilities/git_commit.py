"""git.commit — create commits inside a pinned repository.

Committing is a real-world side effect, so this capability is conservative by
construction:

  - the repo root is pinned at grant time; the model cannot point elsewhere
  - only paths inside the repo are stageable (`safe_join`)
  - `git push` is not implemented and not reachable — this commits locally only
  - the message is validated (non-empty, bounded) so history stays useful
  - nothing is auto-committed: the model must name the paths it wants staged

Deliberately absent: push, force, reset, amend, tag, branch deletion. Each of
those rewrites or publishes history, and none of them belong to a capability a
model can call. If you need them, add them explicitly rather than widening this.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from factory.capabilities.registry import Capability, object_schema, safe_join

MAX_MESSAGE_CHARS = 4096
MAX_PATHS = 50

# Subcommands this capability will run. Anything else is refused, so a bug or a
# surprising argument cannot turn a commit into a push or a history rewrite.
ALLOWED_SUBCOMMANDS = {"status", "add", "commit", "diff", "rev-parse"}


class GitNotARepository(RuntimeError):
    pass


class GitCommit(Capability):
    """Stage paths and create a local commit in a pinned repository."""

    name = "git.commit"
    description = (
        "Stage the named files and create a LOCAL commit in the workspace "
        "repository. Nothing is published: push, reset, amend and tag are not "
        "reachable. Paths must be relative to the repository root."
    )
    pinned_params = ("root", "allow_empty", "author_name")
    required_params = ("root",)

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return object_schema(
            {
                "message": {
                    "type": "string",
                    "description": (
                        "Commit message. Must be non-empty and under "
                        f"{MAX_MESSAGE_CHARS} characters."
                    ),
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Repository-relative paths to stage, e.g. "
                        "['src/main.py']. Nothing is committed implicitly."
                    ),
                },
            },
            required=["message"],
        )

    def __init__(
        self,
        root: Path,
        allow_empty: bool = False,
        author_name: str = "agent-factory",
        timeout: float = 30.0,
    ) -> None:
        self._root = Path(root).resolve()
        self._allow_empty = bool(allow_empty)
        self._author_name = author_name
        self._timeout = float(timeout)

        if not (self._root / ".git").exists():
            raise GitNotARepository(f"{self._root} is not a git repository")

    # ── plumbing ────────────────────────────────────────────────

    def _git_env(self) -> dict[str, str]:
        """A predictable identity, so commits do not depend on host git config."""
        env = dict(os.environ)
        env["GIT_AUTHOR_NAME"] = self._author_name
        env["GIT_AUTHOR_EMAIL"] = f"{self._author_name}@localhost"
        env["GIT_COMMITTER_NAME"] = self._author_name
        env["GIT_COMMITTER_EMAIL"] = f"{self._author_name}@localhost"
        return env

    def _run_env(self, *args: str) -> subprocess.CompletedProcess:
        if args and args[0] not in ALLOWED_SUBCOMMANDS:
            raise PermissionError(
                f"git {args[0]} is not permitted by this capability. "
                f"Allowed: {', '.join(sorted(ALLOWED_SUBCOMMANDS))}"
            )
        return subprocess.run(
            ["git", *args],
            cwd=str(self._root),
            capture_output=True,
            text=True,
            timeout=self._timeout,
            check=False,
            shell=False,
            env=self._git_env(),
        )

    # ── capability ──────────────────────────────────────────────

    def invoke(
        self,
        message: str = "",
        paths: list[str] | None = None,
        allow_empty: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        if not isinstance(message, str) or not message.strip():
            return {"ok": False, "error": "commit message must be a non-empty string"}
        if len(message) > MAX_MESSAGE_CHARS:
            return {
                "ok": False,
                "error": f"commit message exceeds {MAX_MESSAGE_CHARS} characters",
            }

        raw_paths = paths or []
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        if not isinstance(raw_paths, list):
            return {"ok": False, "error": "paths must be a list of strings"}
        if len(raw_paths) > MAX_PATHS:
            return {"ok": False, "error": f"at most {MAX_PATHS} paths per commit"}

        # Resolve and refuse anything outside the repo before touching git.
        resolved: list[str] = []
        for candidate in raw_paths:
            if not isinstance(candidate, str) or not candidate.strip():
                return {"ok": False, "error": "each path must be a non-empty string"}
            try:
                target = safe_join(self._root, candidate)
            except Exception as exc:
                return {"ok": False, "error": str(exc), "capability": self.name}
            if not target.exists():
                return {
                    "ok": False,
                    "error": f"no such path: {candidate}",
                    "capability": self.name,
                }
            rel = target.relative_to(self._root).as_posix()
            resolved.append(rel)

        try:
            if resolved:
                add = self._run_env("add", "--", *resolved)
                if add.returncode != 0:
                    return {
                        "ok": False,
                        "error": f"git add failed: {add.stderr.strip()[:300]}",
                        "capability": self.name,
                    }

            status = self._run_env("status", "--porcelain")
            if status.returncode != 0:
                return {
                    "ok": False,
                    "error": f"git status failed: {status.stderr.strip()[:300]}",
                    "capability": self.name,
                }

            staged = [
                    line
                    for line in status.stdout.splitlines()
                    if line[:1] in {"M", "A", "D", "R", "C"}
                ]

            # `allow_empty` is a PINNED parameter, so the gate strips it from
            # model-supplied arguments before invoke. In practice this is
            # therefore `self._allow_empty`, decided by the factory at grant
            # time — an empty commit is never something a model can ask for.
            #
            # This used to read `if allow_empty or self._allow_empty:` and then
            # refuse *because* empty commits were disabled, which is backwards:
            # `allow_empty=True` returned "empty commits are disabled" and
            # `would_need: allow_empty=true`. The flag could never do anything.
            permit_empty = bool(allow_empty or self._allow_empty)

            if not staged and not permit_empty:
                return {
                    "ok": False,
                    "error": "nothing staged to commit",
                    "capability": self.name,
                    "would_need": (
                        "allow_empty=true on the grant; it is a pinned parameter "
                        "and cannot be supplied by the model"
                    ),
                }

            # Only ask git for an empty commit when there is nothing staged.
            # Passing --allow-empty unconditionally would mask a genuine
            # "git add matched nothing" as a successful empty commit.
            commit_args = ["commit"]
            if not staged:
                commit_args.append("--allow-empty")
            commit_args += ["-m", message]

            commit = self._run_env(*commit_args)
            if commit.returncode != 0:
                return {
                    "ok": False,
                    "error": f"git commit failed: {(commit.stderr or commit.stdout).strip()[:300]}",
                    "capability": self.name,
                }

            sha = self._run_env("rev-parse", "HEAD")
            return {
                "ok": True,
                "capability": self.name,
                "sha": sha.stdout.strip() if sha.returncode == 0 else None,
                "short_sha": sha.stdout.strip()[:12] if sha.returncode == 0 else None,
                "message": message,
                "paths": resolved,
                "files_changed": len(staged),
                "status": staged,
            }

        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"git exceeded {self._timeout}s timeout",
                "capability": self.name,
            }

    def describe(self) -> str:
        return (
            f"git.commit(root={self._root}, "
            f"allow_empty={self._allow_empty}, author={self._author_name})"
        )


