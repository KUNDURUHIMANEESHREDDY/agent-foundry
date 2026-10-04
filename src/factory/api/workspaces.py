"""Named workspaces for the API.

The capability root is the single most security-relevant value in this system.
`safe_join` confines paths to it, `filesystem.write` writes inside it,
`python.execute` runs there, and `git.commit` commits there. The model can never
change it: the factory pins it at grant time and the gate strips any attempt.

That guarantee is worthless if an HTTP client can supply the root directly.
`RunRequest.workspace` used to be a free-text path, so the caller — not the
model, not the operator — chose the directory every capability was bound to:

    POST /compile {"workspace": "/"}
      -> filesystem.read(root=/)   -> reads /etc/hosts

This module removes paths from the request entirely. A client sends an opaque
identifier; the server decides what that identifier means. The identifier is
never joined to a base directory and never touches the filesystem as a path, so
traversal in the identifier is not a concept that can exist.

Server-side configuration, for example:

    FACTORY_WORKSPACES="default=.,docs=./docs,logs=./var/logs"

An unknown or malformed identifier is refused. There is deliberately no
fallback to the process working directory: a silent fallback is how a
misconfigured deployment ends up serving `/`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: Identifiers are opaque keys, never paths. The pattern admits no separators,
#: no dots, and no traversal, so an id cannot address anything on disk even if
#: a future change accidentally joins it onto a base path.
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

DEFAULT_ID = "default"
ENV_VAR = "FACTORY_WORKSPACES"


class UnknownWorkspace(KeyError):
    """A client asked for a workspace the server has not approved."""


class WorkspaceRegistry:
    """Server-side mapping of opaque id -> approved directory."""

    def __init__(self, default_id: str = DEFAULT_ID) -> None:
        self._dirs: dict[str, Path] = {}
        self._default_id = default_id

    def add(self, workspace_id: str, path: str | Path) -> Path:
        """Approve a directory under an identifier. Returns the resolved path.

        The path is resolved once, here. Nothing downstream re-resolves it, so a
        symlink swapped in afterwards cannot move the capability root.
        """
        if not ID_PATTERN.match(workspace_id):
            raise ValueError(
                f"workspace id {workspace_id!r} must match {ID_PATTERN.pattern} "
                f"(lowercase letters, digits, '-', '_'; no path separators)"
            )

        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"workspace {workspace_id!r} -> {resolved} is not a directory")

        self._dirs[workspace_id] = resolved
        return resolved

    def resolve(self, workspace_id: str | None) -> Path:
        """The approved directory for an id.

        `None` or empty selects the default workspace — it does NOT select the
        process working directory, which would reintroduce the original bug with
        an extra step.
        """
        wanted = (workspace_id or "").strip() or self._default_id
        try:
            return self._dirs[wanted]
        except KeyError:
            raise UnknownWorkspace(
                f"unknown workspace {wanted!r}. Approved: {', '.join(self.ids()) or '(none)'}"
            ) from None

    def ids(self) -> list[str]:
        return sorted(self._dirs)

    @property
    def default_id(self) -> str:
        return self._default_id

    def describe(self) -> dict[str, str]:
        return {i: str(self._dirs[i]) for i in self.ids()}

    @classmethod
    def from_env(cls, base: str | Path | None = None) -> "WorkspaceRegistry":
        """Build from `FACTORY_WORKSPACES`, else a single `default` workspace.

        Format: `id=path,id=path`. Relative paths resolve against `base`.
        An unparseable or empty configuration yields no workspaces rather than a
        permissive default, so a typo cannot silently widen access.
        """
        base_dir = Path(base).resolve() if base else Path.cwd()
        registry = cls()

        raw = os.environ.get(ENV_VAR, "").strip()
        if not raw:
            registry.add(DEFAULT_ID, base_dir)
            return registry

        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(
                    f"{ENV_VAR} entry {chunk!r} must look like 'id=path'"
                )
            workspace_id, _, raw_path = chunk.partition("=")
            candidate = Path(raw_path.strip()).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            registry.add(workspace_id.strip(), candidate)

        if not registry.ids():
            raise ValueError(f"{ENV_VAR} was set but defined no usable workspaces")

        return registry