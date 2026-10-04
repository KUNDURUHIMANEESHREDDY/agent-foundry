"""Built-in capabilities.

`filesystem.read` is the reference implementation: it accepts a model-supplied
path, so it demonstrates the boundary properly — the root is pinned at grant
time and traversal is refused on the resolved path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from factory.capabilities.registry import Capability, object_schema, safe_join

MAX_READ_BYTES = 256 * 1024


class FilesystemRead(Capability):
    """Read a UTF-8 file from inside a pinned workspace root."""

    name = "filesystem.read"
    description = (
        "Read a UTF-8 text file from the workspace. Paths are relative to the "
        "workspace root; absolute paths and paths escaping it are refused."
    )
    pinned_params = ("root",)

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return object_schema(
            {
                "path": {
                    "type": "string",
                    "description": (
                        "File to read, relative to the workspace root, e.g. "
                        "'src/main.py' or 'README.md'."
                    ),
                }
            },
            required=["path"],
        )

    def __init__(self, root: Path, max_bytes: int = MAX_READ_BYTES) -> None:
        self._root = Path(root).resolve()
        self._max_bytes = max_bytes

    def invoke(self, path: str = "", **_: Any) -> dict[str, Any]:
        if not isinstance(path, str):
            return {"ok": False, "error": "path must be a string"}

        try:
            target = safe_join(self._root, path)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        if not target.exists():
            return {"ok": False, "error": f"No such file: {path}"}
        if target.is_dir():
            return {
                "ok": True,
                "kind": "directory",
                "path": path,
                "entries": sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())[:200],
            }

        raw = target.read_bytes()
        truncated = len(raw) > self._max_bytes
        content = raw[: self._max_bytes].decode("utf-8", errors="replace")

        return {
            "ok": True,
            "kind": "file",
            "path": path,
            "size": len(raw),
            "truncated": truncated,
            "content": content,
        }

    def describe(self) -> str:
        return f"filesystem.read(root={self._root}, max_bytes={self._max_bytes})"
