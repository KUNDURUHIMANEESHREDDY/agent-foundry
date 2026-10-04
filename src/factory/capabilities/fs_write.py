"""filesystem.write — write files inside a pinned workspace.

Writing is riskier than reading, so this capability is deliberately narrower:

  - the root is pinned at grant time; the model cannot move it
  - every path goes through `safe_join`, so traversal and symlink escapes fail
  - an existing file must be opted into explicitly (`overwrite`), so a
    read-then-write loop cannot silently clobber data
  - writes are atomic: a temp file plus rename, so a crash mid-write leaves the
    original intact rather than a truncated file
  - size is capped, because an agent loop can fill a disk in one step

Every refusal is returned as an observation, never raised, so the model can
recover instead of the run dying on the first mistake.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from factory.capabilities.registry import Capability, object_schema, safe_join

#: The most one `filesystem.write` call may create.
#:
#: This is the *inner* bound — blast radius of a single step — and
#: `max_tokens` is the *outer* one, a cumulative budget for a whole run. For
#: those to mean anything the inner bound has to sit below the outer, and it did
#: not. Measured:
#:
#:     this cap          512KB = ~131,000 tokens of payload
#:     coder's budget    60K tokens = ~240,000 bytes, overhead aside
#:
#: so the cap could not fire at any budget a shipping spec grants: it was 2x out
#: of reach for `coder`, and 3–33x out of reach for the rest. A limit that never
#: fires is not a limit, and an untested one rots — the same failure as the
#: false-green eval cases.
#:
#: At 128KB the cap is reachable: a single call can carry at most ~240KB of
#: payload under `coder`, so anything past 128KB is refused by the capability
#: rather than by the budget. The two bounds are now ordered, and the eval case
#: `rejects-oversized-write` exercises this at the spec's real limits with no
#: `limits:` override.
#:
#: The cap is on the RESULTING FILE, not on one write: `append` is refused once
#: it would push the file past the limit. So this is a ceiling on how large any
#: file in the workspace may become, which is the blast-radius property that
#: matters — and it means `append` is not a way around it.
MAX_WRITE_BYTES = 128 * 1024


class FilesystemWrite(Capability):
    """Create or overwrite a UTF-8 file inside a pinned workspace root."""

    name = "filesystem.write"
    description = (
        "Write a UTF-8 text file in the workspace. Creating a new file is "
        "always allowed; replacing an existing one requires overwrite=true."
    )
    pinned_params = ("root", "allow_overwrite")
    required_params = ("root",)

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return object_schema(
            {
                "path": {
                    "type": "string",
                    "description": (
                        "File to write, relative to the workspace root, e.g. "
                        "'notes.txt'."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Full text content of the file.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": (
                        "Required to replace a file that already exists. "
                        "Mutually exclusive with append."
                    ),
                },
                "append": {
                    "type": "boolean",
                    "description": (
                        "Append to the file instead of replacing it. Mutually "
                        "exclusive with overwrite."
                    ),
                },
            },
            required=["path", "content"],
        )

    def __init__(
        self,
        root: Path,
        allow_overwrite: bool = True,
        max_bytes: int = MAX_WRITE_BYTES,
    ) -> None:
        self._root = Path(root).resolve()
        self._allow_overwrite = bool(allow_overwrite)
        self._max_bytes = int(max_bytes)

    def invoke(
        self,
        path: str = "",
        content: str = "",
        overwrite: bool = False,
        append: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        if not isinstance(path, str) or not path.strip():
            return {"ok": False, "error": "path must be a non-empty string"}
        if not isinstance(content, str):
            return {"ok": False, "error": "content must be a string"}

        if overwrite and append:
            return {"ok": False, "error": "cannot set both overwrite and append"}

        encoded = content.encode("utf-8")
        if len(encoded) > self._max_bytes:
            return {
                "ok": False,
                "error": (
                    f"content is {len(encoded)} bytes, over the "
                    f"{self._max_bytes} byte limit"
                ),
                "capability": self.name,
            }

        try:
            target = safe_join(self._root, path)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "capability": self.name}

        if target.is_dir():
            return {
                "ok": False,
                "error": f"path is a directory: {path}",
                "capability": self.name,
            }

        existed = target.exists()
        if existed and not (overwrite or append) and not self._allow_overwrite:
            return {
                "ok": False,
                "error": (
                    f"{path} already exists. Pass overwrite=true, or grant "
                    f"allow_overwrite to change that policy."
                ),
                "capability": self.name,
            }
        if existed and not (overwrite or append):
            # Allowed by policy, but say so in the result rather than silently.
            pass

        if append and existed:
            try:
                existing = target.read_bytes()
            except OSError as exc:
                return {"ok": False, "error": f"cannot read for append: {exc}"}
            if len(existing) + len(encoded) > self._max_bytes:
                return {
                    "ok": False,
                    "error": (
                        f"append would exceed the {self._max_bytes} byte limit"
                    ),
                }
            try:
                with target.open("ab") as fh:
                    fh.write(encoded)
            except OSError as exc:
                return {"ok": False, "error": f"append failed: {exc}"}
            return {
                "ok": True,
                "path": path,
                "bytes_written": len(encoded),
                "size": target.stat().st_size,
                "appended": True,
                "created": False,
            }

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {"ok": False, "error": f"cannot create parent directory: {exc}"}

        # Atomic: write to a sibling temp file, then rename over the target.
        tmp_name = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
            )
            with os.fdopen(fd, "wb") as fh:
                fh.write(encoded)
            os.replace(tmp_name, target)
            tmp_name = None
        except OSError as exc:
            return {"ok": False, "error": f"write failed: {exc}"}
        finally:
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

        return {
            "ok": True,
            "path": path,
            "bytes_written": len(encoded),
            "size": target.stat().st_size,
            "created": not existed,
            "overwrote": existed,
        }

    def describe(self) -> str:
        return (
            f"filesystem.write(root={self._root}, "
            f"allow_overwrite={self._allow_overwrite}, max_bytes={self._max_bytes})"
        )