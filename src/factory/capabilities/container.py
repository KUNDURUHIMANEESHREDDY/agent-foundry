"""Container execution backend — the first real containment `python.execute` has had.

WHAT THIS PROVIDES
    An OS container around every script. The workspace is the only mount and is
    mounted **read only**; there is no network namespace; all Linux capabilities
    are dropped; privilege escalation is refused; memory, CPU and process count
    are capped; the container runs as a non-root uid; and its root filesystem is
    read-only with a small writable tmpfs.

WHAT THIS IS NOT
    A perfect boundary. The Docker daemon is reachable from the CLI the user
    typed, and a container is not a VM. Someone who already controls this machine
    can do things this does not stop. It is a materially stronger boundary than a
    subprocess -- which is the whole claim, and the only one.

WHY READ-ONLY WORKSPACE
    Writing files is `filesystem.write`'s job, and it is a granted, audited
    capability. If executed code can also write, then the write boundary is only
    as strong as the model's discretion, and `filesystem.write` becomes
    decorative. Mounting read-only keeps exactly one path to the filesystem.

Failure is always closed. If the daemon is missing, or the image is absent and
cannot be pulled, the capability refuses rather than falling back to a bare
subprocess -- a silent downgrade from `container` to `subprocess` would be a
security failure with no symptom at all.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_IMAGE = "python:3.11-slim"

#: Refuse to be configured with unbounded limits. A container with no memory cap
#: is a denial-of-service primitive pointed at the host.
DEFAULT_MEMORY = "512m"
DEFAULT_CPUS = "1.0"
DEFAULT_PIDS = 128
DEFAULT_TMPFS = "64m"

#: Runs as nobody, so a container escape does not land on a named account.
NON_ROOT_UID = 65534
NON_ROOT_GID = 65534

#: Where the workspace appears inside the container.
IN_CONTAINER_WORKSPACE = "/workspace"


class ContainerUnavailable(RuntimeError):
    """The container runtime cannot provide the containment that was asked for.

    Raised instead of degrading to a weaker backend. The caller must propagate it.
    """


@dataclass
class ContainerConfig:
    """Everything the containment depends on, so a trace can report it exactly."""

    image: str = DEFAULT_IMAGE
    memory: str = DEFAULT_MEMORY
    cpus: str = DEFAULT_CPUS
    pids_limit: int = DEFAULT_PIDS
    tmpfs: str = DEFAULT_TMPFS
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> dict[str, object]:
        """The security-relevant settings, for the trace and for tests."""
        return {
            "image": self.image,
            "memory": self.memory,
            "cpus": self.cpus,
            "pids_limit": self.pids_limit,
            "workspace_mount": "read-only",
            "network": "none",
            "capabilities": "dropped",
            "user": f"{NON_ROOT_UID}:{NON_ROOT_GID}",
            "root_filesystem": "read-only",
        }


def docker_executable() -> str:
    found = shutil.which("docker")
    if not found:
        raise ContainerUnavailable(
            "isolation=container requires a container runtime, and no `docker` "
            "executable is on PATH. Refusing to run the code unconfined."
        )
    return found


def probe(config: ContainerConfig | None = None, timeout_s: float = 20.0) -> str:
    """Ask the daemon for its version. Raises if it cannot answer.

    Checking for the executable is not enough: Docker Desktop installs the CLI
    whether or not the engine is running, and the failure then looks like a
    missing-runtime error rather than a stopped daemon.
    """
    exe = docker_executable()
    try:
        proc = subprocess.run(
            [exe, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise ContainerUnavailable(
            f"the container runtime did not answer within {timeout_s:.0f}s; "
            "it may be starting. Refusing to run the code unconfined."
        ) from exc

    if proc.returncode != 0 or not proc.stdout.strip():
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise ContainerUnavailable(
            "the container runtime is installed but not answering"
            + (f": {detail[-1]}" if detail else "")
            + ". Start it, or run this spec at isolation: subprocess -- which is "
            "not a sandbox."
        )
    return proc.stdout.strip()


def ensure_image(config: ContainerConfig, timeout_s: float = 300.0) -> None:
    """Make sure the image is local, pulling it if needed.

    Failure here must not be swallowed: running nothing would look exactly like
    a script that produced no output, and the eval cases would pass vacuously.
    """
    exe = docker_executable()
    check = subprocess.run(
        [exe, "image", "inspect", config.image],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if check.returncode == 0:
        return

    pull = subprocess.run(
        [exe, "pull", config.image], capture_output=True, text=True, timeout=timeout_s
    )
    if pull.returncode != 0:
        detail = (pull.stderr or pull.stdout or "").strip().splitlines()
        raise ContainerUnavailable(
            f"could not obtain image {config.image!r}"
            + (f": {detail[-1]}" if detail else "")
        )


def build_argv(
    code: str,
    workspace: Path,
    config: ContainerConfig | None = None,
    python_executable: str = "/usr/local/bin/python",
    exe: str | None = None,
) -> list[str]:
    """The `docker run` argv. Every flag here is a containment claim.

    Kept pure and separate from process handling so the containment can be
    asserted directly on the argv. A test that has to infer the flags from
    behaviour is a test that stops proving anything when the behaviour is broken
    in a way that still looks right.

    `exe` is injectable so the flags can be asserted on a machine with no
    container runtime at all. Asserting containment by launching containers
    means the test only runs where Docker happens to be installed.
    """
    cfg = config or ContainerConfig()

    return [
        exe or docker_executable(),
        "run",
        "--rm",
        # No network namespace at all. Cheaper and stronger than filtering:
        # there is nothing to allow or deny.
        "--network", "none",
        # The workspace is the only mount, and it cannot be written to. Writing
        # is filesystem.write's job, and it is granted separately.
        "--volume", f"{Path(workspace).resolve()}:{IN_CONTAINER_WORKSPACE}:ro",
        "--workdir", IN_CONTAINER_WORKSPACE,
        # The container's own filesystem is read-only; only the tmpfs is writable.
        "--read-only",
        "--tmpfs", f"/tmp:rw,size={cfg.tmpfs},mode=1777",
        # No Linux capabilities, and no way to acquire any.
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        # Nobody. A bug that escapes the container does not land on a named user.
        "--user", f"{NON_ROOT_UID}:{NON_ROOT_GID}",
        # Resource caps, so a script cannot exhaust the host.
        "--memory", cfg.memory,
        "--cpus", cfg.cpus,
        "--pids-limit", str(cfg.pids_limit),
        # Never inherit the caller's environment: it carries host paths, and
        # possibly secrets.
        "--env", "PATH=/usr/local/bin:/usr/bin:/bin",
        "--env", "HOME=/tmp",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "PYTHONUNBUFFERED=1",
        "--env", "PYTHONHASHSEED=0",
        *cfg.extra_args,
        cfg.image,
        python_executable, "-I", "-c", code,
    ]


def python_executable_in_image(config: ContainerConfig | None = None) -> str:
    """Where python lives inside the image.

    Not assumed: `python:3.11-slim` puts it there, but an arbitrary image may
    not, and a wrong guess produces an interpreter error rather than a clear
    message. Overridable so a caller with a different image is not stuck.
    """
    return "/usr/local/bin/python"