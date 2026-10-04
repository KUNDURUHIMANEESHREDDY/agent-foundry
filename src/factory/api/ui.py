"""Serving a frontend from the same origin as the API.

WHY SAME-ORIGIN RATHER THAN CORS
    This API can execute code on the machine it runs on. A browser enforces
    same-origin policy, so a page from anywhere else cannot read a response from
    `127.0.0.1` -- which is what stops a hostile tab from driving `/run`.

    Adding permissive CORS would remove that protection: any page the user has
    open could then POST to `/run` and get code execution with no user intent
    involved. For a local app that can execute code, that is a much worse problem
    than the cross-origin one it solves.

    So the frontend is served *from* the API. One origin, no CORS headers, no
    wildcard, nothing to misconfigure.

WHY A HOST GUARD
    Serving the UI normally means authentication off, because pasting a key into
    a browser is a bad trade for a local app. With the key gone, the remaining
    defence against a hostile page is DNS rebinding: an attacker-controlled
    hostname that resolves to 127.0.0.1. The browser treats it as same-origin --
    no CORS involved -- and the Host header carries the attacker's domain, not
    localhost.

    Rejecting any Host that is not loopback closes that. It costs one header
    check and it is the difference between "no key" and "no key, and only from a
    page you served".
"""

from __future__ import annotations

from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.staticfiles import StaticFiles

#: Binds that cannot be reached from another machine. Anything else makes an
#: unauthenticated API reachable from the network.
LOOPBACK_BINDS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

#: Host header values accepted when the UI is served without authentication.
ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class UnsafeBindError(RuntimeError):
    """A UI was requested on a bind other machines can reach."""


class MissingUiError(RuntimeError):
    """The UI directory does not exist, so the app would serve a bare API."""


def is_loopback_bind(host: str) -> bool:
    return host.strip().lower() in LOOPBACK_BINDS


def host_name(value: str) -> str:
    """The hostname from a Host header, without the port.

    Handles `[::1]:8000` and `example.com:8000` as well as a bare name, because
    the three forms all appear and a naive `split(":")` mangles the first.
    """
    v = value.strip()
    if v.startswith("["):
        end = v.find("]")
        return v[1:end] if end != -1 else v
    if v.count(":") == 1:
        return v.split(":", 1)[0]
    return v


def host_is_allowed(value: str | None) -> bool:
    """Whether a Host header names the loopback interface.

    A missing Host is refused. Under HTTP/1.1 it should not happen, and a
    request that cannot say where it came from is not one to trust when the
    authentication that would otherwise cover for it is switched off.
    """
    if not value:
        return False
    return host_name(value).lower() in ALLOWED_HOSTS


class LoopbackHostGuard(BaseHTTPMiddleware):
    """Refuse requests whose Host header is not loopback.

    Applied only when authentication is off. With a key required, the key is the
    defence and this would be redundant; with the key gone, this is what stands
    between a hostile page and code execution.
    """

    async def dispatch(self, request, call_next):
        if not host_is_allowed(request.headers.get("host")):
            return _plain(
                403,
                "This server only answers requests addressed to localhost. "
                "A request naming another host was refused.",
            )
        return await call_next(request)


def _plain(status: int, message: str):
    from starlette.responses import PlainTextResponse

    return PlainTextResponse(message, status_code=status)


def resolve_ui_dir(directory: str | Path) -> Path:
    """Resolve and sanity-check the UI directory before anything is mounted."""
    path = Path(directory).expanduser().resolve()

    if not path.is_dir():
        raise MissingUiError(
            f"UI directory {path} does not exist or is not a directory."
        )

    if not (path / "index.html").is_file():
        raise MissingUiError(
            f"{path} has no index.html, so there is nothing to serve at /."
        )

    return path


def mount_ui(
    app,
    directory: str | Path,
    *,
    bind_host: str = "127.0.0.1",
    guard_hosts: bool = True,
) -> Path:
    """Serve a built frontend from the API's own origin.

    Mounted last so the API routes registered earlier keep matching first: a
    catch-all at "/" must not shadow `/run`.

    `bind_host` is checked rather than trusted. A UI with authentication off,
    served on `0.0.0.0`, would hand code execution to the network -- and the
    operator who typed that bind should be told here rather than discovering it
    from someone else's agent run.
    """
    if not is_loopback_bind(bind_host):
        raise UnsafeBindError(
            f"--ui requires a loopback bind, not {bind_host!r}. Serving a UI "
            f"means authentication can be off, and an unauthenticated API that "
            f"executes code must not be reachable from the network. Bind "
            f"127.0.0.1."
        )

    path = resolve_ui_dir(directory)

    if guard_hosts:
        app.add_middleware(LoopbackHostGuard)

    # html=True serves index.html for directory requests, which is what a
    # single-page app needs at "/".
    app.mount("/", StaticFiles(directory=str(path), html=True), name="ui")

    return path