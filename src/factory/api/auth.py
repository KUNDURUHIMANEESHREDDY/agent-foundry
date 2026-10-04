"""Authentication for the factory API.

Every endpoint except `/health` can compile and execute an agent, persist a spec,
or read run history. None of that should be reachable by an anonymous network
client, so authentication is on by default and there is no code path that
silently disables it.

Three modes, in order of precedence:

  1. `FACTORY_API_INSECURE=1` — authentication off. Opt-in, logged loudly, and
     only ever appropriate for a loopback bind.
  2. `FACTORY_API_KEY` set — that key is required.
  3. Neither set — a key is generated at startup and printed once to stderr.

Mode 3 is the important one: a developer who forgets to configure a key gets a
server that works, with a key they can read from the console — not a server
that is wide open. The failure mode of missing configuration should be "you must
authenticate", never "everyone may".

Keys are compared with `hmac.compare_digest`, so a caller cannot discover the
key by measuring response time.
"""

from __future__ import annotations

import hmac
import os
import secrets
import sys

from fastapi import HTTPException, Request

ENV_KEY = "FACTORY_API_KEY"
ENV_INSECURE = "FACTORY_API_INSECURE"

#: Endpoints reachable without a key. Deliberately minimal: liveness only.
#: Nothing that can execute code, read traces, or mutate the registry.
PUBLIC_PATHS = frozenset({"/health"})

_GENERATED_NOTICE = (
    "Agent Factory API: generated a temporary API key because neither "
    f"{ENV_KEY} nor {ENV_INSECURE} was set.\n"
    "  key: {key}\n"
    "  send it as 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.\n"
    f"  set {ENV_KEY} to pin a key, or {ENV_INSECURE} to disable auth "
    "(loopback only)."
)


class AuthSettings:
    """Resolved once at startup so request handling cannot change policy.

    Holds the map from API key to tenant id. With no map, the single `key`
    argument still works and behaves exactly as before — one key, one tenant.
    """

    def __init__(
        self,
        key: str | None = None,
        *,
        insecure: bool = False,
        key_tenants: dict[str, str] | None = None,
        insecure_tenant: str = "default",
    ) -> None:
        self.insecure = bool(insecure)
        self.generated = False
        self._insecure_tenant = insecure_tenant

        # An empty map means "nothing configured", not "an explicit empty
        # allowlist". Treating it as explicit suppressed the generated-key path
        # and left the server accepting no credential at all — which reads as a
        # lockout rather than a misconfiguration.
        if key_tenants:
            self._key_tenants = dict(key_tenants)
            self._key = key or ""
            return

        if self.insecure:
            self._key = ""
            self._key_tenants = {}
        elif key:
            self._key = key
            self._key_tenants = {key: insecure_tenant}
        else:
            self._key = secrets.token_urlsafe(32)
            self.generated = True
            self._key_tenants = {self._key: insecure_tenant}

    @classmethod
    def from_env(cls, key_tenants: dict[str, str] | None = None) -> "AuthSettings":
        insecure = os.environ.get(ENV_INSECURE, "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        key = os.environ.get(ENV_KEY, "").strip()
        return cls(key or None, insecure=insecure, key_tenants=key_tenants)

    @property
    def enabled(self) -> bool:
        return not self.insecure

    @property
    def key(self) -> str:
        return self._key

    def announce(self) -> None:
        """Say something on startup. Silence is how an open server goes unnoticed."""
        if self.insecure:
            print(
                f"WARNING: Agent Factory API authentication is DISABLED "
                f"({ENV_INSECURE} is set). Anyone who can reach this port can "
                f"execute agents and read traces. Do not bind it to a public "
                f"interface.",
                file=sys.stderr,
            )
        elif self.generated:
            print(_GENERATED_NOTICE.format(key=self._key), file=sys.stderr)
        else:
            count = len(self._key_tenants) or 1
            print(
                f"Agent Factory API: authentication enabled ({ENV_KEY}), "
                f"{count} tenant(s).",
                file=sys.stderr,
            )

    def check(self, presented: str | None) -> bool:
        if self.insecure:
            return True
        return self.tenant_for(presented) is not None

    def tenant_for(self, presented: str | None) -> str | None:
        """Which tenant a presented key belongs to, or None.

        Every configured key is compared, so the number of comparisons does not
        leak which key was closer. Returns the tenant id rather than a bool,
        because "is this key valid" and "who is this" are the same question.
        """
        if self.insecure:
            return self._insecure_tenant
        if not presented or not self._key_tenants:
            return None

        matched: str | None = None
        for key, tenant_id in self._key_tenants.items():
            if hmac.compare_digest(presented, key):
                matched = tenant_id
        return matched


def presented_key(request: Request) -> str | None:
    """Extract a key from either accepted header form."""
    header = request.headers.get("authorization", "")
    if header[:7].lower() == "bearer ":
        return header[7:].strip()

    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key.strip()

    return None