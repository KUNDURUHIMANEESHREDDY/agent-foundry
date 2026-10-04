"""Callers: who is asking, and what they may do.

The API had one API key and no notion of a caller. That made two problems that
look like one:

  * every caller could request any capability, including `python.execute` — so
    the key was effectively a grant of code execution, with nothing narrower to
    hand out;
  * every caller could read every trace and every registered spec.

A tenant is the answer to both: a named principal with a capability allowlist
and its own data scope. It is not a full RBAC system and does not pretend to be
— there are no roles, no inheritance, no per-resource ACLs. It is the smallest
thing that makes "this caller may not do that" expressible.

Configuration, either of:

    FACTORY_API_KEY=secret                      # one tenant, everything
    FACTORY_TENANT_KEYS="k1=alice,k2=bob"       # keys -> tenant ids

and, per tenant:

    FACTORY_TENANTS="alice=filesystem.read,filesystem.write;bob=*"

Tenants are separated by `;` and capabilities by `,`. Commas cannot do both jobs:
the first version used commas for both and `alice=filesystem.read,filesystem.write`
parsed as a tenant `alice` plus a garbage tenant `filesystem.write`. A format
that mis-parses into *more* access is worse than one that fails loudly, so the
separator is unambiguous.

`*` means "every capability the registry can grant", which is what the
single-tenant deployment always allowed. Absent configuration yields one tenant
named `default` with everything, so a local `factory serve` behaves as before.

WHY A DENIAL IS NOT SILENT

If a spec asks for a capability the tenant may not have, the request is refused
with 403 and says which tenant and which capability. Dropping it and running the
weaker agent would let the caller believe they had something stronger than they
did — the same shape as the `memory.path` problem, where a request was quietly
ignored. A refusal is inconvenient; a silent downgrade is a lie.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

#: Grants every capability the registry can issue.
ALL = "*"

ENV_KEYS = "FACTORY_TENANT_KEYS"
ENV_TENANTS = "FACTORY_TENANTS"

DEFAULT_TENANT = "default"


@dataclass(frozen=True)
class Tenant:
    """One caller identity and its policy."""

    id: str
    #: Allowed capability names. `frozenset({ALL})` means unrestricted.
    capabilities: frozenset[str] = frozenset({ALL})
    description: str = ""

    @property
    def unrestricted(self) -> bool:
        return ALL in self.capabilities

    def allows(self, capability: str) -> bool:
        return self.unrestricted or capability in self.capabilities

    def forbidden(self, requested: list[str]) -> list[str]:
        """Requested capabilities this tenant may not have."""
        return sorted(c for c in requested if not self.allows(c))

    def describe(self) -> dict[str, object]:
        return {
            "id": self.id,
            "capabilities": (
                "all" if self.unrestricted else sorted(self.capabilities)
            ),
            "description": self.description,
        }


class UnknownTenant(KeyError):
    """A key that resolves to no configured tenant."""


def _parse_ids(raw: str) -> list[str]:
    """`tenant=cap,cap;tenant2=cap` -> tenant ids.

    Semicolon between tenants, comma between capabilities. Commas cannot do both:
    an earlier comma-only version read `alice=a,b` as tenant `alice` plus a
    nonsense tenant `b`, which is a policy that widens access by accident.
    """
    out: list[str] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"{ENV_TENANTS} entry {chunk!r} must look like 'tenant=cap,cap'"
            )
        name, _, caps = chunk.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"{ENV_TENANTS} entry {chunk!r} has an empty tenant id")
        granted = frozenset(
            c.strip() for c in caps.split(",") if c.strip()
        ) or frozenset({ALL})
        out.append(name)
        _TENANT_CAPABILITIES[name] = granted
    return out


#: Populated by `_parse_ids`, read by `TenantRegistry.from_env`. Module-level
#: only because the two steps must agree; it is written once at construction and
#: never mutated afterwards.
_TENANT_CAPABILITIES: dict[str, frozenset[str]] = {}


@dataclass
class TenantRegistry:
    """Tenant id -> policy."""

    tenants: dict[str, Tenant] = field(default_factory=dict)

    def get(self, tenant_id: str | None) -> Tenant | None:
        if not tenant_id:
            return None
        return self.tenants.get(tenant_id)

    def require(self, tenant_id: str | None) -> Tenant:
        tenant = self.get(tenant_id)
        if tenant is None:
            raise UnknownTenant(f"unknown tenant {tenant_id!r}")
        return tenant

    def ids(self) -> list[str]:
        return sorted(self.tenants)

    def describe(self) -> list[dict[str, object]]:
        return [self.tenants[i].describe() for i in self.ids()]

    @classmethod
    def single(cls, tenant_id: str = DEFAULT_TENANT, description: str = "") -> "TenantRegistry":
        """One tenant with everything. The default, and what `FACTORY_API_KEY` means."""
        return cls(
            {tenant_id: Tenant(id=tenant_id, capabilities=frozenset({ALL}), description=description)}
        )

    @classmethod
    def from_env(cls) -> "TenantRegistry":
        """Build from `FACTORY_TENANTS`, else one unrestricted `default`.

        A malformed configuration raises rather than falling back: silently
        granting everything because a policy failed to parse is precisely the
        failure mode this module exists to prevent.
        """
        _TENANT_CAPABILITIES.clear()
        raw = os.environ.get(ENV_TENANTS, "").strip()
        if not raw:
            return cls.single()

        ids = _parse_ids(raw)
        if not ids:
            raise ValueError(f"{ENV_TENANTS} was set but defined no tenants")

        return cls(
            {
                i: Tenant(id=i, capabilities=_TENANT_CAPABILITIES.get(i, frozenset({ALL})))
                for i in ids
            }
        )


# ── keys ────────────────────────────────────────────────────────────────


def key_map_from_env(tenants: TenantRegistry) -> dict[str, str]:
    """API key -> tenant id.

    `FACTORY_TENANT_KEYS="k1=alice,k2=bob"` maps explicitly.
    `FACTORY_API_KEY=k` alone maps that key to the sole tenant, so a
    single-tenant deployment needs no new configuration.
    """
    explicit = os.environ.get(ENV_KEYS, "").strip()
    pairs: dict[str, str] = {}

    if explicit:
        for chunk in explicit.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(
                    f"{ENV_KEYS} entry {chunk!r} must look like 'key=tenant'"
                )
            key, _, tenant_id = chunk.partition("=")
            key, tenant_id = key.strip(), tenant_id.strip()
            if not key or not tenant_id:
                raise ValueError(f"{ENV_KEYS} entry {chunk!r} is incomplete")
            if tenant_id not in tenants.tenants:
                raise ValueError(
                    f"{ENV_KEYS} names tenant {tenant_id!r}, which {ENV_TENANTS} "
                    f"does not define. Known: {', '.join(tenants.ids())}"
                )
            pairs[key] = tenant_id
        return pairs

    single = os.environ.get("FACTORY_API_KEY", "").strip()
    if single and len(tenants.tenants) == 1:
        only = tenants.ids()[0]
        return {single: only}

    if single and len(tenants.tenants) > 1:
        raise ValueError(
            f"FACTORY_API_KEY is set but {ENV_TENANTS} defines "
            f"{len(tenants.tenants)} tenants. Use {ENV_KEYS} to say which key "
            f"belongs to which tenant, otherwise every key would grant the first."
        )

    return pairs