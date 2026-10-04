"""Isolation levels: what the runtime actually provides, not what it promises.

`python.execute` originally carried a `sandbox_level` string that was recorded in
the result and never checked -- a flag that did nothing, the same defect class as
`CompiledAgent.denied` and `git.commit`'s inverted `allow_empty`.

The levels here are ordered claims about *containment*, and each one is either
demonstrably true of a backend or the backend is not allowed to claim it. The
distinction that matters:

    subprocess   Bounded execution. A crash, hang, output flood, env leak or
                 escaped process tree is handled. Filesystem and network are NOT
                 confined: the child is the calling user.
    container    OS-level containment. The workspace is the only mount, read
                 only; there is no network namespace; capabilities are dropped
                 and privileges cannot be raised; memory, CPU and process count
                 are capped.

`subprocess` is not a weaker `container`. It is a different claim, and calling it
a sandbox is the overclaim this project has spent its life refusing.

Ordering exists so a spec can state a *requirement* and the runtime can refuse to
run when it cannot meet it. Without that, asking for `container` and silently
receiving `subprocess` is a security failure with no symptom.
"""

from __future__ import annotations

from enum import Enum


class Isolation(str, Enum):
    """Ordered containment levels. Higher is stronger.

    All four comparisons are defined explicitly, and `total_ordering` is
    deliberately NOT used.

    `total_ordering` only fills in operators that are absent, and it looks for
    them with `getattr` -- which finds the ones inherited from `str`. So
    `Isolation.SUBPROCESS >= Isolation.CONTAINER` silently fell through to
    string comparison, where `"subprocess" >= "container"` is True, and
    `confines_filesystem` returned True for the level that confines nothing.

    That is the exact overclaim this project exists to prevent, introduced by
    the convenience decorator rather than by a wrong decision. Defining the
    operators directly means no comparison can leak to the string value.
    """

    NONE = "none"
    #: Today's behaviour, honestly named. No filesystem or network confinement.
    SUBPROCESS = "subprocess"
    #: OS container: workspace-only read-only mount, no network, dropped caps.
    CONTAINER = "container"

    def _rank(self, other: object) -> int | None:
        if not isinstance(other, Isolation):
            return None
        return _ORDER.index(self), _ORDER.index(other)

    def __lt__(self, other: object) -> bool:
        pair = self._rank(other)
        if pair is None:
            return NotImplemented
        return pair[0] < pair[1]

    def __le__(self, other: object) -> bool:
        pair = self._rank(other)
        if pair is None:
            return NotImplemented
        return pair[0] <= pair[1]

    def __gt__(self, other: object) -> bool:
        pair = self._rank(other)
        if pair is None:
            return NotImplemented
        return pair[0] > pair[1]

    def __ge__(self, other: object) -> bool:
        pair = self._rank(other)
        if pair is None:
            return NotImplemented
        return pair[0] >= pair[1]

    @property
    def confines_filesystem(self) -> bool:
        return self >= Isolation.CONTAINER

    @property
    def denies_network(self) -> bool:
        return self >= Isolation.CONTAINER

    def describe(self) -> str:
        return _DESCRIPTIONS[self]


_ORDER = (Isolation.NONE, Isolation.SUBPROCESS, Isolation.CONTAINER)

_DESCRIPTIONS: dict[Isolation, str] = {
    Isolation.NONE: (
        "no containment: a bare subprocess with the caller's privileges"
    ),
    Isolation.SUBPROCESS: (
        "bounded execution only (timeout, output cap, scrubbed env, killed "
        "process tree). No filesystem or network confinement -- the child is "
        "the calling user"
    ),
    Isolation.CONTAINER: (
        "OS container: workspace mounted read-only as the only filesystem view, "
        "no network, all capabilities dropped, no privilege escalation, and "
        "memory/CPU/process-count caps"
    ),
}


def parse(value: object, default: Isolation = Isolation.SUBPROCESS) -> Isolation:
    """Read a level from a spec or config value, rejecting nonsense loudly.

    An unknown level is an error rather than a fallback. Defaulting an
    unrecognised `isolation: contianer` to something weaker is precisely the
    silent downgrade this enum exists to prevent.
    """
    if isinstance(value, Isolation):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return Isolation(value.strip().lower())
        except ValueError:
            pass
    raise ValueError(
        f"unknown isolation level {value!r}; expected one of "
        + ", ".join(level.value for level in _ORDER)
    )


#: Levels whose whole point is that a weaker backend must not be substituted.
#: `none` and `subprocess` need no runtime, so nothing is refused for them.
REQUIRES_RUNTIME = {Isolation.CONTAINER}