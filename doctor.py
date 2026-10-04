"""Preflight: what this machine can and cannot prove.

Reports the state of the three external dependencies the project's own claims
rest on:

    a model      live-model qualification (P2) -- unproven without one
    a container  `isolation: container` -- refuse rather than run unconfined
    Langfuse     the live round trip, claimed nowhere but worth knowing

P2 has been described as "blocked" in conversation for a while. That is a claim,
and this is the command that settles it: it prints which backends the shipped
specs point at and whether anything is listening. Cheap on purpose, so it is
never a reason the answer is stale.

The model check is a TCP connect, not a completion request. It answers the
question that actually gets asked -- is anything listening -- without spending a
token or depending on anyone's credentials. It therefore cannot prove a model
*works*, only that something is there, and the output says so.
"""

from __future__ import annotations

import json
import os
import sys

from factory.eval.runner import live_model_status

AGENTS = "agents"


def _langfuse_status() -> tuple[bool, str]:
    keys = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    present = [k for k in keys if os.environ.get(k)]
    if len(present) == len(keys):
        return True, "credentials present; run verify_langfuse_live.py to confirm"
    missing = [k for k in keys if k not in present]
    return False, "missing " + ", ".join(missing)


def _container_status() -> tuple[bool, str]:
    from factory.capabilities import container

    try:
        version = container.probe()
    except container.ContainerUnavailable as exc:
        return False, str(exc)
    return True, f"usable (daemon {version})"


def report(as_json: bool = False) -> int:
    models = live_model_status(AGENTS)
    any_model = any(r["reachable"] for r in models)
    has_container, container_detail = _container_status()
    has_langfuse, langfuse_detail = _langfuse_status()

    if as_json:
        print(json.dumps(
            {
                "live_model_qualification": {
                    "possible": any_model,
                    "backends": models,
                    "note": (
                        "reachable means something is listening on the configured "
                        "base_url; it does not prove the model answers correctly"
                    ),
                },
                "container_isolation": {"possible": has_container, "detail": container_detail},
                "langfuse_live": {"possible": has_langfuse, "detail": langfuse_detail},
            },
            indent=2,
        ))
    else:
        print("live-model qualification (P2)")
        for row in models:
            mark = "reachable " if row["reachable"] else "unreachable"
            print(f"  [{mark}] {row['provider']:<14} {row['base_url'] or '-':<32} {row['detail']}")

        # Printed whether or not anything is reachable. The moment a backend is up
        # is precisely when someone would read this as "P2 is done", and an open
        # socket says nothing about whether the model answers usefully.
        print("\n  A reachable socket does not prove a model answers correctly.")
        print("  `factory eval --live` is what proves it; nothing here substitutes.")
        if not any_model:
            print("\n  => NOT RUNNABLE. Every live-model claim stays unproven here.")

        print("\ncontainer isolation")
        mark = "usable     " if has_container else "unavailable "
        print(f"  [{mark}] {container_detail}")

        print("\nlangfuse live round trip")
        mark = "configured " if has_langfuse else "unconfigured"
        print(f"  [{mark}] {langfuse_detail}")

    # Exit 0 when the ordinary suite can run. A machine without a model is a
    # normal machine, not a broken one; refusing to exit 0 would make this
    # useless as a preflight and would train people to ignore it.
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    return report(as_json="--json" in argv)


if __name__ == "__main__":
    raise SystemExit(main())