"""Command line entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from factory.capabilities.registry import CapabilityRegistry
from factory.compiler import CompileError, compile_agent
from factory.models.fake import ScriptedAdapter
from factory.models.base import ModelResponse
from factory.runtime.trace import MemoryTraceStore
from factory.spec.loader import SpecError, load_spec


def _registry(workspace):
    from factory.capabilities.registry import registry_for

    return registry_for(workspace or ".")


def _sabotage_audit(suites, progress, jobs):
    """Seam for tests: run the audit with progress reporting wired through."""
    from factory.eval.sabotage import audit

    return audit(suites, progress=progress, max_workers=jobs)


def cmd_validate(args: argparse.Namespace) -> int:
    from factory.compiler import validate

    spec = load_spec(args.spec)
    report = validate(spec)
    for w in report.warnings:
        print(f"warn: {w}")
    for e in report.errors:
        print(f"error: {e}")
    print("ok" if report.ok else "invalid")
    return 0 if report.ok else 1


def cmd_compile(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    workspace = args.workspace or spec.memory.path
    compiled = compile_agent(
        spec, _registry(workspace), workspace=workspace, strict=not args.lenient
    )
    print(f"agent:   {compiled.spec.name} v{compiled.spec.version}")
    print(f"model:   {compiled.model.describe()}")
    print(f"granted: {', '.join(compiled.granted) or '(none)'}")
    for w in compiled.warnings:
        print(f"warn:    {w}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    workspace = args.workspace or spec.memory.path

    model = None
    if args.dry_run:
        # Proves the loop, gating and tracing without a model server.
        model = ScriptedAdapter([ModelResponse(text="dry-run: no model called")])

    compiled = compile_agent(
        spec, _registry(workspace), workspace=workspace, model=model
    )
    result = asyncio.run(compiled.runtime.run(args.task))

    if args.json:
        print(json.dumps(
            {
                "status": result.status,
                "text": result.text,
                "halt_reason": result.trace.halt_reason,
                "steps": len(result.trace.steps),
                "tokens": result.trace.total_tokens,
                "violations": [
                    v for s in result.trace.steps for v in s.violations
                ],
            },
            indent=2,
        ))
    else:
        print(f"status:  {result.status}")
        if result.trace.halt_reason:
            print(f"halted:  {result.trace.halt_reason}")
        print(f"steps:   {len(result.trace.steps)}")
        print(f"tokens:  {result.trace.total_tokens}")
        print(f"text:    {result.text}")

    return 0 if result.ok else 1


def cmd_eval(args: argparse.Namespace) -> int:
    import asyncio

    from factory.eval import (
        compare,
        format_text,
        load_all,
        run_suite,
        save_baseline,
        security_failures,
        to_json,
    )

    suites = load_all(args.suites)
    base = Path(args.suites)

    async def run_all() -> list:
        out = []
        for suite in suites:
            out.append(await run_suite(suite, base, use_script=not args.live))
        return out

    summaries = asyncio.run(run_all())

    print(to_json(summaries) if args.json else format_text(summaries))

    if args.baseline:
        save_baseline(summaries, args.baseline)
        print(f"\nbaseline saved: {args.baseline}")

    if args.compare:
        baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        print("\n--- vs baseline ---")
        print(compare(baseline, summaries))

    sec = sum(len(security_failures(s)) for s in summaries)
    return 1 if sec or any(s.failed for s in summaries) else 0


def cmd_sabotage(args: argparse.Namespace) -> int:
    """Prove the eval suites actually detect broken mitigations."""
    from factory.eval.sabotage import (
        CASE_TIMEOUT_S,
        SABOTAGES,
        audit,
        format_report,
    )

    print(
        f"running {len(SABOTAGES)} sabotages, up to {CASE_TIMEOUT_S}s per case, "
        f"{args.jobs} at a time...",
        file=sys.stderr,
    )

    def tick(result) -> None:
        mark = "ok " if result.detected else "GAP"
        print(f"  [{mark}] {result.sabotage.id}: {result.verdict}", file=sys.stderr)

    results, errors = _sabotage_audit(args.suites, tick, args.jobs)
    print(format_report(results, errors))

    unhealthy = [r for r in results if not r.detected]
    return 1 if unhealthy else 0


def cmd_agents(args: argparse.Namespace) -> int:
    """Registry subcommands: list / show / versions / diff / register / from-goal."""
    from factory.registry import AgentRegistry, RegistryError

    reg = AgentRegistry(args.db) if args.db else AgentRegistry()

    try:
        if args.agents_cmd == "list":
            agents = reg.list()
            if not agents:
                print("no agents registered")
                return 0
            print(f"{'NAME':<24} {'VER':>4}  {'LATEST':<9} {'PRI':<9} CAPABILITIES")
            for a in agents:
                latest = reg.latest(a.name)
                caps = ",".join(latest.spec.capabilities) if latest else ""
                print(
                    f"{a.name:<24} {a.versions:>4}  {str(a.latest):<9} "
                    f"{a.priority:<9} {caps}"
                )
            return 0

        if args.agents_cmd == "show":
            rec = (
                reg.get_version(args.name, args.version)
                if args.version
                else reg.latest(args.name)
            )
            if rec is None:
                print(f"error: {args.name} not found", file=sys.stderr)
                return 2
            print(json.dumps(rec.spec.model_dump(mode="json"), indent=2))
            return 0

        if args.agents_cmd == "versions":
            recs = reg.versions(args.name)
            if not recs:
                print(f"error: {args.name} not found", file=sys.stderr)
                return 2
            for r in recs:
                tags = f"  [{', '.join(r.tags)}]" if r.tags else ""
                print(f"  {r.version:<10} {r.notes or '(no notes)'}{tags}")
            return 0

        if args.agents_cmd == "diff":
            d = reg.diff(args.name, args.version, args.to)
            if not d["changed"]:
                print(f"{d['agent']}: no changes between {d['from']} and {d['to']}")
                return 0
            for change in d["changes"]:
                print(f"  {change['field']}:")
                print(f"    from: {json.dumps(change['from'])}")
                print(f"    to:   {json.dumps(change['to'])}")
            if d["capabilities_added"]:
                print(f"  capabilities added:   {', '.join(d['capabilities_added'])}")
            if d["capabilities_removed"]:
                print(f"  capabilities removed: {', '.join(d['capabilities_removed'])}")
            return 0

        if args.agents_cmd == "register":
            rec = reg.register_file(args.path, notes=args.notes, tags=args.tag, replace=args.replace)
            print(f"registered {rec.agent}@{rec.version}  ({len(rec.spec.capabilities)} capabilities)")
            return 0

        if args.agents_cmd == "from-goal":
            from factory.compiler import compile_goal_to_spec, plan_goal
            from factory.capabilities.registry import registry_for
            from factory.spec.agent_spec import AgentSpec, GoalSpec

            requirements = {}
            for pair in args.req or []:
                key, _, value = pair.partition("=")
                requirements[key] = value if value else True

            known = set(registry_for(Path.cwd()).known())

            if args.template:
                template = reg.get(args.template)
                if template is None:
                    print(f"error: template '{args.template}' not found", file=sys.stderr)
                    return 2
            else:
                template = AgentSpec(
                    name=args.name or "generated",
                    version=args.version,
                    system_prompt="You are an agent. Use your capabilities to accomplish the goal.",
                )

            constraints = {"max_steps": int(args.max_steps)} if args.max_steps else {}
            goal = GoalSpec(
                goal=args.goal, requirements=requirements, constraints=constraints
            )

            capabilities, errors, proposal = plan_goal(goal, template, known)
            if errors:
                for key, msg in errors.items():
                    print(f"error: {key}: {msg}", file=sys.stderr)
                return 2

            if args.dry_run:
                print(f"goal:    {args.goal}")
                print(f"would add: {', '.join(capabilities) or '(nothing)'}")
                for p in proposal:
                    print(f"  + {p['capability']}  ({p['reason']})")
                return 0

            spec = compile_goal_to_spec(
                goal, template, known, name=args.name, version=args.version
            )
            rec = reg.register(
                spec,
                notes=f"derived from goal: {args.goal}",
                tags=["generated"],
                replace=args.replace,
            )
            print(f"registered {rec.agent}@{rec.version}")
            for p in proposal:
                print(f"  + {p['capability']}  ({p['reason']})")
            return 0

        return 2

    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def cmd_traces(args: argparse.Namespace) -> int:
    from factory.runtime.trace import MemoryTraceStore

    store = MemoryTraceStore()
    print(json.dumps({"note": "traces are per-process; use the API to persist"}, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the API.

    Refuses a non-loopback bind without auth. Every endpoint past `/health` can
    execute an agent, and the API key is effectively a grant of code execution,
    so an accidental `0.0.0.0` with `FACTORY_API_INSECURE=1` is the kind of
    mistake that is only noticed afterwards.
    """
    import uvicorn

    from factory.api.auth import AuthSettings
    from factory.api.workspaces import WorkspaceRegistry

    settings = AuthSettings.from_env()
    exposed = args.host not in ("127.0.0.1", "localhost", "::1")

    if exposed and not settings.enabled:
        print(
            f"refusing to bind {args.host} with authentication disabled.\n"
            "  Anyone who can reach this port can execute agents, write files in "
            "the approved workspaces, and read traces.\n"
            f"  Set FACTORY_API_KEY, or bind 127.0.0.1 to use insecure mode.",
            file=sys.stderr,
        )
        return 2

    workspaces = WorkspaceRegistry.from_env()
    print(f"workspaces: {workspaces.describe()}", file=sys.stderr)

    uvicorn.run("factory.api.server:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="factory", description="Agent Factory")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Check a spec without compiling")
    p_val.add_argument("spec")
    p_val.set_defaults(func=cmd_validate)

    p_serve = sub.add_parser("serve", help="Run the API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    p_comp = sub.add_parser("compile", help="Compile a spec, report grants")
    p_comp.add_argument("spec")
    p_comp.add_argument("--workspace")
    p_comp.add_argument("--lenient", action="store_true", help="drop ungranted capabilities")
    p_comp.set_defaults(func=cmd_compile)

    p_run = sub.add_parser("run", help="Run an agent against a task")
    p_run.add_argument("spec")
    p_run.add_argument("task")
    p_run.add_argument("--workspace")
    p_run.add_argument("--json", action="store_true")
    p_run.add_argument("--dry-run", action="store_true", help="use scripted model, no server")
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("eval", help="Run eval suites against a spec")
    p_eval.add_argument("--suites", default="evals")
    p_eval.add_argument("--json", action="store_true")
    p_eval.add_argument(
        "--live", action="store_true", help="use the real model instead of scripted turns"
    )
    p_eval.add_argument("--baseline", help="save this run as a baseline")
    p_eval.add_argument("--compare", help="compare against a saved baseline")
    p_eval.set_defaults(func=cmd_eval)

    p_sab = sub.add_parser(
        "sabotage",
        help="Prove the eval suites detect broken mitigations",
    )
    p_sab.add_argument("--suites", default="evals")
    p_sab.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="sabotages to run concurrently (each uses its own sandbox)",
    )
    p_sab.set_defaults(func=cmd_sabotage)

    p_agents = sub.add_parser("agents", help="Agent registry")
    p_agents.add_argument("--db", help="Registry database (default .factory/agents.db)")
    a_sub = p_agents.add_subparsers(dest="agents_cmd", required=True)

    a_list = a_sub.add_parser("list", help="List registered agents")
    a_list.set_defaults(_noop=True)

    a_show = a_sub.add_parser("show", help="Print an agent spec as JSON")
    a_show.add_argument("name")
    a_show.add_argument("--version")
    a_show.set_defaults(_noop=True)

    a_vers = a_sub.add_parser("versions", help="List versions of an agent")
    a_vers.add_argument("name")
    a_vers.set_defaults(_noop=True)

    a_diff = a_sub.add_parser("diff", help="Diff two versions")
    a_diff.add_argument("name")
    a_diff.add_argument("version", help="from version")
    a_diff.add_argument("--to", required=True, help="to version")
    a_diff.set_defaults(_noop=True)

    a_reg = a_sub.add_parser("register", help="Register a spec file")
    a_reg.add_argument("path")
    a_reg.add_argument("--notes", default="")
    a_reg.add_argument("--tag", action="append", default=[])
    a_reg.add_argument("--replace", action="store_true")
    a_reg.set_defaults(_noop=True)

    a_goal = a_sub.add_parser("from-goal", help="Derive an agent from a goal")
    a_goal.add_argument("goal")
    a_goal.add_argument("--name", default="generated")
    a_goal.add_argument("--version", default="0.1.0")
    a_goal.add_argument(
        "--req", action="append", help="requirement, repeatable (e.g. --req files --req python)"
    )
    a_goal.add_argument("--max-steps", type=int)
    a_goal.add_argument("--template", help="use a registered agent as the template")
    a_goal.add_argument("--dry-run", action="store_true", help="show the plan only")
    a_goal.add_argument("--replace", action="store_true")
    a_goal.set_defaults(_noop=True)

    # A subparser with no command has no `func`; fall back to printing help.
    p_agents.set_defaults(func=cmd_agents)
    for p in (a_list, a_show, a_vers, a_diff, a_reg, a_goal):
        p.set_defaults(func=cmd_agents)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        args.func.print_help()
        return 0
    try:
        return args.func(args)
    except (SpecError, CompileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
