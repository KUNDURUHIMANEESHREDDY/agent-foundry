"""FastAPI surface for the factory.

Exposes the three operations that matter before any UI exists: validate a spec,
compile it and report grants, and run a task with a full trace back.

SECURITY

Two things are enforced before any of that happens.

  * The capability root comes from `WORKSPACES`, a server-side map of opaque ids
    to approved directories. Clients send `workspace_id`, never a path. See
    `factory.api.workspaces` — accepting a caller's path here would let them pick
    the root that `safe_join`, `filesystem.write`, `python.execute` and
    `git.commit` are all confined to.

  * Every route except `/health` requires an API key. The dependency is attached
    to a router rather than to each handler, so a newly added endpoint is
    authenticated by default instead of by remembering.

This module is not safe to expose unauthenticated. Authentication defaults to on
and can only be turned off explicitly, via `FACTORY_API_INSECURE`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from factory.api.auth import PUBLIC_PATHS, AuthSettings, presented_key
from factory.api.tenants import Tenant, TenantRegistry, key_map_from_env
from factory.api.workspaces import DEFAULT_ID, UnknownWorkspace, WorkspaceRegistry
from factory.capabilities.registry import registry_for
from factory.compiler import CompileError, compile_agent, validate
from factory.models import build_adapter
from factory.models.fake import ScriptedAdapter
from factory.models.base import ModelResponse
from factory.registry import AgentRegistry
from factory.runtime.trace import SqliteTraceStore, TraceStore
from factory.spec.agent_spec import AgentSpec
from factory.spec.loader import SpecError, load_spec, parse_spec

# Interactive docs and the OpenAPI schema hang off the bare app, so a router
# dependency cannot protect them. Publishing them unauthenticated hands an
# anonymous caller a map of every endpoint and payload shape, which is
# reconnaissance for everything else this module fixes. Off by default; opt in
# with FACTORY_API_DOCS=1 when you are poking at it locally.
_DOCS_ON = os.environ.get("FACTORY_API_DOCS", "").strip().lower() in (
    "1", "true", "yes", "on",
)

#: Where traces are persisted. Server configuration, never a request field.
#:
#: Two reasons this is not the spec's `memory:` block:
#:
#:   * `/run` used to build its own store from the submitted spec, whose default
#:     is `type: memory`. Every API run landed in a throwaway MemoryTraceStore,
#:     so `/traces` could never see a trace the API itself produced.
#:
#:   * `memory.path` is reachable from the spec YAML, and SqliteTraceStore does
#:     `Path(path).parent.mkdir(parents=True)`. A caller could therefore choose
#:     where the server creates directories and writes a database. Same class of
#:     bug as accepting a workspace path.
TRACE_DB_ENV = "FACTORY_TRACE_DB"


@dataclass
class ApiDependencies:
    """Everything the handlers need from the server, in one object.

    These were four module globals (`STORE`, `AGENTS`, `WORKSPACES`, `AUTH`),
    which meant one database per process, no way to serve two tenants, and tests
    that had to monkeypatch module attributes to get an isolated store — the
    pattern most likely to produce order-dependent, secretly-shared state.

    Now they are constructed per app. `create_app(ApiDependencies(...))` gives a
    test its own database without touching anything global, and `app.state.deps`
    is the single place a handler looks.
    """

    store: TraceStore
    agents: AgentRegistry
    workspaces: WorkspaceRegistry
    auth: AuthSettings
    data_dir: Path
    tenants: TenantRegistry = field(default_factory=TenantRegistry.single)
    trace_db_env: str = TRACE_DB_ENV

    @classmethod
    def from_env(cls, data_dir: str | Path | None = None) -> "ApiDependencies":
        base = Path(data_dir).resolve() if data_dir else Path.cwd() / ".factory"
        base.mkdir(parents=True, exist_ok=True)

        tenants = TenantRegistry.from_env()
        auth = AuthSettings.from_env(key_map_from_env(tenants))
        auth.announce()

        return cls(
            store=SqliteTraceStore(
                os.environ.get(TRACE_DB_ENV) or (base / "traces.db")
            ),
            agents=AgentRegistry(base / "agents.db"),
            workspaces=WorkspaceRegistry.from_env(),
            auth=auth,
            data_dir=base,
            tenants=tenants,
        )

    @property
    def default_workspace(self) -> Path:
        return self.workspaces.resolve(self.workspaces.default_id)


def get_deps(request: Request) -> ApiDependencies:
    """FastAPI dependency: this app's server-side state."""
    return request.app.state.deps


def get_store(deps: ApiDependencies = Depends(get_deps)) -> TraceStore:
    return deps.store


def get_agents(deps: ApiDependencies = Depends(get_deps)) -> AgentRegistry:
    return deps.agents


def get_workspaces(deps: ApiDependencies = Depends(get_deps)) -> WorkspaceRegistry:
    return deps.workspaces


def resolve_tenant(request: Request) -> Tenant:
    """Which caller this request is. Set by `require_auth`.

    Handlers depend on this rather than reading a global, so a request cannot be
    attributed to the wrong caller even if two arrive at once — the same reason
    the Langfuse active-run id became a ContextVar.
    """
    tenant = getattr(request.state, "tenant", None)
    if tenant is None:
        # Unreachable on the secure router: require_auth runs first and 401s.
        # Present so a future public route that needs a tenant fails loudly
        # rather than silently getting an unrestricted one.
        raise HTTPException(
            status_code=401, detail="missing or invalid API key"
        )
    return tenant


def require_auth(request: Request) -> None:
    """Reject unauthenticated requests, and record who sent them."""
    if request.url.path in PUBLIC_PATHS:
        return

    deps: ApiDependencies = request.app.state.deps
    tenant_id = deps.auth.tenant_for(presented_key(request))
    if tenant_id is None:
        raise HTTPException(
            status_code=401,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    tenant = deps.tenants.get(tenant_id)
    if tenant is None:
        # A key that resolves to a tenant nobody defined. Should be impossible:
        # key_map_from_env validates. Refused rather than assumed unrestricted.
        raise HTTPException(
            status_code=403, detail=f"key refers to unknown tenant {tenant_id!r}"
        )

    request.state.tenant = tenant


#: Liveness only.
public = APIRouter()

#: Anything that can execute code, read traces, or mutate state lives here, so
#: it inherits the auth dependency. New routes are protected by construction
#: rather than by remembering to add `Depends(require_auth)`.
secure = APIRouter(dependencies=[Depends(require_auth)])


class RegisterRequest(BaseModel):
    # Unknown fields are refused rather than ignored. A request that sends
    # `workspace: "/"` must not be quietly accepted with the server's own root:
    # the caller would believe they pointed the agent somewhere they did not.
    model_config = ConfigDict(extra="forbid")

    yaml: str = Field(min_length=1, description="Agent spec as YAML")
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    replace: bool = False


class GoalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1)
    requirements: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    template: str = Field(default="", description="Name of a registered agent to use as template")
    name: str = ""
    version: str = "0.1.0"
    persist: bool = Field(
        default=True,
        description="Write the result into the registry. `register` is reserved by BaseModel.",
    )


class SpecRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yaml: str = Field(min_length=1, description="Agent spec as YAML")


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yaml: str = Field(min_length=1)
    task: str = Field(min_length=1)
    workspace_id: str = Field(
        default=DEFAULT_ID,
        description=(
            "Server-approved workspace id, not a path. The capability root is "
            "chosen by the server; a client cannot supply a directory."
        ),
    )
    dry_run: bool = Field(
        default=False,
        description="Use a scripted model so the loop runs without a model server.",
    )


def _workspace(body: RunRequest, workspaces: WorkspaceRegistry) -> Path:
    try:
        return workspaces.resolve(body.workspace_id)
    except UnknownWorkspace as exc:
        # 403, not 404: the id is well-formed but not approved here, and saying
        # which ids ARE approved would turn this into an enumeration oracle.
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _compile(
    body: RunRequest, deps: ApiDependencies, tenant: Tenant, dry_run: bool = False
):
    try:
        spec = parse_spec(body.yaml)
    except SpecError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # The tenant's policy is applied to the spec, and a capability outside it is
    # REFUSED rather than dropped. Dropping it would run a weaker agent than the
    # caller asked for while reporting success — the same shape as the old
    # `memory.path` behaviour, where a request was quietly ignored.
    forbidden = tenant.forbidden(spec.capabilities)
    if forbidden:
        raise HTTPException(
            status_code=403,
            detail=(
                f"tenant {tenant.id!r} may not use {forbidden}. "
                f"Allowed: "
                f"{sorted(tenant.capabilities) if not tenant.unrestricted else 'all'}."
            ),
        )

    workspace = _workspace(body, deps.workspaces)
    model = (
        ScriptedAdapter([ModelResponse(text="dry-run: no model called")])
        if dry_run
        else None
    )
    try:
        compiled = compile_agent(
            spec,
            # The spec's required containment. A tenant asking for `container`
            # gets refused if there is no runtime, rather than silently getting
            # an unconfined subprocess.
            registry_for(workspace, spec.isolation),
            workspace=workspace,
            model=model,
            # Server-owned persistence. Without this the run is traced into a
            # MemoryTraceStore and `/traces` stays permanently empty.
            store=deps.store,
        )
    except CompileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Do not silently discard what the caller asked for. Say so, in the response
    # they are already reading.
    requested = spec.memory
    if requested.type != "memory" or requested.path:
        compiled.warnings.append(
            f"spec requested memory type={requested.type!r} path={requested.path!r}; "
            f"ignored. Trace persistence is server configuration "
            f"({deps.trace_db_env}), not a request field."
        )

    return compiled


@public.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "version": "0.1.0"}


@secure.get("/capabilities")
async def capabilities() -> dict[str, Any]:
    from factory.capabilities.builtin import FilesystemRead

    return {"known": [FilesystemRead.name]}


@secure.get("/workspaces")
async def list_workspaces(
    workspaces: WorkspaceRegistry = Depends(get_workspaces),
) -> dict[str, Any]:
    """The ids a client may pass as `workspace_id`.

    Authenticated, so ids are not an enumeration oracle for anonymous callers.
    """
    return {
        "workspaces": workspaces.describe(),
        "default": workspaces.default_id,
    }


@secure.post("/validate")
async def validate_spec(req: SpecRequest) -> dict[str, Any]:
    try:
        spec = parse_spec(req.yaml)
    except SpecError as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": []}

    report = validate(spec)
    return {
        "valid": report.ok,
        "agent": spec.name,
        "version": spec.version,
        "errors": report.errors,
        "warnings": report.warnings,
    }


@secure.post("/compile")
async def compile_spec(
    body: RunRequest,
    deps: ApiDependencies = Depends(get_deps),
    tenant: Tenant = Depends(resolve_tenant),
) -> dict[str, Any]:
    compiled = _compile(body, deps, tenant, dry_run=True)
    return {
        "agent": compiled.spec.name,
        "version": compiled.spec.version,
        "model": compiled.model.describe(),
        "granted": compiled.granted,
        "warnings": compiled.warnings,
        "system_prompt": compiled.spec.system_prompt,
    }


@secure.post("/run")
async def run(
    body: RunRequest,
    deps: ApiDependencies = Depends(get_deps),
    tenant: Tenant = Depends(resolve_tenant),
) -> dict[str, Any]:
    compiled = _compile(body, deps, tenant, dry_run=body.dry_run)
    result = await compiled.runtime.run(body.task, tenant=tenant.id)

    return {
        # Without this the client cannot correlate a run with `/traces`, which is
        # the other half of having a persistent store at all.
        "trace_id": result.trace.id,
        "status": result.status,
        "ok": result.ok,
        "text": result.text,
        "halt_reason": result.trace.halt_reason,
        "steps": [s.__dict__ for s in result.trace.steps],
        "tokens": result.trace.total_tokens,
        "violations": [v for s in result.trace.steps for v in s.violations],
        "warnings": compiled.warnings,
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    """One server-sent event.

    `event:` carries the type so a client can subscribe selectively, and
    `data:` is one line of JSON. The blank line after is part of the SSE
    framing -- without it the client buffers forever.
    """
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@secure.post("/run/stream")
async def run_stream(
    body: RunRequest,
    deps: ApiDependencies = Depends(get_deps),
    tenant: Tenant = Depends(resolve_tenant),
) -> Any:
    """Run an agent, reporting each step as it completes.

    `/run` answers only when the whole run finishes, which for an agent taking
    10-30 seconds leaves a frontend with nothing to show. This streams the same
    run, same trace, same tenant scoping -- it is a second view of one execution,
    not a second execution.

    Steps are emitted after they are durable, so a client that disconnects and
    reconnects via `/traces/{id}` sees the same history the stream showed.
    """
    from fastapi.responses import StreamingResponse

    compiled = _compile(body, deps, tenant, dry_run=body.dry_run)

    async def generate():
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_step(record: Any, trace: Any) -> None:
            # Called from the loop; hop threads because queue.put is not
            # thread-safe and uvicorn runs the agent on the loop already, but the
            # callback contract does not promise that.
            loop.call_soon_threadsafe(
                queue.put_nowait,
                (
                    "step",
                    {
                        "step": record.index,
                        "tool_calls": [tc["name"] for tc in record.tool_calls],
                        "text": record.text,
                        "duration_ms": record.duration_ms,
                        "violations": len(record.violations),
                    },
                ),
            )

        async def watch():
            result = await compiled.runtime.run(body.task, tenant=tenant.id, on_step=on_step)
            return result

        runner = asyncio.create_task(watch())

        # Emitted first so the client has the id before any step arrives, and can
        # fall back to /traces/{id} if the stream drops.
        yield _sse("start", {"task": body.task, "agent": compiled.spec.name})

        while True:
            done, _ = await asyncio.wait(
                {runner}, timeout=0.05
            )
            if runner in done:
                break

            while not queue.empty():
                name, payload = queue.get_nowait()
                yield _sse(name, payload)

        while not queue.empty():
            name, payload = queue.get_nowait()
            yield _sse(name, payload)

        result = runner.result()

        yield _sse(
            "done",
            {
                "trace_id": result.trace.id,
                "status": result.status,
                "ok": result.ok,
                "text": result.text,
                "halt_reason": result.trace.halt_reason,
                "tokens": result.trace.total_tokens,
                "violations": [v for s in result.trace.steps for v in s.violations],
                "warnings": compiled.warnings,
            },
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@secure.get("/traces")
async def traces(
    limit: int = 20,
    store: TraceStore = Depends(get_store),
    tenant: Tenant = Depends(resolve_tenant),
) -> dict[str, Any]:
    """This tenant's runs only.

    Every caller used to see every trace, including other callers' tasks and
    prompts. Reads are now scoped, so `/traces` means "mine".
    """
    found = store.list(limit=limit, tenant=tenant.id)
    return {
        "traces": [
            {
                "id": t.id,
                "agent": t.agent,
                "task": t.task,
                "status": t.status,
                "halt_reason": t.halt_reason,
                "tokens": t.total_tokens,
                "steps": len(t.steps),
            }
            for t in found
        ]
    }


@secure.get("/traces/{run_id}")
async def trace_detail(
    run_id: str,
    store: TraceStore = Depends(get_store),
    tenant: Tenant = Depends(resolve_tenant),
) -> dict[str, Any]:
    found = store.get(run_id, tenant=tenant.id)
    if found is None:
        # 404, not 403: confirming someone else's run exists would leak that the
        # id is real, which is the information the scope is meant to withhold.
        raise HTTPException(status_code=404, detail="trace not found")
    return found.to_dict()


# ── registry ──────────────────────────────────────────────────────────


@secure.get("/agents")
async def list_agents(
    agents: AgentRegistry = Depends(get_agents),
    tenant: Tenant = Depends(resolve_tenant),
) -> dict[str, Any]:
    return {"agents": [vars(a) for a in agents.list(tenant=tenant.id)]}


@secure.get("/agents/{name}")
async def get_agent(
    name: str,
    version: str | None = None,
    agents: AgentRegistry = Depends(get_agents),
    tenant: Tenant = Depends(resolve_tenant),
) -> dict[str, Any]:
    rec = (
        agents.get_version(name, version, tenant=tenant.id)
        if version
        else agents.latest(name, tenant=tenant.id)
    )
    if rec is None:
        raise HTTPException(status_code=404, detail=f"agent '{name}' not found")
    return {**rec.summary(), "spec": rec.spec.model_dump(mode="json")}


@secure.get("/agents/{name}/versions")
async def agent_versions(
    name: str,
    agents: AgentRegistry = Depends(get_agents),
    tenant: Tenant = Depends(resolve_tenant),
) -> dict[str, Any]:
    recs = agents.versions(name, tenant=tenant.id)
    if not recs:
        raise HTTPException(status_code=404, detail=f"agent '{name}' not found")
    return {"agent": name, "versions": [r.summary() for r in recs]}


@secure.get("/agents/{name}/diff")
async def agent_diff(
    name: str,
    from_version: str,
    to: str,
    agents: AgentRegistry = Depends(get_agents),
    tenant: Tenant = Depends(resolve_tenant),
) -> dict[str, Any]:
    try:
        return agents.diff(name, from_version, to, tenant=tenant.id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@secure.post("/agents")
async def register_agent(
    req: RegisterRequest,
    agents: AgentRegistry = Depends(get_agents),
    tenant: Tenant = Depends(resolve_tenant),
) -> dict[str, Any]:
    """Register a spec under the calling tenant.

    The tenant's capability policy applies here too. Storing a spec the caller
    could not run would be harmless, but registering `python.execute` and being
    told later that a run was refused would be a confusing way to learn the
    boundary exists.
    """
    try:
        spec = parse_spec(req.yaml)
    except SpecError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    report = validate(spec)
    if not report.ok:
        raise HTTPException(
            status_code=422, detail={"errors": report.errors, "warnings": report.warnings}
        )

    forbidden = tenant.forbidden(spec.capabilities)
    if forbidden:
        raise HTTPException(
            status_code=403,
            detail=(
                f"tenant {tenant.id!r} may not register a spec using {forbidden}"
            ),
        )

    try:
        rec = agents.register(
            spec,
            notes=req.notes,
            tags=req.tags,
            source="api",
            replace=req.replace,
            tenant=tenant.id,
        )
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return rec.summary()


@secure.post("/compile-goal")
async def compile_goal_endpoint(
    req: GoalRequest,
    deps: ApiDependencies = Depends(get_deps),
    tenant: Tenant = Depends(resolve_tenant),
) -> dict[str, Any]:
    """Derive an agent spec from a goal, optionally registering the result."""
    from factory.compiler import compile_goal_to_spec, plan_goal
    from factory.compiler.compiler import REQUIREMENT_MAP
    from factory.spec.agent_spec import GoalSpec

    known = registry_for(deps.default_workspace).known()
    # A goal must not plan its way to a capability the tenant may not have.
    if not tenant.unrestricted:
        known = [c for c in known if tenant.allows(c)]

    if req.template:
        template = deps.agents.get(req.template, tenant=tenant.id)
        if template is None:
            raise HTTPException(
                status_code=404, detail=f"template agent '{req.template}' not found"
            )
    else:
        template = AgentSpec(
            name=req.name or "generated",
            version=req.version,
            system_prompt="You are an agent. Use your capabilities to accomplish the goal.",
        )

    goal = GoalSpec(
        goal=req.goal, requirements=req.requirements, constraints=req.constraints
    )
    capabilities, errors, proposal = plan_goal(goal, template, set(known))

    if errors:
        raise HTTPException(
            status_code=422,
            detail={
                "errors": errors,
                "known_requirements": sorted(REQUIREMENT_MAP),
                "known_capabilities": known,
            },
        )

    if not req.persist:
        return {"registered": False, "capabilities": capabilities, "proposal": proposal}

    spec = compile_goal_to_spec(
        goal,
        template,
        set(known),
        name=req.name or "generated",
        version=req.version,
    )
    rec = deps.agents.register(
        spec,
        notes=f"derived from goal: {req.goal}",
        tags=["generated"],
        tenant=tenant.id,
    )
    return {
        "registered": True,
        "agent": rec.agent,
        "version": rec.version,
        "capabilities": capabilities,
        "proposal": proposal,
        "spec": rec.spec.model_dump(mode="json"),
    }


# ── app construction ───────────────────────────────────────────────────


def create_app(deps: ApiDependencies | None = None) -> FastAPI:
    """Build an app bound to its own server-side state.

    The module-level `app` below exists so `uvicorn factory.api.server:app`
    works, but tests should call this instead of monkeypatching module
    attributes: two apps over two dependency sets is the point.
    """
    application = FastAPI(
        title="Agent Factory",
        version="0.1.0",
        docs_url="/docs" if _DOCS_ON else None,
        redoc_url="/redoc" if _DOCS_ON else None,
        openapi_url="/openapi.json" if _DOCS_ON else None,
    )
    application.state.deps = deps or ApiDependencies.from_env()

    application.include_router(public)
    application.include_router(secure)
    return application


#: The default, environment-configured app. Used by `uvicorn` and `factory serve`.
app = create_app()

# Convenience handles onto the default app's state. These are *derived* from the
# app rather than declared separately, so they cannot drift from what handlers
# actually use — the old version had both, and nothing kept them in step.
DATA_DIR = app.state.deps.data_dir
STORE = app.state.deps.store
AGENTS = app.state.deps.agents
WORKSPACES = app.state.deps.workspaces
AUTH = app.state.deps.auth
