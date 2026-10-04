"""Langfuse observability.

Two integration points, both additive — the runtime and compiler are untouched:

  LangfuseModelAdapter  wraps any ModelAdapter; each `complete()` becomes a
                        Langfuse `generation` with model, params, usage, and
                        the tool calls the model asked for.

  LangfuseTraceStore    implements the existing TraceStore protocol; a finished
                        RunTrace becomes a Langfuse `agent` observation with a
                        `tool` observation per capability call.

Both attach to the same Langfuse trace by passing the factory run id as the
Langfuse `trace_id`, so an LLM call and the run that caused it sit together.

ID SPACES
    Langfuse display ids (`observation.trace_id`, `.id`) are hex strings; the
    underlying OTel ids (`span.context.trace_id`, `span.parent.span_id`) are
    ints. They are NOT interchangeable. Nesting must be asserted against the
    OTel fields, which is what the tests here do.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable

from factory.models.base import Message, ModelAdapter, ModelResponse
from factory.runtime.trace import RunTrace, TraceStore


class LangfuseNotConfigured(RuntimeError):
    """Raised when Langfuse is requested but no credentials are present."""


@dataclass(frozen=True)
class LangfuseSettings:
    public_key: str
    secret_key: str
    host: str = "https://cloud.langfuse.com"
    environment: str | None = None
    release: str | None = None

    @classmethod
    def from_env(cls) -> "LangfuseSettings":
        public = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        secret = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
        if not public or not secret:
            raise LangfuseNotConfigured(
                "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY. "
                "Keys are at langfuse.com → Settings → API Keys."
            )
        return cls(
            public_key=public,
            secret_key=secret,
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            environment=os.environ.get("LANGFUSE_ENVIRONMENT") or None,
            release=os.environ.get("LANGFUSE_RELEASE") or None,
        )

    @property
    def configured(self) -> bool:
        return bool(
            os.environ.get("LANGFUSE_PUBLIC_KEY")
            and os.environ.get("LANGFUSE_SECRET_KEY")
        )


def build_client(settings: LangfuseSettings, span_exporter: Any | None = None) -> Any:
    """Create a Langfuse client. `span_exporter` is the seam tests use."""
    from langfuse import Langfuse

    kwargs: dict[str, Any] = {
        "public_key": settings.public_key,
        "secret_key": settings.secret_key,
        "host": settings.host,
    }
    if settings.environment:
        kwargs["environment"] = settings.environment
    if settings.release:
        kwargs["release"] = settings.release
    if span_exporter is not None:
        kwargs["span_exporter"] = span_exporter
        # Never let a test exporter also hit the network.
        kwargs["flush_at"] = 1
        kwargs["flush_interval"] = 0.1
    return Langfuse(**kwargs)


def _usage(prompt: int, completion: int) -> dict[str, int]:
    usage = {"input": prompt, "output": completion, "total": prompt + completion}
    return usage


def to_langfuse_id(value: str) -> str:
    """Map an arbitrary factory id to a valid 32-char lowercase hex trace id.

    Langfuse validates trace ids strictly and raises `ValueError` on
    `int(trace_id, 16)` for anything non-hex — so `run-abc123` cannot be used
    directly. SHA-256 keeps the mapping deterministic, which matters because
    scores are attached later by run id.
    """
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


class LangfuseModelAdapter(ModelAdapter):
    """Decorator that traces every model call as a Langfuse generation.

    Bind a factory run id with `bind_trace` so generations join the run's trace.
    Without a binding they still export, as standalone traces.
    """

    def __init__(
        self,
        inner: ModelAdapter,
        client: Any,
        model_name: str | None = None,
        bind_trace: Callable[[], str | None] | None = None,
    ) -> None:
        super().__init__(f"langfuse({inner.describe()})")
        self._inner = inner
        self._client = client
        self._model_name = model_name
        self._bind_trace = bind_trace or (lambda: None)
        self._bound: str | None = None

    def bind_trace(self, trace_id: str | None) -> None:
        """Pin subsequent generations to a specific Langfuse trace."""
        self._bound = trace_id

    @property
    def bound_trace(self) -> str | None:
        return self._bound

    def _trace_context(self) -> dict[str, str] | None:
        trace_id = self._bound or self._bind_trace()
        return {"trace_id": to_langfuse_id(trace_id)} if trace_id else None

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        observation = self._client.start_observation(
            name="model.complete",
            as_type="generation",
            trace_context=self._trace_context(),
            model=self._model_name or self._inner.name,
            model_parameters={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
            input={
                "messages": [
                    {"role": m.role, "content": m.content} for m in messages
                ],
                "tools": [t.get("name") for t in tools if isinstance(t, dict)],
            },
        )

        try:
            response = await self._inner.complete(
                messages, tools, max_output_tokens=max_output_tokens, temperature=temperature
            )
        except Exception as exc:
            observation.update(
                output=None,
                level="ERROR",
                status_message=str(exc),
                metadata={"error": type(exc).__name__},
            )
            observation.end()
            raise

        output: dict[str, Any] = {"text": response.text}
        if response.tool_calls:
            output["tool_calls"] = [
                {"name": c.name, "arguments": c.arguments} for c in response.tool_calls
            ]

        observation.update(
            output=output,
            usage_details=_usage(response.prompt_tokens, response.completion_tokens),
            metadata={"stop_reason": response.stop_reason},
        )
        observation.end()
        return response

    async def health(self) -> bool:
        return await self._inner.health()

    def describe(self) -> str:
        return f"langfuse({self._inner.describe()})"


def active_run_var(name: str = "factory_active_run") -> ContextVar[str | None]:
    """A fresh ContextVar for holding "which run is executing right now".

    This was a shared `list[str | None]` with one slot, written by `begin_run`
    and read by every model call. That is a single global, so two concurrent
    runs overwrite each other:

        run A -> begin_run(A)          holder = A
        run B -> begin_run(B)          holder = B
        run A -> model.complete()      sees B, files its generation under B

    Reproduced with two runs sharing one observability: all three of run A's
    generations landed on run B's trace, and run A produced none of its own. It
    needed a model adapter that actually suspends (`await asyncio.sleep`) —
    `ScriptedAdapter` never yields, so the runs executed back to back and never
    interleaved. That gap is why the bug survived: the test double could not
    produce the one condition that triggers it.

    A ContextVar is copied into every `asyncio.Task`, so concurrent runs each
    see only their own binding. The caveat is honest: isolation is per task.
    Two runs interleaved inside a *single* task would still share a binding,
    but that cannot happen — coroutines only interleave at suspension points,
    and `gather`/`create_task` give each run its own task.

    One variable per `LangfuseObservability`, so two observability objects do
    not read each other's runs.
    """
    return ContextVar(name, default=None)


class LangfuseTraceStore(TraceStore):
    """Exports finished runs to Langfuse. Implements the existing protocol.

    A run is exported exactly once, when the runtime marks its last save with
    `final=True`.

    Exporting on every save was wrong in a way that looked harmless: the runtime
    saves after each step so a killed run stays recoverable, so one 5-step run
    produced six `factory.run` observations carrying 1, 2, 3, 4, 5, 5 steps.
    Duplicates are the least of it — the set is cumulative, so any aggregate
    over "steps per run" or "tokens per run" reads a sawtooth and over-counts by
    roughly half.

    Runs that never reach a terminal status are not exported. That used to mean a
    crashed run was visible here, and now it is not; the local SQLite store keeps
    those rows, which is where a half-finished run belongs.
    """

    def __init__(
        self,
        client: Any,
        keep_local: bool = True,
        active_trace: ContextVar[str | None] | None = None,
    ) -> None:
        self._client = client
        self._local: dict[str, RunTrace] = {}
        self._keep_local = keep_local
        # Run ids already exported. Guards the "export exactly once" promise
        # against a caller that marks a final save twice — cheap insurance
        # against the one failure this whole change exists to prevent.
        self._exported: set[str] = set()
        # The active run, per execution context. Not a shared list: see
        # `active_run_var`.
        self._active = active_trace if active_trace is not None else active_run_var()

    @property
    def current_trace_id(self) -> str | None:
        return self._active.get()

    def begin_run(self, trace_id: str) -> None:
        self._active.set(trace_id)

    def save(self, trace: RunTrace, *, final: bool = False) -> None:
        if self._keep_local:
            self._local[trace.id] = trace
        if not final:
            return
        if trace.id in self._exported:
            return
        self._exported.add(trace.id)
        self._export(trace)

    @property
    def exported_runs(self) -> list[str]:
        """Run ids sent to Langfuse. Exposed so tests can assert cardinality."""
        return sorted(self._exported)

    def get(self, run_id: str) -> RunTrace | None:
        return self._local.get(run_id)

    def list(self, limit: int = 50) -> list[RunTrace]:
        return sorted(self._local.values(), key=lambda t: t.started_at, reverse=True)[:limit]

    def _export(self, trace: RunTrace) -> None:
        violations = [v for s in trace.steps for v in s.violations]
        level = "ERROR" if trace.status != "success" else "DEFAULT"
        lf_trace_id = to_langfuse_id(trace.id)

        agent = self._client.start_observation(
            name=f"factory.run:{trace.agent}",
            as_type="agent",
            trace_context={"trace_id": lf_trace_id},
            input={"task": trace.task},
            metadata={
                "factory_run_id": trace.id,
                "agent": trace.agent,
                "agent_version": trace.agent_version,
                "status": trace.status,
                "halt_reason": trace.halt_reason,
                "steps": len(trace.steps),
                "total_tokens": trace.total_tokens,
                "violations": len(violations),
            },
            version=trace.agent_version,
            level=level,
        )

        with agent.start_as_current_observation(
            name="run.result", as_type="span"
        ) as result:
            result.update(
                output={
                    "status": trace.status,
                    "text": trace.final_text,
                    "halt_reason": trace.halt_reason,
                }
            )

        for step in trace.steps:
            # `generation`, not `span`: Langfuse only retains usage_details on
            # generations, so a step token count would be silently dropped.
            # A step *is* one model call, so this is also the honest type.
            with agent.start_as_current_observation(
                name=f"step-{step.index}", as_type="generation", input={"index": step.index}
            ) as span:
                if step.error:
                    span.update(level="ERROR", status_message=step.error)
                span.update(
                    output={
                        "text": step.text,
                        "truncated": step.truncated,
                        "duration_ms": step.duration_ms,
                    },
                    usage_details=_usage(step.prompt_tokens, step.completion_tokens),
                    metadata={
                        "tool_calls": [tc["name"] for tc in step.tool_calls],
                        "violations": step.violations,
                    },
                )

            for call in step.tool_calls:
                match = next(
                    (
                        o
                        for o in step.observations
                        if o.get("capability") == call["name"] or o.get("path") == call["arguments"].get("path")
                    ),
                    None,
                )
                with agent.start_as_current_observation(
                    name=call["name"], as_type="tool", input=call["arguments"]
                ) as tool:
                    tool.update(
                        output=match,
                        level="WARNING" if step.violations else "DEFAULT",
                    )

        agent.end()

    def score(self, run_id: str, name: str, value: float, comment: str | None = None) -> None:
        """Attach a score to an exported run, e.g. from the eval harness.

        Accepts the factory run id; the Langfuse trace id is derived from it.
        """
        self._client.create_score(
            trace_id=to_langfuse_id(run_id),
            name=name,
            value=value,
            comment=comment,
        )

    def read_observations(self, run_id: str, minutes: int = 15) -> list[Any]:
        """Read this run's observations back from Langfuse.

        Uses the v2 observations endpoint; `api.trace.get` is legacy and returns
        410 on organisations created after 2026-09-16.
        """
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        resp = self._client.api.observations.get_many(
            trace_id=to_langfuse_id(run_id),
            from_start_time=now - dt.timedelta(minutes=minutes),
            to_start_time=now + dt.timedelta(minutes=5),
        )
        return list(getattr(resp, "data", resp) or [])

    def read_scores(self, run_id: str, minutes: int = 15) -> list[Any]:
        """Read scores for this run.

        Uses the v3 scores endpoint; `api.scores.get_many` is v2 and 410s on
        newer organisations. Note the differing time-parameter names: the v2
        observations endpoint takes from_start_time, this one from_timestamp.
        """
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        resp = self._client.api.scores_v3.get_many_v3(
            trace_id=to_langfuse_id(run_id),
            from_timestamp=now - dt.timedelta(minutes=minutes),
            to_timestamp=now + dt.timedelta(minutes=5),
        )
        return list(getattr(resp, "data", resp) or [])

    def flush(self) -> None:
        self._client.flush()

    def shutdown(self) -> None:
        self._client.shutdown()


class LangfuseObservability:
    """Pairs a trace store with a model wrapper sharing one trace binding.

    This is the intended entry point. Without it the model generations and the
    run land on separate Langfuse traces, which defeats the point: you want the
    LLM calls nested under the run that caused them.

        obs = LangfuseObservability(client)
        compiled = compile_agent(
            spec, registry, workspace,
            model=obs.wrap_model(inner, "qwen2.5:7b"),
            store=obs.store,
        )

    The store and the wrapped models share one `ContextVar`, so concurrent runs
    on the same observability keep their own generations. Each instance gets its
    own variable, so two observability objects do not read each other's runs.
    """

    def __init__(self, client: Any, keep_local: bool = True) -> None:
        self._active = active_run_var()
        self.store = LangfuseTraceStore(
            client, keep_local=keep_local, active_trace=self._active
        )
        self._client = client

    @property
    def current_trace_id(self) -> str | None:
        return self._active.get()

    def wrap_model(self, inner: ModelAdapter, model_name: str | None = None) -> LangfuseModelAdapter:
        return LangfuseModelAdapter(
            inner,
            self._client,
            model_name=model_name,
            bind_trace=self._active.get,
        )

    def flush(self) -> None:
        self._client.flush()

    def shutdown(self) -> None:
        self._client.shutdown()
