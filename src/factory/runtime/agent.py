"""The agent runtime loop.

Every limit in the spec is read *inside* the loop, on every iteration. A limit
that is only checked before the loop is a comment, not a control.

Halting is explicit and always reported: the caller gets a status and a reason,
never a silent truncation of work.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

from factory.capabilities.registry import CapabilityDenied, CapabilityGate
from factory.models.base import Message, ModelAdapter, ModelResponse
from factory.runtime.context import (
    ContextBudget,
    estimate_completion_tokens,
    estimate_request_tokens,
    truncate,
)
from factory.runtime.trace import RunTrace, StepRecord, TraceStore
from factory.spec.agent_spec import AgentSpec, Limits


class RunStatus:
    SUCCESS = "success"
    MAX_STEPS = "max_steps_exceeded"
    MAX_TOKENS = "max_tokens_exceeded"
    MAX_TOOL_CALLS = "max_tool_calls_exceeded"
    MODEL_ERROR = "model_error"
    TIMEOUT = "timeout"
    #: The model call itself exceeded its per-step allowance. Distinct from
    #: TIMEOUT, which is the run's total wall clock running out: one means a
    #: slow provider, the other means a long run. Collapsing them would make the
    #: trace unable to say which bound was hit.
    MODEL_TIMEOUT = "model_timeout"


@dataclass
class RunResult:
    status: str
    text: str
    trace: RunTrace
    observations: list[dict[str, Any]]

    @property
    def ok(self) -> bool:
        return self.status == RunStatus.SUCCESS


def tool_schema(capability_name: str, description: str) -> dict[str, Any]:
    """REMOVED — do not reintroduce.

    This used to build one generic `{path: string}` schema for every capability,
    which told a real model that `git.commit` takes `path` and never `message`,
    and that `python.execute` takes `path` and never `code`. Scripted evals hid
    it completely, because they manufacture `ToolCall` objects directly instead
    of going through a schema.

    Tool schemas now come from the capability itself:
    `CapabilityRegistry.tool_schema(name)` -> `Capability.parameter_schema()`.
    """
    raise NotImplementedError(
        "Generic tool schemas are gone. Use "
        "CapabilityRegistry.tool_schema(name), which reads each capability's own "
        "parameter_schema()."
    )


class AgentRuntime:
    """Executes one task against one model, bounded by its spec and its grants."""

    def __init__(
        self,
        spec: AgentSpec,
        model: ModelAdapter,
        gate: CapabilityGate,
        store: TraceStore,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> None:
        self._spec = spec
        self._model = model
        self._gate = gate
        self._store = store
        self._tools = tool_schemas or []

    async def run(self, task: str, *, tenant: str | None = None) -> RunResult:
        limits = self._spec.limits
        budget = ContextBudget(
            window=self._spec.model.context_window,
            reserve=limits.context_reserve_tokens,
        )

        trace = RunTrace(
            id=f"run-{uuid.uuid4().hex[:12]}",
            agent=self._spec.name,
            agent_version=self._spec.version,
            task=task,
            # Part of the run's identity from birth. Stamped afterwards it would
            # be missing from every intermediate save, and a run killed
            # mid-flight would be unattributable to anyone.
            tenant=tenant,
        )

        # Let observability backends attach later observations to this run.
        # Duck-typed so a plain store without begin_run still works.
        begin = getattr(self._store, "begin_run", None)
        if callable(begin):
            begin(trace.id)

        messages: list[Message] = [
            Message(role="system", content=self._spec.system_prompt),
            Message(role="user", content=task),
        ]

        observations: list[dict[str, Any]] = []
        tool_calls_made = 0
        started = time.monotonic()

        # `step` is bounded above. The check is inside the loop on purpose.
        for index in range(1, limits.max_steps + 1):
            if time.monotonic() - started > limits.step_timeout_s * limits.max_steps:
                return self._halt(
                    trace, RunStatus.TIMEOUT,
                    f"exceeded {limits.step_timeout_s * limits.max_steps:.0f}s wall clock",
                )

            step_started = time.monotonic()
            record = StepRecord(index=index, model=self._model.describe(), prompt_tokens=0, completion_tokens=0)

            window, report = truncate(messages, budget)
            record.truncated = report.truncated

            try:
                response = await asyncio.wait_for(
                    self._model.complete(
                        window,
                        self._tools,
                        max_output_tokens=self._spec.model.max_output_tokens,
                        temperature=self._spec.model.temperature,
                    ),
                    # Whichever is tighter: this step's own allowance, or what is
                    # left of the run's wall clock. Neither bound may be
                    # overrun by waiting on the model.
                    timeout=self._step_deadline(started=started, limits=limits, index=index),
                )
            except (asyncio.TimeoutError, TimeoutError):
                # Must be caught before the generic `except Exception` below,
                # which would misreport a timeout as a model error.
                record.error = "model call exceeded its time budget"
                record.duration_ms = int((time.monotonic() - step_started) * 1000)
                trace.steps.append(record)
                self._store.save(trace)
                return self._halt(
                    trace, RunStatus.MODEL_TIMEOUT,
                    f"model call on step {index} exceeded "
                    f"{limits.step_timeout_s:g}s",
                )
            except Exception as exc:
                record.error = str(exc)
                record.duration_ms = int((time.monotonic() - step_started) * 1000)
                trace.steps.append(record)
                self._store.save(trace)
                return self._halt(
                    trace, RunStatus.MODEL_ERROR,
                    f"model call failed on step {index}: {exc}",
                )

            record.prompt_tokens = response.prompt_tokens
            record.completion_tokens = response.completion_tokens
            record.text = response.text
            prompt_charge, completion_charge = self._charge_tokens(
                response, window, self._tools
            )
            trace.total_prompt_tokens += prompt_charge
            trace.total_completion_tokens += completion_charge

            if trace.total_tokens > limits.max_tokens:
                record.duration_ms = int((time.monotonic() - step_started) * 1000)
                trace.steps.append(record)
                return self._halt(
                    trace, RunStatus.MAX_TOKENS,
                    f"cumulative {trace.total_tokens} tokens exceeds max_tokens={limits.max_tokens}",
                )

            if response.text:
                messages.append(Message(role="assistant", content=response.text))

            if not response.wants_tool():
                record.duration_ms = int((time.monotonic() - step_started) * 1000)
                trace.steps.append(record)
                trace.status = RunStatus.SUCCESS
                trace.completed_at = time.time()
                trace.final_text = response.text
                self._store.save(trace, final=True)
                return RunResult(
                    status=RunStatus.SUCCESS,
                    text=response.text,
                    trace=trace,
                    observations=observations,
                )

            record.tool_calls = [
                {"name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls
            ]
            messages.append(
                Message(role="assistant", content="", tool_calls=response.tool_calls)
            )

            for call in response.tool_calls:
                tool_calls_made += 1
                if tool_calls_made > limits.max_tool_calls:
                    record.duration_ms = int((time.monotonic() - step_started) * 1000)
                    trace.steps.append(record)
                    return self._halt(
                        trace, RunStatus.MAX_TOOL_CALLS,
                        f"exceeded max_tool_calls={limits.max_tool_calls}",
                    )

                # Boundary check on every call. No path around this.
                clean_args, strip_violation = self._gate.sanitize_arguments(
                    call.name, call.arguments
                )
                result, violation = self._invoke(call.name, clean_args)

                if strip_violation:
                    result = {
                        **result,
                        "stripped_args": sorted(strip_violation["stripped"]),
                    }
                    violation = (
                        strip_violation
                        if violation is None
                        else {**violation, "also_stripped": sorted(strip_violation["stripped"])}
                    )

                record.observations.append(result)
                if violation:
                    record.violations.append(violation)
                observations.append(result)

                messages.append(
                    Message(
                        role="observation",
                        content=f"{call.name}: {_summarize(result)}",
                    )
                )

            record.duration_ms = int((time.monotonic() - step_started) * 1000)
            trace.steps.append(record)
            self._store.save(trace)

        return self._halt(
            trace, RunStatus.MAX_STEPS,
            f"exceeded max_steps={limits.max_steps}",
        )

    def _invoke(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        try:
            capability = self._gate.check(name)
        except CapabilityDenied as exc:
            violation = {"capability": name, "reason": str(exc), "arguments": arguments}
            # Denial is fed back to the model as an observation so it can adapt,
            # and recorded separately so it is never mistaken for success.
            return (
                {"ok": False, "error": f"denied: {exc}", "capability": name},
                violation,
            )

        try:
            return capability.invoke(**arguments), None
        except Exception as exc:
            return (
                {"ok": False, "error": f"tool error: {exc}", "capability": name},
                None,
            )

    @staticmethod
    def _step_deadline(
        started: float, limits: Limits, index: int
    ) -> float:
        """How long this model call may take.

        Two bounds, and the wait must respect the tighter one:

          * `step_timeout_s` — the per-step allowance. A single slow provider
            must not consume a whole run's budget.
          * the run's remaining wall clock (`step_timeout_s * max_steps`, minus
            what has already elapsed).

        Before this existed, `await model.complete(...)` had no deadline at all.
        The loop checked elapsed time at the top of each iteration, so an adapter
        that never returned blocked forever and neither `max_steps` nor
        `step_timeout_s` was reachable — the limits were decorative.

        Measured with a hanging adapter and a 1.0s budget: the run never halted.
        `wait_for` cancels the coroutine, so a stuck socket is actually torn
        down rather than leaked.
        """
        run_budget = limits.step_timeout_s * limits.max_steps
        remaining = run_budget - (time.monotonic() - started)
        return max(0.001, min(limits.step_timeout_s, remaining))

    def _charge_tokens(
        self,
        response: ModelResponse,
        window: list[Message],
        tools: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """What to bill this step for, against the cumulative ceiling.

        Provider-reported usage is a *refinement*, not the authority. It was the
        only input before, which meant a provider returning `0` — or omitting
        `usage`, or counting only completion — silently disabled `max_tokens`
        entirely. Local inference servers do all three.

        So each half is charged as `max(reported, estimated)`. An honest provider
        is unaffected: its figure is at least as large as the estimate. A
        provider that under-reports cannot under-charge, and the ceiling still
        trips.
        """
        local_prompt = estimate_request_tokens(window, tools)
        local_completion = estimate_completion_tokens(
            response.text, list(response.tool_calls)
        )

        prompt = max(int(response.prompt_tokens or 0), local_prompt)
        completion = max(int(response.completion_tokens or 0), local_completion)
        return prompt, completion

    def _halt(self, trace: RunTrace, status: str, reason: str) -> RunResult:
        trace.status = status
        trace.halt_reason = reason
        trace.completed_at = time.time()
        # Terminal status, so this is the run's last save. Exporting stores act
        # on that; the earlier per-step saves exist only for crash recovery.
        self._store.save(trace, final=True)
        return RunResult(
            status=status,
            text=trace.final_text or (trace.steps[-1].text if trace.steps else ""),
            trace=trace,
            observations=[],
        )


def _summarize(result: dict[str, Any]) -> str:
    if not result.get("ok", False):
        return str(result.get("error", "error"))
    kind = result.get("kind")
    if kind == "file":
        head = str(result.get("content", ""))[:200]
        more = "…" if result.get("truncated") else ""
        return f"file ({result.get('size')} bytes)\n{head}{more}"
    if kind == "directory":
        entries = result.get("entries", [])
        return f"directory with {len(entries)} entries: {', '.join(entries[:20])}"
    return str(result)[:200]
