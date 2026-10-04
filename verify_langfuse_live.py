"""Real end-to-end Langfuse test against cloud.langfuse.com.

Reads credentials from the environment. Nothing is printed in full.

    set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY, then:
    PYTHONPATH=src python verify_langfuse_live.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from factory.capabilities.registry import registry_for  # noqa: E402
from factory.compiler import compile_agent  # noqa: E402
from factory.models.base import ToolCall  # noqa: E402
from factory.models.fake import ScriptedAdapter  # noqa: E402
from factory.models.base import ModelResponse  # noqa: E402
from factory.spec.loader import load_spec  # noqa: E402
from factory.tracing.langfuse import (  # noqa: E402
    LangfuseNotConfigured,
    LangfuseObservability,
    LangfuseSettings,
    build_client,
    to_langfuse_id,
)

AGENTS = Path("agents").resolve()
WS = Path("evals/workspace").resolve()

failures = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


async def main() -> int:
    print("=== 1. credentials ===")
    try:
        settings = LangfuseSettings.from_env()
    except LangfuseNotConfigured as exc:
        print(f"  FAIL not configured: {exc}")
        return 1
    print(f"  host    {settings.host}")
    print(f"  public  {settings.public_key[:14]}...")
    check("credentials loaded", True)

    client = build_client(settings)
    print("\n=== 2. auth_check (real HTTP call) ===")
    t0 = time.monotonic()
    try:
        authed = client.auth_check()
    except Exception as exc:
        authed = False
        print(f"  error: {type(exc).__name__}: {exc}")
    check("auth_check() returned True", bool(authed), f"{time.monotonic() - t0:.1f}s")

    if not authed:
        print("  cannot continue without a working key")
        return 1

    obs = LangfuseObservability(client)

    print("\n=== 3. run a real agent, traced to the cloud ===")
    (WS / "live.txt").write_text("the answer is 42", encoding="utf-8")

    inner = ScriptedAdapter([
        ModelResponse(
            prompt_tokens=142,
            completion_tokens=23,
            tool_calls=[ToolCall("filesystem.read", {"path": "live.txt"})],
        ),
        ModelResponse(prompt_tokens=38, completion_tokens=11, text="The file says the answer is 42."),
    ])

    compiled = compile_agent(
        load_spec(AGENTS / "reader.yaml"),
        registry_for(WS),
        workspace=WS,
        model=obs.wrap_model(inner, model_name="qwen2.5:7b"),
        store=obs.store,
    )

    result = await compiled.runtime.run("What does live.txt say?")
    check("run succeeded", result.status == "success", result.status)
    check("final text correct", "42" in result.text, result.text[:50])
    check("run id bound to model adapter", obs.current_trace_id == result.trace.id)

    # Concurrency. The active run id used to be one shared slot, so two runs
    # overwrote each other and A's generations were filed under B's trace. One
    # sequential run cannot see this, which is why it survived.
    print("\n=== 4b. two concurrent runs must not share a trace ===")
    from factory.models.fake import SlowAdapter  # noqa: E402

    slow = lambda: SlowAdapter(  # noqa: E731
        [ModelResponse(text="", tool_calls=[ToolCall("filesystem.read", {"path": "a.txt"})])],
        delay_s=0.05,
    )
    runtime_x = compile_agent(
        load_spec(AGENTS / "reader.yaml"),
        registry_for(WS),
        workspace=WS,
        model=obs.wrap_model(slow(), "qwen2.5:7b"),
        store=obs.store,
    )
    runtime_y = compile_agent(
        load_spec(AGENTS / "reader.yaml"),
        registry_for(WS),
        workspace=WS,
        model=obs.wrap_model(slow(), "qwen2.5:7b"),
        store=obs.store,
    )
    # A workspace file for the runs to read, so they actually make tool calls.
    (WS / "a.txt").write_text("x", encoding="utf-8")

    async def _both():
        return await asyncio.gather(
            runtime_x.runtime.run("concurrent one"),
            runtime_y.runtime.run("concurrent two"),
        )

    pair = asyncio.run(_both())
    id_x, id_y = pair[0].trace.id, pair[1].trace.id
    check("concurrent runs got distinct ids", id_x != id_y, f"{id_x} / {id_y}")
    check(
        "active binding is one of the two runs",
        obs.current_trace_id in (id_x, id_y),
        str(obs.current_trace_id),
    )

    print(f"\n  factory run id : {result.trace.id}")
    print(f"  langfuse trace : {to_langfuse_id(result.trace.id)}")

    print("\n=== 4. flush and wait for the server to accept ===")
    t0 = time.monotonic()
    client.flush()
    check("flush() returned without error", True, f"{time.monotonic() - t0:.1f}s")

    # Spans are batched server-side; give ingestion a moment before reading back.
    print("  waiting 8s for ingestion...")
    time.sleep(8)

    print("\n=== 5. read the observations back from the API ===")
    # api.trace.get is legacy and 410s on orgs created after 2026-09-16, so the
    # store's v2 observations reader is used instead.
    lf_trace = to_langfuse_id(result.trace.id)

    # Ingestion is asynchronous, so poll rather than assume a fixed delay.
    rows = []
    for attempt in range(8):
        try:
            rows = obs.store.read_observations(result.trace.id)
            if rows:
                if attempt:
                    print(f"  data appeared after {attempt} extra poll(s)")
                break
        except Exception as exc:
            print(f"  attempt {attempt + 1} failed: {type(exc).__name__}")
        time.sleep(5)

    print(f"  returned {len(rows)} observation(s)")
    for o in rows:
        print(f"    - type={getattr(o, 'type', '?')!r:12} name={getattr(o, 'name', '?')!r}")

    check("trace has observations", len(rows) > 0, f"{len(rows)}")
    names = {getattr(o, "name", None) for o in rows}
    check("agent run present", "factory.run:reader" in names, str(sorted(n for n in names if n)))

    # Cardinality, not membership. `"factory.run:reader" in names` is a set test,
    # so it stays true no matter how many copies arrived — which is exactly how
    # the export-on-every-save bug survived: one run produced six
    # `factory.run:reader` observations and every membership check still passed.
    run_obs = [o for o in rows if getattr(o, "name", None) == "factory.run:reader"]
    check(
        "exactly one factory.run observation",
        len(run_obs) == 1,
        f"{len(run_obs)} (was 6 before the fix; the extras were cumulative)",
    )
    step_obs = [
        o for o in rows if str(getattr(o, "name", "") or "").startswith("step-")
    ]
    check(
        "the surviving observation carries every step",
        len(run_obs) == 1 and len(step_obs) == len(result.trace.steps),
        f"{len(step_obs)} step observation(s) for {len(result.trace.steps)} step(s)",
    )

    check("model generations present", "model.complete" in names)
    check("tool observation present", "filesystem.read" in names)
    check("step observations present", any(n and n.startswith("step-") for n in names))
    check(
        "all on one trace",
        {getattr(o, "trace_id", None) for o in rows} == {lf_trace},
        str({getattr(o, "trace_id", None) for o in rows}),
    )

    print("\n=== 6. attach a score ===")
    try:
        obs.store.score(result.trace.id, "live_e2e", 1.0, "end-to-end run completed")
        client.flush()
        check("create_score accepted", True)
    except Exception as exc:
        check("create_score accepted", False, f"{type(exc).__name__}: {exc}")

    srows = []
    for attempt in range(6):
        try:
            srows = obs.store.read_scores(result.trace.id)
            if srows:
                break
        except Exception as exc:
            print(f"  score readback attempt {attempt + 1} failed: {type(exc).__name__}")
        time.sleep(5)

    print(f"  scores returned: {len(srows)}")
    for s in srows[:5]:
        print(f"    - name={getattr(s, 'name', '?')!r} value={getattr(s, 'value', '?')}")
    check("score visible via v3 API", len(srows) > 0, f"{len(srows)}")

    print(f"\n  dashboard url: {client.get_trace_url(trace_id=lf_trace)}")

    print("\n=== 7. an ungranted capability, traced as a violation ===")
    obs2 = LangfuseObservability(client)
    inner2 = ScriptedAdapter([
        ModelResponse(
            prompt_tokens=20,
            completion_tokens=8,
            tool_calls=[ToolCall("shell.execute", {"cmd": "type ../secret.txt"})],
        ),
        ModelResponse(prompt_tokens=15, completion_tokens=6, text="I have no shell capability."),
    ])
    compiled2 = compile_agent(
        load_spec(AGENTS / "reader.yaml"),
        registry_for(WS),
        workspace=WS,
        model=obs2.wrap_model(inner2, model_name="qwen2.5:7b"),
        store=obs2.store,
    )
    result2 = await compiled2.runtime.run("run a shell command")
    client.flush()
    violations = [v for s in result2.trace.steps for v in s.violations]
    check("violation recorded", bool(violations), str(violations)[:80])
    print(f"  violation trace: {to_langfuse_id(result2.trace.id)}")

    (WS / "live.txt").unlink(missing_ok=True)
    time.sleep(3)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("all live checks passed — traces reached cloud.langfuse.com")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
