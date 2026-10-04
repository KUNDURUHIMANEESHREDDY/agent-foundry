"""End-to-end proof of the capability boundary, no model server required.

Run:  PYTHONPATH=src python demo_sandbox.py
"""

import asyncio
from pathlib import Path

from factory.capabilities.registry import registry_for
from factory.compiler import compile_agent
from factory.models.base import ModelResponse, ToolCall
from factory.models.fake import ScriptedAdapter
from factory.spec.loader import load_spec

WS = Path("tmp-ws").resolve()


def model_trying_to_escape():
    """A model that immediately tries to read outside its workspace."""
    return ScriptedAdapter(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall("filesystem.read", {"path": "../secret-outside.txt"}),
                    ToolCall("filesystem.read", {"path": "../../etc/hosts"}),
                    ToolCall("filesystem.read", {"path": "public.txt"}),
                ]
            ),
            ModelResponse(text="Only the in-workspace file is readable."),
        ]
    )


def model_claiming_a_shell():
    """A model that hallucinates a tool it was never granted."""
    return ScriptedAdapter(
        [
            ModelResponse(tool_calls=[ToolCall("shell.execute", {"cmd": "cat /etc/passwd"})]),
            ModelResponse(text="I have no shell capability."),
        ]
    )


def model_trying_to_rebind_the_root():
    """A model that tries to set the factory-pinned root itself."""
    return ScriptedAdapter(
        [
            ModelResponse(
                tool_calls=[ToolCall("filesystem.read", {"path": "public.txt", "root": "/"})]
            ),
            ModelResponse(text="I cannot move the root."),
        ]
    )


async def probe(label, adapter):
    compiled = compile_agent(
        load_spec("agents/reader.yaml"),
        registry_for(WS),
        workspace=WS,
        model=adapter,
    )
    result = await compiled.runtime.run("demo task")

    print(f"\n=== {label} ===")
    print(f"granted: {compiled.granted}")
    print(f"status:  {result.status}")

    step = result.trace.steps[0]
    for obs in step.observations:
        if obs.get("ok"):
            print(f"  ALLOW  {obs.get('path')} -> {str(obs.get('content'))[:40]!r}")
        else:
            print(f"  DENY   {obs.get('error')[:80]}")
    for v in step.violations:
        print(f"  VIOLATION {v.get('capability')}: {v.get('reason', '')[:70]}")


async def main():
    for label, adapter in [
        ("path traversal", model_trying_to_escape()),
        ("ungranted capability", model_claiming_a_shell()),
        ("rebind factory-pinned root", model_trying_to_rebind_the_root()),
    ]:
        await probe(label, adapter)


asyncio.run(main())
