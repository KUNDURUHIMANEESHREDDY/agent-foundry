import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factory.capabilities.builtin import FilesystemRead
from factory.capabilities.registry import CapabilityRegistry
from factory.compiler import compile_agent
from factory.models.base import ModelResponse, ToolCall
from factory.models.fake import ScriptedAdapter
from factory.runtime.agent import RunStatus
from factory.spec.loader import load_spec

AGENTS = Path(__file__).resolve().parents[1] / "agents"


# ── API fixtures: one isolated app per test ────────────────────────────
#
# The API used four module globals (STORE, AGENTS, WORKSPACES, AUTH), so an
# isolated test had to monkeypatch module attributes and restore them — the
# pattern most likely to leak state between tests or make them
# order-dependent. `create_app(ApiDependencies(...))` replaces that: each test
# gets its own in-memory store, its own registry, and its own workspace root,
# with nothing global to restore.


@pytest.fixture
def api_workspace(tmp_path):
    """A workspace directory an agent is allowed to read from."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "a.txt").write_text("hello", encoding="utf-8")
    return ws


@pytest.fixture
def api_deps(tmp_path, api_workspace):
    from factory.api.auth import AuthSettings
    from factory.api.server import ApiDependencies
    from factory.api.workspaces import WorkspaceRegistry
    from factory.registry import AgentRegistry
    from factory.runtime.trace import MemoryTraceStore

    workspaces = WorkspaceRegistry()
    workspaces.add("default", api_workspace)

    return ApiDependencies(
        store=MemoryTraceStore(),
        agents=AgentRegistry(":memory:"),
        workspaces=workspaces,
        auth=AuthSettings("test-key"),
        data_dir=tmp_path / "data",
    )


@pytest.fixture
def api_app(api_deps):
    from factory.api.server import create_app

    return create_app(api_deps)


@pytest.fixture
def api_client(api_app):
    from fastapi.testclient import TestClient

    return TestClient(api_app)


@pytest.fixture
def auth_headers(api_deps):
    return {"Authorization": f"Bearer {api_deps.auth.key}"}


# ── Langfuse fixtures: one exporter, one client, whole session ─────────
#
# These live here rather than in test_langfuse.py because a second session-scoped
# client breaks every other Langfuse test. Langfuse registers a *global* OTel
# tracer provider on first client construction; a second client re-registers it
# and spans start landing in whichever exporter was built last. That produced an
# order-dependent failure — 10 passing alone, 10 failing alongside, and the
# mirror image when the two files ran in the opposite order.
#
# One client, never shut down mid-session: `shutdown()` kills the global batch
# processor, after which any later flush() blocks forever in queue.join().


@pytest.fixture(scope="session")
def lf_exporter():
    from tests.langfuse_helpers import CapturingExporter

    return CapturingExporter()


@pytest.fixture(scope="session")
def lf_client(lf_exporter):
    from tests.langfuse_helpers import make_test_client

    return make_test_client(lf_exporter)


@pytest.fixture(autouse=True)
def _clear_spans(lf_exporter):
    """Isolate assertions without rebuilding the client."""
    lf_exporter.reset()
    yield
    lf_exporter.reset()


def registry(workspace) -> CapabilityRegistry:
    r = CapabilityRegistry(binder=lambda n: {"root": workspace})
    r.register(FilesystemRead)
    return r


class TestShippedSpecs:
    def test_reader_spec_compiles(self, tmp_path):
        compiled = compile_agent(load_spec(AGENTS / "reader.yaml"), registry(workspace), workspace=tmp_path)
        assert compiled.granted == ["filesystem.read"]
        assert compiled.spec.limits.max_steps == 8

    def test_locked_down_grants_nothing(self, tmp_path):
        compiled = compile_agent(load_spec(AGENTS / "locked-down.yaml"), registry(workspace), workspace=tmp_path)
        assert compiled.granted == []

    @pytest.mark.asyncio
    async def test_reader_cannot_escape_its_workspace(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (tmp_path / "secret.txt").write_text("classified")
        (workspace / "ok.txt").write_text("public")

        adapter = ScriptedAdapter([
            ModelResponse(tool_calls=[ToolCall("filesystem.read", {"path": "../secret.txt"})]),
            ModelResponse(text="denied"),
        ])
        compiled = compile_agent(
            load_spec(AGENTS / "reader.yaml"),
            registry(workspace),
            workspace=workspace,
            model=adapter,
        )
        result = await compiled.runtime.run("read the secret")

        assert result.status == RunStatus.SUCCESS
        obs = result.trace.steps[0].observations[0]
        assert obs["ok"] is False
        assert "outside the workspace" in obs["error"]
        assert "classified" not in str(obs)

    @pytest.mark.asyncio
    async def test_locked_down_agent_cannot_use_any_tool(self, tmp_path):
        adapter = ScriptedAdapter([
            ModelResponse(tool_calls=[ToolCall("filesystem.read", {"path": "x"})]),
            ModelResponse(text="I have no tools"),
        ])
        compiled = compile_agent(
            load_spec(AGENTS / "locked-down.yaml"),
            registry(workspace),
            workspace=tmp_path,
            model=adapter,
        )
        result = await compiled.runtime.run("read a file")

        violations = result.trace.steps[0].violations
        assert violations
        assert violations[0]["capability"] == "filesystem.read"
        assert result.trace.steps[0].observations[0]["ok"] is False
