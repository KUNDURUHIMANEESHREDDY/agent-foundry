"""The tool schema is the capability's contract with the model.

Every capability used to be advertised with one generic `{path: string}`
schema. A real model was therefore told that `git.commit` takes `path` and
never `message`, and that `python.execute` takes `path` and never `code`. No
capability except `filesystem.read` was callable as described.

Nothing caught it, because the scripted evals manufacture `ToolCall` objects
directly and never go through a schema. These tests close that gap by deriving
the expected contract from `invoke` itself, so a new parameter cannot be added
to a capability without the schema following.
"""

from __future__ import annotations

import inspect
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from factory.capabilities.builtin import FilesystemRead
from factory.capabilities.fs_write import FilesystemWrite
from factory.capabilities.git_commit import GitCommit
from factory.capabilities.python_exec import PythonExecute
from factory.capabilities.registry import (
    Capability,
    CapabilityRegistry,
    object_schema,
)

ALL = [FilesystemRead, FilesystemWrite, PythonExecute, GitCommit]


def real_registry(tmp_path: Path | None = None) -> CapabilityRegistry:
    reg = CapabilityRegistry(binder=lambda _n: {})
    reg.register(FilesystemRead)
    reg.register(FilesystemWrite)
    reg.register(PythonExecute)
    reg.register(GitCommit)
    return reg


def invoke_params(impl: type[Capability]) -> set[str]:
    """Named parameters `invoke` actually reads, excluding **kwargs catch-alls."""
    sig = inspect.signature(impl.invoke)
    return {
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }


class TestSchemaMatchesInvoke:
    """The core guard. Each capability's schema must cover its real signature."""

    @pytest.mark.parametrize("impl", ALL, ids=lambda c: c.name)
    def test_every_invoke_parameter_is_advertised(self, impl):
        """Anything `invoke` reads must be in the schema or be factory-pinned.

        This is the assertion that would have failed for `git.commit`
        (`message`, `paths`) and `python.execute` (`code`).
        """
        schema = impl.parameter_schema()
        advertised = set(schema["properties"]) | set(impl.pinned_params)

        missing = sorted(invoke_params(impl) - advertised)
        assert not missing, (
            f"{impl.name}.invoke reads {missing} but its schema never mentions "
            f"them. A real model cannot supply these arguments. Schema "
            f"advertises: {sorted(schema['properties'])}"
        )

    @pytest.mark.parametrize("impl", ALL, ids=lambda c: c.name)
    def test_schema_advertises_no_invented_parameters(self, impl):
        """The schema may not promise arguments `invoke` would ignore."""
        schema = impl.parameter_schema()
        invented = sorted(
            set(schema["properties"]) - invoke_params(impl) - set(impl.pinned_params)
        )
        assert not invented, (
            f"{impl.name} advertises {invented}, which invoke() does not read. "
            f"A model would send them and watch them be silently dropped."
        )

    @pytest.mark.parametrize("impl", ALL, ids=lambda c: c.name)
    def test_required_is_a_subset_of_properties(self, impl):
        schema = impl.parameter_schema()
        missing = sorted(set(schema["required"]) - set(schema["properties"]))
        assert not missing, f"{impl.name} requires undeclared params {missing}"


class TestSchemasAreDistinct:
    def test_no_two_capabilities_share_a_schema(self, tmp_path):
        """The original bug: one generic schema for every capability."""
        reg = real_registry(tmp_path)
        seen: dict[str, str] = {}
        for name in reg.known():
            fingerprint = str(reg.tool_schema(name)["parameters"])
            assert fingerprint not in seen, (
                f"{name} has the same schema as {seen.get(fingerprint)}: "
                f"{fingerprint}"
            )
            seen[fingerprint] = name

    def test_every_schema_advertises_something(self, tmp_path):
        """A schema with no properties is not a usable contract."""
        reg = real_registry(tmp_path)
        for name in reg.known():
            params = reg.tool_schema(name)["parameters"]
            assert params["properties"], f"{name} advertises no parameters at all"


class TestPinnedParamsAreNeverAdvertised:
    @pytest.mark.parametrize("impl", ALL, ids=lambda c: c.name)
    def test_no_pinned_param_appears_in_the_schema(self, impl):
        leaked = sorted(set(impl.parameter_schema()["properties"]) & set(impl.pinned_params))
        assert not leaked, f"{impl.name} advertises pinned param(s) {leaked}"

    def test_registration_rejects_a_schema_that_leaks(self):
        """Enforced centrally, so a future capability cannot get this wrong."""

        class Leaky(Capability):
            name = "leaky"
            pinned_params = ("root",)

            @classmethod
            def parameter_schema(cls):
                return object_schema(
                    {"path": {"type": "string"}, "root": {"type": "string"}},
                    required=["path"],
                )

            def invoke(self, **kwargs):
                return {}

        with pytest.raises(ValueError, match="factory-pinned"):
            CapabilityRegistry().register(Leaky)

    def test_registration_rejects_a_capability_with_no_schema(self):
        """A capability with no contract cannot be called by a real model."""

        class Schemaless(Capability):
            name = "schemaless"

            def invoke(self, **kwargs):
                return {}

        with pytest.raises(ValueError, match="parameter_schema"):
            CapabilityRegistry().register(Schemaless)

    def test_registration_rejects_a_non_object_schema(self):
        class Weird(Capability):
            name = "weird"

            @classmethod
            def parameter_schema(cls):
                return {"type": "string"}

            def invoke(self, **kwargs):
                return {}

        with pytest.raises(ValueError, match="must describe an object"):
            CapabilityRegistry().register(Weird)


class TestKnownCapabilitiesAreCorrect:
    """The specific contracts, asserted explicitly.

    Broad property tests above keep new capabilities honest; these catch a
    capability whose schema is *self-consistent but wrong* — for example
    requiring `path` on a capability that has no such concept.
    """

    def test_git_commit_takes_message_and_paths(self, tmp_path):
        params = real_registry(tmp_path).tool_schema("git.commit")["parameters"]
        assert set(params["properties"]) == {"message", "paths"}
        assert params["required"] == ["message"]
        assert params["properties"]["paths"]["type"] == "array"

    def test_python_execute_takes_code_not_path(self, tmp_path):
        params = real_registry(tmp_path).tool_schema("python.execute")["parameters"]
        assert set(params["properties"]) == {"code"}
        assert "path" not in params["properties"]

    def test_filesystem_write_exposes_content(self, tmp_path):
        params = real_registry(tmp_path).tool_schema("filesystem.write")["parameters"]
        assert {"path", "content", "overwrite", "append"} <= set(params["properties"])
        assert set(params["required"]) == {"path", "content"}

    def test_filesystem_read_takes_path(self, tmp_path):
        params = real_registry(tmp_path).tool_schema("filesystem.read")["parameters"]
        assert set(params["properties"]) == {"path"}

    def test_no_schema_requires_a_path_it_cannot_use(self, tmp_path):
        """`path` was required on all four capabilities, including git.commit."""
        reg = real_registry(tmp_path)
        for name in reg.known():
            params = reg.tool_schema(name)["parameters"]
            if "path" in params["required"]:
                assert "path" in invoke_params(
                    next(c for c in ALL if c.name == name)
                ), f"{name} requires 'path' but invoke() does not read it"


class TestRegistryToolSchema:
    def test_unknown_capability_raises(self):
        with pytest.raises(Exception, match="No implementation"):
            real_registry().tool_schema("nope")

    def test_schema_carries_name_and_description(self, tmp_path):
        for name in real_registry(tmp_path).known():
            schema = real_registry(tmp_path).tool_schema(name)
            assert schema["name"] == name
            assert schema["description"].strip(), f"{name} has no description"


class TestCompilerUsesRegistrySchemas:
    def test_compiled_tool_schemas_match_the_registry(self):
        """`compile_agent` must not synthesise its own schema again."""
        from factory.compiler import compile_agent
        from factory.spec.loader import load_spec

        repo = Path(tempfile.mkdtemp(prefix="factory-schema-test-"))
        try:
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            # The real registry, so its binder supplies workspace roots the way
            # production does.
            from factory.capabilities.registry import registry_for

            reg = registry_for(repo)

            root = Path(__file__).resolve().parents[1]
            spec = load_spec(root / "agents" / "coder.yaml")

            compiled = compile_agent(spec, reg)
            built = {s["name"]: s for s in compiled.runtime._tools}

            for name in built:
                assert built[name]["parameters"] == reg.tool_schema(name)["parameters"], (
                    f"{name}: compiler emitted a schema that differs from the "
                    f"capability's own"
                )
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_generic_tool_schema_helper_is_disabled(self):
        """The old helper is a trap; it must not silently come back."""
        from factory.runtime.agent import tool_schema

        with pytest.raises(NotImplementedError, match="Generic tool schemas"):
            tool_schema("filesystem.read", "desc")