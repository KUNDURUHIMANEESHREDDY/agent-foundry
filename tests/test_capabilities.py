from __future__ import annotations

import pytest

from factory.capabilities.registry import (
    Capability,
    CapabilityDenied,
    CapabilityGate,
    CapabilityNotFound,
    CapabilityRegistry,
    object_schema,
    safe_join,
)


class Echo(Capability):
    name = "echo"
    pinned_params = ("token",)
    required_params = ("token",)

    def __init__(self, token: str = "none") -> None:
        self.token = token

    @classmethod
    def parameter_schema(cls):
        return object_schema({"text": {"type": "string"}}, required=["text"])

    def invoke(self, **kwargs):
        return {"ok": True, "echo": kwargs, "token": self.token}


class Unpinned(Capability):
    name = "unpinned"

    @classmethod
    def parameter_schema(cls):
        return object_schema({}, required=[])

    def invoke(self, **kwargs):
        return {"ok": True, **kwargs}


@pytest.fixture
def registry() -> CapabilityRegistry:
    reg = CapabilityRegistry(binder=lambda name: {"token": "factory-set"} if name == "echo" else {})
    reg.register(Echo)
    reg.register(Unpinned)
    return reg


class TestRegistry:
    def test_registers_and_lists(self, registry):
        assert registry.known() == ["echo", "unpinned"]

    def test_rejects_duplicate_registration(self, registry):
        with pytest.raises(ValueError, match="already registered"):
            registry.register(Echo)

    def test_rejects_missing_name(self):
        class Nameless(Capability):
            def invoke(self, **kwargs):
                return {}

        with pytest.raises(ValueError, match="non-empty name"):
            CapabilityRegistry().register(Nameless)

    def test_unknown_capability_is_rejected_not_ignored(self, registry):
        with pytest.raises(CapabilityNotFound, match="No implementation"):
            registry.issue("shell.execute")

    def test_factory_may_bind_pinned_params(self, registry):
        grant = registry.issue("echo", token="factory-set")
        assert grant.params == {"token": "factory-set"}

    def test_grant_requires_pinned_params(self):
        """A registry with no binder cannot mint a grant with missing bounds."""
        bare = CapabilityRegistry()
        bare.register(Echo)
        with pytest.raises(ValueError, match="missing factory-bound parameter"):
            bare.issue("echo")

    def test_reports_pinned_params(self, registry):
        assert registry.pinned_params("echo") == ("token",)
        assert registry.pinned_params("unpinned") == ()

    def test_agent_cannot_rebind_pinned_params(self, registry):
        """The model asks for a different token; the factory's value stands."""
        gate = CapabilityGate.of(registry, [registry.issue("echo", token="factory-set")])

        clean, violation = gate.sanitize_arguments("echo", {"token": "forged", "x": 1})

        assert clean == {"x": 1}
        assert violation is not None
        assert "token" in violation["reason"]
        assert gate.check("echo").invoke(**clean)["token"] == "factory-set"

    def test_no_violation_when_pinned_params_absent(self, registry):
        gate = CapabilityGate.of(registry, [registry.issue("echo", token="t")])
        clean, violation = gate.sanitize_arguments("echo", {"x": 1})
        assert clean == {"x": 1}
        assert violation is None

    def test_sanitize_leaves_ungranted_arguments_untouched(self, registry):
        gate = CapabilityGate.of(registry, [])
        args = {"cmd": "ls"}
        clean, violation = gate.sanitize_arguments("unpinned", args)
        assert clean == args
        assert violation is None

    def test_grant_ignores_params_not_declared_pinned(self, registry):
        grant = registry.issue("unpinned", extra="allowed")
        assert grant.params == {"extra": "allowed"}


class TestGate:
    def test_denies_ungranted_capability(self, registry):
        gate = CapabilityGate.of(registry, [registry.issue("echo")])
        with pytest.raises(CapabilityDenied, match="not granted"):
            gate.check("unpinned")

    def test_returns_bound_instance_for_granted(self, registry):
        grant = registry.issue("echo", token="factory-set")
        gate = CapabilityGate.of(registry, [grant])
        cap = gate.check("echo")
        result = cap.invoke(x=1)
        assert result["token"] == "factory-set"
        assert result["echo"] == {"x": 1}

    def test_gate_with_no_grants_denies_everything(self, registry):
        gate = CapabilityGate.of(registry, [])
        with pytest.raises(CapabilityDenied, match=r"\(none\)"):
            gate.check("echo")

    def test_denied_error_lists_only_granted(self, registry):
        gate = CapabilityGate.of(registry, [registry.issue("echo")])
        with pytest.raises(CapabilityDenied) as exc:
            gate.check("unpinned")
        assert "echo" in str(exc.value)
        assert "unpinned" not in str(exc.value).split("Granted:")[1]

    def test_is_granted(self, registry):
        gate = CapabilityGate.of(registry, [registry.issue("echo")])
        assert gate.is_granted("echo")
        assert not gate.is_granted("unpinned")

    def test_instance_is_cached(self, registry):
        gate = CapabilityGate.of(registry, [registry.issue("echo")])
        assert gate.check("echo") is gate.check("echo")


class TestSafeJoin:
    def test_accepts_path_inside_root(self, tmp_path):
        (tmp_path / "a.txt").write_text("hi")
        assert safe_join(tmp_path, "a.txt") == (tmp_path / "a.txt").resolve()

    def test_refuses_parent_traversal(self, tmp_path):
        with pytest.raises(CapabilityDenied, match="outside the workspace"):
            safe_join(tmp_path, "../escape.txt")

    def test_refuses_nested_traversal_that_escapes(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        with pytest.raises(CapabilityDenied):
            safe_join(root, "a/../../outside")

    def test_refuses_absolute_path(self, tmp_path):
        with pytest.raises(CapabilityDenied, match="outside the workspace"):
            safe_join(tmp_path, "/etc/passwd")

    def test_refuses_empty(self, tmp_path):
        with pytest.raises(ValueError, match="non-empty"):
            safe_join(tmp_path, "   ")

    def test_symlink_cannot_escape_root(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("classified")
        try:
            (root / "link.txt").symlink_to(secret)
        except OSError:
            pytest.skip("symlinks unavailable on this platform")
        with pytest.raises(CapabilityDenied, match="outside the workspace"):
            safe_join(root, "link.txt")
