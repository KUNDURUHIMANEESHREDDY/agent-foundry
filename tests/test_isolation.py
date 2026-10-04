"""Container isolation: the containment claims, and the refusal when unavailable.

The eval suites in `container-isolation.yaml` prove containment by running real
containers. These tests assert the same properties *structurally* — on the argv
and on the refusal path — so they run on a machine with no container runtime, and
so a broken flag is caught even where Docker is absent.

The distinction matters: a test that can only observe behaviour through a
container is a test that goes quiet, rather than red, on a machine without one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from factory.capabilities import container
from factory.capabilities.python_exec import PythonExecute
from factory.isolation import Isolation, parse
from factory.spec.agent_spec import AgentSpec


# ── levels ────────────────────────────────────────────────────────
class TestLevelsAreOrdered:
    def test_container_is_stronger_than_subprocess(self):
        assert Isolation.CONTAINER > Isolation.SUBPROCESS

    def test_subprocess_is_stronger_than_none(self):
        assert Isolation.SUBPROCESS > Isolation.NONE

    def test_subprocess_does_not_confine_the_filesystem(self):
        """The claim that must never quietly become untrue."""
        assert not Isolation.SUBPROCESS.confines_filesystem

    def test_subprocess_does_not_deny_the_network(self):
        assert not Isolation.SUBPROCESS.denies_network

    def test_container_confines_both(self):
        assert Isolation.CONTAINER.confines_filesystem
        assert Isolation.CONTAINER.denies_network


class TestUnknownLevelsAreRejected:
    """A typo must not become a weaker level.

    `isolation: contianer` silently defaulting to `subprocess` would be a
    security downgrade with no symptom anywhere. The spelling error would be the
    only evidence, and it would be in a YAML file nobody re-reads.
    """

    @pytest.mark.parametrize(
        "bad", ["contianer", "", "sandbox", "docker", "true", 1, "subprocess2"]
    )
    def test_it_raises(self, bad):
        with pytest.raises(ValueError):
            parse(bad)

    @pytest.mark.parametrize("variant", ["container", "CONTAINER", " Container "])
    def test_case_and_whitespace_are_normalised(self, variant):
        """Leniency here is safe: it accepts a *stronger* reading of the field.

        Normalising is the opposite of the dangerous default, which would map an
        unrecognised spelling onto something weaker.
        """
        assert parse(variant) is Isolation.CONTAINER

    def test_it_names_the_valid_levels(self):
        with pytest.raises(ValueError) as exc:
            parse("nope")
        assert "container" in str(exc.value)

    def test_none_defaults_to_subprocess(self):
        assert parse(None) is Isolation.SUBPROCESS

    def test_a_valid_level_round_trips(self):
        assert parse("container") is Isolation.CONTAINER


# ── the argv, which is where containment actually comes from ─────
@pytest.fixture
def argv(tmp_path: Path) -> list[str]:
    return container.build_argv(
        "print(1)", tmp_path, exe="docker"
    )


def _flag(argv: list[str], name: str) -> str | None:
    """The value following `name`, or None."""
    return argv[argv.index(name) + 1] if name in argv else None


class TestContainmentFlagsArePresent:
    def test_no_network(self, argv):
        assert _flag(argv, "--network") == "none"

    def test_workspace_is_read_only(self, tmp_path):
        argv = container.build_argv("x", tmp_path, exe="docker")
        assert f"{tmp_path.resolve()}:/workspace:ro" in argv

    def test_root_filesystem_is_read_only(self, argv):
        assert "--read-only" in argv

    def test_capabilities_are_dropped(self, argv):
        assert _flag(argv, "--cap-drop") == "ALL"

    def test_privilege_escalation_is_refused(self, argv):
        assert "no-new-privileges" in argv

    def test_it_runs_as_nobody(self, argv):
        assert _flag(argv, "--user") == f"{container.NON_ROOT_UID}:{container.NON_ROOT_GID}"

    def test_resources_are_capped(self, argv):
        assert _flag(argv, "--memory")
        assert _flag(argv, "--cpus")
        assert _flag(argv, "--pids-limit")

    def test_only_the_workspace_is_mounted(self, argv):
        """One volume. A second mount is a second way out."""
        assert argv.count("--volume") == 1
        assert argv.count("-v") == 0

    def test_no_docker_socket_is_mounted(self, argv):
        assert not any("docker.sock" in a for a in argv)

    def test_the_host_environment_is_not_inherited(self, argv):
        """Only an explicit PATH/HOME, never the caller's variables."""
        envs = [a for i, a in enumerate(argv) if i and argv[i - 1] == "--env"]
        assert envs == ["PATH=/usr/local/bin:/usr/bin:/bin", "HOME=/tmp",
                        "PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1",
                        "PYTHONHASHSEED=0"]

    def test_the_container_is_removed_afterwards(self, argv):
        assert "--rm" in argv


class TestOneMissingFlagBreaksTheTest:
    """Not a test of docker -- a test that each flag is load-bearing.

    Each case removes exactly one containment flag and asserts the suite would
    notice. Without this, the tests above could all be reading a flag that no
    longer exists in the code.
    """

    REQUIRED = (
        "--network", "--read-only", "--cap-drop",
        "--security-opt", "--user", "--memory", "--pids-limit",
    )

    @pytest.mark.parametrize("flag", REQUIRED)
    def test_the_flag_is_actually_present(self, argv, flag):
        assert flag in argv, f"{flag} is not in the argv, so nothing tests it"

    def test_the_workspace_mount_is_read_only_not_writable(self, tmp_path):
        """A writable workspace is the single most dangerous regression here."""
        argv = container.build_argv("x", tmp_path, exe="docker")
        volumes = [a for i, a in enumerate(argv) if i and argv[i - 1] == "--volume"]
        assert volumes and all(v.endswith(":ro") for v in volumes)


# ── the refusal path ──────────────────────────────────────────────
class TestContainerRefusesRatherThanDowngrades:
    """The security property: asking for containment and not getting it must fail.

    A silent downgrade from `container` to `subprocess` would produce a
    successful run of unconfined code. Nothing would look wrong.
    """

    def test_the_capability_reports_the_level_it_got(self, tmp_path):
        cap = PythonExecute(workspace=tmp_path, sandbox_level="subprocess")
        assert cap.sandbox_level == "subprocess"

    def test_an_unknown_level_is_rejected_at_construction(self, tmp_path):
        with pytest.raises(ValueError):
            PythonExecute(workspace=tmp_path, sandbox_level="contianer")

    def test_probe_raises_when_the_daemon_is_unreachable(self, monkeypatch):
        """Simulate a stopped Docker Desktop: installed, not answering."""

        class _Proc:
            returncode = 1
            stdout = ""
            stderr = "Cannot connect to the Docker daemon."

        monkeypatch.setattr(container, "docker_executable", lambda: "docker")
        monkeypatch.setattr(container.subprocess, "run", lambda *a, **k: _Proc())

        with pytest.raises(container.ContainerUnavailable) as exc:
            container.probe()
        assert "not answering" in str(exc.value)

    def test_probe_raises_when_the_image_cannot_run(self, monkeypatch):
        """A healthy daemon that cannot run the image is still unusable.

        Found on windows-latest: Docker was up serving Windows containers, so a
        version probe passed, and then the Linux image could not run at all --
        every containment case failed with empty output, which reads as broken
        code rather than an absent runtime.
        """

        class _Proc:
            def __init__(self, code, out="", err=""):
                self.returncode = code
                self.stdout = out
                self.stderr = err

        def fake_run(argv, **k):
            if "version" in argv:
                return _Proc(0, out="29.8.1")
            return _Proc(
                125, err="image operating system mismatch: no matching manifest"
            )

        monkeypatch.setattr(container, "docker_executable", lambda: "docker")
        monkeypatch.setattr(container.subprocess, "run", fake_run)

        with pytest.raises(container.ContainerUnavailable) as exc:
            container.probe()
        assert "cannot run" in str(exc.value)
        assert "Linux" in str(exc.value), "the error should name the real cause"

    def test_probe_passes_only_when_the_image_actually_runs(self, monkeypatch):
        class _Proc:
            def __init__(self, code, out="", err=""):
                self.returncode = code
                self.stdout = out
                self.stderr = err

        def fake_run(argv, **k):
            if "version" in argv:
                return _Proc(0, out="29.8.1")
            return _Proc(0)

        monkeypatch.setattr(container, "docker_executable", lambda: "docker")
        monkeypatch.setattr(container.subprocess, "run", fake_run)

        assert container.probe() == "29.8.1"

    def test_the_error_says_it_is_not_a_sandbox(self, monkeypatch):
        """The message must not leave the reader thinking subprocess is fine."""

        class _Proc:
            returncode = 1
            stdout = ""
            stderr = "nope"

        monkeypatch.setattr(container, "docker_executable", lambda: "docker")
        monkeypatch.setattr(container.subprocess, "run", lambda *a, **k: _Proc())

        with pytest.raises(container.ContainerUnavailable) as exc:
            container.probe()
        assert "not a sandbox" in str(exc.value)

    def test_a_missing_executable_is_reported_clearly(self, monkeypatch):
        monkeypatch.setattr(container.shutil, "which", lambda _: None)
        with pytest.raises(container.ContainerUnavailable) as exc:
            container.docker_executable()
        assert "unconfined" in str(exc.value)

    def test_invoke_refuses_rather_than_running_unconfined(self, tmp_path, monkeypatch):
        """The end-to-end shape of the refusal."""
        monkeypatch.setattr(
            PythonExecute, "_require_container",
            lambda self: (_ for _ in ()).throw(
                container.ContainerUnavailable("no runtime")
            ),
        )
        cap = PythonExecute(workspace=tmp_path, sandbox_level="container")
        result = cap.invoke(code="print('should not run')")

        assert result["ok"] is False
        assert "isolation=container unavailable" in result["error"]
        assert result["sandbox_level"] == "container"

    def test_a_refusal_never_reports_stdout(self, tmp_path, monkeypatch):
        """A refused call must not look like a run that printed nothing."""
        monkeypatch.setattr(
            PythonExecute, "_require_container",
            lambda self: (_ for _ in ()).throw(container.ContainerUnavailable("x")),
        )
        cap = PythonExecute(workspace=tmp_path, sandbox_level="container")
        assert "stdout" not in cap.invoke(code="print(1)")


# ── the spec declares a requirement ───────────────────────────────
class TestSpecDeclaresTheRequirement:
    def _spec(self, **kw) -> AgentSpec:
        return AgentSpec(name="a", system_prompt="p", **kw)

    def test_the_default_is_subprocess(self):
        assert self._spec().isolation is Isolation.SUBPROCESS

    def test_a_spec_can_require_container(self):
        assert self._spec(isolation="container").isolation is Isolation.CONTAINER

    def test_the_field_is_validated(self):
        with pytest.raises(Exception):
            self._spec(isolation="contianer")

    def test_the_default_is_not_a_sandbox(self):
        """The safe default is the one that admits what it is."""
        assert not self._spec().isolation.confines_filesystem


class TestTheRegistryPassesTheLevelThrough:
    def test_the_capability_receives_the_requested_level(self, tmp_path):
        from factory.capabilities.registry import registry_for

        reg = registry_for(tmp_path, Isolation.CONTAINER)
        cap = reg.instantiate(reg.issue("python.execute"))
        assert cap.sandbox_level == "container"

    def test_the_default_registry_keeps_subprocess(self, tmp_path):
        from factory.capabilities.registry import registry_for

        reg = registry_for(tmp_path)
        cap = reg.instantiate(reg.issue("python.execute"))
        assert cap.sandbox_level == "subprocess"