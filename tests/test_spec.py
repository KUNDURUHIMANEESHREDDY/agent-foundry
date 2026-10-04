from __future__ import annotations

import pytest

from factory.spec.agent_spec import AgentSpec, Limits, ModelRef
from factory.spec.loader import SpecError, load_spec, parse_spec

VALID = """
name: tester
version: 1.0.0
system_prompt: You are a test agent.
capabilities:
  - filesystem.read
  - filesystem.read
model:
  provider: fake
  name: none
limits:
  max_steps: 3
"""


class TestLoading:
    def test_parses_valid_spec(self):
        spec = parse_spec(VALID)
        assert spec.name == "tester"
        assert spec.model.provider == "fake"

    def test_deduplicates_capabilities(self):
        spec = parse_spec(VALID)
        assert spec.capabilities == ["filesystem.read"]

    def test_rejects_empty_spec(self):
        with pytest.raises(SpecError, match="empty"):
            parse_spec("")

    def test_rejects_non_mapping(self):
        with pytest.raises(SpecError, match="must be a mapping"):
            parse_spec("- a\n- b\n")

    def test_reports_all_validation_errors(self):
        with pytest.raises(SpecError) as exc:
            parse_spec("name: Bad_Name!\nsystem_prompt: hi\nversion: nope\n")
        msg = str(exc.value)
        assert "name" in msg
        assert "version" in msg

    def test_missing_file(self, tmp_path):
        with pytest.raises(SpecError, match="not found"):
            load_spec(tmp_path / "nope.yaml")

    def test_loads_from_disk(self, tmp_path):
        p = tmp_path / "a.yaml"
        p.write_text(VALID, encoding="utf-8")
        assert load_spec(p).name == "tester"

    def test_invalid_yaml_is_reported_as_such(self):
        with pytest.raises(SpecError, match="Invalid YAML"):
            parse_spec("name: [unclosed\n")


class TestSpecInvariants:
    def test_wants_reflects_capabilities(self):
        spec = parse_spec(VALID)
        assert spec.wants("filesystem.read")
        assert not spec.wants("shell.execute")

    def test_extra_fields_are_rejected(self):
        with pytest.raises(SpecError, match="(?i)extra"):
            parse_spec(VALID + "\nsurprise: yes\n")

    def test_defaults_are_applied(self):
        spec = parse_spec(VALID)
        assert spec.memory.type == "memory"
        assert spec.limits.max_tool_calls > 0
        assert spec.model.temperature == 0.0

    def test_rejects_negative_max_steps(self):
        with pytest.raises(SpecError, match="max_steps"):
            parse_spec(VALID.replace("max_steps: 3", "max_steps: 0"))
