"""Capture exporter and shared client for the Langfuse tests.

WHY ONE SHARED CLIENT
    Langfuse registers a *global* OpenTelemetry tracer provider on first client
    construction. Calling `shutdown()` kills that provider's batch processor, so
    any later client enqueues spans that nobody drains and `flush()` blocks
    forever in `queue.join()`. Building a client per test therefore hangs the
    suite. One session-scoped client, never shut down until the session ends.

    Spans are cleared between tests instead, so assertions stay isolated.
"""

from __future__ import annotations

import json
from typing import Any

from opentelemetry.sdk.trace.export import SpanExportResult

from factory.tracing.langfuse import LangfuseSettings, build_client


class CapturingExporter:
    """Collects finished OTel spans instead of shipping them anywhere."""

    def __init__(self) -> None:
        self.spans: list[Any] = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    # ── query helpers ───────────────────────────────────────────

    def names(self) -> list[str]:
        return [s.name for s in self.spans]

    def find(self, name: str):
        for s in self.spans:
            if s.name == name:
                return s
        return None

    def find_all(self, prefix: str) -> list[Any]:
        return [s for s in self.spans if s.name.startswith(prefix)]

    def attrs(self, name: str) -> dict[str, Any]:
        span = self.find(name)
        return dict(span.attributes) if span else {}

    def attr(self, name: str, key: str) -> Any:
        return self.attrs(name).get(key)

    def json_attr(self, name: str, key: str) -> Any:
        raw = self.attr(name, key)
        if raw is None:
            return None
        return json.loads(raw)

    def meta(self, name: str, key: str) -> Any:
        """Langfuse flattens metadata into `...metadata.<key>` attributes,
        it does not store one JSON blob. Verified against SDK 4.15.6."""
        return self.attr(name, f"langfuse.observation.metadata.{key}")

    def children_of(self, span) -> list[Any]:
        """Children in OTel-id space. Langfuse display ids are a different space."""
        return [s for s in self.spans if s.parent and s.parent.span_id == span.context.span_id]

    def otel_trace_ids(self) -> set[int]:
        return {s.context.trace_id for s in self.spans}

    def reset(self) -> None:
        self.spans.clear()


def make_test_client(exporter: CapturingExporter):
    """A real Langfuse client wired to a local exporter, never to a network host."""
    settings = LangfuseSettings(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        # Unroutable on purpose: the exporter is local, so nothing is sent.
        host="http://127.0.0.1:1",
    )
    return build_client(settings, span_exporter=exporter)
