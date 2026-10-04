"""Langfuse API-shape notes, gathered empirically from a live org.

Recorded because two of these cost real debugging time and neither is
discoverable from the SDK's type hints.

LIVE API SURFACE (SDK 4.15.6, org created after 2026-09-16)
    GET  /api/public/traces/{id}    -> 410 LEGACY_API_UNAVAILABLE
    GET  /api/public/v2/scores      -> 410 LEGACY_API_UNAVAILABLE
    GET  /api/public/v2/observations -> works  (from_start_time/to_start_time)
    GET  /api/public/v3/scores       -> works  (from_timestamp/to_timestamp)

    Read observations via client.api.observations.get_many(...)
    Read scores       via client.api.scores_v3.get_many_v3(...)
    NOT client.api.scores.get_many(...)  -- v2, 410s

    Note the inconsistent time-parameter names between the two endpoints.
    Passing from_start_time to scores_v3 raises TypeError.

TRACING
    The global OTel provider means client.shutdown() poisons every later
    client: spans queue forever and flush() blocks in queue.join().

    Langfuse stores an observation's trace id under a different id space than
    OTel uses. Agent run ids must be mapped to 32 lowercase hex chars before
    they are accepted as trace_context.trace_id.
"""

OBSERVATIONS_V2 = "client.api.observations.get_many"
SCORES_V3 = "client.api.scores_v3.get_many_v3"
LEGACY = "client.api.trace.get / client.api.scores.get_many -> HTTP 410"
