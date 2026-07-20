---
applies_to:
  - inference/crates/icn-server/src/telemetry.rs
  - inference/crates/icn-server/src/main.rs
  - inference/crates/icn-api/src/lib.rs
  - inference/crates/icn-engine/src/**
---

# ICN observability

ICN emits structured diagnostics through Rust's `tracing` facade and exports both OpenTelemetry
traces and OpenTelemetry log records from the server binary. Domain crates create spans and events
without knowing the exporter. The `magnitude-icn` binary exclusively owns subscriber installation,
OTLP configuration, batching, resource identity, and shutdown flushing.

## Service identity and activation

The OpenTelemetry resource service name is exactly `magnitude-icn`. The resource also reports the
ICN package version and executable identity.

Local Motel export is enabled by `MAGNITUDE_OTEL=1` and targets
`http://127.0.0.1:27686/v1/traces` and `/v1/logs`. `MAGNITUDE_OTEL_ENDPOINT` selects another OTLP
HTTP base URL. Standard generic and signal-specific OTLP endpoint variables are also honored.
Export uses OTLP HTTP/JSON and batches each signal with a one-second scheduled delay. When OTLP is
disabled, structured diagnostics remain available on standard error without constructing providers
or exporters.

The server flushes and shuts down both providers during orderly process exit. Telemetry delivery is
best effort: an unavailable collector must not change inference results or keep ICN alive after its
normal shutdown boundary.

## Traces and logs

Every HTTP request receives an information-level server span containing its method, matched route,
path without query parameters, and protocol version. Incoming W3C trace context is extracted before
the span is created, allowing an ACN request to remain the parent of ICN work. Response completion
emits status and latency information.

Important model and inference operations use function spans. Context is carried explicitly across
blocking task and dedicated executor-thread boundaries; thread changes must not detach request work
from its originating completion span. Continuous batching may combine several requests, so a shared
native batch must not be falsely represented as the child of only one request.

`tracing` events are exported as real OpenTelemetry log records in addition to appearing as span
events where applicable. Log records retain active trace and span correlation. Exporter-internal
HTTP and OpenTelemetry diagnostics are excluded from the OTLP log pipeline to prevent recursive
telemetry.

## Data policy

Instrumentation records stable operational metadata such as model and completion identifiers,
token counts, finish reason, queue duration, prompt duration, decode duration, cache behavior,
backend selection, status, and stable error diagnostics.

Instrumentation must never record prompts, generated or reasoning text, image contents,
authorization values, request headers, token arrays, tool arguments, full request structures, or
query strings. Function instrumentation therefore skips arguments by default and opts safe fields
in explicitly. Per-token spans are prohibited; token activity belongs in aggregate fields, metrics,
or bounded lifecycle events.

The server disables the native backend's unbounded diagnostic callback. Per-layer tensor placement,
graph construction, and device initialization dumps are not service telemetry and must not be
forwarded to standard error or OTLP. ICN emits bounded structured summaries and errors at its own
operation boundaries instead.

## Acceptance criteria

- With `MAGNITUDE_OTEL=1`, Motel lists traces and logs under service `magnitude-icn`.
- One ICN request produces a server span and correlated log records without exposing request
  content or authorization material.
- A valid incoming `traceparent` becomes the parent of the ICN server span.
- Completion work remains correlated across Axum, `spawn_blocking`, and the model executor command
  boundary.
- Trace and log providers flush during graceful shutdown.
- With OTLP disabled, ICN remains functional and retains standard-error diagnostics.
- Collector failure never changes an HTTP or inference domain result.
