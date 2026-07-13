import { Otlp } from "@effect/opentelemetry"
import { FetchHttpClient } from "@effect/platform"
import { Layer, Effect } from "effect"

/**
 * The default OTLP base URL when motel is running locally.
 * motel listens on http://127.0.0.1:27686 and accepts traces at /v1/traces
 * and logs at /v1/logs.
 */
const MOTEL_BASE_URL = "http://127.0.0.1:27686"

/**
 * When `MAGNITUDE_OTEL_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) is
 * set, creates a unified OTLP layer that exports both spans AND logs to the
 * configured endpoint (defaults to motel's local URL).
 *
 * `@effect/rpc` creates a span for every RPC call, `@effect/platform` creates
 * HTTP spans, and `Effect.log*` calls create log records. With this layer
 * provided, all of them are exported via OTLP to motel instead of being
 * discarded by Effect's default no-op tracer/logger.
 *
 * When the env var is unset, returns `Layer.empty` so there is zero overhead.
 */
export const TracingLayer = Layer.unwrapEffect(
  Effect.gen(function* () {
    const endpoint =
      process.env.MAGNITUDE_OTEL_ENDPOINT ??
      process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT ??
      (process.env.MAGNITUDE_OTEL === "1" ? MOTEL_BASE_URL : undefined)

    if (!endpoint) return Layer.empty

    return Otlp.layerJson({
      baseUrl: endpoint,
      resource: { serviceName: "magnitude-acn" },
      tracerExportInterval: "1 seconds",
      loggerExportInterval: "1 seconds",
    }).pipe(
      Layer.provide(FetchHttpClient.layer),
    )
  }),
)
