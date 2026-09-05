import { Otlp } from "@effect/opentelemetry"
import { FetchHttpClient } from "@effect/platform"
import { Layer, Logger } from "effect"

/**
 * Replaces the default console logger with a no-op so logs go to OTLP only.
 */
const NoConsoleLogger = Logger.replace(Logger.defaultLogger, Logger.none)

const MOTEL_BASE_URL = "http://127.0.0.1:27686"

/**
 * Reads the OTLP endpoint from environment variables.
 * Wrapped in try/catch so this is safe to call in environments
 * where `process.env` is not available (browser).
 */
const readOtelEndpoint = (): string | undefined => {
  try {
    return (
      process.env.MAGNITUDE_OTEL_ENDPOINT ??
      process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT ??
      (process.env.MAGNITUDE_OTEL === "1" ? MOTEL_BASE_URL : undefined)
    )
  } catch {
    return undefined
  }
}

export interface MakeTracingLayerOptions {
  readonly endpoint?: string
}

/**
 * OTLP layer providing both tracer and logger.
 *
 * When `MAGNITUDE_OTEL=1` (or `MAGNITUDE_OTEL_ENDPOINT` is set), all
 * spans and `Effect.log*` records are exported to motel (or any OTLP
 * collector at the configured endpoint).
 *
 * `Layer.empty` when disabled — zero overhead.
 *
 * Consumers should add this layer to their runtime if they want tracing.
 * It is NOT included in the core SDK protocol layer.
 */
export const makeTracingLayer = (options?: MakeTracingLayerOptions): Layer.Layer<never, never, never> => {
  const otelEndpoint = options?.endpoint ?? readOtelEndpoint()
  return otelEndpoint
    ? Layer.mergeAll(
        Otlp.layerJson({
          baseUrl: otelEndpoint,
          resource: { serviceName: "magnitude-cli" },
          tracerExportInterval: "1 seconds",
          loggerExportInterval: "1 seconds",
        }).pipe(Layer.provide(FetchHttpClient.layer)),
        NoConsoleLogger,
      )
    : Layer.empty
}

export const TracingLayer: Layer.Layer<never, never, never> = makeTracingLayer()
