import { HttpClient, HttpClientRequest } from "@effect/platform"
import { Context, Effect, Layer, Schedule, Schema } from "effect"
import { randomBytes } from "node:crypto"
import { AssertionFailure, InfrastructureFailure } from "./domain"
import { executionTelemetry, ExecutionTraceId, NativeExecution } from "./execution-telemetry"
import { endpointTests, EndpointTests, Generation } from "./suites/endpoint"

export const GenerationExecution = Schema.Struct({ generation: Generation, native: NativeExecution })
type Collector = Effect.Effect.Success<ReturnType<typeof executionTelemetry>>

/** Trace correlation is one part of backend verification, not a substitute for module/device checks. */
export const correlateGeneration = (trace: typeof ExecutionTraceId.Type, generation: Generation, records: readonly NativeExecution[]) => Effect.gen(function* () {
  const matches = records.filter(record => record.traceId === trace)
  if (matches.length !== 1) return yield* new AssertionFailure({ message: "Expected exactly one native completion for the generation trace" })
  if (matches[0]!.model !== generation.model) return yield* new AssertionFailure({ message: "Native completion model differs from the public generation" })
  return GenerationExecution.make({ generation, native: matches[0]! })
})

/** Generate through the app's public endpoint with a fresh trace, then wait for its native exporter. */
export const observeGeneration = (origin: string, model: string, collector: Pick<Collector, "observations">) => Effect.scoped(Effect.gen(function* () {
  const http = yield* HttpClient.HttpClient
  const traceId = ExecutionTraceId.make(randomBytes(16).toString("hex"))
  const traceparent = `00-${traceId}-${randomBytes(8).toString("hex")}-01`
  const client = http.pipe(HttpClient.mapRequest(HttpClientRequest.setHeader("traceparent", traceparent)), HttpClient.withTracerPropagation(false))
  const context = yield* Layer.build(endpointTests(origin, model).pipe(Layer.provide(Layer.succeed(HttpClient.HttpClient, client))))
  const generation = yield* Context.get(context, EndpointTests).generate
  yield* collector.observations.pipe(Effect.map(records => records.some(record => record.traceId === traceId)),
    Effect.repeat({ until: observed => observed, schedule: Schedule.spaced("100 millis") }),
    Effect.timeoutFail({ duration: "10 seconds", onTimeout: () => new InfrastructureFailure({ operation: "execution-telemetry", message: "No native completion arrived for the generation trace" }) }),
  )
  return yield* correlateGeneration(traceId, generation, yield* collector.observations)
}))
