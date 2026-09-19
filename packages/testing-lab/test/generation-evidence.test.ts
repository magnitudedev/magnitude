import { FetchHttpClient } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { expect, test } from "vitest"
import { ExecutionTraceId, NativeExecution } from "../src/execution-telemetry"
import { correlateGeneration, observeGeneration } from "../src/generation-evidence"
import { Generation } from "../src/suites/endpoint"

const generation = Generation.make({ requestId: "public-completion", model: "fixture-model", text: "HELLO", chunks: 1 })
const record = (traceId: string, patch: Record<string, unknown> = {}) => Schema.decodeUnknownSync(NativeExecution)({
  traceId, model: "fixture-model", workerPid: 42, workerGeneration: "2", requestId: "7", allocations: [{ kind: "host", model_bytes: 1024 }], ...patch,
})
test("correlation rejects a different model, missing trace or ambiguous native requests", async () => {
  const trace = ExecutionTraceId.make("a".repeat(32))
  for (const records of [[], [record("b".repeat(32))], [record(trace, { model: "different" })], [record(trace), record(trace, { requestId: "8" })]]) {
    expect((await Effect.runPromise(correlateGeneration(trace, generation, records).pipe(Effect.either)))._tag).toBe("Left")
  }
  const result = await Effect.runPromise(correlateGeneration(trace, generation, [record("b".repeat(32)), record(trace)]))
  expect(result.generation.requestId).toBe("public-completion")
  expect(result.native.requestId).toBe("7")
})

test("public generation carries a unique trace and preserves both public and native request identities", async () => {
  const records: NativeExecution[] = []
  const headers: string[] = []
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
    const header = request.headers.get("traceparent") ?? ""
    headers.push(header)
    if (!/^00-[a-f0-9]{32}-[a-f0-9]{16}-01$/.test(header)) return new Response(null, { status: 400 })
    records.push(record(header.split("-")[1]!))
    return Response.json({ id: "public-completion", model: "fixture-model", choices: [{ index: 0,
      message: { role: "assistant", content: "HELLO" }, finish_reason: "stop" }], usage: { prompt_tokens: 12, completion_tokens: 1 } })
  } })
  try {
    for (let index = 0; index < 2; index++) {
      let reads = 0
      const result = await Effect.runPromise(observeGeneration(`http://127.0.0.1:${server.port}`, "fixture-model", {
        observations: Effect.sync(() => ++reads <= 2 ? [] : records),
      }).pipe(Effect.provide(FetchHttpClient.layer)))
      expect(result.generation).toEqual(generation)
      expect(result.native.traceId).toBe(headers[index]!.split("-")[1])
    }
    expect(headers[0]).not.toBe(headers[1])
    const cancelled = await Effect.runPromise(observeGeneration(`http://127.0.0.1:${server.port}`, "fixture-model", {
      observations: Effect.succeed([]),
    }).pipe(Effect.timeoutOption("150 millis"), Effect.provide(FetchHttpClient.layer)))
    expect(Option.isNone(cancelled)).toBe(true)
    expect(headers).toHaveLength(3) // Waiting for telemetry must not repeat generation.
  } finally { await server.stop(true) }
})
