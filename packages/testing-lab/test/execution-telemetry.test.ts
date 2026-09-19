import { Effect, Option } from "effect"
import { expect, test } from "vitest"
import { decodeNativeExecutions, executionTelemetry } from "../src/execution-telemetry"

const traceId = "a".repeat(32)
const allocation = [{ kind: "device", backend: "Metal", physical_id: null, native_index: 1, model_bytes: 4096 }]
const attribute = (key: string, value: string) => ({ key, value: { stringValue: value } })
const envelope = (patch: Record<string, unknown> = {}, service = "magnitude-icn") => ({ resourceLogs: [{
  resource: { attributes: [attribute("service.name", service)] }, scopeLogs: [{ logRecords: [{
    traceId, body: { stringValue: "arbitrary unretained body" }, attributes: [attribute("event.name", "icn.inference.completed"),
      attribute("model.id", "fixture-model"), attribute("native.target.allocations", JSON.stringify(allocation)),
      { key: "worker.pid", value: { intValue: "42" } }, { key: "worker.generation", value: { intValue: "2" } },
      { key: "worker.request.id", value: { intValue: "7" } }, attribute("unreviewed", "must-not-be-retained")], ...patch,
  }] }],
}] })

test("retains only native completion identity and typed target allocations", async () => {
  const records = await Effect.runPromise(decodeNativeExecutions(envelope()))
  expect(records).toEqual([{ traceId, model: "fixture-model", workerPid: 42, workerGeneration: "2", requestId: "7", allocations: allocation, modules: Option.none() }])
  expect(await Effect.runPromise(decodeNativeExecutions(envelope({}, "other-service")))).toEqual([])
  expect(await Effect.runPromise(decodeNativeExecutions(envelope({ attributes: [attribute("event.name", "unrelated")] })))).toEqual([])
  for (const changed of [envelope({ traceId: "0".repeat(32) }), envelope({ traceId: "unrelated" }), envelope({ attributes: [
    ...envelope().resourceLogs[0]!.scopeLogs[0]!.logRecords[0]!.attributes, attribute("worker.pid", "changed"),
  ] })]) expect((await Effect.runPromise(decodeNativeExecutions(changed).pipe(Effect.either)))._tag).toBe("Left")
})

test("retains loaded module evidence while rejecting malformed supplied observations", async () => {
  const modules = [{ name: "libggml-metal.dylib", sha256: "b".repeat(64), bytes: 1234 }]
  const withModules = (value: string) => envelope({ attributes: [...envelope().resourceLogs[0]!.scopeLogs[0]!.logRecords[0]!.attributes,
    attribute("native.backend.modules", value)] })
  const records = await Effect.runPromise(decodeNativeExecutions(withModules(JSON.stringify(modules))))
  expect(records[0]!.modules).toEqual(Option.some(modules))
  for (const value of ["malformed", JSON.stringify([{ ...modules[0], name: "../other.dylib" }]), JSON.stringify([{ ...modules[0], sha256: "invalid" }]), " ".repeat(16 * 1024 + 1)]) {
    expect((await Effect.runPromise(decodeNativeExecutions(withModules(value)).pipe(Effect.either)))._tag).toBe("Left")
  }
})

test("collector isolates its routes, deduplicates delivery and rejects conflicting evidence", async () => {
  let endpoint = ""
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const collector = yield* executionTelemetry()
    endpoint = collector.endpoint
    const send = (value: unknown, route = `${endpoint}/v1/logs`) => Effect.tryPromise(() => fetch(route, {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(value),
    }))
    expect((yield* send(envelope(), new URL("/v1/logs", endpoint).href)).status).toBe(404)
    const replies = yield* Effect.all([send(envelope()), send(envelope()), send(envelope({ traceId: "b".repeat(32) }))], { concurrency: "unbounded" })
    expect(replies.map(reply => reply.status)).toEqual([200, 200, 200])
    const records = yield* collector.observations
    expect(records).toHaveLength(2)
    expect(records.map(record => record.traceId).sort()).toEqual([traceId, "b".repeat(32)])
    const conflicting = envelope()
    conflicting.resourceLogs[0]!.scopeLogs[0]!.logRecords[0]!.attributes[1] = attribute("model.id", "different-model")
    expect((yield* send(conflicting)).status).toBe(400)
    expect((yield* collector.observations.pipe(Effect.either))._tag).toBe("Left")
  })))
  await expect(fetch(`${endpoint}/v1/logs`, { method: "POST" })).rejects.toThrow()
})
