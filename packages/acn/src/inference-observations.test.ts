import { Effect, Option, Runtime, Schedule, Schema, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { InferenceObservationGroupIdSchema } from "@magnitudedev/acn-protocol"
import { makeInferenceObservations, OBSERVATION_GROUP_HEADER, OBSERVATION_ID_HEADER } from "./inference-observations"
import { InferenceGatewayFailed, proxyOpenAiInferenceRequest } from "./inference-gateway"
import * as HttpServerResponse from "@effect/platform/HttpServerResponse"

const group = Schema.decodeUnknownSync(InferenceObservationGroupIdSchema)(crypto.randomUUID())
const otherGroup = Schema.decodeUnknownSync(InferenceObservationGroupIdSchema)(crypto.randomUUID())
const request = (id: string = crypto.randomUUID(), groupId = group) => new Request(
  "http://127.0.0.1:10100/inference/v1/chat/completions", {
    method: "POST", headers: { [OBSERVATION_ID_HEADER]: id, [OBSERVATION_GROUP_HEADER]: groupId },
  },
)
const sse = (body: BodyInit) => new Response(body, {
  status: 201, statusText: "Created", headers: { "content-type": "text/event-stream", "x-test": "preserved" },
})
const frame = (value: unknown) => `data: ${JSON.stringify(value)}\n\n`
const run = <A>(program: Effect.Effect<A, unknown, import("effect").Scope.Scope>) =>
  Effect.runPromise(Effect.scoped(program))

describe("optional inference observation", () => {
  it("preserves bytes, status and headers, even across every UTF-8/SSE split", async () => {
    const wire = frame({ choices: [{ delta: { content: "private text 🦋" } }], progress: { phase: "model_loading", fraction: 0.5, private: "discard" } }) +
      frame({ progress: { phase: "prefill", cached_tokens: 3, completed_tokens: 5, total_tokens: 10 } }) +
      frame({ timings: { prompt_ms: 1, time_to_first_token_ms: 2, predicted_n: 3, predicted_ms: 4, predicted_per_second: 750 } }) + "data: [DONE]\n\n"
    const bytes = new TextEncoder().encode(wire)
    await run(Effect.gen(function* () {
      const service = yield* makeInferenceObservations
      const input = sse(new ReadableStream({ start(c) { for (const byte of bytes) c.enqueue(Uint8Array.of(byte)); c.close() } }))
      const output = yield* service.observe(request(), input)
      expect(output.status).toBe(201)
      expect(output.statusText).toBe("Created")
      expect(output.headers.get("x-test")).toBe("preserved")
      expect(yield* Effect.promise(() => output.text())).toBe(wire)
      const snapshot = yield* service.read(group)
      expect(snapshot).toHaveLength(1)
      expect(snapshot[0]!.state).toBe("Ended")
      expect(Option.getOrThrow(snapshot[0]!.progress)).toEqual({ phase: "prefill", cached_tokens: 3, completed_tokens: 5, total_tokens: 10 })
      expect(Option.getOrThrow(snapshot[0]!.timings).predicted_n).toBe(3)
      expect(JSON.stringify(snapshot)).not.toContain("private")
      expect(yield* service.read(otherGroup)).toEqual([])
    }))
  })

  it.each(["data: invalid\n\n", "data: " + "x".repeat(1024 * 1024 + 1) + "\n\n",
    frame({ progress: { phase: "prefill", completed_tokens: "bad" } }),
    'data: {"progress":{"phase":"generating"}}',
  ])("never changes malformed, oversized or incomplete stream bytes", async wire => {
    await run(Effect.gen(function* () {
      const service = yield* makeInferenceObservations
      const output = yield* service.observe(request(), sse(wire))
      expect(yield* Effect.promise(() => output.text())).toBe(wire)
      const [entry] = yield* service.read(group)
      expect(entry!.state).toBe("Ended")
      expect(Option.isNone(entry!.progress)).toBe(true)
    }))
  })

  it("does not observe absent/invalid identity, other paths, JSON or compressed responses", async () => {
    await run(Effect.gen(function* () {
      const service = yield* makeInferenceObservations
      const input = sse(frame({ progress: { phase: "generating" } }))
      for (const req of [new Request(request().url), request("invalid"), new Request("http://localhost/other", request())]) {
        expect(yield* service.observe(req, input)).toBe(input)
      }
      const json = Response.json({ progress: { phase: "generating" } })
      expect(yield* service.observe(request(), json)).toBe(json)
      const compressed = sse("opaque")
      compressed.headers.set("content-encoding", "gzip")
      expect(yield* service.observe(request(), compressed)).toBe(compressed)
      expect(yield* service.read(group)).toEqual([])
    }))
  })

  it("isolates concurrent groups and refuses duplicate IDs and exhausted capacity", async () => {
    await run(Effect.gen(function* () {
      const service = yield* makeInferenceObservations
      const first = request()
      yield* service.observe(first, sse(""))
      const duplicate = sse("duplicate")
      expect(yield* service.observe(first, duplicate)).toBe(duplicate)
      for (let i = 1; i < 128; i++) yield* service.observe(request(crypto.randomUUID(), otherGroup), sse(""))
      const overflow = sse("overflow")
      expect(yield* service.observe(request(), overflow)).toBe(overflow)
      expect(yield* service.read(group)).toHaveLength(1)
      expect(yield* service.read(otherGroup)).toHaveLength(127)
    }))
  })

  it("propagates cancellation to the original response and closes its observation", async () => {
    let cancelled = false
    await run(Effect.gen(function* () {
      const service = yield* makeInferenceObservations
      const original = sse(new ReadableStream({ pull(c) { c.enqueue(new TextEncoder().encode(frame({ progress: { phase: "generating" } }))) }, cancel() { cancelled = true } }))
      const output = yield* service.observe(request(), original)
      const reader = output.body!.getReader()
      yield* Effect.promise(() => reader.read())
      yield* Effect.promise(() => reader.cancel("caller interrupted"))
      expect(cancelled).toBe(true)
      expect((yield* service.read(group))[0]!.state).toBe("Ended")
    }))
  })

  it("expires ended observations without evicting active work", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const service = yield* makeInferenceObservations
      const ended = yield* service.observe(request(), sse(""))
      yield* Effect.promise(() => ended.text())
      const active = yield* service.observe(request(), sse(new ReadableStream()))
      yield* TestClock.adjust("61 seconds")
      expect(yield* service.read(group)).toHaveLength(1)
      expect((yield* service.read(group))[0]!.state).toBe("Active")
      yield* Effect.promise(() => active.body!.cancel())
    })).pipe(Effect.provide(TestContext.TestContext)))
  })

  it.each(["\r\n", "\r"])("observes SSE with %j line endings without rewriting it", async newline => {
    await run(Effect.gen(function* () {
      const service = yield* makeInferenceObservations
      const wire = `: heartbeat${newline}data: {"progress":{"phase":"generating"}}${newline}${newline}`
      const output = yield* service.observe(request(), sse(wire))
      expect(yield* Effect.promise(() => output.text())).toBe(wire)
      expect(Option.getOrThrow((yield* service.read(group))[0]!.progress)).toEqual({ phase: "generating" })
    }))
  })

  it("ends observations on upstream failure without manufacturing a successful response", async () => {
    await run(Effect.gen(function* () {
      const service = yield* makeInferenceObservations
      const failure = new Error("upstream disconnected")
      const original = sse(new ReadableStream({ start(controller) { controller.error(failure) } }))
      const output = yield* service.observe(request(), original)
      const result = yield* Effect.tryPromise(() => output.text()).pipe(Effect.either)
      expect(result._tag).toBe("Left")
      expect((yield* service.read(group))[0]!.state).toBe("Ended")
      expect(Option.isNone((yield* service.read(group))[0]!.timings)).toBe(true)
    }))
  })

  it("cancels the source when a real HTTP consumer aborts the observed response", async () => {
    await run(Effect.gen(function* () {
      const service = yield* makeInferenceObservations
      const runRequest = Runtime.runPromise(yield* Effect.runtime<never>())
      let cancelled = false
      const server = yield* Effect.acquireRelease(Effect.sync(() => Bun.serve({
        hostname: "127.0.0.1", port: 0,
        fetch: source => runRequest(service.observe(source, sse(new ReadableStream({
          async pull(controller) {
            await Bun.sleep(5)
            if (!cancelled) controller.enqueue(new TextEncoder().encode(frame({ progress: { phase: "generating" } })))
          },
          cancel() { cancelled = true },
        })))),
      })), server => Effect.promise(async () => { await server.stop(true) }))
      const controller = new AbortController()
      const response = yield* Effect.promise(() => fetch(`http://127.0.0.1:${server.port}/inference/v1/chat/completions`, {
        method: "POST", signal: controller.signal,
        headers: { [OBSERVATION_ID_HEADER]: crypto.randomUUID(), [OBSERVATION_GROUP_HEADER]: group },
      }))
      yield* Effect.promise(() => response.body!.getReader().read())
      controller.abort()
      yield* Effect.suspend(() => cancelled ? Effect.void : Effect.sleep("10 millis").pipe(
        Effect.andThen(Effect.suspend(() => cancelled ? Effect.void : Effect.fail("Source was not cancelled"))),
      )).pipe(Effect.retry({ times: 100 }), Effect.timeout("2 seconds"))
      expect(cancelled).toBe(true)
      expect((yield* service.read(group))[0]!.state).toBe("Ended")
    }))
  })

  it.each([false, true])("propagates HTTP proxy cancellation with observation enabled=%s", async observed => {
    await run(Effect.gen(function* () {
      const service = yield* makeInferenceObservations
      const runRequest = Runtime.runPromise(yield* Effect.runtime<never>())
      let cancelled = false
      const upstream = yield* Effect.acquireRelease(Effect.sync(() => Bun.serve({
        hostname: "127.0.0.1", port: 0,
        fetch: () => sse(new ReadableStream({
          async pull(controller) {
            await Bun.sleep(5)
            if (!cancelled) controller.enqueue(new TextEncoder().encode(frame({ progress: { phase: "generating" } })))
          },
          cancel() { cancelled = true },
        })),
      })), server => Effect.promise(async () => { await server.stop(true) }))
      const proxy = yield* Effect.acquireRelease(Effect.sync(() => Bun.serve({
        hostname: "127.0.0.1", port: 0,
        async fetch(source) {
          const response = await runRequest(Effect.tryPromise({
            try: signal => proxyOpenAiInferenceRequest(source, {
              origin: new URL(`http://127.0.0.1:${upstream.port}`), clientOptions: {},
            }, fetch, signal),
            catch: cause => new InferenceGatewayFailed({ cause }),
          }))
          const presented = observed ? await runRequest(service.observe(source, response)) : response
          return HttpServerResponse.toWeb(HttpServerResponse.fromWeb(presented))
        },
      })), server => Effect.promise(async () => { await server.stop(true) }))
      const controller = new AbortController()
      const response = yield* Effect.promise(() => fetch(`http://127.0.0.1:${proxy.port}/inference/v1/chat/completions`, {
        method: "POST", signal: controller.signal,
        headers: { [OBSERVATION_ID_HEADER]: crypto.randomUUID(), [OBSERVATION_GROUP_HEADER]: group },
      }))
      yield* Effect.promise(() => response.body!.getReader().read())
      controller.abort()
      yield* Effect.suspend(() => cancelled ? Effect.void : Effect.fail("Upstream was not cancelled"))
        .pipe(Effect.retry({ times: 100, schedule: Schedule.spaced("10 millis") }), Effect.timeout("2 seconds"))
      expect(cancelled).toBe(true)
    }))
  })
})
