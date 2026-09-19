import { expect, test } from "vitest"
import { Effect, Layer, Schema } from "effect"
import { decodeGenerationStream } from "../src/suites/endpoint"
const event = (content: string | null, finish: string | null = null, id = "completion-one", model = "fixture-model") => `data: ${JSON.stringify({ id, model, choices: [{ index: 0, delta: content === null ? {} : { content }, finish_reason: finish }] })}\r\n\r\n`
test("validates complete SSE content, finish and request identity", async () => {
  const result = await Effect.runPromise(decodeGenerationStream(`: keepalive\r\n\r\n${event("HEL")}${event("LO")}${event(null, "stop")}data: [DONE]\r\n\r\n`, "fixture-model"))
  expect(result.text).toBe("HELLO")
  expect(result.chunks).toBe(2)
})
test.each([
  event("HELLO"),
  event("HELLO") + "data: [DONE]\n\n",
  event("HELLO") + event(null, "length") + "data: [DONE]\n\n",
  event("HELLO") + event("again", null, "another-request") + event(null, "stop") + "data: [DONE]\n\n",
  event("HELLO", null, "completion-one", "another-model") + event(null, "stop") + "data: [DONE]\n\n",
  event("HELLO") + event(null, "stop") + "data: [DONE]\n\n" + event("after"),
  "data: malformed\n\n",
])("rejects truncated, malformed or misattributed generation", async wire => {
  const result = await Effect.runPromise(decodeGenerationStream(wire, "fixture-model").pipe(Effect.either))
  expect(result._tag).toBe("Left")
})

test("endpoint engine executes requests and cancels its stream before a subsequent generation", async () => {
  const { FetchHttpClient } = await import("@effect/platform")
  const { endpointTests, EndpointTests } = await import("../src/suites/endpoint")
  let cancelled = false, completions = 0
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request) {
    if (new URL(request.url).pathname.endsWith("/models")) return Response.json({ data: [{ id: "fixture-model" }] })
    const body = await request.json() as { model: string; messages: { content: string }[] | string; stream?: boolean }
    if (body.model !== "fixture-model") return Response.json({ error: "unknown model" }, { status: 404 })
    if (typeof body.messages === "string") return Response.json({ error: "invalid messages" }, { status: 400 })
    if (body.stream && body.messages[0]!.content.startsWith("Count slowly")) return new Response(new ReadableStream({
      start(controller) { controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({ id: "completion-one", model: "fixture-model", choices: [{ index: 0, delta: { reasoning_content: "Let me count" } }] })}\n\n`)) }, cancel() { cancelled = true },
    }), { headers: { "content-type": "text/event-stream" } })
    if (body.stream) return new Response(event("HELLO") + event(null, "stop") + "data: [DONE]\n\n", { headers: { "content-type": "text/event-stream" } })
    completions++
    return Response.json({ id: "completion-one", model: "fixture-model", choices: [{ index: 0, message: { role: "assistant", content: "HELLO" }, finish_reason: "stop" }], usage: { prompt_tokens: 12, completion_tokens: 1 } })
  } })
  try {
    await Effect.runPromise(Effect.gen(function* () {
      const tests = yield* EndpointTests
      yield* tests.discover
      expect((yield* tests.generate).text).toBe("HELLO")
      expect((yield* tests.stream).text).toBe("HELLO")
      yield* tests.invalid
      yield* tests.cancelAndRetry
    }).pipe(Effect.provide(endpointTests(`http://127.0.0.1:${server.port}`, "fixture-model").pipe(Layer.provide(FetchHttpClient.layer)))))
    expect(completions).toBe(3)
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(cancelled).toBe(true)
  } finally { server.stop(true) }
})

test("accepts the live API contract omitting finish_reason on nonterminal events", async () => {
  const chunk = { id: "completion-one", model: "fixture-model", choices: [{ index: 0, delta: { content: "HELLO" } }] }
  const result = await Effect.runPromise(decodeGenerationStream(`data: ${JSON.stringify(chunk)}\n\n${event(null, "stop")}data: [DONE]\n\n`, "fixture-model"))
  expect(result.text).toBe("HELLO")
})

test("failed generation preserves a bounded redacted response without replaying the request", async () => {
  const { FetchHttpClient } = await import("@effect/platform")
  const { endpointTests, EndpointTests } = await import("../src/suites/endpoint")
  let calls = 0
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch() {
    calls++
    return new Response(calls === 1 ? "native tool failure Bearer secret-token" : "x".repeat(20 * 1024), { status: 500 })
  } })
  try {
    await Effect.runPromise(Effect.gen(function* () {
      const tests = yield* EndpointTests
      const first = yield* tests.generate.pipe(Effect.either)
      expect(first).toMatchObject({ _tag: "Left", left: { message: "Generation returned HTTP 500: native tool failure Bearer [REDACTED]" } })
      expect(calls).toBe(1)
      const second = yield* tests.generate.pipe(Effect.either)
      expect(second).toMatchObject({ _tag: "Left", left: { message: "Generation returned HTTP 500: Error response unavailable or larger than 16 Ki characters" } })
      expect(calls).toBe(2)
    }).pipe(Effect.provide(endpointTests(`http://127.0.0.1:${server.port}`, "fixture-model").pipe(Layer.provide(FetchHttpClient.layer)))))
  } finally { server.stop(true) }
})


test.each([false, true])("tool follow-up requires the newly supplied result, echo input=%s", async echoInput => {
  const { FetchHttpClient } = await import("@effect/platform")
  const { endpointTests, EndpointTests } = await import("../src/suites/endpoint")
  let key = "", calls = 0
  const requestSchema = Schema.Struct({ messages: Schema.Array(Schema.Struct({ role: Schema.String, content: Schema.NullOr(Schema.String) })) })
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request) {
    calls++
    const body = Schema.decodeUnknownSync(requestSchema)(await request.json())
    const tool = body.messages.find(message => message.role === "tool")
    if (!tool) key = body.messages[0]!.content!.match(/[a-f0-9-]{36}/)![0]
    else expect(tool.content).not.toBe(key)
    return Response.json({ id: `completion-${calls}`, model: "fixture-model", choices: [{ index: 0,
      message: tool ? { role: "assistant", content: echoInput ? key : tool.content } : { role: "assistant", content: null,
        tool_calls: [{ id: "lookup-call", type: "function", function: { name: "lab_lookup", arguments: JSON.stringify({ key }) } }] },
      finish_reason: tool ? "stop" : "tool_calls" }], usage: { prompt_tokens: 12, completion_tokens: 12 } })
  } })
  try {
    const result = await Effect.runPromise(EndpointTests.pipe(Effect.flatMap(tests => tests.tools), Effect.either,
      Effect.provide(endpointTests(`http://127.0.0.1:${server.port}`, "fixture-model").pipe(Layer.provide(FetchHttpClient.layer)))))
    expect(result._tag).toBe(echoInput ? "Left" : "Right")
    expect(calls).toBe(2)
  } finally { server.stop(true) }
})
