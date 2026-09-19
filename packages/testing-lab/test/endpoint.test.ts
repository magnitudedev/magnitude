import { expect, test } from "vitest"
import { Effect, Layer } from "effect"
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
