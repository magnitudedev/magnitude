import { expect, it } from "vitest"
import { Effect, Option } from "effect"
import { makeUsageFetch, makeUsageObservation, makeUsageWebSocket } from "./serving-usage-observer"
import type { ServingUsage, ServingUsageRecord } from "./serving-usage"
const timings = { cache_n: 40, prompt_n: 60, predicted_n: 20, predicted_ms: 1000, time_to_first_token_ms: 150, parser_ms: 0, prompt_ms: 20, predicted_per_second: 20, predicted_per_token_ms: 50, prompt_per_second: 3000, prompt_per_token_ms: 1, sampler_ms: 0 }
const chat = { model: "magnitude-local/model-a", usage: { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120, prompt_tokens_details: { cached_tokens: 40 } }, timings }
const responsesUsage = { input_tokens: 100, input_tokens_details: { cached_tokens: 40 }, output_tokens: 20, output_tokens_details: { reasoning_tokens: 0 }, total_tokens: 120 }
const sink = () => {
  const records: ServingUsageRecord[] = []
  const service: ServingUsage = { record: value => Effect.sync(() => { records.push(value) }), read: () => Effect.succeed({ _tag: "Unavailable", message: "test" }) }
  return { records, service }
}
it("uses terminal timings without a separate usage block and records once", () => {
  const observation = makeUsageObservation({ streaming: true, startedAt: 0, now: () => 1500 })
  observation.push({ model: "model-a", choices: [{ delta: { content: "hello" }, finish_reason: null }], timings })
  observation.push({ model: "model-a", choices: [{ delta: {}, finish_reason: "stop" }], timings })
  expect(Option.getOrThrow(observation.finish())).toMatchObject({ input: 100, cached: 40, output: 20, generationMs: 1000, firstTokenMs: 150, complete: true })
  expect(observation.finish()).toEqual(Option.none())
})
it("does not mistake intermediate timing for terminal evidence", () => {
  const observation = makeUsageObservation({ streaming: true, startedAt: 0 })
  observation.push({ model: "model-a", choices: [{ delta: { content: "partial" }, finish_reason: null }], timings })
  expect(Option.getOrThrow(observation.finish())).toMatchObject({ input: null, output: null, complete: false })
})
it("measures first output rather than response creation", () => {
  let now = 100
  const observation = makeUsageObservation({ streaming: true, startedAt: 0, now: () => now })
  observation.push({ type: "response.created", response: { model: "magnitude-local/model-a" } })
  now = 200; observation.push({ type: "response.output_text.delta", delta: "hello" })
  now = 1200; observation.push({ type: "response.completed", response: { model: "magnitude-local/model-a", usage: responsesUsage } })
  expect(Option.getOrThrow(observation.finish())).toMatchObject({ model: "model-a", input: 100, cached: 40, output: 20, generationMs: 1000, firstTokenMs: 200, complete: true })
})
it("does not invent Anthropic streaming cache evidence or double-count cached input", () => {
  const message = { model: "anthropic-local/model-a", usage: { input_tokens: 100, cache_creation_input_tokens: 0, cache_read_input_tokens: 40, output_tokens: 20 } }
  const streamed = makeUsageObservation({ streaming: true, startedAt: 0 })
  streamed.push({ type: "message_start", message: { ...message, usage: { ...message.usage, output_tokens: 0, cache_read_input_tokens: 0 } } })
  streamed.push({ type: "message_delta", usage: { output_tokens: 20 } }); streamed.push({ type: "message_stop" })
  expect(Option.getOrThrow(streamed.finish())).toMatchObject({ input: 100, cached: null, output: 20, complete: true })
  const plain = makeUsageObservation({ streaming: false, startedAt: 0 }); plain.push(message)
  expect(Option.getOrThrow(plain.finish())).toMatchObject({ input: 100, cached: 40, output: 20, firstTokenMs: null, generationMs: null })
})
it.each([1, 7, 8192])("preserves SSE bytes at chunk size %i and counts once", async size => {
  const { records, service } = sink()
  const bytes = new TextEncoder().encode(`: comment\r\ndata: ${JSON.stringify({ choices: [{ delta: { content: "☃ hello" } }] })}\r\n\r\ndata: ${JSON.stringify(chat)}\n\ndata: [DONE]\n\n`)
  let offset = 0
  const fetcher = makeUsageFetch(new URL("http://local"), service, async () => new Response(new ReadableStream({ pull(c) { if (offset >= bytes.length) c.close(); else { c.enqueue(bytes.slice(offset, offset + size)); offset += size } } }), { headers: { "content-type": "text/event-stream", "x-test": "preserved" } }))
  const response = await fetcher("http://local/v1/chat/completions")
  expect(response.headers.get("x-test")).toBe("preserved")
  expect(new Uint8Array(await response.arrayBuffer())).toEqual(bytes)
  expect(records).toHaveLength(1); expect(records[0]).toMatchObject({ model: "model-a", input: 100, cached: 40, output: 20, complete: true })
})
it("forwards cancellation and records incomplete evidence", async () => {
  const { records, service } = sink(); let cancelled = false
  const fetcher = makeUsageFetch(new URL("http://local"), service, async () => new Response(new ReadableStream({ pull(c) { c.enqueue(new TextEncoder().encode('data: {"model":"model-a"}\n\n')) }, cancel() { cancelled = true } }), { headers: { "content-type": "text/event-stream" } }))
  const reader = (await fetcher("http://local/v1/chat/completions")).body!.getReader()
  await reader.read(); await reader.cancel()
  expect(cancelled).toBe(true); expect(records).toHaveLength(1); expect(records[0]?.complete).toBe(false)
})
it("excludes cloud traffic, model lists, token counting, and rejected requests", async () => {
  const { records, service } = sink()
  for (const [url, status] of [["https://api.openai.com/v1/responses", 200], ["http://local/v1/models", 200], ["http://local/v1/messages/count_tokens", 200], ["http://local/v1/chat/completions", 400]] as const) {
    await (await makeUsageFetch(new URL("http://local"), service, async () => Response.json(chat, { status }))(url)).text()
  }
  expect(records).toEqual([])
})
it("records sequential WebSocket generations once and skips warm-up and rejected requests", async () => {
  const { records, service } = sink(); const socket = makeUsageWebSocket(service)
  for (const generate of [false, true, true]) {
    await Effect.runPromise(socket.sent(JSON.stringify({ type: "response.create", model: "model-a", generate })))
    await Effect.runPromise(socket.received(JSON.stringify({ type: "response.created", response: { model: "model-a" } })))
    await Effect.runPromise(socket.received(JSON.stringify({ type: "response.completed", response: { model: "model-a", usage: responsesUsage } })))
  }
  await Effect.runPromise(socket.sent(JSON.stringify({ type: "response.create", model: "model-a" })))
  await Effect.runPromise(socket.received('{"type":"error"}')); await Effect.runPromise(socket.close())
  expect(records).toHaveLength(2); expect(new Set(records.map(record => record.id)).size).toBe(2)
})
it("finalizes the active WebSocket request when the previously registered close effect runs", async () => {
  const { records, service } = sink(); const socket = makeUsageWebSocket(service)
  const finalizer = socket.close()
  await Effect.runPromise(socket.sent('{"type":"response.create","model":"model-a"}'))
  await Effect.runPromise(socket.received('{"type":"response.created","response":{"model":"model-a"}}'))
  await Effect.runPromise(finalizer)
  expect(records).toHaveLength(1); expect(records[0]).toMatchObject({ model: "model-a", complete: false })
})
