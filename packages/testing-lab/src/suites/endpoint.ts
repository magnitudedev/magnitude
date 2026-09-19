import { HttpClient, HttpClientRequest } from "@effect/platform"
import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import { AssertionFailure } from "../domain"

const optional = <A, I>(schema: Schema.Schema<A, I>) => Schema.optionalWith(schema, { as: "Option", exact: true })
const Tool = Schema.Struct({ id: Schema.String, type: Schema.Literal("function"), function: Schema.Struct({ name: Schema.String, arguments: Schema.String }) })
const Message = Schema.Struct({ role: Schema.Literal("assistant"), content: Schema.NullOr(Schema.String), tool_calls: optional(Schema.Array(Tool)) })
const Usage = Schema.Struct({ prompt_tokens: Schema.Int.pipe(Schema.nonNegative()), completion_tokens: Schema.Int.pipe(Schema.positive()) })
const Completion = Schema.Struct({ id: Schema.NonEmptyString, model: Schema.NonEmptyString,
  choices: Schema.NonEmptyArray(Schema.Struct({ index: Schema.Int, message: Message, finish_reason: Schema.NullOr(Schema.String) })), usage: Usage })
const Chunk = Schema.Struct({ id: Schema.NonEmptyString, model: Schema.NonEmptyString, choices: Schema.Array(Schema.Struct({ index: Schema.Int,
  delta: Schema.Struct({ content: optional(Schema.NullOr(Schema.String)), role: optional(Schema.String), reasoning_content: optional(Schema.String) }), finish_reason: optional(Schema.NullOr(Schema.String)) })) })
export const Generation = Schema.Struct({ requestId: Schema.NonEmptyString, model: Schema.NonEmptyString, text: Schema.NonEmptyString, chunks: Schema.Int.pipe(Schema.positive()) })
export type Generation = typeof Generation.Type
const assertion = (message: string) => new AssertionFailure({ message })
export interface EndpointTests {
  readonly discover: Effect.Effect<void, AssertionFailure>
  readonly generate: Effect.Effect<Generation, AssertionFailure>
  readonly stream: Effect.Effect<Generation, AssertionFailure>
  readonly tools: Effect.Effect<Generation, AssertionFailure>
  readonly invalid: Effect.Effect<void, AssertionFailure>
  readonly cancelAndRetry: Effect.Effect<Generation, AssertionFailure>
}
export const EndpointTests = Context.GenericTag<EndpointTests>("@magnitudedev/testing-lab/EndpointTests")

/** Decode complete SSE events, including CRLF and multi-line data. Never accept a truncated stream. */
export const decodeGenerationStream = (wire: string, model: string) => Effect.gen(function* () {
  let text = "", id = "", count = 0, finished = false, done = false
  for (const event of wire.replace(/\r\n/g, "\n").split("\n\n")) {
    const data = event.split("\n").filter(l => l.startsWith("data:")).map(l => l.slice(5).replace(/^ /, "")).join("\n")
    if (!data) continue
    if (done) return yield* assertion("Generation stream continued after [DONE]")
    if (data === "[DONE]") { done = true; continue }
    const chunk = yield* Schema.decodeUnknown(Schema.parseJson(Chunk))(data).pipe(Effect.mapError(error => assertion(`Malformed generation stream event: ${String(error).slice(-1200)}; data=${data.slice(0, 1200)}`)))
    if (chunk.model !== model || (id && chunk.id !== id)) return yield* assertion("Generation stream changed model or request identity")
    id = chunk.id
    for (const choice of chunk.choices) {
      if (choice.index !== 0) return yield* assertion("Unexpected generation choice index")
      if (Option.isSome(choice.delta.content) && choice.delta.content.value) {
        if (finished) return yield* assertion("Generation emitted text after its finish event")
        text += choice.delta.content.value; count++
      }
      if (Option.isSome(choice.finish_reason) && choice.finish_reason.value !== null) {
        if (choice.finish_reason.value !== "stop") return yield* assertion(`Generation did not finish normally: ${choice.finish_reason.value}`)
        finished = true
      }
    }
  }
  if (!done || !finished || !text.trim() || !id || count === 0) return yield* assertion("Generation stream was empty, truncated or unfinished")
  return Generation.make({ requestId: id, model, text, chunks: count })
})
export const endpointTests = (origin: string, model: string) => Layer.effect(EndpointTests, Effect.gen(function* () {
  const http = yield* HttpClient.HttpClient
  const url = yield* Effect.try({ try: () => new URL(origin), catch: () => assertion("Invalid inference origin") })
  if (url.protocol !== "http:" || !["127.0.0.1", "[::1]"].includes(url.hostname) || url.pathname !== "/" || url.username || url.password || url.search || url.hash) return yield* assertion("Inference tests must target the isolated app's loopback origin")
  const bounded = <A, E, R>(effect: Effect.Effect<A, E, R>) => effect.pipe(Effect.timeoutFail({ duration: "3 minutes", onTimeout: () => assertion("Inference operation exceeded three minutes") }))
  const request = (body: unknown) => Effect.gen(function* () {
    const json = yield* Schema.encode(Schema.parseJson(Schema.Unknown))(body).pipe(Effect.mapError(() => assertion("Invalid inference test request")))
    return yield* http.execute(HttpClientRequest.post(`${url.origin}/inference/v1/chat/completions`).pipe(HttpClientRequest.bodyText(json, "application/json"))).pipe(
      Effect.mapError(() => assertion("Packaged application's inference endpoint was unreachable")))
  })
  const messages = [{ role: "user", content: "Reply with exactly the word HELLO." }]
  const body = { model, messages, temperature: 0, max_tokens: 128 }
  const check = (value: Generation) => value.text.includes("HELLO") ? Effect.succeed(value) : Effect.fail(assertion("Generation did not follow the basic instruction"))
  const chat = (input: unknown) => bounded(Effect.gen(function* () {
    const response = yield* request(input)
    if (response.status !== 200) {
      const detail = yield* response.stream.pipe(Stream.decodeText(), Stream.runFoldEffect("", (all, part) => all.length + part.length > 16 * 1024
        ? Effect.fail(assertion("Error response exceeded its diagnostic bound")) : Effect.succeed(all + part)),
        Effect.catchAll(() => Effect.succeed("Error response unavailable or larger than 16 Ki characters")))
      return yield* assertion(`Generation returned HTTP ${response.status}: ${detail.replace(/Bearer\s+[^\s"']+/gi, "Bearer [REDACTED]")}`)
    }
    const value = yield* response.json.pipe(Effect.flatMap(Schema.decodeUnknown(Completion)), Effect.mapError(() => assertion("Invalid nonstreamed completion or missing token usage")))
    if (value.model !== model || value.choices.length !== 1 || value.choices[0]!.index !== 0) return yield* assertion("Completion did not use the requested model and single choice")
    return value
  }))
  const generate = chat(body).pipe(Effect.flatMap(c => {
    const choice = c.choices[0]!
    return choice.finish_reason === "stop" && choice.message.content?.trim()
      ? check(Generation.make({ requestId: c.id, model: c.model, text: choice.message.content, chunks: 1 }))
      : Effect.fail(assertion("Nonstreamed completion was empty or unfinished"))
  }))
  const streamResponse = Effect.gen(function* () {
    const response = yield* request({ ...body, stream: true })
    if (response.status !== 200 || !response.headers["content-type"]?.includes("text/event-stream")) return yield* assertion("Streaming endpoint did not return an SSE success response")
    return response
  })
  const stream = bounded(Effect.gen(function* () {
    const response = yield* streamResponse
    const wire = yield* response.stream.pipe(Stream.decodeText(), Stream.runFoldEffect("", (all, part) => all.length + part.length > 1024 * 1024
      ? Effect.fail(assertion("Generation stream exceeded its bounded output")) : Effect.succeed(all + part)), Effect.mapError(() => assertion("Generation stream failed or exceeded its output bound")))
    return yield* decodeGenerationStream(wire, model).pipe(Effect.flatMap(check))
  }))
  return {
    discover: bounded(Effect.gen(function* () {
      const response = yield* http.get(`${url.origin}/inference/v1/models`).pipe(Effect.mapError(() => assertion("Model discovery endpoint was unreachable")))
      if (response.status !== 200) return yield* assertion(`Model discovery returned HTTP ${response.status}`)
      const models = yield* response.json.pipe(Effect.flatMap(Schema.decodeUnknown(Schema.Struct({ data: Schema.Array(Schema.Struct({ id: Schema.String })) }))), Effect.mapError(() => assertion("Malformed model discovery response")))
      if (!models.data.some(m => m.id === model)) return yield* assertion("Model downloaded through the app is absent from endpoint discovery")
    })),
    generate, stream,
    tools: bounded(Effect.gen(function* () {
      const key = crypto.randomUUID(), result = crypto.randomUUID()
      const conversation = [{ role: "user", content: `Call lab_lookup with key ${key}, then repeat the returned value exactly.` }]
      const completion = yield* chat({ ...body, messages: conversation, tools: [{ type: "function", function: { name: "lab_lookup", description: "Look up a stored value by key", parameters: { type: "object", properties: { key: { type: "string" } }, required: ["key"], additionalProperties: false } } }], tool_choice: { type: "function", function: { name: "lab_lookup" } } }).pipe(
        Effect.mapError(error => assertion(`Tool invocation: ${error.message}`)))
      const choice = completion.choices[0]!
      const calls = Option.getOrElse(choice.message.tool_calls, () => [])
      if (choice.finish_reason !== "tool_calls" || calls.length !== 1 || calls[0]!.function.name !== "lab_lookup") return yield* assertion("Model did not produce the requested tool call")
      const args = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ key: Schema.String })))(calls[0]!.function.arguments).pipe(Effect.mapError(() => assertion("Tool arguments were not valid JSON")))
      if (args.key !== key) return yield* assertion("Model changed the tool's input key")
      const message = yield* Schema.encode(Message)(choice.message).pipe(Effect.orDie)
      const followup = yield* chat({ ...body, messages: [...conversation, message, { role: "tool", tool_call_id: calls[0]!.id, content: result }] }).pipe(
        Effect.mapError(error => assertion(`Tool result follow-up: ${error.message}`)))
      const answer = followup.choices[0]!
      if (answer.finish_reason !== "stop" || !answer.message.content?.includes(result)) return yield* assertion("Generation did not consume the actual tool result")
      return Generation.make({ requestId: followup.id, model, text: answer.message.content, chunks: 1 })
    })),
    invalid: bounded(Effect.gen(function* () {
      for (const malformed of [{ ...body, model: `missing-${crypto.randomUUID()}` }, { model, messages: "invalid" }]) {
        const response = yield* request(malformed)
        yield* response.text.pipe(Effect.mapError(() => assertion("Invalid-request error body failed")))
        if (response.status < 400 || response.status >= 500) return yield* assertion("Invalid inference request was not rejected as a client error")
      }
      yield* generate
    })),
    cancelAndRetry: bounded(Effect.gen(function* () {
      const response = yield* request({ ...body, messages: [{ role: "user", content: "Count slowly from 1 to 500, one number per line." }], max_tokens: 2048, stream: true })
      if (response.status !== 200) return yield* assertion("Cancellation generation failed to start")
      // Closing the scoped response after the first generated text or reasoning token aborts this exact request.
      yield* response.stream.pipe(Stream.decodeText(), Stream.splitLines, Stream.filter(line => line.startsWith("data:") && line.slice(5).trim() !== "[DONE]"),
        Stream.mapEffect(line => Schema.decodeUnknown(Schema.parseJson(Chunk))(line.slice(5).trim())),
        Stream.filter(chunk => chunk.choices.some(choice => (Option.isSome(choice.delta.content) && Boolean(choice.delta.content.value)) || (Option.isSome(choice.delta.reasoning_content) && choice.delta.reasoning_content.value.length > 0))), Stream.take(1), Stream.runHead, Effect.flatMap(value => Option.isSome(value) ? Effect.void : Effect.fail(assertion("Cancelled generation produced no generated token"))),
        Effect.mapError(error => assertion(`Could not cancel an active generation stream: ${String(error).slice(-1800)}`)))
      return yield* generate
    })),
  } satisfies EndpointTests
}))
