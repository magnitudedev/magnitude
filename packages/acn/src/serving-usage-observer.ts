import * as Native from "@magnitudedev/icn-protocol/schemas"
import { Effect, Option, Schema } from "effect"
import { createParser } from "eventsource-parser"
import { ServingModelId } from "@magnitudedev/acn-protocol"
import { ServingUsage, ServingUsageRecord } from "./serving-usage"
import type { InferenceFetch } from "./inference-gateway"

const object = (value: unknown): Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : {}
const count = Schema.decodeUnknownOption(Schema.Number.pipe(Schema.int(), Schema.nonNegative()))
const json = Schema.decodeUnknownOption(Schema.parseJson(Schema.Unknown))
const canonicalModel = (value: string) => ServingModelId.make(value.replace(/^(?:magnitude-local|anthropic-local)\//, ""))

/** Observes evidence only; never rewrites a request, response, or native counter. */
export const makeUsageObservation = (options: {
  readonly streaming: boolean
  readonly startedAt: number
  readonly now?: () => number
  readonly id?: string
}) => {
  const now = options.now ?? performance.now.bind(performance)
  const id = options.id ?? crypto.randomUUID()
  let model = ServingModelId.make("unknown")
  let input: number | null = null
  let cached: number | null = null
  let output: number | null = null
  let generationMs: number | null = null
  let firstTokenMs: number | null = null
  let firstOutputAt: number | null = null
  let terminalAt: number | null = null
  let complete = false
  let finished = false
  const push = (value: unknown) => {
    const frame = object(value)
    const response = frame.response === undefined ? frame : object(frame.response)
    const message = frame.message === undefined ? response : object(frame.message)
    if (typeof message.model === "string" && message.model.length > 0) model = canonicalModel(message.model)
    const choices = Array.isArray(frame.choices) ? frame.choices.map(object) : []
    const eventType = typeof frame.type === "string" ? frame.type : ""
    const delta = object(frame.delta)
    const hasOutput = choices.some(choice => {
      const part = object(choice.delta)
      return [part.content, part.reasoning_content].some(text => typeof text === "string" && text.length > 0) || Array.isArray(part.tool_calls) && part.tool_calls.length > 0
    }) || eventType === "content_block_delta" && Object.values(delta).some(text => typeof text === "string" && text.length > 0 && text !== delta.type)
      || eventType.startsWith("response.") && eventType.endsWith(".delta") && typeof frame.delta === "string" && frame.delta.length > 0
    if (options.streaming && hasOutput && firstOutputAt === null) firstOutputAt = now()
    const chat = Schema.decodeUnknownOption(Native.Usage)(message.usage)
    const responses = Schema.decodeUnknownOption(Native.ResponseUsage)(message.usage)
    const anthropic = Schema.decodeUnknownOption(Native.UsageResponse)(message.usage)
    const chatTerminal = choices.some(choice => choice.finish_reason != null)
    if (Option.isSome(chat)) {
      input = chat.value.prompt_tokens; cached = chat.value.prompt_tokens_details.cached_tokens; output = chat.value.completion_tokens
      complete = true
    } else if (Option.isSome(responses)) {
      input = responses.value.input_tokens; cached = responses.value.input_tokens_details.cached_tokens; output = responses.value.output_tokens
      complete = true
    } else if (Option.isSome(anthropic)) {
      // ICN's Messages input count includes cache reads. Its streaming start event
      // does not yet report cache evidence; zero there is not a measured cache miss.
      input = anthropic.value.input_tokens
      if (!options.streaming) { cached = anthropic.value.cache_read_input_tokens; output = anthropic.value.output_tokens; complete = true }
    }
    if (eventType === "message_delta") {
      const tokens = count(object(frame.usage).output_tokens)
      if (Option.isSome(tokens)) output = tokens.value
    }
    if (eventType === "message_stop") complete = input !== null && output !== null
    const timings = Schema.decodeUnknownOption(Native.Timings)(frame.timings)
    if (Option.isSome(timings) && (chatTerminal || Option.isSome(chat) || !options.streaming)) {
      const t = timings.value
      input = t.prompt_n + t.cache_n; cached = t.cache_n; output = t.predicted_n
      generationMs = Number.isFinite(t.predicted_ms) && t.predicted_ms > 0 ? t.predicted_ms : null
      firstTokenMs = Number.isFinite(t.time_to_first_token_ms) && t.time_to_first_token_ms >= 0 ? t.time_to_first_token_ms : null
      complete = true
    }
    if (complete && terminalAt === null) terminalAt = now()
  }
  return {
    push,
    pushJson: (text: string) => { const decoded = json(text); if (Option.isSome(decoded)) push(decoded.value) },
    finish: (): Option.Option<ServingUsageRecord> => {
      if (finished) return Option.none()
      finished = true
      if (firstOutputAt !== null) {
        firstTokenMs ??= Math.max(0, firstOutputAt - options.startedAt)
        const elapsed = (terminalAt ?? now()) - firstOutputAt
        if (complete && output !== null && elapsed > 0) generationMs ??= elapsed
      }
      return Option.some({ id: ServingUsageRecord.fields.id.make(id), model, completedAt: Date.now(), input, cached, output, generationMs, firstTokenMs, complete })
    },
  }
}

/** A pull-through observer preserves backpressure and forwards the original bytes. */
export const makeUsageFetch = (origin: URL, usage: ServingUsage, fetchTarget: InferenceFetch = fetch): InferenceFetch => async (input, init) => {
  const url = new URL(input instanceof Request ? input.url : String(input))
  const startedAt = performance.now()
  const response = await fetchTarget(input, init)
  if (url.origin !== origin.origin || !/\/(?:chat\/completions|responses|messages)$/.test(url.pathname) || !response.ok || !response.body) return response
  const streaming = response.headers.get("content-type")?.includes("text/event-stream") === true
  const observation = makeUsageObservation({ streaming, startedAt })
  const decoder = new TextDecoder()
  // Observation is bounded independently of serving; unusually large or encoded
  // bodies continue unchanged and leave an explicitly incomplete usage record.
  const limit = 16 * 1024 * 1024
  let buffered = ""
  let observing = !response.headers.has("content-encoding")
  const parser = createParser({ maxBufferSize: limit, onEvent: event => observation.pushJson(event.data) })
  const reader = response.body.getReader()
  const save = () => Effect.runPromise(Option.match(observation.finish(), { onNone: () => Effect.void, onSome: usage.record }))
  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const chunk = await reader.read()
        if (chunk.done) {
          if (observing) {
            try {
              if (streaming) parser.feed(decoder.decode())
              else observation.pushJson(buffered + decoder.decode())
            } catch { /* Missing usage evidence must not fail the response. */ }
          }
          await save()
          controller.close()
          reader.releaseLock()
          return
        }
        if (observing) {
          try {
            const text = decoder.decode(chunk.value, { stream: true })
            if (streaming) parser.feed(text)
            else if (buffered.length + text.length <= limit) buffered += text
            else { observing = false; buffered = "" }
          } catch { observing = false; buffered = "" }
        }
        controller.enqueue(chunk.value)
      } catch (error) {
        await save()
        controller.error(error)
        reader.releaseLock()
      }
    },
    async cancel(reason) { try { await reader.cancel(reason) } finally { await save(); reader.releaseLock() } },
  })
  return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers })
}

/** ICN serializes Responses requests on each WebSocket, including warm-up replies. */
export const makeUsageWebSocket = (usage: ServingUsage) => {
  const pending: Array<ReturnType<typeof makeUsageObservation> | null> = []
  let active: ReturnType<typeof makeUsageObservation> | null = null
  const finish = () => {
    const record = active?.finish() ?? Option.none()
    active = null
    return Option.match(record, { onNone: () => Effect.void, onSome: usage.record })
  }
  return {
    sent: (message: string | Uint8Array) => Effect.sync(() => {
      const decoded = json(typeof message === "string" ? message : new TextDecoder().decode(message))
      if (Option.isNone(decoded)) return
      const request = object(decoded.value)
      if (request.type !== "response.create") return
      const observation = request.generate === false ? null : makeUsageObservation({ streaming: true, startedAt: performance.now() })
      observation?.push({ model: request.model })
      pending.push(observation)
    }),
    received: (message: string | Uint8Array) => Effect.suspend(() => {
      const decoded = json(typeof message === "string" ? message : new TextDecoder().decode(message))
      if (Option.isNone(decoded)) return Effect.void
      const frame = object(decoded.value)
      if (frame.type === "response.created") active = pending.shift() ?? null
      if (frame.type === "error") {
        if (active === null) pending.shift()
        return finish()
      }
      active?.push(frame)
      return ["response.completed", "response.incomplete", "response.failed"].includes(String(frame.type)) ? finish() : Effect.void
    }),
    close: () => Effect.suspend(finish),
  }
}
