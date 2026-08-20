import { Context, Data, Effect, Layer, Option, Schema } from "effect"
import {
  ChatCompletionsRequestExtensionsSchema,
  finalizeChatCompletionsRequest,
} from "@magnitudedev/ai"
import type { JsonRecord } from "@magnitudedev/utils/schema"
import type {
  ExpectedToolCall,
  PlannedRequest,
  RequestObservation,
  TerminalEvidence,
  ToolCallObservation,
} from "./domain"
import { stableStringify } from "./hash"

export class EndpointError extends Data.TaggedError("EndpointError")<{
  readonly operation: string
  readonly message: string
}> {}

export interface EndpointConfiguration {
  readonly endpoint: string
  readonly servedModel: string
  readonly apiKey?: string
  readonly timeoutMs?: number
  readonly requestBody?: JsonRecord
}

interface MutableToolCall { id: string; name: string; arguments: string }

function endpointUrl(base: string, path: string): string {
  return `${base.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`
}

function allowed(actual: unknown, alternatives: readonly unknown[]): boolean {
  return alternatives.some((candidate) => stableStringify(candidate) === stableStringify(actual))
}

export function validateToolCalls(expected: readonly ExpectedToolCall[], actual: readonly ToolCallObservation[]): string | undefined {
  if (actual.length !== expected.length) return `expected ${expected.length} tool calls, received ${actual.length}`
  const remaining = [...actual]
  for (const expectedCall of expected) {
    const index = remaining.findIndex((call) => call.name === expectedCall.name)
    if (index < 0) return `missing tool call ${expectedCall.name}`
    const observed = remaining.splice(index, 1)[0]!
    let args: unknown
    try {
      args = JSON.parse(observed.arguments)
    } catch {
      return `${observed.name} arguments are not valid JSON`
    }
    if (!args || typeof args !== "object" || Array.isArray(args)) return `${observed.name} arguments are not an object`
    for (const [key, alternatives] of Object.entries(expectedCall.arguments)) {
      if (!(key in args)) {
        if (alternatives.some((candidate) => candidate === "" || candidate === null)) continue
        return `${observed.name} is missing argument ${key}`
      }
      if (!allowed((args as Record<string, unknown>)[key], alternatives)) return `${observed.name}.${key} is outside the BFCL allowed values`
    }
  }
  return undefined
}

function parseSseBlock(block: string): readonly string[] {
  const data = block.split(/\r?\n/).filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart())
  return data.length === 0 ? [] : [data.join("\n")]
}

function record(value: unknown, field: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new EndpointError({ operation: "terminal-evidence", message: `${field} must be an object` })
  return value as Record<string, unknown>
}

function count(value: unknown, field: string): number {
  if (!Number.isSafeInteger(value) || Number(value) < 0) throw new EndpointError({ operation: "terminal-evidence", message: `${field} must be a non-negative integer` })
  return Number(value)
}

const NonNegativeInt = Schema.Int.pipe(Schema.nonNegative())
const NonNegativeFinite = Schema.Number.pipe(Schema.finite(), Schema.nonNegative())
const RawTerminalEvidence = Schema.Struct({
  choices: Schema.Array(Schema.Unknown),
  usage: Schema.Struct({
    prompt_tokens: NonNegativeInt,
    completion_tokens: NonNegativeInt,
    total_tokens: NonNegativeInt,
    prompt_tokens_details: Schema.Struct({ cached_tokens: NonNegativeInt }),
  }),
  timings: Schema.Struct({
    cache_n: NonNegativeInt,
    prompt_n: NonNegativeInt,
    prompt_ms: NonNegativeFinite,
    predicted_n: NonNegativeInt,
    predicted_ms: NonNegativeFinite,
    draft_n: Schema.optionalWith(NonNegativeInt, { as: "Option", exact: true }),
    draft_n_accepted: Schema.optionalWith(NonNegativeInt, { as: "Option", exact: true }),
  }),
})

export function parseTerminalEvidence(payload: unknown): TerminalEvidence {
  let terminal: typeof RawTerminalEvidence.Type
  try {
    terminal = Schema.decodeUnknownSync(RawTerminalEvidence)(payload)
  } catch (error) {
    throw new EndpointError({
      operation: "terminal-evidence",
      message: error instanceof Error ? error.message : String(error),
    })
  }
  if (terminal.choices.length !== 0) throw new EndpointError({ operation: "terminal-evidence", message: "terminal usage chunk must have empty choices" })
  const evidence: TerminalEvidence = {
    usage: {
      promptTokens: terminal.usage.prompt_tokens,
      cachedPromptTokens: terminal.usage.prompt_tokens_details.cached_tokens,
      completionTokens: terminal.usage.completion_tokens,
      totalTokens: terminal.usage.total_tokens,
    },
    timings: {
      cacheTokens: terminal.timings.cache_n,
      evaluatedPromptTokens: terminal.timings.prompt_n,
      promptMs: terminal.timings.prompt_ms,
      generatedTokens: terminal.timings.predicted_n,
      generationMs: terminal.timings.predicted_ms,
      ...(Option.isSome(terminal.timings.draft_n) ? { draftTokens: terminal.timings.draft_n.value } : {}),
      ...(Option.isSome(terminal.timings.draft_n_accepted) ? { acceptedDraftTokens: terminal.timings.draft_n_accepted.value } : {}),
    },
  }
  if (evidence.usage.totalTokens !== evidence.usage.promptTokens + evidence.usage.completionTokens) {
    throw new EndpointError({ operation: "terminal-evidence", message: "usage.total_tokens does not equal prompt_tokens + completion_tokens" })
  }
  if (evidence.usage.promptTokens !== evidence.timings.cacheTokens + evidence.timings.evaluatedPromptTokens) {
    throw new EndpointError({ operation: "terminal-evidence", message: "usage.prompt_tokens does not equal timings.cache_n + timings.prompt_n" })
  }
  if (evidence.usage.cachedPromptTokens !== evidence.timings.cacheTokens) {
    throw new EndpointError({ operation: "terminal-evidence", message: "cached prompt token counts disagree" })
  }
  if (evidence.usage.completionTokens !== evidence.timings.generatedTokens) {
    throw new EndpointError({ operation: "terminal-evidence", message: "completion token counts disagree" })
  }
  return evidence
}

async function requestWithFetch(config: EndpointConfiguration, request: PlannedRequest): Promise<RequestObservation> {
  const started = performance.now()
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort("request timeout"), config.timeoutMs ?? 300_000)
  const events: { atMs: number; payload: unknown }[] = []
  const toolCalls = new Map<number, MutableToolCall>()
  let status: number | undefined
  let headersMs: number | undefined
  let ttftMs: number | undefined
  let outputText = ""
  let finishReason: string | undefined
  let terminalPayload: unknown
  let sawDone = false
  let streamId: string | undefined

  try {
    const extensions = await Effect.runPromise(Schema.decodeUnknown(
      ChatCompletionsRequestExtensionsSchema,
    )(config.requestBody ?? {}))
    const body = await Effect.runPromise(finalizeChatCompletionsRequest({
      ...extensions,
      model: config.servedModel,
      messages: request.messages,
      ...(request.tools.length > 0 ? { tools: request.tools, tool_choice: "required" } : {}),
      parallel_tool_calls: true,
      max_tokens: request.maxOutputTokens,
      temperature: request.temperature ?? 0,
      top_p: request.topP ?? 1,
      seed: request.seed ?? 42,
      stream: true,
      stream_options: { include_usage: true },
      chat_template_kwargs: { enable_thinking: request.enableThinking ?? false },
    }))
    const response = await fetch(endpointUrl(config.endpoint, "/v1/chat/completions"), {
      method: "POST",
      headers: { "content-type": "application/json", accept: "text/event-stream", ...(config.apiKey ? { authorization: `Bearer ${config.apiKey}` } : {}) },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    status = response.status
    headersMs = performance.now() - started
    if (!response.ok) {
      return { requestId: request.id, outcome: "rejected", status, headersMs, completedMs: performance.now() - started, outputText: "", toolCalls: [], events, error: (await response.text()).slice(0, 4096) }
    }
    if (!response.body) throw new EndpointError({ operation: "stream", message: "response had no body" })
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ""
    let done = false
    while (!done) {
      const read = await reader.read()
      done = read.done
      buffer += decoder.decode(read.value ?? new Uint8Array(), { stream: !done })
      const blocks = buffer.split(/\r?\n\r?\n/)
      buffer = blocks.pop() ?? ""
      if (done && buffer.trim().length > 0) blocks.push(buffer)
      for (const block of blocks) for (const data of parseSseBlock(block)) {
        if (data === "[DONE]") { sawDone = true; continue }
        let payload: unknown
        try { payload = JSON.parse(data) } catch { throw new EndpointError({ operation: "decode-sse", message: `invalid SSE JSON: ${data.slice(0, 256)}` }) }
        const atMs = performance.now() - started
        events.push({ atMs, payload })
        const object = record(payload, "stream chunk")
        if (object.error !== undefined && object.error !== null) throw new EndpointError({ operation: "stream", message: `target stream error: ${JSON.stringify(object.error)}` })
        if (typeof object.id !== "string" || object.id.length === 0) throw new EndpointError({ operation: "stream", message: "stream chunk is missing id" })
        if (streamId === undefined) streamId = object.id
        else if (streamId !== object.id) throw new EndpointError({ operation: "stream", message: "stream chunk id changed during request" })
        const choices = object.choices
        if (!Array.isArray(choices)) throw new EndpointError({ operation: "stream", message: "stream chunk choices must be an array" })
        if (object.usage !== undefined && object.usage !== null) terminalPayload = payload
        for (const choiceValue of choices) {
          const choice = record(choiceValue, "choice")
          if (choice.index !== 0) throw new EndpointError({ operation: "stream", message: "benchmark requires exactly choice index 0" })
          if (typeof choice.finish_reason === "string") finishReason = choice.finish_reason
          if (choice.delta === undefined || choice.delta === null) continue
          const delta = record(choice.delta, "choice.delta")
          let semantic = false
          if (typeof delta.content === "string" && delta.content.length > 0) { outputText += delta.content; semantic = true }
          const deltas = delta.tool_calls === undefined || delta.tool_calls === null ? [] : delta.tool_calls
          if (!Array.isArray(deltas)) throw new EndpointError({ operation: "stream", message: "delta.tool_calls must be an array" })
          for (const callValue of deltas) {
            const call = record(callValue, "tool call delta")
            const index = count(call.index, "tool call delta index")
            const current = toolCalls.get(index) ?? { id: "", name: "", arguments: "" }
            if (typeof call.id === "string") current.id += call.id
            if (call.function !== undefined && call.function !== null) {
              const fn = record(call.function, "tool call function")
              if (typeof fn.name === "string") current.name += fn.name
              if (typeof fn.arguments === "string") current.arguments += fn.arguments
            }
            toolCalls.set(index, current)
            semantic ||= current.name.length > 0 || current.arguments.length > 0
          }
          if (semantic && ttftMs === undefined) ttftMs = atMs
        }
      }
    }
    if (!sawDone) throw new EndpointError({ operation: "stream", message: "stream ended without [DONE]" })
    if (terminalPayload === undefined) throw new EndpointError({ operation: "terminal-evidence", message: "stream ended without terminal usage and timings" })
    const terminal = parseTerminalEvidence(terminalPayload)
    const completedMs = performance.now() - started
    const observedCalls = [...toolCalls.entries()].sort(([left], [right]) => left - right).map(([, call]) => call)
    const validationError = validateToolCalls(request.expected, observedCalls)
    return { requestId: request.id, outcome: validationError ? "invalid" : "valid", status, headersMs, ttftMs, completedMs, outputText, toolCalls: observedCalls, finishReason, terminal, events, error: validationError }
  } catch (error) {
    return {
      requestId: request.id,
      outcome: controller.signal.aborted ? "timeout" : error instanceof EndpointError ? "protocol-error" : "error",
      status,
      headersMs,
      ttftMs,
      completedMs: performance.now() - started,
      outputText,
      toolCalls: [...toolCalls.entries()].sort(([left], [right]) => left - right).map(([, call]) => call),
      finishReason,
      events,
      error: error instanceof Error ? error.message : String(error),
    }
  } finally {
    clearTimeout(timeout)
  }
}

async function probeWithFetch(config: EndpointConfiguration, readinessPath = "/health"): Promise<void> {
  let lastError = "endpoint is not ready"
  for (const path of [readinessPath, "/v1/models"]) {
    try {
      const response = await fetch(endpointUrl(config.endpoint, path), { headers: config.apiKey ? { authorization: `Bearer ${config.apiKey}` } : undefined })
      if (response.ok) return
      lastError = `${path} returned ${response.status}`
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error)
    }
  }
  throw new EndpointError({ operation: "readiness", message: lastError })
}

export interface EndpointClientService {
  readonly execute: (
    config: EndpointConfiguration,
    request: PlannedRequest,
  ) => Effect.Effect<RequestObservation>
  readonly probe: (
    config: EndpointConfiguration,
    readinessPath?: string,
  ) => Effect.Effect<void, EndpointError>
}

export class EndpointClient extends Context.Tag("@magnitudedev/inference-benchmark/EndpointClient")<
  EndpointClient,
  EndpointClientService
>() {}

export const EndpointClientLive = Layer.succeed(EndpointClient, EndpointClient.of({
  execute: (config, request) => Effect.promise(() => requestWithFetch(config, request)),
  probe: (config, readinessPath) => Effect.tryPromise({
    try: () => probeWithFetch(config, readinessPath),
    catch: (error) => error instanceof EndpointError
      ? error
      : new EndpointError({
          operation: "readiness",
          message: error instanceof Error ? error.message : String(error),
        }),
  }),
}))

export const executeRequest = (
  config: EndpointConfiguration,
  request: PlannedRequest,
): Effect.Effect<RequestObservation, never, EndpointClient> =>
  Effect.flatMap(EndpointClient, (client) => client.execute(config, request))

export const probeEndpoint = (
  config: EndpointConfiguration,
  readinessPath?: string,
): Effect.Effect<void, EndpointError, EndpointClient> =>
  Effect.flatMap(EndpointClient, (client) => client.probe(config, readinessPath))
