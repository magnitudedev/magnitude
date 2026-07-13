import { Cause, Stream } from "effect"
import type { Schema } from "effect"
import { createStreamingFieldParser, type StreamingFieldParser } from "../../streaming/field-parser"
import { createToolCallId, type ProviderToolCallId, type ToolCallId } from "../../prompt/ids"
import type { FinishReason, ResponseStreamEvent, ValidationIssue } from "../../response/events"
import type { ResponseUsage } from "../../response/usage"
import {
  type StreamFailure,
  type StreamFailureContext,
  type StreamProgress,
  type AcceptedHttpResponse,
  type ProviderCall,
  payloadSample,
  ModelStreamTerminal,
  StreamClientCorrectnessViolation,
  StreamOperationalFailure,
  StreamProviderCorrectnessViolation,
  StreamProviderError,
  toCauseInfo,
  type UsageAtTermination,
  type UsageMissingReason,
} from "../../errors/failure"
import type { ToolDefinition } from "../../tools/tool-definition"
import type { ChatCompletionsStreamChunk } from "../../wire/chat-completions"
import type { FieldEvent } from "../../streaming/types"
import type { TokenLogprob } from "../../trace"
import type { RawInputToken, RawOutputToken } from "../../response/events"

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

interface ToolCallState {
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly toolName: string
  readonly parser: StreamingFieldParser
}

// ---------------------------------------------------------------------------
// Terminal construction helpers
// ---------------------------------------------------------------------------

function usageAtTermination(usage: ResponseUsage | null, reasonIfMissing: UsageMissingReason): UsageAtTermination {
  return usage === null
    ? { _tag: "UsageNotReported" as const, reason: reasonIfMissing }
    : { _tag: "UsageReported" as const, usage }
}

function buildTerminal(
  pending: PendingTerminal,
  call: ProviderCall,
  response: AcceptedHttpResponse,
  progress: StreamProgress,
  usage: ResponseUsage | null,
): ModelStreamTerminal {
  const usageAt = usageAtTermination(usage, "usage_chunk_never_arrived")

  switch (pending._tag) {
    case "completed":
      return ModelStreamTerminal.StreamCompleted({
        call,
        response,
        finishReason: pending.finishReason,
        progress,
        usage: usageAt,
      })
    case "validation_failure":
      return ModelStreamTerminal.StreamFailed({
        cause: new StreamProviderCorrectnessViolation({
          call,
          response,
          violation: {
            _tag: "InvalidConstrainedOutput",
            output: {
              _tag: "InvalidToolInput",
              toolCallId: pending.toolCallId,
              providerToolCallId: pending.providerToolCallId,
              toolName: pending.toolName,
              issue: pending.issue,
            },
          },
          progress,
        }),
        usage: usageAt,
      })
  }
}

function makeTerminatedStreamTerminal(
  failure: StreamFailure,
  usage: ResponseUsage | null,
): ModelStreamTerminal {
  const usageAt = usageAtTermination(usage, "stream_failed_before_usage")
  return ModelStreamTerminal.StreamFailed({
    cause: failure,
    usage: usageAt,
  })
}

// ---------------------------------------------------------------------------
// Decoder phase — three states
//
//   STREAMING  → processing content, thoughts, tool calls
//   FINISHING  → received finish_reason, waiting for usage chunk
//   DONE       → emitted stream_end, terminal
// ---------------------------------------------------------------------------

type DecoderPhase =
  | { readonly _tag: 'streaming' }
  | { readonly _tag: 'finishing'; readonly pending: PendingTerminal }
  | { readonly _tag: 'done' }

/** The terminal info that can be deferred until the usage chunk arrives. */
type PendingTerminal =
  | { readonly _tag: "completed"; readonly finishReason: FinishReason }
  | { readonly _tag: "validation_failure"; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId; readonly toolName: string; readonly issue: ValidationIssue }

interface DecoderState {
  readonly thoughtOpen: boolean
  readonly messageOpen: boolean
  readonly openToolCalls: ReadonlyMap<number, ToolCallState>
  readonly toolSchemas: ReadonlyMap<string, Schema.Schema.AnyNoContext>
  readonly phase: DecoderPhase
  readonly rawInput: ReadonlyArray<RawInputToken> | null
  readonly rawOutput: ReadonlyArray<RawOutputToken> | null
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeInitialState(
  tools?: readonly ToolDefinition[],
): DecoderState {
  const toolSchemas = new Map<string, Schema.Schema.AnyNoContext>()
  if (tools) {
    for (const tool of tools) {
      toolSchemas.set(tool.name, tool.inputSchema)
    }
  }
  return {
    thoughtOpen: false,
    messageOpen: false,
    openToolCalls: new Map(),
    toolSchemas,
    phase: { _tag: 'streaming' },
    rawInput: null,
    rawOutput: null,
  }
}

function toUsage(
  usage: NonNullable<ChatCompletionsStreamChunk["usage"]>,
): ResponseUsage {
  return {
    inputTokens: usage.prompt_tokens,
    outputTokens: usage.completion_tokens,
    cacheReadTokens: usage.prompt_tokens_details?.cached_tokens ?? 0,
    cacheWriteTokens: 0,
    cost: usage.cost ?? null,
  }
}

function mapReason(reason: string | null | undefined): FinishReason {
  switch (reason) {
    case "stop":
    case "tool_calls":
    case "length":
    case "content_filter":
    case "end_turn":
      return reason
    default:
      return "unknown"
  }
}

/** Wrap FieldEvents from the parser with a toolCallId to produce ResponseStreamEvents. */
function wrapFieldEvents(
  fieldEvents: readonly FieldEvent[],
  toolCallId: ToolCallId,
  providerToolCallId: ProviderToolCallId,
): ResponseStreamEvent[] {
  return fieldEvents.map((fe) => {
    switch (fe._tag) {
      case "field_start":
        return { _tag: "tool_call_field_start" as const, toolCallId, providerToolCallId, path: fe.path }
      case "field_delta":
        return { _tag: "tool_call_field_delta" as const, toolCallId, providerToolCallId, path: fe.path, delta: fe.delta }
      case "field_end":
        return { _tag: "tool_call_field_end" as const, toolCallId, providerToolCallId, path: fe.path, value: fe.value }
    }
  })
}

function decoderProgress(chunksObserved: number, modelEventsEmitted: number): StreamProgress {
  return { dataPayloadsDecoded: chunksObserved, modelEventsEmitted }
}

// ---------------------------------------------------------------------------
// Chunk processing
// ---------------------------------------------------------------------------

function processChunk(
  chunk: ChatCompletionsStreamChunk,
  state: DecoderState,
  parsers: Map<ToolCallId, StreamingFieldParser>,
  logprobs: TokenLogprob[],
  generateToolCallId: () => ToolCallId,
  streamContext: StreamFailureContext,
  progress: StreamProgress,
): readonly [DecoderState, readonly ResponseStreamEvent[]] {
  const events: ResponseStreamEvent[] = []
  let nextState = state

  // ── DONE: terminal, skip everything ────────────────────────────────────────
  if (nextState.phase._tag === 'done') {
    return [nextState, events]
  }

  // ── Accumulate raw data from every chunk ────────────────────────────────────
  nextState = {
    ...nextState,
    rawInput: chunk.raw_input ?? nextState.rawInput,
    rawOutput: chunk.raw_output ?? nextState.rawOutput,
  }

  // ── Server-side error envelope: terminate the stream with a typed error ────
  if (chunk.error) {
    const error = chunk.error
    const failure = new StreamProviderError({
      call: streamContext.call,
      response: streamContext.response,
      providerError: {
        message: error.message,
        type: error.type ?? null,
        code: error.code ?? null,
        param: error.param ?? null,
      },
      payload: payloadSample(JSON.stringify({ error })),
      progress,
    })
    events.push({
      _tag: "stream_end",
      terminal: makeTerminatedStreamTerminal(failure, null),
      rawInput: nextState.rawInput ?? undefined,
      rawOutput: nextState.rawOutput ?? undefined,
    })
    return [{ ...nextState, phase: { _tag: 'done' } }, events]
  }

  // ── Usage chunk: emit stream_end with the pending terminal + raw data ────────
  if (chunk.usage) {
    if (nextState.phase._tag === 'finishing') {
      const { pending } = nextState.phase
      events.push({
        _tag: "stream_end",
        terminal: buildTerminal(
          pending,
          streamContext.call,
          streamContext.response,
          progress,
          toUsage(chunk.usage),
        ),
        rawInput: nextState.rawInput ?? undefined,
        rawOutput: nextState.rawOutput ?? undefined,
      })
      return [{ ...nextState, phase: { _tag: 'done' } }, events]
    }
  }

  // ── FINISHING: waiting for usage only, skip content chunks ──────────────────
  if (nextState.phase._tag === 'finishing') {
    return [nextState, events]
  }

  // ── STREAMING: normal processing ───────────────────────────────────────────
  const choice = chunk.choices[0]
  if (!choice) {
    return [nextState, events]
  }

  // Accumulate logprobs from chunk
  if (choice.logprobs?.content) {
    for (const lp of choice.logprobs.content) {
      logprobs.push({
        token: lp.token,
        logprob: lp.logprob,
        topLogprobs: lp.top_logprobs.map((tp) => ({ token: tp.token, logprob: tp.logprob })),
      })
    }
  }

  const delta = choice.delta

  // Thought content
  if (delta.reasoning_content) {
    if (!nextState.thoughtOpen) {
      nextState = { ...nextState, thoughtOpen: true }
      events.push({ _tag: "thought_start", level: "medium" })
    }
    events.push({ _tag: "thought_delta", text: delta.reasoning_content })
  }

  // Message content
  if (delta.content) {
    if (nextState.thoughtOpen) {
      events.push({ _tag: "thought_end" })
      nextState = { ...nextState, thoughtOpen: false }
    }
    if (!nextState.messageOpen) {
      nextState = { ...nextState, messageOpen: true }
      events.push({ _tag: "message_start" })
    }
    events.push({ _tag: "message_delta", text: delta.content })
  }

  // Tool calls
  if (delta.tool_calls && delta.tool_calls.length > 0) {
    if (nextState.thoughtOpen) {
      events.push({ _tag: "thought_end" })
      nextState = { ...nextState, thoughtOpen: false }
    }
    if (nextState.messageOpen) {
      events.push({ _tag: "message_end" })
      nextState = { ...nextState, messageOpen: false }
    }

    const calls = new Map(nextState.openToolCalls)

    for (const toolCallDelta of delta.tool_calls) {
      let toolCall = calls.get(toolCallDelta.index)

      if (!toolCall) {
        // Finalize all open tool calls with lower indices — they won't receive more deltas
        for (const [idx, openCall] of calls.entries()) {
          if (idx < toolCallDelta.index) {
            const fieldEvents = openCall.parser.end()
            events.push(...wrapFieldEvents(fieldEvents, openCall.toolCallId, openCall.providerToolCallId))

            if (!openCall.parser.valid) {
              const pending: PendingTerminal = {
                _tag: "validation_failure",
                toolCallId: openCall.toolCallId,
                providerToolCallId: openCall.providerToolCallId,
                toolName: openCall.toolName,
                issue: openCall.parser.validationIssue!,
              }
              return [{ ...nextState, phase: { _tag: 'finishing', pending }, openToolCalls: new Map() }, events]
            }

            events.push({
              _tag: "tool_call_ready",
              toolCallId: openCall.toolCallId,
              providerToolCallId: openCall.providerToolCallId,
            })
            calls.delete(idx)
          }
        }

        const name = toolCallDelta.function?.name ?? ""
        const schema = nextState.toolSchemas.get(name)
        const parser = schema
          ? createStreamingFieldParser(schema)
          : createStreamingFieldParser()
        const toolCallId = generateToolCallId()
        const providerToolCallId = (toolCallDelta.id ?? toolCallId) as ProviderToolCallId
        toolCall = { toolCallId, providerToolCallId, toolName: name, parser }
        calls.set(toolCallDelta.index, toolCall)
        parsers.set(toolCallId, parser)
        events.push({
          _tag: "tool_call_start",
          toolCallId: toolCall.toolCallId,
          providerToolCallId: toolCall.providerToolCallId,
          toolName: toolCall.toolName,
        })
      } else if (toolCallDelta.function?.name && toolCall.toolName.length === 0) {
        const name = toolCallDelta.function.name
        const schema = nextState.toolSchemas.get(name)
        const parser = schema
          ? createStreamingFieldParser(schema)
          : createStreamingFieldParser()
        toolCall = { ...toolCall, toolName: name, parser }
        calls.set(toolCallDelta.index, toolCall)
        parsers.set(toolCall.toolCallId, parser)
      }

      if (toolCallDelta.function?.arguments) {
        const fieldEvents = toolCall.parser.push(toolCallDelta.function.arguments)
        events.push(...wrapFieldEvents(fieldEvents, toolCall.toolCallId, toolCall.providerToolCallId))
      }
    }

    nextState = {
      ...nextState,
      openToolCalls: calls,
    }
  }

  // ── Finish reason ──────────────────────────────────────────────────────────
  if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
    // Close open blocks
    if (nextState.thoughtOpen) {
      events.push({ _tag: "thought_end" })
      nextState = { ...nextState, thoughtOpen: false }
    }
    if (nextState.messageOpen) {
      events.push({ _tag: "message_end" })
      nextState = { ...nextState, messageOpen: false }
    }

    // Finalize open tool calls
    for (const toolCall of nextState.openToolCalls.values()) {
      const fieldEvents = toolCall.parser.end()
      events.push(...wrapFieldEvents(fieldEvents, toolCall.toolCallId, toolCall.providerToolCallId))

      if (!toolCall.parser.valid) {
        const pending: PendingTerminal = {
          _tag: "validation_failure",
          toolCallId: toolCall.toolCallId,
          providerToolCallId: toolCall.providerToolCallId,
          toolName: toolCall.toolName,
          issue: toolCall.parser.validationIssue!,
        }
        if (chunk.usage) {
          events.push({
            _tag: "stream_end",
            terminal: buildTerminal(
              pending,
              streamContext.call,
              streamContext.response,
              progress,
              toUsage(chunk.usage),
            ),
            rawInput: nextState.rawInput ?? undefined,
            rawOutput: nextState.rawOutput ?? undefined,
          })
          return [{ ...nextState, phase: { _tag: 'done' }, openToolCalls: new Map() }, events]
        }
        return [{ ...nextState, phase: { _tag: 'finishing', pending }, openToolCalls: new Map() }, events]
      }

      events.push({
        _tag: "tool_call_ready",
        toolCallId: toolCall.toolCallId,
        providerToolCallId: toolCall.providerToolCallId,
      })
    }

    const finishReason = mapReason(choice.finish_reason)

    if (chunk.usage) {
      nextState = {
        ...nextState,
        rawInput: chunk.raw_input ?? nextState.rawInput,
        rawOutput: chunk.raw_output ?? nextState.rawOutput,
      }
      events.push({
        _tag: "stream_end",
        terminal: buildTerminal(
          { _tag: "completed", finishReason },
          streamContext.call,
          streamContext.response,
          progress,
          toUsage(chunk.usage),
        ),
        rawInput: nextState.rawInput ?? undefined,
        rawOutput: nextState.rawOutput ?? undefined,
      })
      nextState = { ...nextState, openToolCalls: new Map(), phase: { _tag: 'done' } }
    } else {
      nextState = { ...nextState, openToolCalls: new Map(), phase: { _tag: 'finishing', pending: { _tag: "completed", finishReason } } }
    }
  }

  return [nextState, events]
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function decode<E>(
  chunks: Stream.Stream<ChatCompletionsStreamChunk, E>,
  options: {
    tools?: readonly ToolDefinition[]
    streamContext: StreamFailureContext
    generateToolCallId?: () => ToolCallId
    toStreamFailure: (error: E) => StreamFailure
  },
): {
  readonly events: Stream.Stream<ResponseStreamEvent, never>
  readonly parsers: ReadonlyMap<ToolCallId, StreamingFieldParser>
  readonly logprobs: TokenLogprob[]
} {
  const generateToolCallId = options.generateToolCallId ?? createToolCallId
  const parsers = new Map<ToolCallId, StreamingFieldParser>()
  const logprobs: TokenLogprob[] = []
  let chunksObserved = 0
  let modelEventsEmitted = 0

  let lastState: DecoderState = makeInitialState(options.tools)
  const tracked = Stream.mapAccum(
    chunks,
    makeInitialState(options.tools),
    (state, chunk): readonly [DecoderState, readonly ResponseStreamEvent[]] => {
      chunksObserved += 1
      const result = processChunk(
        chunk,
        state,
        parsers,
        logprobs,
        generateToolCallId,
        options.streamContext,
        decoderProgress(chunksObserved, modelEventsEmitted),
      )
      lastState = result[0]
      modelEventsEmitted += result[1].length
      return result
    },
  )

  const flattened = Stream.flatMap(tracked, (events) => Stream.fromIterable(events))

  // Fallback: if stream closes while FINISHING (usage never arrived), emit
  // stream_end with UsageNotReported. If it closes before finish_reason, report
  // a retryable stream operational failure.
  const raw: Stream.Stream<ResponseStreamEvent, E> = Stream.concat(
    flattened,
    Stream.suspend(() => {
      if (lastState.phase._tag === 'finishing') {
        const endEvent: ResponseStreamEvent = {
          _tag: "stream_end",
          terminal: buildTerminal(
            lastState.phase.pending,
            options.streamContext.call,
            options.streamContext.response,
            decoderProgress(chunksObserved, modelEventsEmitted),
            null,
          ),
          rawInput: lastState.rawInput ?? undefined,
          rawOutput: lastState.rawOutput ?? undefined,
        }
        return Stream.make(endEvent)
      }
      if (lastState.phase._tag === 'streaming') {
        const failure = new StreamOperationalFailure({
          call: options.streamContext.call,
          response: options.streamContext.response,
          reason: {
            _tag: "ConnectionClosedWithoutTerminalOutcome",
            expectation: chunksObserved === 0
              ? { _tag: "InitialChunk" }
              : { _tag: "FinishReasonOrMoreChunks" },
          },
          progress: decoderProgress(chunksObserved, modelEventsEmitted),
        })
        const endEvent: ResponseStreamEvent = {
          _tag: "stream_end",
          terminal: makeTerminatedStreamTerminal(failure, null),
          rawInput: lastState.rawInput ?? undefined,
          rawOutput: lastState.rawOutput ?? undefined,
        }
        return Stream.make(endEvent)
      }
      return Stream.empty
    }),
  )

  // Expected stream failures become terminal events. Generic upstream errors (E)
  // are mapped to StreamFailure via toStreamFailure, then wrapped
  // in the corresponding stream terminal. Defects/dies are not caught here; they remain defects for the
  // harness dispatcher to report.
  const withErrorHandling = Stream.catchAll(raw, (error) => {
    const streamFailure = options.toStreamFailure(error)
    const endEvent: ResponseStreamEvent = {
      _tag: "stream_end",
      terminal: makeTerminatedStreamTerminal(streamFailure, null),
      rawInput: lastState.rawInput ?? undefined,
      rawOutput: lastState.rawOutput ?? undefined,
    }
    return Stream.make(endEvent)
  })

  const withDefectHandling = Stream.catchAllCause(withErrorHandling, (cause) => {
    if (!Cause.isDie(cause)) {
      return Stream.failCause(cause as Cause.Cause<never>)
    }

    const streamFailure = new StreamClientCorrectnessViolation({
      call: options.streamContext.call,
      response: options.streamContext.response,
      component: "model_event_reducer",
      message: "Native chat-completions decoder defect while reducing model stream",
      evidence: { _tag: "UnexpectedDefectCaught", cause: toCauseInfo(cause) },
      progress: decoderProgress(chunksObserved, modelEventsEmitted),
    })
    const endEvent: ResponseStreamEvent = {
      _tag: "stream_end",
      terminal: makeTerminatedStreamTerminal(streamFailure, null),
      rawInput: lastState.rawInput ?? undefined,
      rawOutput: lastState.rawOutput ?? undefined,
    }
    return Stream.make(endEvent)
  })

  // stream_end is the terminal event — include it then stop
  const events = Stream.takeUntil(withDefectHandling, (event) => event._tag === "stream_end")

  return { events, parsers, logprobs }
}
