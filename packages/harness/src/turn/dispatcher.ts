import { Effect, Cause, Data, Layer, Stream, Schema, Option } from "effect"
import { JsonValueSchema, type JsonValue } from "@magnitudedev/utils/schema"
import type { ProviderToolCallId, ResponseStreamEvent, ToolCallId, StreamingFieldParser, FinishReason, ValidationIssue } from "@magnitudedev/ai"
import { formatStreamFailureMessage, formatValidationIssue, type ModelStreamTerminal } from "@magnitudedev/ai"
import type { HarnessEvent, ToolError, ToolResult, TurnOutcome } from "../events"
import type { HarnessHooks, ExecuteHookContext } from "../hooks"
import type { Toolkit } from "../tool/toolkit"
import type { HarnessToolErased, ToolContext, StreamHook } from "../tool/tool"
import type { EngineState, ToolOutcome } from "./reducers"


// ── TurnAbort — planned abort control flow ────────────────────────────

/**
 * Typed error used to abort the dispatch event loop.
 *
 * When the dispatcher detects a terminal condition during tool execution
 * (error, rejection, defect), it emits all relevant lifecycle events first,
 * then fails with TurnAbort carrying the terminal outcome. Stream.runForEach
 * stops consuming — no more events are processed. The outer catch handler
 * extracts the outcome and emits TurnEnd.
 *
 * This is a planned abort, not a crash. It never crosses package boundaries.
 */
export class TurnAbort extends Data.TaggedError("TurnAbort")<{
  readonly outcome: TurnOutcome
}> {}

// ── Config ───────────────────────────────────────────────────────────

export interface DispatchConfig<TDenial extends JsonValue = JsonValue> {
  readonly events: Stream.Stream<ResponseStreamEvent, never>
  readonly parsers: ReadonlyMap<ToolCallId, StreamingFieldParser>
  readonly toolkit: Toolkit
  readonly hooks?: HarnessHooks<unknown, TDenial>
  // Erased layer — createHarness enforces type coverage at compile time.
  readonly layer?: Layer.Layer<unknown>
  readonly initialEngineState?: EngineState
  readonly emit: (event: HarnessEvent) => Effect.Effect<void>
  readonly maxThoughtChars?: number
  readonly requestId: string | null
}

// ── Per-tool-call accumulator ────────────────────────────────────────

interface ToolCallAccumulator {
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly streamState: unknown
  readonly streamHook: StreamHook<any, any, any, any> | undefined
}

// ── Dispatch ─────────────────────────────────────────────────────────

export function dispatch<TDenial extends JsonValue = JsonValue>(config: DispatchConfig<TDenial>): Effect.Effect<void> {
  const { toolkit, hooks, emit, initialEngineState } = config

  const withRequestId = <T extends object>(outcome: T): T & { readonly requestId: string | null } => ({
    ...outcome,
    requestId: config.requestId,
  })

  // Build lookup maps from toolkit
  const toolNameToKey = new Map<string, string>()
  const toolKeyToEntry = new Map<string, { tool: HarnessToolErased }>()
  for (const key of toolkit.keys) {
    const entry = toolkit.entries[key]
    const tool = entry.tool as HarnessToolErased
    toolNameToKey.set(tool.definition.name, key)
    toolKeyToEntry.set(key, { tool })
  }

  // Build cached outcomes map from initial engine state
  const cachedOutcomes = new Map<ToolCallId, ToolOutcome>()
  if (initialEngineState) {
    for (const [toolCallId, outcome] of initialEngineState.toolOutcomes) {
      cachedOutcomes.set(toolCallId, outcome)
    }
  }

  // Mutable dispatch state (scoped to this dispatch invocation)
  const accumulators = new Map<ToolCallId, ToolCallAccumulator>()
  let toolCallCount = 0
  let thoughtCharCount = 0

  // ── Provide layer to erased effect ───────────────────────────────

  // Erased-boundary layer provision — type coverage enforced by createHarness at compile time.
  function provideLayer<A, E, R>(effect: Effect.Effect<A, E, R>): Effect.Effect<A, E, never> {
    if (config.layer) {
      return Effect.provide(effect, config.layer) as Effect.Effect<A, E, never>
    }
    return effect as Effect.Effect<A, E, never>
  }

  const jsonDecoder = Schema.decodeUnknown(JsonValueSchema)

  function defect(message: string): TurnAbort {
    return new TurnAbort({ outcome: withRequestId({ _tag: "EngineDefect", message }) })
  }

  function encodeSchemaJson(
    schema: Schema.Schema.AnyNoContext,
    value: unknown,
    label: string,
  ): Effect.Effect<JsonValue, TurnAbort> {
    return Schema.encodeUnknown(schema)(value).pipe(
      Effect.flatMap(jsonDecoder),
      Effect.mapError((error) => defect(`${label} did not encode to JSON\n${error.message}`)),
    )
  }

  function ensureJson(value: unknown, label: string): Effect.Effect<JsonValue, TurnAbort> {
    return jsonDecoder(value).pipe(
      Effect.mapError((error) => defect(`${label} is not JSON serializable\n${error.message}`)),
    )
  }

  function decodeToolInput(
    tool: HarnessToolErased,
    value: unknown,
  ): Effect.Effect<unknown, { readonly _tag: "InputRejected"; readonly issue: ValidationIssue }> {
    return Schema.decodeUnknown(tool.definition.inputSchema)(value).pipe(
      Effect.mapError((error) => ({ _tag: "InputRejected" as const, issue: formatValidationIssue(error) })),
    )
  }

  function toToolError(value: unknown): ToolError {
    if (typeof value === "object" && value !== null && "message" in value && typeof value.message === "string") {
      return value as ToolError
    }
    return { message: String(value) }
  }

  function encodeToolError(tool: HarnessToolErased, error: unknown): Effect.Effect<ToolError, TurnAbort> {
    if (!tool.errorSchema) return Effect.succeed(toToolError(error))
    return encodeSchemaJson(tool.errorSchema, error, `Error from tool ${tool.definition.name}`).pipe(
      Effect.map(toToolError),
    )
  }

  // ── Emit helper for tool context ─────────────────────────────────

  function makeToolEmit(tool: HarnessToolErased, toolCallId: ToolCallId, providerToolCallId: ProviderToolCallId, toolName: string, toolKey: string) {
    return (value: unknown): Effect.Effect<void, TurnAbort> =>
      Effect.gen(function* () {
        const encoded = tool.emissionSchema
          ? yield* encodeSchemaJson(tool.emissionSchema, value, `Emission from tool ${toolName}`)
          : yield* ensureJson(value, `Emission from tool ${toolName}`)
        yield* emit({ _tag: "ToolEmission", toolCallId, providerToolCallId, toolName, toolKey, value: encoded })
        if (hooks?.onEmission) {
          yield* provideLayer(hooks.onEmission({ toolCallId, toolName, toolKey, value: encoded }))
        }
      })
  }

  // ── Tool execution pipeline ──────────────────────────────────────

  function executeTool(
    toolCallId: ToolCallId,
    providerToolCallId: ProviderToolCallId,
    toolName: string,
    toolKey: string,
    input: unknown,
  ): Effect.Effect<void, TurnAbort> {
    const lookup = toolKeyToEntry.get(toolKey)
    if (!lookup) {
      return Effect.fail(new TurnAbort({ outcome: withRequestId({ _tag: "EngineDefect", message: `Unknown tool key: ${toolKey}` }) }))
    }
    const { tool } = lookup
    const encodedInput = encodeSchemaJson(tool.definition.inputSchema, input, `Input for tool ${toolName}`)

    // Check cached outcome
    const cached = cachedOutcomes.get(toolCallId)
    if (cached && cached._tag === "Completed") {
      return Effect.gen(function* () {
        yield* emit({
          _tag: "ToolExecutionStarted",
          toolCallId, providerToolCallId, toolName, toolKey,
          input: yield* encodedInput,
          cached: true,
        })
        yield* emit({
          _tag: "ToolExecutionEnded",
          toolCallId, providerToolCallId, toolName, toolKey,
          result: cached.result,
        })

        // afterExecute hook for cached results
        // Note: cached results carry ToolResultErased (denial: unknown) but the hook
        // signature expects ToolResult<unknown, ToolError, TDenial>. Cached results
        // are inherently untyped, so we upcast here.
        if (hooks?.afterExecute) {
          yield* provideLayer(hooks.afterExecute({ toolCallId, toolName, toolKey, input, result: cached.result as ToolResult<JsonValue, ToolError, TDenial> }))
        }

        // Fast-fail on cached error outcomes
        if (cached.result._tag === "Error") {
          return yield* new TurnAbort({
            outcome: withRequestId({ _tag: "ToolExecutionError", toolCallId, providerToolCallId, toolName, toolKey, error: cached.result.error }),
          })
        }
      })
    }

    // beforeExecute hook
    const hookCtx: ExecuteHookContext = { toolCallId, toolName, toolKey, input }

    return Effect.gen(function* () {
      const decision = hooks?.beforeExecute
        ? yield* provideLayer(hooks.beforeExecute(hookCtx))
        : { _tag: "Proceed" as const }

      if (decision._tag === "Deny") {
        yield* emit({
          _tag: "ToolExecutionStarted",
          toolCallId, providerToolCallId, toolName, toolKey,
          input: yield* encodedInput,
          cached: false,
        })
        const denial = yield* ensureJson(decision.denial, `Denial from beforeExecute hook for tool ${toolName}`)
        const result: ToolResult = { _tag: "Denied", denial }
        yield* emit({ _tag: "ToolExecutionEnded", toolCallId, providerToolCallId, toolName, toolKey, result })
        return yield* new TurnAbort({ outcome: withRequestId({ _tag: "GateRejected", toolCallId, providerToolCallId, toolName }) })
      }

      const effectiveInput = decision._tag === "Proceed" && decision.modifiedInput !== undefined
        ? yield* decodeToolInput(tool, decision.modifiedInput).pipe(
          Effect.mapError(({ issue }) =>
            new TurnAbort({
              outcome: withRequestId({ _tag: "ToolInputValidationFailure", toolCallId, providerToolCallId, toolName, toolKey, issue }),
            }),
          ),
        )
        : input

      const encodedEffectiveInput = yield* encodeSchemaJson(tool.definition.inputSchema, effectiveInput, `Input for tool ${toolName}`)

      yield* emit({
        _tag: "ToolExecutionStarted",
        toolCallId, providerToolCallId, toolName, toolKey,
        input: encodedEffectiveInput,
        cached: false,
      })

      // Build ToolContext with working emit
      const toolCtx: ToolContext<unknown> = {
        emit: makeToolEmit(tool, toolCallId, providerToolCallId, toolName, toolKey) as any,
      }

      const rawResult = yield* Effect.gen(function* () {
        const toolEffect = tool.execute(effectiveInput, toolCtx)
        const provided = provideLayer(toolEffect)
        const output = yield* provided
        return { _tag: "Success" as const, output }
      }).pipe(
        Effect.catchAllCause((cause) =>
          Effect.gen(function* () {
            const failure = Cause.failureOption(cause)
            if (Option.isSome(failure) && failure.value instanceof TurnAbort) {
              return yield* failure.value
            }
            const squashed = Cause.squash(cause)
            const error = yield* encodeToolError(tool, squashed)
            return { _tag: "Error" as const, error }
          }),
        ),
      )

      const result: ToolResult = rawResult._tag === "Success"
        ? {
          _tag: "Success",
          output: yield* encodeSchemaJson(tool.definition.outputSchema, rawResult.output, `Output from tool ${toolName}`),
        }
        : rawResult

      yield* emit({ _tag: "ToolExecutionEnded", toolCallId, providerToolCallId, toolName, toolKey, result })

      // afterExecute hook
      if (hooks?.afterExecute) {
        yield* provideLayer(hooks.afterExecute({ ...hookCtx, result }))
      }

      // Fast-fail on tool execution errors
      if (result._tag === "Error") {
        return yield* new TurnAbort({
          outcome: withRequestId({ _tag: "ToolExecutionError", toolCallId, providerToolCallId, toolName, toolKey, error: result.error }),
        })
      }
    })
  }

  // ── Terminal handling ─────────────────────────────────────────────

  function terminalToOutcome(terminal: ModelStreamTerminal): TurnOutcome {
    switch (terminal._tag) {
      case "StreamCompleted":
        return mapFinishReasonToOutcome(terminal.finishReason, toolCallCount, config.requestId)

      case "StreamFailed":
        return withRequestId({
          _tag: "StreamFailed",
          message: formatStreamFailureMessage(terminal.cause),
          terminal,
        })
    }
  }

  // ── Stream event processing ──────────────────────────────────────

  function processEvent(event: ResponseStreamEvent): Effect.Effect<void, TurnAbort> {
    switch (event._tag) {
      case "thought_start": {
        thoughtCharCount = 0
        return emit({ _tag: "ThoughtStart", level: event.level })
      }

      case "thought_delta": {
        thoughtCharCount += event.text.length
        if (config.maxThoughtChars !== undefined && thoughtCharCount > config.maxThoughtChars) {
          return Effect.fail(new TurnAbort({
            outcome: withRequestId({ _tag: "ThoughtLimitExceeded", limit: config.maxThoughtChars }),
          }))
        }
        return emit({ _tag: "ThoughtDelta", text: event.text })
      }

      case "thought_end":
        return emit({ _tag: "ThoughtEnd" })

      case "message_start":
        return emit({ _tag: "MessageStart" })

      case "message_delta":
        return emit({ _tag: "MessageDelta", text: event.text })

      case "message_end":
        return emit({ _tag: "MessageEnd" })

      case "tool_call_start": {
        const toolKey = toolNameToKey.get(event.toolName)
        if (!toolKey) {
          return Effect.fail(new TurnAbort({ outcome: withRequestId({ _tag: "EngineDefect", message: `Unknown tool name: ${event.toolName}` }) }))
        }
        const entry = toolKeyToEntry.get(toolKey)
        if (!entry) {
          return Effect.fail(new TurnAbort({ outcome: withRequestId({ _tag: "EngineDefect", message: `No entry for tool key: ${toolKey}` }) }))
        }
        toolCallCount++

        const acc: ToolCallAccumulator = {
          toolCallId: event.toolCallId,
          providerToolCallId: event.providerToolCallId,
          toolName: event.toolName,
          toolKey,
          streamState: entry.tool.stream?.initial,
          streamHook: entry.tool.stream,
        }
        accumulators.set(event.toolCallId, acc)

        return emit({
          _tag: "ToolInputStarted",
          toolCallId: event.toolCallId,
          providerToolCallId: event.providerToolCallId,
          toolName: event.toolName,
          toolKey,
        })
      }

      case "tool_call_field_start":
        return Effect.void

      case "tool_call_field_delta": {
        const acc = accumulators.get(event.toolCallId)
        if (!acc) return Effect.void

        const field = event.path[0] ?? ""

        return Effect.gen(function* () {
          yield* emit({
            _tag: "ToolInputFieldChunk",
            toolCallId: event.toolCallId,
            providerToolCallId: event.providerToolCallId,
            field,
            path: event.path,
            delta: event.delta,
          })

          // Invoke stream hook onInput if present — read partial from parser
          if (acc.streamHook) {
            const parser = config.parsers.get(event.toolCallId)
            if (parser) {
              const entry = toolKeyToEntry.get(acc.toolKey)
              if (!entry) {
                return yield* new TurnAbort({ outcome: withRequestId({ _tag: "EngineDefect", message: `No entry for tool key: ${acc.toolKey}` }) })
              }
              const toolCtx: ToolContext<unknown> = {
                emit: makeToolEmit(entry.tool, acc.toolCallId, acc.providerToolCallId, acc.toolName, acc.toolKey) as any,
              }
              const partial = parser.partial
              if (partial) {
                const newStreamState = yield* provideLayer(
                  acc.streamHook.onInput(partial, acc.streamState, toolCtx),
                ).pipe(
                  Effect.catchTag("StreamValidationError", (e) =>
                    Effect.gen(function* () {
                      const issue: ValidationIssue = { path: [], message: e.message }
                      yield* emit({
                        _tag: "ToolInputRejected",
                        toolCallId: event.toolCallId,
                        providerToolCallId: event.providerToolCallId,
                        toolName: acc.toolName,
                        toolKey: acc.toolKey,
                        issue,
                      })
                      // No formatting — the reducer produces ToolResultEntry from this event
                      return yield* new TurnAbort({
                        outcome: withRequestId({ _tag: "ToolInputValidationFailure", toolCallId: acc.toolCallId, providerToolCallId: acc.providerToolCallId, toolName: acc.toolName, toolKey: acc.toolKey, issue }),
                      })
                    }),
                  ),
                )
                const current = accumulators.get(event.toolCallId)!
                accumulators.set(event.toolCallId, { ...current, streamState: newStreamState })
              }
            }
          }
        })
      }

      case "tool_call_field_end": {
        const acc = accumulators.get(event.toolCallId)
        if (!acc) return Effect.void

        const field = event.path[0] ?? ""

        return emit({
          _tag: "ToolInputFieldComplete",
          toolCallId: event.toolCallId,
          providerToolCallId: event.providerToolCallId,
          field,
          path: event.path,
          value: event.value,
        })
      }

      case "tool_call_ready": {
        const acc = accumulators.get(event.toolCallId)
        if (!acc) return Effect.void

        const parser = config.parsers.get(event.toolCallId)
        if (!parser || parser.decoded === null) {
          return Effect.fail(new TurnAbort({ outcome: withRequestId({ _tag: "EngineDefect", message: `No decoded input for ${event.toolCallId}` }) }))
        }

        return Effect.gen(function* () {
          yield* emit({
            _tag: "ToolInputReady",
            toolCallId: acc.toolCallId,
            providerToolCallId: acc.providerToolCallId,
          })

          // Execute tool inline — sequential ordering required for dependent tools
          yield* executeTool(acc.toolCallId, acc.providerToolCallId, acc.toolName, acc.toolKey, parser.decoded!)
        })
      }

      case "stream_end": {
        return Effect.gen(function* () {
          const outcome = terminalToOutcome(event.terminal)

          // Extract usage from terminal when available
          const usage = event.terminal._tag === "StreamCompleted" && event.terminal.usage._tag === "UsageReported"
            ? event.terminal.usage.usage
            : null

          yield* emit({
            _tag: "TurnEnd",
            outcome,
            usage,
          })
        })
      }

      default: {
        const _exhaustive: never = event
        return _exhaustive
      }
    }
  }

  // ── Main processing ──────────────────────────────────────────────

  return Stream.runForEach(config.events, processEvent).pipe(
    // Planned abort — emit TurnEnd with the abort's outcome
    Effect.catchTag("TurnAbort", (abort) =>
      emit({ _tag: "TurnEnd", outcome: abort.outcome, usage: null }),
    ),
    // Fiber interruption (user ESC) — emit TurnEnd with Interrupted
    // Defects/crashes — emit TurnEnd with EngineDefect so the UI shows
    // a real error instead of silently marking it as "interrupted".
    Effect.catchAllCause((cause) => {
      if (Cause.isInterrupted(cause)) {
        return emit({ _tag: "TurnEnd", outcome: withRequestId({ _tag: "Interrupted" }), usage: null })
      }

      const message = `Harness dispatcher defect\n${Cause.pretty(cause)}`
      return Effect.logError("[harness] Dispatcher defect", { message, cause: Cause.pretty(cause) }).pipe(
        Effect.zipRight(emit({ _tag: "TurnEnd", outcome: withRequestId({ _tag: "EngineDefect", message }), usage: null })),
      )
    }),
  )
}

// ── Helpers ──────────────────────────────────────────────────────────

function mapFinishReasonToOutcome(reason: FinishReason, toolCallCount: number, requestId: string | null): TurnOutcome {
  switch (reason) {
    case "stop":
    case "end_turn":
    case "tool_calls":
      return { _tag: "Completed", toolCallsCount: toolCallCount, requestId }
    case "length":
      return { _tag: "OutputTruncated", requestId }
    case "content_filter":
      return { _tag: "ContentFiltered", requestId }
    case "unknown":
    default:
      return { _tag: "Completed", toolCallsCount: toolCallCount, requestId }
  }
}
