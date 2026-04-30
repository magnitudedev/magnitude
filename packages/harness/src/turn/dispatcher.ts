import { Effect, Fiber, Cause, Schema, Layer, Stream } from "effect"
import type { ResponseStreamEvent, StreamError, ToolCallId } from "@magnitudedev/ai"
import type { StreamingPartial } from "../tool/streaming-partial"
import { applyFieldChunk, extractStreamingPartialValues } from "../tool/streaming-partial"
import type { HarnessEvent, ToolResult, TurnOutcome } from "../events"
import type { HarnessHooks, ExecuteHookContext } from "../hooks"
import type { Toolkit } from "../tool/toolkit"
import type { HarnessToolErased, ToolContext, StreamHook } from "../tool/tool"
import type { EngineState, ToolOutcome } from "./reducers"
import { formatToolResult } from "./result-formation"

// ── Config ───────────────────────────────────────────────────────────

export interface DispatchConfig {
  readonly modelStream: Stream.Stream<ResponseStreamEvent, StreamError>
  readonly toolkit: Toolkit
  readonly hooks?: HarnessHooks<unknown>
  // Erased layer — createHarness enforces type coverage at compile time.
  readonly layer?: Layer.Layer<unknown>
  readonly initialEngineState?: EngineState
  readonly emit: (event: HarnessEvent) => Effect.Effect<void>
}

// ── Per-tool-call accumulator ────────────────────────────────────────

interface ToolCallAccumulator {
  readonly toolCallId: ToolCallId
  readonly toolName: string
  readonly toolKey: string
  readonly group: string
  readonly streamingPartial: StreamingPartial<unknown>
  readonly streamState: unknown
  readonly streamHook: StreamHook<unknown, unknown, unknown, unknown, unknown> | undefined
}

// ── Dispatch ─────────────────────────────────────────────────────────

export function dispatch(config: DispatchConfig): Effect.Effect<void> {
  const { toolkit, hooks, emit, initialEngineState } = config

  // Build lookup maps from toolkit
  const toolNameToKey = new Map<string, string>()
  const toolKeyToEntry = new Map<string, { tool: HarnessToolErased; group: string }>()
  for (const key of toolkit.keys) {
    const entry = toolkit.entries[key]
    const tool = entry.tool as HarnessToolErased
    toolNameToKey.set(tool.definition.name, key)
    toolKeyToEntry.set(key, { tool, group: entry.group ?? "" })
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
  const toolFibers = new Map<ToolCallId, Fiber.RuntimeFiber<void, never>>()
  let terminalOverride: TurnOutcome | null = null
  let toolCallCount = 0

  // ── Provide layer to erased effect ───────────────────────────────

  // Erased-boundary layer provision — type coverage enforced by createHarness at compile time.
  function provideLayer<A, E>(effect: Effect.Effect<A, E, unknown>): Effect.Effect<A, E, never> {
    if (config.layer) {
      return Effect.provide(effect, config.layer) as Effect.Effect<A, E, never>
    }
    return effect as Effect.Effect<A, E, never>
  }

  // ── Emit helper for tool context ─────────────────────────────────

  function makeToolEmit(toolCallId: ToolCallId, toolName: string, toolKey: string) {
    return (value: unknown): Effect.Effect<void> =>
      Effect.gen(function* () {
        yield* emit({ _tag: "ToolEmission", toolCallId, toolName, toolKey, value })
        if (hooks?.onEmission) {
          yield* provideLayer(hooks.onEmission({ toolCallId, toolName, toolKey, value }))
        }
      })
  }

  // ── Tool execution pipeline ──────────────────────────────────────

  function executeTool(
    toolCallId: ToolCallId,
    toolName: string,
    toolKey: string,
    group: string,
    input: unknown,
  ): Effect.Effect<void> {
    const lookup = toolKeyToEntry.get(toolKey)
    if (!lookup) {
      terminalOverride = { _tag: "EngineDefect", message: `Unknown tool key: ${toolKey}` }
      return Effect.void
    }
    const { tool } = lookup

    // Check cached outcome
    const cached = cachedOutcomes.get(toolCallId)
    if (cached && cached._tag === "Completed") {
      return Effect.gen(function* () {
        yield* emit({
          _tag: "ToolExecutionStarted",
          toolCallId, toolName, toolKey, group,
          input,
          cached: true,
        })
        yield* emit({
          _tag: "ToolExecutionEnded",
          toolCallId, toolName, toolKey, group,
          result: cached.result,
        })
        const parts = yield* provideLayer(formatToolResult(toolCallId, toolName, toolKey, cached.result, hooks))
        yield* emit({ _tag: "ToolResultFormatted", toolCallId, toolName, toolKey, parts })
      })
    }

    // beforeExecute hook
    const hookCtx: ExecuteHookContext = { toolCallId, toolName, toolKey, group, input }

    return Effect.gen(function* () {
      const decision = hooks?.beforeExecute
        ? yield* provideLayer(hooks.beforeExecute(hookCtx))
        : { _tag: "Proceed" as const }

      if (decision._tag === "Reject") {
        yield* emit({
          _tag: "ToolExecutionStarted",
          toolCallId, toolName, toolKey, group,
          input,
          cached: false,
        })
        const result: ToolResult = { _tag: "Rejected", rejection: decision.rejection }
        yield* emit({ _tag: "ToolExecutionEnded", toolCallId, toolName, toolKey, group, result })
        const parts = yield* provideLayer(formatToolResult(toolCallId, toolName, toolKey, result, hooks))
        yield* emit({ _tag: "ToolResultFormatted", toolCallId, toolName, toolKey, parts })
        terminalOverride = { _tag: "GateRejected", toolCallId, toolName }
        return
      }

      const effectiveInput = decision.modifiedInput ?? input

      yield* emit({
        _tag: "ToolExecutionStarted",
        toolCallId, toolName, toolKey, group,
        input: effectiveInput,
        cached: false,
      })

      // Build ToolContext with working emit
      const toolCtx: ToolContext<unknown> = {
        emit: makeToolEmit(toolCallId, toolName, toolKey),
      }

      // Execute tool with layer provision
      const result: ToolResult = yield* Effect.gen(function* () {
        const toolEffect = tool.execute(effectiveInput, toolCtx)
        const provided = provideLayer(toolEffect)
        const output = yield* provided
        return { _tag: "Success" as const, output }
      }).pipe(
        Effect.catchAllCause((cause) => {
          const error = Cause.squash(cause)
          return Effect.succeed({ _tag: "Error" as const, error })
        }),
      )

      yield* emit({ _tag: "ToolExecutionEnded", toolCallId, toolName, toolKey, group, result })

      // afterExecute hook
      if (hooks?.afterExecute) {
        yield* provideLayer(hooks.afterExecute({ ...hookCtx, result }))
      }

      // Format result
      const parts = yield* provideLayer(formatToolResult(toolCallId, toolName, toolKey, result, hooks))
      yield* emit({ _tag: "ToolResultFormatted", toolCallId, toolName, toolKey, parts })
    })
  }

  // ── Stream event processing ──────────────────────────────────────

  function processEvent(event: ResponseStreamEvent): Effect.Effect<void> {
    switch (event._tag) {
      case "thought_start":
        return emit({ _tag: "ThoughtStart", level: event.level })

      case "thought_delta":
        return emit({ _tag: "ThoughtDelta", text: event.text })

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
          terminalOverride = { _tag: "EngineDefect", message: `Unknown tool name: ${event.toolName}` }
          return Effect.void
        }
        const entry = toolKeyToEntry.get(toolKey)
        if (!entry) {
          terminalOverride = { _tag: "EngineDefect", message: `No entry for tool key: ${toolKey}` }
          return Effect.void
        }
        toolCallCount++

        const acc: ToolCallAccumulator = {
          toolCallId: event.toolCallId,
          toolName: event.toolName,
          toolKey,
          group: entry.group,
          streamingPartial: {},
          streamState: entry.tool.stream?.initial,
          streamHook: entry.tool.stream,
        }
        accumulators.set(event.toolCallId, acc)

        return emit({
          _tag: "ToolInputStarted",
          toolCallId: event.toolCallId,
          toolName: event.toolName,
          toolKey,
          group: entry.group,
        })
      }

      case "tool_call_field_start":
        return Effect.void

      case "tool_call_field_delta": {
        const acc = accumulators.get(event.toolCallId)
        if (!acc) return Effect.void

        const field = event.path[0] ?? ""
        const newPartial = applyFieldChunk(acc.streamingPartial, event.path, event.delta)
        const updatedAcc: ToolCallAccumulator = { ...acc, streamingPartial: newPartial }
        accumulators.set(event.toolCallId, updatedAcc)

        return Effect.gen(function* () {
          yield* emit({
            _tag: "ToolInputFieldChunk",
            toolCallId: event.toolCallId,
            field,
            path: event.path,
            delta: event.delta,
          })

          // Invoke stream hook onInput if present
          if (updatedAcc.streamHook) {
            const toolCtx: ToolContext<unknown> = {
              emit: makeToolEmit(acc.toolCallId, acc.toolName, acc.toolKey),
            }
            const newStreamState = yield* provideLayer(
              updatedAcc.streamHook.onInput(newPartial, updatedAcc.streamState, toolCtx),
            ).pipe(
              Effect.catchAllCause((cause) =>
                Effect.as(
                  Effect.logWarning("Stream hook onInput failed", { cause: Cause.squash(cause) }),
                  updatedAcc.streamState,
                ),
              ),
            )
            accumulators.set(event.toolCallId, { ...updatedAcc, streamState: newStreamState })
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
          field,
          path: event.path,
          value: event.value,
        })
      }

      case "tool_call_end": {
        const acc = accumulators.get(event.toolCallId)
        if (!acc) return Effect.void

        const entry = toolKeyToEntry.get(acc.toolKey)
        if (!entry) return Effect.void

        return Effect.gen(function* () {
          // Reconstruct input from streaming partial and decode against schema
          const rawInput = extractStreamingPartialValues(acc.streamingPartial)
          const decodeResult = yield* provideLayer(
            Schema.decodeUnknown(entry.tool.definition.inputSchema)(rawInput),
          ).pipe(
            Effect.map((input) => ({ _tag: "ok" as const, input })),
            Effect.catchAll((error) => Effect.succeed({ _tag: "fail" as const, error })),
          )

          if (decodeResult._tag === "fail") {
            yield* emit({
              _tag: "ToolInputDecodeFailure",
              toolCallId: acc.toolCallId,
              toolName: acc.toolName,
              toolKey: acc.toolKey,
              group: acc.group,
              detail: decodeResult.error,
            })
            terminalOverride = {
              _tag: "ToolInputDecodeFailure",
              toolCallId: acc.toolCallId,
              toolName: acc.toolName,
              detail: decodeResult.error,
            }
            return
          }

          yield* emit({
            _tag: "ToolInputReady",
            toolCallId: acc.toolCallId,
            input: decodeResult.input,
          })

          // Fork tool execution concurrently
          const fiber = yield* Effect.fork(
            executeTool(acc.toolCallId, acc.toolName, acc.toolKey, acc.group, decodeResult.input),
          )
          toolFibers.set(acc.toolCallId, fiber)
        })
      }

      case "response_done": {
        return Effect.gen(function* () {
          // Join all in-flight tool fibers
          for (const [, fiber] of toolFibers) {
            yield* Fiber.join(fiber)
          }
          toolFibers.clear()

          const outcome: TurnOutcome = terminalOverride ?? mapReasonToOutcome(event.reason, toolCallCount)
          yield* emit({
            _tag: "TurnEnd",
            outcome,
            usage: event.usage ?? null,
          })
        })
      }

      default: {
        const _exhaustive: never = event
        return _exhaustive
      }
    }
  }

  // ── Main processing with error/interrupt handling ────────────────

  function interruptAllTools(): Effect.Effect<void> {
    return Effect.gen(function* () {
      for (const [toolCallId, fiber] of toolFibers) {
        yield* Fiber.interrupt(fiber)
        const acc = accumulators.get(toolCallId)
        yield* emit({
          _tag: "ToolExecutionEnded",
          toolCallId,
          toolName: acc?.toolName ?? "",
          toolKey: acc?.toolKey ?? "",
          group: acc?.group ?? "",
          result: { _tag: "Interrupted" },
        })
      }
      toolFibers.clear()
    })
  }

  return Stream.runForEach(config.modelStream, processEvent).pipe(
    Effect.catchAllCause(() =>
      Effect.gen(function* () {
        yield* interruptAllTools()
        yield* emit({ _tag: "TurnEnd", outcome: { _tag: "Interrupted" }, usage: null })
      }),
    ),
  )
}

// ── Helpers ──────────────────────────────────────────────────────────

function mapReasonToOutcome(reason: string, toolCallCount: number): TurnOutcome {
  switch (reason) {
    case "stop":
    case "end_turn":
    case "tool_use":
      return { _tag: "Completed", toolCallsCount: toolCallCount }
    case "max_tokens":
    case "length":
      return { _tag: "OutputTruncated" }
    case "content_filter":
      return { _tag: "ContentFiltered" }
    default:
      return { _tag: "Completed", toolCallsCount: toolCallCount }
  }
}
