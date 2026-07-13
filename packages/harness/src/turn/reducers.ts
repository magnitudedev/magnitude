import type {
  AssistantMessage,
  ToolCallPart,
  ToolCallId,
  ProviderToolCallId,
  ResponseUsage,
  JsonValue,
} from "@magnitudedev/ai"
import {
  AssistantMessageSchema,
  JsonValueSchema,
  ToolCallPartSchema,
  ProviderToolCallIdSchema as AiProviderToolCallIdSchema,
  ToolCallIdSchema as AiToolCallIdSchema,
} from "@magnitudedev/ai"
import { makeSchemaUnionFromEntries, type NonEmptySchemaEntries } from "@magnitudedev/utils/schema"
import type {
  HarnessEvent,
  StreamFailedTerminal,
  ToolResult,
  ToolResultEntry,
  TurnOutcome,
} from "../events"
import type { Toolkit } from "../tool/toolkit"
import { BaseStateSchema, type BaseState, type StateModel } from "../tool/state-model"
import { createToolHandle, interruptToolHandle, processToolHandle, type ToolHandle } from "../tool/tool-handle"
import type { StreamingPartial } from "@magnitudedev/ai"
import { applyFieldChunk, extractStreamingPartialValues } from "../tool/streaming-partial"
import { Schema, Option } from "effect"

const ToolCallIdSchema = AiToolCallIdSchema
const ProviderToolCallIdSchema = AiProviderToolCallIdSchema

// ── Reducer Interface ────────────────────────────────────────────────

export interface Reducer<TState> {
  readonly initial: TState
  readonly step: (state: TState, event: HarnessEvent) => TState
}

// ── CanonicalTurnState (public) ──────────────────────────────────────

export const ResponseUsageSchema = Schema.Struct({
  inputTokens: Schema.Number,
  outputTokens: Schema.Number,
  cacheReadTokens: Schema.Number,
  cacheWriteTokens: Schema.Number,
  cost: Schema.NullOr(Schema.Number),
})
export type ResponseUsageState = typeof ResponseUsageSchema.Type

export const ValidationIssueSchema = Schema.Struct({
  path: Schema.Array(Schema.Union(Schema.String, Schema.Number)),
  message: Schema.String,
})

export const ToolErrorSchema = Schema.Struct({
  message: Schema.String,
})

export const ToolResultSchema = Schema.Union(
  Schema.TaggedStruct("Success", { output: JsonValueSchema }),
  Schema.TaggedStruct("Error", { error: ToolErrorSchema }),
  Schema.TaggedStruct("Denied", { denial: JsonValueSchema }),
  Schema.TaggedStruct("Interrupted", {}),
  Schema.TaggedStruct("InputRejected", { issue: ValidationIssueSchema, partialInput: JsonValueSchema }),
)

export const ToolResultEntrySchema = Schema.Struct({
  toolCallId: ToolCallIdSchema,
  providerToolCallId: ProviderToolCallIdSchema,
  toolName: Schema.String,
  result: ToolResultSchema,
})

export const SafetyStopReasonSchema = Schema.Union(
  Schema.TaggedStruct("IdenticalResponseCircuitBreaker", { threshold: Schema.Number }),
  Schema.TaggedStruct("Other", { message: Schema.String }),
)

export const StreamFailedTerminalSchema = Schema.TaggedStruct("StreamFailed", {
  cause: Schema.Unknown,
  usage: Schema.Unknown,
}) as Schema.Schema<StreamFailedTerminal>

export const TurnOutcomeSchema = Schema.Union(
  Schema.extend(Schema.TaggedStruct("Completed", { toolCallsCount: Schema.Number }), Schema.Struct({ requestId: Schema.NullOr(Schema.String) })),
  Schema.extend(Schema.TaggedStruct("OutputTruncated", {}), Schema.Struct({ requestId: Schema.NullOr(Schema.String) })),
  Schema.extend(Schema.TaggedStruct("ContentFiltered", {}), Schema.Struct({ requestId: Schema.NullOr(Schema.String) })),
  Schema.extend(Schema.TaggedStruct("SafetyStop", { reason: SafetyStopReasonSchema }), Schema.Struct({ requestId: Schema.NullOr(Schema.String) })),
  Schema.extend(Schema.TaggedStruct("ToolInputValidationFailure", {
    toolCallId: ToolCallIdSchema,
    providerToolCallId: ProviderToolCallIdSchema,
    toolName: Schema.String,
    toolKey: Schema.String,
    issue: ValidationIssueSchema,
  }), Schema.Struct({ requestId: Schema.NullOr(Schema.String) })),
  Schema.extend(Schema.TaggedStruct("ToolExecutionError", {
    toolCallId: ToolCallIdSchema,
    providerToolCallId: ProviderToolCallIdSchema,
    toolName: Schema.String,
    toolKey: Schema.String,
    error: ToolErrorSchema,
  }), Schema.Struct({ requestId: Schema.NullOr(Schema.String) })),
  Schema.extend(Schema.TaggedStruct("GateRejected", {
    toolCallId: ToolCallIdSchema,
    providerToolCallId: ProviderToolCallIdSchema,
    toolName: Schema.String,
  }), Schema.Struct({ requestId: Schema.NullOr(Schema.String) })),
  Schema.extend(Schema.TaggedStruct("StreamFailed", { message: Schema.String, terminal: StreamFailedTerminalSchema }), Schema.Struct({ requestId: Schema.NullOr(Schema.String) })),
  Schema.extend(Schema.TaggedStruct("EngineDefect", { message: Schema.String }), Schema.Struct({ requestId: Schema.NullOr(Schema.String) })),
  Schema.extend(Schema.TaggedStruct("Interrupted", {}), Schema.Struct({ requestId: Schema.NullOr(Schema.String) })),
  Schema.extend(Schema.TaggedStruct("ThoughtLimitExceeded", { limit: Schema.Number }), Schema.Struct({ requestId: Schema.NullOr(Schema.String) })),
)

export const CanonicalTurnStateSchema = Schema.Struct({
  assistantMessage: AssistantMessageSchema,
  toolResults: Schema.Array(ToolResultEntrySchema),
  outcome: Schema.NullOr(TurnOutcomeSchema),
  usage: Schema.NullOr(ResponseUsageSchema),
})
export type CanonicalTurnState = typeof CanonicalTurnStateSchema.Type
type ToolResultState = ToolResult
type ToolResultEntryState = ToolResultEntry
type TurnOutcomeState = TurnOutcome

// ── CanonicalAccumulator ──────────────────────────────────────────────

/**
 * Internal accumulation state for building canonical turns.
 * Exported for consumers who need manual replay via CanonicalAccumulatorReducer + projectCanonical.
 * Most consumers should use the harness's Ref<CanonicalTurnState> instead.
 */
export const ToolCallMetaSchema = Schema.Struct({
  providerToolCallId: ProviderToolCallIdSchema,
  toolName: Schema.String,
  toolKey: Schema.String,
})

export const StreamingLeafSchema = Schema.Struct({
  isFinal: Schema.Boolean,
  value: JsonValueSchema,
})

export const StreamingPartialSchema: Schema.Schema<StreamingPartial<unknown>> = Schema.suspend(() =>
  Schema.Record({ key: Schema.String, value: Schema.Union(StreamingLeafSchema, StreamingPartialSchema) }) as Schema.Schema<StreamingPartial<unknown>>
)

export const CanonicalAccumulatorSchema = Schema.Struct({
  reasoning: Schema.String,
  messageText: Schema.String,
  toolCallMeta: Schema.ReadonlyMap({ key: Schema.String, value: ToolCallMetaSchema }),
  toolCallInputs: Schema.ReadonlyMap({ key: Schema.String, value: JsonValueSchema }),
  toolCallInputChunks: Schema.ReadonlyMap({ key: Schema.String, value: StreamingPartialSchema }),
  readyToolCalls: Schema.ReadonlySet(Schema.String),
  assistantMessage: AssistantMessageSchema,
  toolResults: Schema.Array(ToolResultEntrySchema),
  outcome: Schema.NullOr(TurnOutcomeSchema),
  usage: Schema.NullOr(ResponseUsageSchema),
})
export type CanonicalAccumulator = typeof CanonicalAccumulatorSchema.Type

/** Project internal accumulator to public CanonicalTurnState */
export function projectCanonical(acc: CanonicalAccumulator): CanonicalTurnState {
  return {
    assistantMessage: acc.assistantMessage,
    toolResults: acc.toolResults,
    outcome: acc.outcome,
    usage: acc.usage,
  }
}

/**
 * Extract JsonValue from a StreamingPartial by unwrapping StreamingLeaf wrappers
 * and requiring the result to be JSON.
 */
function extractPartialAsJson(partial: StreamingPartial<unknown>): JsonValue {
  const value = extractStreamingPartialValues(partial)
  const isJsonValue = Schema.is(JsonValueSchema)
  if (isJsonValue(value)) return value
  throw new Error("Streaming tool input partial decoded to a non-JSON value")
}

function updateToolCall(
  toolCalls: readonly ToolCallPart[],
  toolCallId: ToolCallId,
  updater: (tc: ToolCallPart) => ToolCallPart,
): readonly ToolCallPart[] {
  return toolCalls.map((tc) => (tc.id === toolCallId ? updater(tc) : tc))
}

const canonicalAccumulatorInitial: CanonicalAccumulator = {
  reasoning: "",
  messageText: "",
  toolCallMeta: new Map(),
  toolCallInputs: new Map(),
  toolCallInputChunks: new Map(),
  readyToolCalls: new Set(),
  assistantMessage: { _tag: "AssistantMessage", reasoning: Option.none(), text: Option.none(), toolCalls: Option.none() },
  toolResults: [],
  outcome: null,
  usage: null,
}

function canonicalAccumulatorStep(state: CanonicalAccumulator, event: HarnessEvent): CanonicalAccumulator {
  switch (event._tag) {
    case "ThoughtDelta": {
      const reasoning = state.reasoning + event.text
      return {
        ...state,
        reasoning,
        assistantMessage: { ...state.assistantMessage, reasoning: Option.some(reasoning) },
      }
    }

    case "MessageDelta": {
      const messageText = state.messageText + event.text
      return {
        ...state,
        messageText,
        assistantMessage: { ...state.assistantMessage, text: messageText ? Option.some(messageText) : Option.none() },
      }
    }

    case "ToolInputStarted": {
      const id = event.toolCallId
      // Placeholder input — will be replaced by ToolInputReady with the actual decoded input.
      // Empty object literal satisfies JsonValue (Record<string, JsonValue>).
      const emptyInput: JsonValue = {}
      const toolCalls: readonly ToolCallPart[] = [
        ...(Option.getOrElse(state.assistantMessage.toolCalls, () => [] as readonly ToolCallPart[])),
        {
          _tag: "ToolCallPart" as const,
          id,
          providerToolCallId: event.providerToolCallId,
          name: event.toolName,
          input: emptyInput,
        },
      ]
      const meta = new Map(state.toolCallMeta)
      meta.set(event.toolCallId, { providerToolCallId: event.providerToolCallId, toolName: event.toolName, toolKey: event.toolKey })
      return {
        ...state,
        toolCallMeta: meta,
        assistantMessage: { ...state.assistantMessage, toolCalls: Option.some(toolCalls) },
      }
    }

    case "ToolInputFieldChunk": {
      const chunks = new Map(state.toolCallInputChunks)
      const existing = chunks.get(event.toolCallId) ?? {}
      chunks.set(event.toolCallId, applyFieldChunk(existing, event.path, event.delta))
      return { ...state, toolCallInputChunks: chunks }
    }

    case "ToolInputReady": {
      // ToolInputReady no longer carries input — the canonical input is assembled
      // from ToolInputFieldChunk/ToolInputFieldComplete events via toolCallInputChunks.
      // Extract the accumulated partial as the final input for the tool call.
      const chunks = state.toolCallInputChunks.get(event.toolCallId)
      const inputAsJson: JsonValue = chunks && Object.keys(chunks).length > 0
        ? extractPartialAsJson(chunks)
        : {}
      const toolCalls = updateToolCall(
        Option.getOrElse(state.assistantMessage.toolCalls, () => [] as readonly ToolCallPart[]),
        event.toolCallId,
        (tc) => ({ ...tc, input: inputAsJson }),
      )
      const ready = new Set(state.readyToolCalls)
      ready.add(event.toolCallId)
      const inputs = new Map(state.toolCallInputs)
      inputs.set(event.toolCallId, inputAsJson)
      return {
        ...state,
        readyToolCalls: ready,
        toolCallInputs: inputs,
        assistantMessage: { ...state.assistantMessage, toolCalls: Option.some(toolCalls) },
      }
    }

    case "ToolExecutionEnded": {
      const result: ToolResultEntryState = {
        toolCallId: event.toolCallId,
        providerToolCallId: event.providerToolCallId,
        toolName: event.toolName,
        result: event.result,
      }
      return {
        ...state,
        toolResults: [...state.toolResults, result],
      }
    }

    case "ToolInputRejected": {
      const chunks = state.toolCallInputChunks.get(event.toolCallId)
      const partialInput: JsonValue = chunks && Object.keys(chunks).length > 0
        ? extractPartialAsJson(chunks)
        : {}
      const result: ToolResultEntryState = {
        toolCallId: event.toolCallId,
        providerToolCallId: event.providerToolCallId,
        toolName: event.toolName,
        result: {
          _tag: "InputRejected",
          issue: event.issue,
          partialInput,
        },
      }
      return {
        ...state,
        toolResults: [...state.toolResults, result],
      }
    }

    case "TurnEnd": {
      let assistantMessage = state.assistantMessage

      // Assemble partial inputs for any tool calls that never got ToolInputReady,
      // regardless of why the turn ended. This ensures the LLM sees what it actually
      // sent (e.g. a path that failed validation) instead of an empty {} placeholder.
      {
        const toolCalls: readonly ToolCallPart[] = Option.getOrElse(assistantMessage.toolCalls, () => [] as readonly ToolCallPart[]).map((tc): ToolCallPart => {
          if (state.readyToolCalls.has(tc.id)) return tc
          const chunks = state.toolCallInputChunks.get(tc.id)
          if (chunks && Object.keys(chunks).length > 0) {
            return { _tag: "ToolCallPart", id: tc.id, providerToolCallId: tc.providerToolCallId, name: tc.name, input: extractPartialAsJson(chunks) }
          }
          return tc
        })
        assistantMessage = { ...assistantMessage, toolCalls: Option.some(toolCalls) }
      }

      // Add synthetic results for tool calls that never got a result event.
      // This ensures every ToolCallPart has a matching ToolResultEntry,
      // satisfying the OpenAI API constraint that every tool call must have a result.
      const toolResults = [...state.toolResults]
      for (const tc of Option.getOrElse(assistantMessage.toolCalls, () => [] as readonly ToolCallPart[])) {
        if (!toolResults.some(r => r.toolCallId === tc.id)) {
          toolResults.push({
              toolCallId: tc.id,
            providerToolCallId: tc.providerToolCallId,
            toolName: tc.name,
            result: { _tag: "Interrupted" },
          })
        }
      }

      return {
        ...state,
        assistantMessage,
        toolResults,
        outcome: event.outcome,
        usage: event.usage,
      }
    }

    default:
      return state
  }
}

/**
 * Canonical accumulator reducer — carries internal accumulation state.
 * The harness uses this internally with dual-ref pattern:
 *   - Internal Ref<CanonicalAccumulator> for state between steps
 *   - Public Ref<CanonicalTurnState> projected via projectCanonical after each step
 *
 * External consumers doing manual replay should use this reducer
 * and call projectCanonical() to extract the public view.
 */
export const CanonicalAccumulatorReducer: Reducer<CanonicalAccumulator> = {
  initial: canonicalAccumulatorInitial,
  step: canonicalAccumulatorStep,
}

// ── ToolOutcome ──────────────────────────────────────────────────────

export const ToolOutcomeSchema = Schema.Union(
  Schema.TaggedStruct("Completed", { result: ToolResultSchema }),
  Schema.TaggedStruct("InputRejected", {}),
)
export type ToolOutcome = typeof ToolOutcomeSchema.Type

// ── EngineState ──────────────────────────────────────────────────────

export const EngineStateSchema = Schema.Struct({
  toolCallMap: Schema.ReadonlyMap({ key: ToolCallIdSchema, value: Schema.String }),
  toolOutcomes: Schema.ReadonlyMap({ key: ToolCallIdSchema, value: ToolOutcomeSchema }),
  deadToolCalls: Schema.ReadonlySet(Schema.String),
  stopped: Schema.Boolean,
})
export type EngineState = typeof EngineStateSchema.Type

const engineStateInitial: EngineState = {
  toolCallMap: new Map(),
  toolOutcomes: new Map(),
  deadToolCalls: new Set(),
  stopped: false,
}

function engineStateStep(state: EngineState, event: HarnessEvent): EngineState {
  switch (event._tag) {
    case "ToolInputStarted": {
      const toolCallMap = new Map(state.toolCallMap)
      toolCallMap.set(event.toolCallId, event.toolKey)
      return { ...state, toolCallMap }
    }

    case "ToolExecutionEnded": {
      const toolOutcomes = new Map(state.toolOutcomes)
      const result: ToolResultState = event.result
      toolOutcomes.set(event.toolCallId, { _tag: "Completed", result })
      return { ...state, toolOutcomes }
    }

    case "ToolInputRejected": {
      const deadToolCalls = new Set(state.deadToolCalls)
      deadToolCalls.add(event.toolCallId)
      return { ...state, deadToolCalls }
    }

    case "TurnEnd": {
      let newState = state
      if (event.outcome._tag === "ToolInputValidationFailure") {
        const toolOutcomes = new Map(state.toolOutcomes)
        toolOutcomes.set(event.outcome.toolCallId, { _tag: "InputRejected" })
        const deadToolCalls = new Set(state.deadToolCalls)
        deadToolCalls.add(event.outcome.toolCallId)
        newState = { ...state, toolOutcomes, deadToolCalls }
      }
      return { ...newState, stopped: true }
    }

    default:
      return state
  }
}

export const EngineStateReducer: Reducer<EngineState> = {
  initial: engineStateInitial,
  step: engineStateStep,
}

// ── TurnState (unified) ──────────────────────────────────────────────

export function makeToolHandleSchema<TToolStateSchema extends Schema.Schema.AnyNoContext>(
  toolStateSchema: TToolStateSchema,
) {
  return Schema.Struct({
    toolCallId: ToolCallIdSchema,
    providerToolCallId: ProviderToolCallIdSchema,
    toolKey: Schema.String,
    state: toolStateSchema,
  })
}

export function makeKeyedToolHandleSchema<
  const TToolKey extends string,
  TToolStateSchema extends Schema.Schema.AnyNoContext,
>(
  toolKey: TToolKey,
  toolStateSchema: TToolStateSchema,
) {
  return Schema.Struct({
    toolCallId: ToolCallIdSchema,
    providerToolCallId: ProviderToolCallIdSchema,
    toolKey: Schema.Literal(toolKey),
    state: toolStateSchema,
  })
}

export type KeyedToolHandleSchemaEntries<TEntries extends NonEmptySchemaEntries> = {
  readonly [Index in keyof TEntries]: TEntries[Index] extends readonly [
    infer TToolKey extends string,
    infer TToolStateSchema extends Schema.Schema.AnyNoContext,
  ]
    ? readonly [TToolKey, ReturnType<typeof makeKeyedToolHandleSchema<TToolKey, TToolStateSchema>>]
    : TEntries[Index]
}

export function makeKeyedToolHandleSchemaEntries<const TEntries extends NonEmptySchemaEntries>(
  entries: TEntries,
): KeyedToolHandleSchemaEntries<TEntries> {
  return entries.map(([toolKey, stateSchema]) => [
    toolKey,
    makeKeyedToolHandleSchema(toolKey, stateSchema),
  ] as const) as KeyedToolHandleSchemaEntries<TEntries>
}

export function makeKeyedToolHandleUnionSchemaFromEntries<const TEntries extends NonEmptySchemaEntries>(
  entries: TEntries,
) {
  return makeSchemaUnionFromEntries(makeKeyedToolHandleSchemaEntries(entries))
}

export function makeToolHandleStateSchema<TToolStateSchema extends Schema.Schema.AnyNoContext>(
  toolStateSchema: TToolStateSchema,
) {
  return Schema.Struct({
    handles: Schema.ReadonlyMap({ key: Schema.String, value: makeToolHandleSchema(toolStateSchema) }),
  })
}

export function makeToolHandleStateSchemaFromHandle<TToolHandleSchema extends Schema.Schema.AnyNoContext>(
  toolHandleSchema: TToolHandleSchema,
) {
  return Schema.Struct({
    handles: Schema.ReadonlyMap({ key: Schema.String, value: toolHandleSchema }),
  })
}

export function makeTurnStateSchema<TToolStateSchema extends Schema.Schema.AnyNoContext>(
  toolStateSchema: TToolStateSchema,
) {
  return Schema.Struct({
    _accumulator: CanonicalAccumulatorSchema,
    canonical: CanonicalTurnStateSchema,
    engine: EngineStateSchema,
    handles: makeToolHandleStateSchema(toolStateSchema),
  })
}

export function makeTurnStateSchemaFromToolHandle<TToolHandleSchema extends Schema.Schema.AnyNoContext>(
  toolHandleSchema: TToolHandleSchema,
) {
  return Schema.Struct({
    _accumulator: CanonicalAccumulatorSchema,
    canonical: CanonicalTurnStateSchema,
    engine: EngineStateSchema,
    handles: makeToolHandleStateSchemaFromHandle(toolHandleSchema),
  })
}

export const TurnStateSchema = makeTurnStateSchema(BaseStateSchema)

export const ToolHandleStateSchema = makeToolHandleStateSchema(BaseStateSchema)
export interface ToolHandleStateForHandle<TToolHandle extends ToolHandle = ToolHandle> {
  readonly handles: ReadonlyMap<string, TToolHandle>
}
export type ToolHandleState = ToolHandleStateForHandle

export interface TurnStateForToolHandle<TToolHandle extends ToolHandle = ToolHandle> {
  readonly _accumulator: CanonicalAccumulator
  readonly canonical: CanonicalTurnState
  readonly engine: EngineState
  readonly handles: ToolHandleStateForHandle<TToolHandle>
}

export type TurnState = TurnStateForToolHandle

export function createTurnReducer<TToolHandle extends ToolHandle = ToolHandle>(
  toolkit: Toolkit,
): Reducer<TurnStateForToolHandle<TToolHandle>> {
  const toolHandleReducer = createToolHandleReducer<TToolHandle>(toolkit)

  const initial: TurnStateForToolHandle<TToolHandle> = {
    _accumulator: CanonicalAccumulatorReducer.initial,
    canonical: projectCanonical(CanonicalAccumulatorReducer.initial),
    engine: EngineStateReducer.initial,
    handles: toolHandleReducer.initial,
  }

  function step(state: TurnStateForToolHandle<TToolHandle>, event: HarnessEvent): TurnStateForToolHandle<TToolHandle> {
    const _accumulator = CanonicalAccumulatorReducer.step(state._accumulator, event)
    const canonical = projectCanonical(_accumulator)
    const engine = EngineStateReducer.step(state.engine, event)
    const handles = toolHandleReducer.step(state.handles, event)
    return { _accumulator, canonical, engine, handles }
  }

  return { initial, step }
}

export function createToolHandleReducer<TToolHandle extends ToolHandle = ToolHandle>(
  toolkit: Toolkit,
): Reducer<ToolHandleStateForHandle<TToolHandle>> {
  // Build toolKey → StateModel lookup from toolkit entries
  const stateModels = new Map<string, StateModel>()
  for (const key of toolkit.keys) {
    const entry = toolkit.entries[key]
    if (entry.state) {
      stateModels.set(key, entry.state)
    }
  }

  const initial: ToolHandleStateForHandle<TToolHandle> = { handles: new Map() }

  function step(state: ToolHandleStateForHandle<TToolHandle>, event: HarnessEvent): ToolHandleStateForHandle<TToolHandle> {
    if (event._tag === "ToolInputStarted") {
      const model = stateModels.get(event.toolKey)
      if (!model) return state
      const handle = createToolHandle(event.toolCallId, event.providerToolCallId, event.toolKey, model)
      const processed = processToolHandle(handle, event, model)
      const handles = new Map(state.handles)
      handles.set(event.toolCallId, processed as TToolHandle)
      return { handles }
    }

    if (event._tag === "TurnEnd") {
      // Interrupt ALL non-terminal handles on any TurnEnd.
      // The turn is over — any handle not in a terminal state should be interrupted.
      const handles = new Map(state.handles)
      for (const [id, handle] of handles) {
        if (handle.state.phase !== "completed" && handle.state.phase !== "error" && handle.state.phase !== "rejected") {
          const model = stateModels.get(handle.toolKey)
          if (model) {
            handles.set(id, interruptToolHandle(handle, model))
          }
        }
      }
      return { handles }
    }

    // Delegate tool lifecycle events to the appropriate handle
    if (
      event._tag === "ToolInputFieldChunk" ||
      event._tag === "ToolInputFieldComplete" ||
      event._tag === "ToolInputReady" ||
      event._tag === "ToolInputRejected" ||
      event._tag === "ToolExecutionStarted" ||
      event._tag === "ToolExecutionEnded" ||
      event._tag === "ToolEmission"
    ) {
      const existing = state.handles.get(event.toolCallId)
      if (!existing) return state
      const model = stateModels.get(existing.toolKey)
      if (!model) return state
      const processed = processToolHandle(existing, event, model)
      if (processed === existing) return state
      const handles = new Map(state.handles)
      handles.set(event.toolCallId, processed)
      return { handles }
    }

    return state
  }

  return { initial, step }
}
