import type {
  AssistantMessage,
  ToolCallPart,
  ToolCallId,
  ToolResultMessage,
  ResponseUsage,
  ToolResultPart,
  JsonValue,
} from "@magnitudedev/ai"
import type {
  HarnessEvent,
  ToolResult,
  TurnOutcome,
} from "../events"
import type { Toolkit } from "../tool/toolkit"
import type { BaseState, StateModel } from "../tool/state-model"
import { createToolHandle, type ToolHandle } from "../tool/tool-handle"
import { applyFieldChunk, extractStreamingPartialValues, type StreamingPartial } from "../tool/streaming-partial"

// ── Reducer Interface ────────────────────────────────────────────────

export interface Reducer<TState> {
  readonly initial: TState
  readonly step: (state: TState, event: HarnessEvent) => TState
}

// ── CanonicalTurnState (public) ──────────────────────────────────────

export interface CanonicalTurnState {
  readonly assistantMessage: AssistantMessage
  readonly toolResults: readonly ToolResultMessage[]
  readonly outcome: TurnOutcome | null
  readonly usage: ResponseUsage | null
}

// ── CanonicalAccumulator ──────────────────────────────────────────────

/**
 * Internal accumulation state for building canonical turns.
 * Exported for consumers who need manual replay via CanonicalAccumulatorReducer + projectCanonical.
 * Most consumers should use the harness's Ref<CanonicalTurnState> instead.
 */
export interface CanonicalAccumulator {
  readonly reasoning: string
  readonly messageText: string
  readonly toolCallMeta: ReadonlyMap<string, { readonly toolName: string; readonly toolKey: string }>
  readonly toolCallInputs: ReadonlyMap<string, JsonValue>
  readonly toolCallInputChunks: ReadonlyMap<string, StreamingPartial<unknown>>
  readonly readyToolCalls: ReadonlySet<string>
  readonly assistantMessage: AssistantMessage
  readonly toolResults: readonly ToolResultMessage[]
  readonly outcome: TurnOutcome | null
  readonly usage: ResponseUsage | null
}

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
 * Safely convert an unknown value to JsonValue via JSON round-trip.
 * Tool inputs from the model stream are always JSON-serializable by construction.
 */
function serializeToJsonValue(value: unknown): JsonValue {
  if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value
  }
  // eslint-disable-next-line @typescript-eslint/no-unsafe-return
  return JSON.parse(JSON.stringify(value))
}

/**
 * Extract JsonValue from a StreamingPartial by unwrapping StreamingLeaf wrappers
 * and coercing the result to JsonValue.
 */
function extractPartialAsJson(partial: StreamingPartial<unknown>): JsonValue {
  return serializeToJsonValue(extractStreamingPartialValues(partial))
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
  assistantMessage: { _tag: "AssistantMessage" },
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
        assistantMessage: { ...state.assistantMessage, reasoning },
      }
    }

    case "MessageDelta": {
      const messageText = state.messageText + event.text
      return {
        ...state,
        messageText,
        assistantMessage: { ...state.assistantMessage, text: messageText || undefined },
      }
    }

    case "ToolInputStarted": {
      const id = event.toolCallId
      // Placeholder input — will be replaced by ToolInputReady with the actual decoded input.
      // Empty object literal satisfies JsonValue (Record<string, JsonValue>).
      const emptyInput: JsonValue = {}
      const toolCalls: readonly ToolCallPart[] = [
        ...(state.assistantMessage.toolCalls ?? []),
        {
          _tag: "ToolCallPart" as const,
          id,
          name: event.toolName,
          input: emptyInput,
        },
      ]
      const meta = new Map(state.toolCallMeta)
      meta.set(event.toolCallId, { toolName: event.toolName, toolKey: event.toolKey })
      return {
        ...state,
        toolCallMeta: meta,
        assistantMessage: { ...state.assistantMessage, toolCalls },
      }
    }

    case "ToolInputFieldChunk": {
      const chunks = new Map(state.toolCallInputChunks)
      const existing = chunks.get(event.toolCallId) ?? {}
      chunks.set(event.toolCallId, applyFieldChunk(existing, event.path, event.delta))
      return { ...state, toolCallInputChunks: chunks }
    }

    case "ToolInputReady": {
      // ToolInputReady.input is the decoded tool input (TInput=unknown at erased level).
      // For canonical turn construction, we need JsonValue for ToolCallPart.input.
      // The raw JSON from the model stream IS JsonValue by construction.
      // At the erased event level, input is unknown — we serialize to JsonValue for storage.
      const inputAsJson: JsonValue = serializeToJsonValue(event.input)
      const toolCalls = updateToolCall(
        state.assistantMessage.toolCalls ?? [],
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
        assistantMessage: { ...state.assistantMessage, toolCalls },
      }
    }

    case "ToolResultFormatted": {
      const id = event.toolCallId
      const resultMsg: ToolResultMessage = {
        _tag: "ToolResultMessage",
        toolCallId: id,
        toolName: event.toolName,
        parts: event.parts,
      }
      return {
        ...state,
        toolResults: [...state.toolResults, resultMsg],
      }
    }

    case "TurnEnd": {
      let assistantMessage = state.assistantMessage
      // On interrupt, assemble partial inputs for tool calls that never got ToolInputReady
      if (event.outcome._tag === "Interrupted") {
        const toolCalls: readonly ToolCallPart[] = (assistantMessage.toolCalls ?? []).map((tc): ToolCallPart => {
          if (state.readyToolCalls.has(tc.id)) return tc
          const chunks = state.toolCallInputChunks.get(tc.id)
          if (chunks && Object.keys(chunks).length > 0) {
            return { _tag: "ToolCallPart", id: tc.id, name: tc.name, input: extractPartialAsJson(chunks) }
          }
          return tc
        })
        assistantMessage = { ...assistantMessage, toolCalls }
      }
      return {
        ...state,
        assistantMessage,
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

export type ToolOutcome =
  | { readonly _tag: "Completed"; readonly result: ToolResult }
  | { readonly _tag: "DecodeFailure" }

// ── EngineState ──────────────────────────────────────────────────────

export interface EngineState {
  readonly toolCallMap: ReadonlyMap<string, string>
  readonly toolOutcomes: ReadonlyMap<ToolCallId, ToolOutcome>
  readonly deadToolCalls: ReadonlySet<string>
  readonly stopped: boolean
}

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
      toolOutcomes.set(event.toolCallId, { _tag: "Completed", result: event.result })
      return { ...state, toolOutcomes }
    }

    case "ToolInputDecodeFailure": {
      const toolOutcomes = new Map(state.toolOutcomes)
      toolOutcomes.set(event.toolCallId, { _tag: "DecodeFailure" })
      const deadToolCalls = new Set(state.deadToolCalls)
      deadToolCalls.add(event.toolCallId)
      return { ...state, toolOutcomes, deadToolCalls }
    }

    case "TurnEnd":
      return { ...state, stopped: true }

    default:
      return state
  }
}

export const EngineStateReducer: Reducer<EngineState> = {
  initial: engineStateInitial,
  step: engineStateStep,
}

// ── ToolHandleState ──────────────────────────────────────────────────

export interface ToolHandleState {
  readonly handles: ReadonlyMap<string, ToolHandle>
}

const toolHandleInitial: ToolHandleState = {
  handles: new Map(),
}

export function createToolHandleReducer(toolkit: Toolkit): Reducer<ToolHandleState> {
  // Build toolKey → StateModel lookup from toolkit entries
  const stateModels = new Map<string, StateModel>()
  for (const key of toolkit.keys) {
    const entry = toolkit.entries[key]
    if (entry.state) {
      stateModels.set(key, entry.state)
    }
  }

  function step(state: ToolHandleState, event: HarnessEvent): ToolHandleState {
    if (event._tag === "ToolInputStarted") {
      const model = stateModels.get(event.toolKey)
      if (!model) return state
      const handle = createToolHandle(event.toolCallId, event.toolKey, model)
      // Process the ToolInputStarted event through the handle
      const processed = handle.process(event)
      const handles = new Map(state.handles)
      handles.set(event.toolCallId, processed)
      return { handles }
    }

    if (event._tag === "TurnEnd" && event.outcome._tag === "Interrupted") {
      const handles = new Map(state.handles)
      for (const [id, handle] of handles) {
        if (handle.state.phase !== "completed" && handle.state.phase !== "error" && handle.state.phase !== "rejected") {
          handles.set(id, handle.interrupt())
        }
      }
      return { handles }
    }

    // Delegate tool lifecycle events to the appropriate handle
    if (
      event._tag === "ToolInputFieldChunk" ||
      event._tag === "ToolInputFieldComplete" ||
      event._tag === "ToolInputReady" ||
      event._tag === "ToolInputDecodeFailure" ||
      event._tag === "ToolExecutionStarted" ||
      event._tag === "ToolExecutionEnded" ||
      event._tag === "ToolEmission" ||
      event._tag === "ToolResultFormatted"
    ) {
      const existing = state.handles.get(event.toolCallId)
      if (!existing) return state
      const processed = existing.process(event)
      if (processed === existing) return state
      const handles = new Map(state.handles)
      handles.set(event.toolCallId, processed)
      return { handles }
    }

    return state
  }

  return { initial: toolHandleInitial, step }
}
