/**
 * HarnessStateProjection (Forked)
 *
 * Unified projection that delegates to the harness's pure `createTurnReducer`.
 * Replaces three separate projections:
 *   - CanonicalTurnProjection (canonical assistant message + tool results)
 *   - ReplayProjection (engine state for crash recovery)
 *   - ToolStateProjection (tool handle lifecycle state)
 *
 * Translates AppEvents → HarnessEvents and steps the composed reducer.
 */

import { Projection } from '@magnitudedev/event-core'
import type { ResponseUsage } from '@magnitudedev/ai'
import {
  createTurnReducer,
  defineToolkit,
  interruptToolHandle,
  makeTurnStateSchemaFromToolHandle,
  type HarnessEvent,
  type ToolHandle,
  type Toolkit,
  type TurnStateForToolHandle,
  type TurnOutcome as HarnessTurnOutcome,
} from '@magnitudedev/harness'
import type { AppEvent, TurnOutcomeEvent } from '../events'
import { ToolUniverseAmbient } from '../ambient/tool-universe-ambient'
import { ToolHandleSchema, type ToolHandleFromSchema } from '../models/tool-handle-schema'

// ── Translation: AppEvent → HarnessEvent ─────────────────────────────

export function translateToHarnessEvent(event: AppEvent): HarnessEvent | null {
  switch (event.type) {
    case 'thinking_start':
      return { _tag: 'ThoughtStart', level: 'medium' }
    case 'thinking_chunk':
      return { _tag: 'ThoughtDelta', text: event.text }
    case 'thinking_end':
      return { _tag: 'ThoughtEnd' }
    case 'message_start':
      return { _tag: 'MessageStart' }
    case 'message_chunk':
      return { _tag: 'MessageDelta', text: event.text }
    case 'message_end':
      return { _tag: 'MessageEnd' }
    case 'tool_event':
      // ToolLifecycleEvent is a subset of HarnessEvent — pass through directly
      return event.event
    default:
      return null
  }
}

// ── Translation: Agent TurnOutcome → Harness TurnEnd ─────────────────

type AgentTurnOutcome = TurnOutcomeEvent['outcome']
type MappableAgentTurnOutcome = Extract<
  AgentTurnOutcome,
  {
    readonly _tag:
      | 'Completed'
      | 'ToolInputValidationFailure'
      | 'ToolExecutionError'
      | 'GateRejected'
      | 'OutputTruncated'
      | 'SafetyStop'
      | 'Cancelled'
      | 'Overthinking'
  }
>
type MappableTurnOutcomeEvent = Omit<TurnOutcomeEvent, 'outcome'> & {
  readonly outcome: MappableAgentTurnOutcome
}

function isMappableTurnOutcome(outcome: AgentTurnOutcome): outcome is MappableAgentTurnOutcome {
  switch (outcome._tag) {
    case 'Completed':
    case 'ToolInputValidationFailure':
    case 'ToolExecutionError':
    case 'GateRejected':
    case 'OutputTruncated':
    case 'SafetyStop':
    case 'Cancelled':
    case 'Overthinking':
      return true
    case 'ProviderNotReady':
    case 'ConnectionFailure':
    case 'StreamFailed':
    case 'ContextWindowExceeded':
    case 'UnexpectedError':
      return false
  }
  const exhaustive: never = outcome
  return exhaustive
}

function translateAgentOutcome(agentOutcome: MappableAgentTurnOutcome): HarnessTurnOutcome {
  switch (agentOutcome._tag) {
    case 'Completed':
      return { _tag: 'Completed', toolCallsCount: agentOutcome.completion.toolCallsCount, requestId: agentOutcome.requestId }
    case 'Cancelled':
      return { _tag: 'Interrupted', requestId: agentOutcome.requestId }
    case 'OutputTruncated':
      return { _tag: 'OutputTruncated', requestId: agentOutcome.requestId }
    case 'SafetyStop':
      return { _tag: 'SafetyStop', reason: agentOutcome.reason, requestId: agentOutcome.requestId }
    case 'ToolInputValidationFailure':
      return {
        _tag: 'ToolInputValidationFailure',
        toolCallId: agentOutcome.toolCallId,
        providerToolCallId: agentOutcome.providerToolCallId,
        toolName: agentOutcome.toolName,
        toolKey: agentOutcome.toolKey,
        issue: agentOutcome.issue,
        requestId: agentOutcome.requestId,
      }
    case 'ToolExecutionError':
      return {
        _tag: 'ToolExecutionError',
        toolCallId: agentOutcome.toolCallId,
        providerToolCallId: agentOutcome.providerToolCallId,
        toolName: agentOutcome.toolName,
        toolKey: agentOutcome.toolKey,
        error: agentOutcome.error,
        requestId: agentOutcome.requestId,
      }
    case 'GateRejected':
      return {
        _tag: 'GateRejected',
        toolCallId: agentOutcome.toolCallId,
        providerToolCallId: agentOutcome.providerToolCallId,
        toolName: agentOutcome.toolName,
        requestId: agentOutcome.requestId,
      }
    case 'Overthinking':
      return { _tag: 'ThoughtLimitExceeded', limit: agentOutcome.limit, requestId: agentOutcome.requestId }
  }
  const exhaustive: never = agentOutcome
  return exhaustive
}

export function translateTurnOutcome(event: MappableTurnOutcomeEvent): HarnessEvent {
  const harnessOutcome = translateAgentOutcome(event.outcome)

  const usage: ResponseUsage | null =
    event.inputTokens != null || event.outputTokens != null
      ? {
          inputTokens: event.inputTokens ?? 0,
          outputTokens: event.outputTokens ?? 0,
          cacheReadTokens: event.cacheReadTokens ?? 0,
          cacheWriteTokens: event.cacheWriteTokens ?? 0,
          cost: event.cost ?? null,
        }
      : null

  return {
    _tag: 'TurnEnd',
    outcome: harnessOutcome,
    usage,
  }
}

// ── Reducer cache (one per toolkit identity) ─────────────────────────

type HarnessTurnReducer = ReturnType<typeof createTurnReducer<ToolHandleFromSchema>>

const reducerCache = new WeakMap<Toolkit, HarnessTurnReducer>()

function getCachedReducer(toolkit: Toolkit): HarnessTurnReducer {
  let r = reducerCache.get(toolkit)
  if (!r) {
    r = createTurnReducer<ToolHandleFromSchema>(toolkit)
    reducerCache.set(toolkit, r)
  }
  return r
}

function stepEvent(fork: HarnessTurnState, event: AppEvent, toolkit: Toolkit): HarnessTurnState {
  const harnessEvent = translateToHarnessEvent(event)
  if (!harnessEvent) return fork
  return getCachedReducer(toolkit).step(fork, harnessEvent)
}

// ── Initial state (toolkit-independent) ──────────────────────────────

const emptyToolkit = defineToolkit({})
const emptyReducer = createTurnReducer<ToolHandleFromSchema>(emptyToolkit)
export const HarnessTurnStateSchema = makeTurnStateSchemaFromToolHandle(ToolHandleSchema)
export type HarnessTurnState = TurnStateForToolHandle<ToolHandleFromSchema>

const emptyHarnessTurnState: HarnessTurnState = {
  _accumulator: emptyReducer.initial._accumulator,
  canonical: emptyReducer.initial.canonical,
  engine: emptyReducer.initial.engine,
  handles: { handles: new Map() },
}

// ── Projection ───────────────────────────────────────────────────────

export const HarnessStateProjection = Projection.defineForked<AppEvent>()({
  name: 'HarnessState',
  forkState: HarnessTurnStateSchema,
  ambients: [ToolUniverseAmbient] as const,
  initialFork: emptyHarnessTurnState,

  eventHandlers: {
    turn_started: ({ fork }) => ({
      // Reset canonical + handles for new turn.
      // Keep engine state for crash recovery (reset on turn_outcome).
      ...emptyHarnessTurnState,
      engine: fork.engine,
    }),

    thinking_start: ({ event, fork, ambient }) => stepEvent(fork, event, ambient.get(ToolUniverseAmbient)),
    thinking_chunk: ({ event, fork, ambient }) => stepEvent(fork, event, ambient.get(ToolUniverseAmbient)),
    thinking_end: ({ event, fork, ambient }) => stepEvent(fork, event, ambient.get(ToolUniverseAmbient)),
    message_start: ({ event, fork, ambient }) => stepEvent(fork, event, ambient.get(ToolUniverseAmbient)),
    message_chunk: ({ event, fork, ambient }) => stepEvent(fork, event, ambient.get(ToolUniverseAmbient)),
    message_end: ({ event, fork, ambient }) => stepEvent(fork, event, ambient.get(ToolUniverseAmbient)),
    tool_event: ({ event, fork, ambient }) => stepEvent(fork, event, ambient.get(ToolUniverseAmbient)),

    turn_outcome: ({ event, fork, ambient }) => {
      // Agent-only errors have no harness semantics — don't step the reducer.
      // The harness is generic and shouldn't know about agent-specific failure modes.
      // event.outcome remains the source of truth for what went wrong.
      if (!isMappableTurnOutcome(event.outcome)) {
        return {
          ...fork,
          engine: { ...emptyHarnessTurnState.engine, stopped: true },
        }
      }

      // Harness-native outcomes: translate and step the reducer
      const toolkit = ambient.get(ToolUniverseAmbient)
      const harnessEvent = translateTurnOutcome({ ...event, outcome: event.outcome })
      const stepped = getCachedReducer(toolkit).step(fork, harnessEvent)

      // Preserve engine state when turn was killed by process crash —
      // this allows the recovery turn to skip already-executed tools.
      const isProcessCrash = event.outcome._tag === 'Cancelled'
        && event.outcome.reason._tag === 'WorkerKilled'

      return {
        ...stepped,
        engine: isProcessCrash ? fork.engine : emptyHarnessTurnState.engine,
      }
    },

    interrupt: ({ fork, ambient }) => {
      // Interrupt non-terminal tool handles.
      // Can't use the composed reducer here because the harness has no standalone
      // "interrupt" event — it handles interrupts via TurnEnd { Interrupted }.
      // But we haven't received TurnEnd yet at interrupt time.
      const handles = new Map(fork.handles.handles)
      const toolkit = ambient.get(ToolUniverseAmbient)
      for (const [id, handle] of handles) {
        const phase = handle.state.phase
        if (phase !== 'completed' && phase !== 'error' && phase !== 'rejected') {
          const model = toolkit.entries[handle.toolKey]?.state
          if (model) {
            handles.set(id, interruptToolHandle(handle, model))
          }
        }
      }
      return { ...fork, handles: { handles } }
    },
  },
})

// ── Helpers for consumers ────────────────────────────────────────────

/** Convert the handles Map to a Record for consumers that expect Record<string, ToolHandle> */
export function getToolHandlesRecord(state: HarnessTurnState): { readonly [callId: string]: ToolHandleFromSchema } {
  return Object.fromEntries(state.handles.handles)
}
