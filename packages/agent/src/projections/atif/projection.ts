/**
 * AtifProjection (Forked)
 *
 * ATIF (Agent Trajectory Interchange Format) v1.7 projection derived from the
 * AppEvent stream. Each fork (leader=null, workers by agentId) accumulates its
 * own steps independently.
 *
 * Ambient-gated: every event handler checks AtifAmbient and short-circuits if
 * disabled. Zero cost when ATIF is not enabled.
 */

import { Projection } from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { outcomeWillChainContinue } from '../../events'
import { AtifAmbient, type AtifConfig } from '../../ambient/atif-ambient'
import type { JsonValue } from '@magnitudedev/ai'
import { AgentRoutingProjection } from '../agent-routing'
import { AgentLifecycleProjection, getAgentByForkId } from '../agent-lifecycle'
import { DEFAULT_AGENT_NAME } from '../../agents/registry'

import type {
  ActiveAtifTurn,
  AtifForkState,
  AtifStep,
  AtifStepDraft,
  PendingToolCall,
} from './types'
import { Option } from 'effect'
import { AtifForkStateSchema } from './types'
import { atifSignals } from './signals'
import {
  userMessageToStep,
  beginActiveTurn,
  accumulateThinkingChunk,
  accumulateMessageChunk,
  addToolCallToStep,
  addObservationToStep,
  finalizeAgentStep,
  agentCreatedToStep,
  compactionPreparedToStep,
  interruptToStep,
  agentKilledToStep,
  observerOutcomeToStep,
} from './mapping'

// =============================================================================
// Helpers
// =============================================================================

function isEnabled(ambient: { get: (a: typeof AtifAmbient) => AtifConfig }): boolean {
  return ambient.get(AtifAmbient).enabled
}

function createInitialFork(forkId: string | null, agentName: string = DEFAULT_AGENT_NAME): AtifForkState {
  return {
    forkId,
    agentName,
    agentRole: forkId === null ? 'leader' : null,
    modelId: null,
    steps: [],
    activeTurns: new Map(),
    compactionBoundaryIndex: null,
    tokenAccumulator: {
      promptTokens: 0,
      completionTokens: 0,
      cachedTokens: 0,
      costUsd: 0,
    },
  }
}

function emitStep(
  fork: AtifForkState,
  draft: AtifStepDraft,
  emit: { readonly stepAdded: (value: { forkId: string | null; step: AtifStep; stepIndex: number }) => void }
): AtifForkState {
  const step: AtifStep = {
    step_id: fork.steps.length + 1,
    ...draft,
  }
  emit.stepAdded({ forkId: fork.forkId, step, stepIndex: fork.steps.length })
  return {
    ...fork,
    steps: [...fork.steps, step],
  }
}

function timestampToIso(ts: number): string {
  return new Date(ts).toISOString()
}

function setActiveTurn(fork: AtifForkState, turn: ActiveAtifTurn): AtifForkState {
  return {
    ...fork,
    activeTurns: new Map(fork.activeTurns).set(turn.turnId, turn),
  }
}

function updateActiveTurn(
  fork: AtifForkState,
  turnId: string,
  update: (turn: ActiveAtifTurn) => ActiveAtifTurn,
): AtifForkState {
  const turn = fork.activeTurns.get(turnId)
  if (!turn) return fork

  return setActiveTurn(fork, update(turn))
}

function deleteActiveTurn(fork: AtifForkState, turnId: string): AtifForkState {
  if (!fork.activeTurns.has(turnId)) return fork
  const activeTurns = new Map(fork.activeTurns)
  activeTurns.delete(turnId)
  return { ...fork, activeTurns }
}

// =============================================================================
// Projection
// =============================================================================

export const AtifProjection = Projection.defineForked<AppEvent>()({
  name: 'Atif',
  forkState: AtifForkStateSchema,

  reads: [AgentRoutingProjection, AgentLifecycleProjection] as const,
  ambients: [AtifAmbient] as const,

  signals: atifSignals,

  initialFork: createInitialFork(null),

  eventHandlers: {
    user_message: ({ event, fork, ambient, emit }) => {
      if (!isEnabled(ambient)) return fork
      const step = userMessageToStep(event)
      return emitStep(fork, step, emit)
    },

    turn_started: ({ event, fork, ambient }) => {
      if (!isEnabled(ambient)) return fork
      // model_name is resolved from turn_outcome (which carries modelId from the provider)
      return setActiveTurn(fork, beginActiveTurn(event, null))
    },

    thinking_chunk: ({ event, fork, ambient }) => {
      if (!isEnabled(ambient)) return fork
      return updateActiveTurn(fork, event.turnId, (turn) => accumulateThinkingChunk(turn, event))
    },

    message_chunk: ({ event, fork, ambient }) => {
      if (!isEnabled(ambient)) return fork
      return updateActiveTurn(fork, event.turnId, (turn) => accumulateMessageChunk(turn, event))
    },

    tool_event: ({ event, fork, ambient }) => {
      if (!isEnabled(ambient)) return fork
      const activeTurn = fork.activeTurns.get(event.turnId)
      if (!activeTurn) return fork

      const lifecycle = event.event as {
        _tag: string
        toolName?: string
        toolKey?: string
        input?: Record<string, JsonValue>
        cached?: boolean
      }

      if (lifecycle._tag === 'ToolInputReady') {
        const toolCall: PendingToolCall = {
          toolCallId: event.toolCallId,
          function_name: lifecycle.toolName ?? String(event.toolKey),
          arguments: {},
        }
        return setActiveTurn(fork, {
          ...addToolCallToStep(activeTurn, event),
          pendingToolCalls: new Map<string, PendingToolCall>(activeTurn.pendingToolCalls).set(event.toolCallId, toolCall),
        })
      }

      if (lifecycle._tag === 'ToolExecutionStarted') {
        const pending = activeTurn.pendingToolCalls.get(event.toolCallId)
        if (pending) {
          const updatedPending: PendingToolCall = { ...pending, arguments: lifecycle.input ?? {} }
          const toolCallIndex = activeTurn.tool_calls.findIndex(
            tc => tc.tool_call_id === event.toolCallId
          )
          let updatedTurn = activeTurn
          if (toolCallIndex >= 0) {
            const updatedToolCalls = [...activeTurn.tool_calls]
            updatedToolCalls[toolCallIndex] = {
              ...updatedToolCalls[toolCallIndex],
              arguments: lifecycle.input ?? {},
              ...(lifecycle.cached != null
                ? { extra: Option.some({ ...Option.getOrElse(updatedToolCalls[toolCallIndex].extra, () => ({})), cached: lifecycle.cached }) }
                : {}),
            }
            updatedTurn = { ...activeTurn, tool_calls: updatedToolCalls }
          }
          return setActiveTurn(fork, {
            ...updatedTurn,
            pendingToolCalls: new Map<string, PendingToolCall>(activeTurn.pendingToolCalls).set(event.toolCallId, updatedPending),
          })
        }
        return fork
      }

      if (lifecycle._tag === 'ToolExecutionEnded') {
        const updatedTurn = addObservationToStep(activeTurn, event)
        const nextPending = new Map<string, PendingToolCall>(activeTurn.pendingToolCalls)
        nextPending.delete(event.toolCallId)
        return setActiveTurn(fork, { ...updatedTurn, pendingToolCalls: nextPending })
      }

      return fork
    },

    turn_outcome: ({ event, fork, ambient, emit }) => {
      if (!isEnabled(ambient)) return fork
      const activeTurn = fork.activeTurns.get(event.turnId)
      if (!activeTurn) return fork

      const step = finalizeAgentStep(activeTurn, event)
      const stepCost = Option.getOrElse(Option.flatMap(step.metrics, m => m.cost_usd), () => 0)

      const nextAccumulator = {
        promptTokens: fork.tokenAccumulator.promptTokens + (event.inputTokens ?? 0),
        completionTokens: fork.tokenAccumulator.completionTokens + (event.outputTokens ?? 0),
        cachedTokens: fork.tokenAccumulator.cachedTokens + (event.cacheReadTokens ?? 0),
        costUsd: fork.tokenAccumulator.costUsd + stepCost,
      }

      const withoutTurn = deleteActiveTurn(fork, event.turnId)
      return emitStep({ ...withoutTurn, tokenAccumulator: nextAccumulator }, step, emit)
    },

    tool_approved: ({ event, fork, ambient, emit }) => {
      if (!isEnabled(ambient)) return fork
      const step: AtifStepDraft = {
        timestamp: Option.some(timestampToIso(Date.now())),
        source: 'user',
        model_name: Option.none(),
        reasoning_effort: Option.none(),
        message: `Approved tool call ${event.toolCallId}`,
        reasoning_content: Option.none(),
        tool_calls: Option.none(),
        observation: Option.none(),
        metrics: Option.none(),
        is_copied_context: Option.none(),
        extra: Option.some({ toolCallId: event.toolCallId, action: 'approved' }),
        llm_call_count: Option.some(0),
      }
      return emitStep(fork, step, emit)
    },

    tool_rejected: ({ event, fork, ambient, emit }) => {
      if (!isEnabled(ambient)) return fork
      const step: AtifStepDraft = {
        timestamp: Option.some(timestampToIso(Date.now())),
        source: 'user',
        model_name: Option.none(),
        reasoning_effort: Option.none(),
        message: event.reason
          ? `Rejected tool call ${event.toolCallId}: ${event.reason}`
          : `Rejected tool call ${event.toolCallId}`,
        reasoning_content: Option.none(),
        tool_calls: Option.none(),
        observation: Option.none(),
        metrics: Option.none(),
        is_copied_context: Option.none(),
        extra: Option.some({ toolCallId: event.toolCallId, action: 'rejected', ...(event.reason ? { reason: event.reason } : {}) }),
        llm_call_count: Option.some(0),
      }
      return emitStep(fork, step, emit)
    },

    interrupt: ({ event, fork, ambient, emit }) => {
      if (!isEnabled(ambient)) return fork
      const step = interruptToStep(event)
      return emitStep(fork, step, emit)
    },

    compaction_prepared: ({ event, fork, ambient, emit }) => {
      if (!isEnabled(ambient)) return fork
      const step = compactionPreparedToStep(event)
      // Record the boundary index — steps before this will get is_copied_context
      // when compaction_injected fires
      return emitStep(
        { ...fork, compactionBoundaryIndex: fork.steps.length },
        step,
        emit,
      )
    },

    compaction_injected: ({ event, fork, ambient }) => {
      if (!isEnabled(ambient)) return fork
      // Mark steps before the compaction boundary as is_copied_context: true
      const boundary = fork.compactionBoundaryIndex
      if (boundary == null || boundary === 0) return fork

      const updatedSteps = fork.steps.map((step, i) =>
        i < boundary ? { ...step, is_copied_context: Option.some(true) } : step
      )
      return { ...fork, steps: updatedSteps, compactionBoundaryIndex: null }
    },

    observer_outcome: ({ event, fork, ambient, emit }) => {
      if (!isEnabled(ambient)) return fork
      const step = observerOutcomeToStep(event)
      return emitStep(fork, step, emit)
    },
  },

  globalEventHandlers: {
    turn_outcome: ({ event, state, ambient, emit }) => {
      if (!isEnabled(ambient)) return state
      if (event.forkId === null) return state
      if (outcomeWillChainContinue(event.outcome)) return state
      const fork = state.forks.get(event.forkId)
      if (!fork) return state
      emit.forkCompleted({ forkId: event.forkId, stepCount: fork.steps.length })
      return state
    },

    agent_killed: ({ event, state, ambient, emit }) => {
      if (!isEnabled(ambient)) return state
      const fork = state.forks.get(event.forkId)
      if (!fork) return state
      // Emit a terminal system step before marking the fork completed
      const terminalStep = agentKilledToStep(event.agentId, event.reason)
      const nextFork = { ...emitStep(fork, terminalStep, emit), activeTurns: new Map<string, ActiveAtifTurn>() }
      emit.forkCompleted({ forkId: event.forkId, stepCount: nextFork.steps.length })
      return { ...state, forks: new Map(state.forks).set(event.forkId, nextFork) }
    },

    worker_user_killed: ({ event, state, ambient, emit }) => {
      if (!isEnabled(ambient)) return state
      const fork = state.forks.get(event.forkId)
      if (!fork) return state
      const terminalStep = agentKilledToStep(event.agentId, 'user_killed')
      const nextFork = { ...emitStep(fork, terminalStep, emit), activeTurns: new Map<string, ActiveAtifTurn>() }
      emit.forkCompleted({ forkId: event.forkId, stepCount: nextFork.steps.length })
      return { ...state, forks: new Map(state.forks).set(event.forkId, nextFork) }
    },

    agent_created: ({ event, state, ambient, emit }) => {
      if (!isEnabled(ambient)) return state

      // The spawnWorker step belongs on the parent's trajectory
      const parentForkId = event.parentForkId
      const parentFork = state.forks.get(parentForkId)
      if (!parentFork) return state

      const step = agentCreatedToStep(event, event.agentId)
      const nextParentFork = emitStep(parentFork, step, emit)

      // Create the child fork with proper agent name derived from role
      const childFork = {
        ...createInitialFork(event.forkId, `${DEFAULT_AGENT_NAME}-${event.role}`),
        agentRole: event.role,
      }

      return {
        ...state,
        forks: new Map(state.forks)
          .set(parentForkId, nextParentFork)
          .set(event.forkId, childFork),
      }
    },
  },

  signalHandlers: (on) => [
    on(AgentRoutingProjection.signals.agentRegistered, ({ value, state, ambient, read }) => {
      if (!isEnabled(ambient)) return state
      const { forkId } = value
      if (state.forks.has(forkId)) return state

      // Derive agentName from AgentLifecycleProjection if available
      let agentName: string = DEFAULT_AGENT_NAME
      const agentStatus = read(AgentLifecycleProjection)
      const agent = getAgentByForkId(agentStatus, forkId)
      if (agent) {
        agentName = agent.name || `${DEFAULT_AGENT_NAME}-${agent.role}`
      }

      const newFork: AtifForkState = {
        ...createInitialFork(forkId),
        agentName,
        agentRole: agent ? agent.role : null,
      }
      return { ...state, forks: new Map(state.forks).set(forkId, newFork) }
    }),
  ],
})
