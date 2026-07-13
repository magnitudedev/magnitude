/**
 * AgentRoutingProjection
 *
 * Global projection tracking child agent message routing.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import { type AppEvent, outcomeWillChainContinue } from '../events'
import { Schema } from 'effect'

export const RoutingEntrySchema = Schema.Struct({
  agentId: Schema.String,
  forkId: Schema.String,
  parentForkId: Schema.NullOr(Schema.String),
})
export type RoutingEntry = typeof RoutingEntrySchema.Type

const MessageDestinationSchema = Schema.Union(
  Schema.Struct({ kind: Schema.Literal('user') }),
  Schema.Struct({ kind: Schema.Literal('coordinator') }),
  Schema.Struct({ kind: Schema.Literal('worker'), agentId: Schema.String }),
)

export const PendingRoutedMessageSchema = Schema.Struct({
  forkId: Schema.NullOr(Schema.String),
  destination: MessageDestinationSchema,
  text: Schema.String,
  order: Schema.Number,
  targetAgentId: Schema.NullOr(Schema.String),
})
export type PendingRoutedMessage = typeof PendingRoutedMessageSchema.Type

export const AgentRoutingStateSchema = Schema.Struct({
  agents: Schema.ReadonlyMap({ key: Schema.String, value: RoutingEntrySchema }),
  agentByForkId: Schema.ReadonlyMap({ key: Schema.String, value: Schema.String }),
  pendingMessages: Schema.ReadonlyMap({ key: Schema.String, value: PendingRoutedMessageSchema }),
  deferredParentMessages: Schema.ReadonlyMap({ key: Schema.String, value: Schema.Array(PendingRoutedMessageSchema) }),
})
export type AgentRoutingState = typeof AgentRoutingStateSchema.Type

export function getRoutingEntry(state: AgentRoutingState, agentId: string): RoutingEntry | undefined {
  return state.agents.get(agentId)
}

export function getRoutingEntryByForkId(state: AgentRoutingState, forkId: string): RoutingEntry | undefined {
  const agentId = state.agentByForkId.get(forkId)
  if (!agentId) return undefined
  return state.agents.get(agentId)
}

export function isActiveRoute(state: AgentRoutingState, agentId: string): boolean {
  return state.agents.has(agentId)
}

export interface AgentMessageSignal {
  readonly targetForkId: string
  readonly agentId: string
  readonly message: string
  readonly timestamp: number
}

export interface AgentResponseSignal {
  readonly targetForkId: string | null
  readonly agentId: string
  readonly message: string
  readonly timestamp: number
}

export interface AgentCommunicationStreamSignal {
  readonly streamId: string
  readonly targetForkId: string
  readonly direction: 'from_agent' | 'to_agent'
  readonly agentId: string
  readonly textDelta: string
  readonly timestamp: number
}

export interface AgentCommunicationStreamCompletedSignal {
  readonly streamId: string
  readonly targetForkId: string
  readonly direction: 'from_agent' | 'to_agent'
  readonly agentId: string
  readonly timestamp: number
}

export interface InvalidExplicitDestinationSignal {
  readonly forkId: string | null
  readonly turnId: string
  readonly messageId: string
  readonly agentId: string | null
  readonly to: string
  readonly reason: string
  readonly timestamp: number
}


function removeAgentRoutingState(
  state: AgentRoutingState,
  { forkId, agentId }: { forkId: string; agentId: string },
): AgentRoutingState {
  const routedAgentId = state.agentByForkId.get(forkId)
  if (!routedAgentId) return state
  if (routedAgentId !== agentId) return state

  const agents = new Map(state.agents)
  agents.delete(agentId)

  const agentByForkId = new Map(state.agentByForkId)
  agentByForkId.delete(forkId)

  const pendingMessages = new Map(state.pendingMessages)
  for (const [id, pending] of pendingMessages.entries()) {
    if (pending.forkId === forkId || pending.targetAgentId === agentId) {
      pendingMessages.delete(id)
    }
  }

  const deferredParentMessages = new Map(state.deferredParentMessages)
  deferredParentMessages.delete(forkId)

  return {
    ...state,
    agents,
    agentByForkId,
    pendingMessages,
    deferredParentMessages,
  }
}

export const AgentRoutingProjection = Projection.define<AppEvent>()(({
  name: 'AgentRouting',
  state: AgentRoutingStateSchema,
  reads: [] as const,

  initial: {
    agents: new Map<string, RoutingEntry>(),
    agentByForkId: new Map<string, string>(),
    pendingMessages: new Map<string, PendingRoutedMessage>(),
    deferredParentMessages: new Map<string, readonly PendingRoutedMessage[]>(),
  },

  signals: {
    agentRegistered: Signal.create<{ forkId: string; parentForkId: string | null; role: string }>('AgentRouting/registered'),
    agentMessage: Signal.create<AgentMessageSignal>('AgentRouting/message'),
    agentResponse: Signal.create<AgentResponseSignal>('AgentRouting/response'),
    communicationStreamStarted: Signal.create<AgentCommunicationStreamSignal>('AgentRouting/communicationStreamStarted'),
    communicationStreamChunk: Signal.create<AgentCommunicationStreamSignal>('AgentRouting/communicationStreamChunk'),
    communicationStreamCompleted: Signal.create<AgentCommunicationStreamCompletedSignal>('AgentRouting/communicationStreamCompleted'),
    invalidExplicitDestination: Signal.create<InvalidExplicitDestinationSignal>('AgentRouting/invalidExplicitDestination'),
  },

  eventHandlers: {
    agent_created: ({ event, state, emit }) => {
      const existingAgent = state.agents.get(event.agentId)
      if (existingAgent) {
        throw new Error(`[AgentRoutingProjection] Invalid state transition: agent_created for already existing agent ${event.agentId} (forkId: ${existingAgent.forkId})`)
      }

      const existingForkAgentId = state.agentByForkId.get(event.forkId)
      if (existingForkAgentId) {
        throw new Error(`[AgentRoutingProjection] Invalid state transition: agent_created for already indexed fork ${event.forkId} (agentId: ${existingForkAgentId})`)
      }

      const entry: RoutingEntry = {
        agentId: event.agentId,
        forkId: event.forkId,
        parentForkId: event.parentForkId,
      }

      emit.agentRegistered({ forkId: event.forkId, parentForkId: event.parentForkId, role: event.role })

      return {
        ...state,
        agents: new Map(state.agents).set(event.agentId, entry),
        agentByForkId: new Map(state.agentByForkId).set(event.forkId, event.agentId),
      }
    },


    message_start: ({ event, state, emit }) => {
      const destination = event.destination
      const source = event.forkId === null ? undefined : getRoutingEntryByForkId(state, event.forkId)

      const targetAgentId = destination.kind === 'worker'
        ? destination.agentId
        : null

      const pendingMessages = new Map(state.pendingMessages)
      pendingMessages.set(event.id, {
        forkId: event.forkId,
        destination,
        text: '',
        order: event.timestamp,
        targetAgentId,
      })

      if (destination.kind !== 'user') {
        if (destination.kind === 'coordinator' && event.forkId !== null && source) {
          emit.communicationStreamStarted({
            streamId: event.id,
            targetForkId: source.forkId,
            direction: 'to_agent',
            agentId: source.agentId,
            textDelta: '',
            timestamp: event.timestamp,
          })
        } else if (destination.kind === 'worker' && targetAgentId && isActiveRoute(state, targetAgentId)) {
          const target = getRoutingEntry(state, targetAgentId)
          if (target) {
            emit.communicationStreamStarted({
              streamId: event.id,
              targetForkId: target.forkId,
              direction: 'from_agent',
              agentId: target.agentId,
              textDelta: '',
              timestamp: event.timestamp,
            })
          }
        }
      }

      return { ...state, pendingMessages }
    },

    message_chunk: ({ event, state, emit }) => {
      const entry = state.pendingMessages.get(event.id)
      if (!entry) return state

      const pendingMessages = new Map(state.pendingMessages)
      pendingMessages.set(event.id, { ...entry, text: entry.text + event.text })

      if (entry.destination.kind !== 'user' && event.text.length > 0) {
        if (entry.destination.kind === 'coordinator' && entry.forkId !== null) {
          const source = getRoutingEntryByForkId(state, entry.forkId)
          if (source) {
            emit.communicationStreamChunk({
              streamId: event.id,
              targetForkId: source.forkId,
              direction: 'to_agent',
              agentId: source.agentId,
              textDelta: event.text,
              timestamp: event.timestamp,
            })
          }
        } else if (entry.destination.kind === 'worker' && entry.targetAgentId && isActiveRoute(state, entry.targetAgentId)) {
          const target = getRoutingEntry(state, entry.targetAgentId)
          if (target) {
            emit.communicationStreamChunk({
              streamId: event.id,
              targetForkId: target.forkId,
              direction: 'from_agent',
              agentId: target.agentId,
              textDelta: event.text,
              timestamp: event.timestamp,
            })
          }
        }
      }

      return { ...state, pendingMessages }
    },

    message_end: ({ event, state, emit, read }) => {
      const entry = state.pendingMessages.get(event.id)
      if (!entry) return state

      const pendingMessages = new Map(state.pendingMessages)
      pendingMessages.delete(event.id)

      let nextState: AgentRoutingState = { ...state, pendingMessages }

      if (entry.destination.kind === 'coordinator' && entry.forkId !== null) {
        const source = getRoutingEntryByForkId(state, entry.forkId)
        if (source) {
          emit.communicationStreamCompleted({
            streamId: event.id,
            targetForkId: source.forkId,
            direction: 'to_agent',
            agentId: source.agentId,
            timestamp: event.timestamp,
          })
        }

        const existing = state.deferredParentMessages.get(entry.forkId) ?? []
        const deferredParentMessages = new Map(state.deferredParentMessages)
        deferredParentMessages.set(entry.forkId, [...existing, { ...entry, text: entry.text, order: event.timestamp }])
        nextState = { ...nextState, deferredParentMessages }
      }

      if (entry.destination.kind === 'worker') {
        // Re-resolve target at message_end to handle same-turn spawn_worker + message flows.
        const resolvedTargetAgentId =
          entry.targetAgentId ?? entry.destination.agentId

        if (resolvedTargetAgentId && isActiveRoute(state, resolvedTargetAgentId)) {
          const target = getRoutingEntry(state, resolvedTargetAgentId)
          if (target) {
            emit.communicationStreamCompleted({
              streamId: event.id,
              targetForkId: target.forkId,
              direction: 'from_agent',
              agentId: target.agentId,
              timestamp: event.timestamp,
            })

            emit.agentMessage({
              targetForkId: target.forkId,
              agentId: resolvedTargetAgentId,
              message: entry.text,
              timestamp: event.timestamp,
            })
          }
        } else {
          emit.invalidExplicitDestination({
            forkId: entry.forkId,
            turnId: event.turnId,
            messageId: event.id,
            agentId: entry.destination.agentId,
            to: entry.destination.agentId,
            reason: 'no active routed worker at message_end',
            timestamp: event.timestamp,
          })
        }
      }

      return nextState
    },

    turn_outcome: ({ event, state, emit }) => {
      if (event.forkId === null) return state

      const messages = state.deferredParentMessages.get(event.forkId)
      if (!messages || messages.length === 0) return state

      const deferredParentMessages = new Map(state.deferredParentMessages)
      deferredParentMessages.delete(event.forkId)

      if (outcomeWillChainContinue(event.outcome)) {
        return { ...state, deferredParentMessages }
      }

      if (event.outcome._tag !== 'Completed') {
        return { ...state, deferredParentMessages }
      }

      const agent = getRoutingEntryByForkId(state, event.forkId)
      if (!agent) {
        return { ...state, deferredParentMessages }
      }

      const fullText = [...messages]
        .sort((a, b) => a.order - b.order)
        .map(message => message.text)
        .join('\n')
        .trim()

      emit.agentResponse({
        targetForkId: agent.parentForkId,
        agentId: agent.agentId,
        message: fullText,
        timestamp: event.timestamp,
      })

      return { ...state, deferredParentMessages }
    },

    agent_killed: ({ event, state }) => removeAgentRoutingState(state, event),

    worker_user_killed: ({ event, state }) => removeAgentRoutingState(state, event),

    worker_idle_closed: ({ event, state }) => removeAgentRoutingState(state, event),
  },
}))
