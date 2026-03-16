/**
 * AgentsViewProjection
 *
 * Global projection providing a unified feed of all agent activity:
 * messages between orchestrator and agents, activity/status lines,
 * and artifact write events.
 */

import { Projection } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { AgentStatusProjection, getAgentByForkId } from './agent-status'
import { AgentRoutingProjection } from './agent-routing'
import { DisplayProjection, EMPTY_TOOL_COUNTS, incrementToolCount, forkToolStepSignalExport as forkToolStepSignal } from './display'
import type { ForkActivityToolCounts } from './display'
import { createId } from '../util/id'

const generateId = () => createId()

// =============================================================================
// Types
// =============================================================================

export interface AgentsViewMessageItem {
  readonly id: string
  readonly type: 'agents_view_message'
  readonly timestamp: number
  readonly direction: 'to_agent' | 'from_agent'
  readonly fromColorIndex: number | null
  readonly fromRole: string
  readonly fromName: string
  readonly toColorIndex: number | null
  readonly toRole: string
  readonly toName: string
  readonly content: string
  readonly attachedArtifacts: readonly string[]
}

export interface AgentsViewActivityItem {
  readonly id: string
  readonly type: 'agents_view_activity'
  readonly timestamp: number
  readonly forkId: string
  readonly agentRole: string
  readonly agentName: string
  readonly colorIndex: number
  readonly status: 'active' | 'settled'
  readonly toolCounts: ForkActivityToolCounts
  readonly startedAt: number
  readonly completedAt?: number
}

export interface AgentsViewArtifactItem {
  readonly id: string
  readonly type: 'agents_view_artifact'
  readonly timestamp: number
  readonly forkId: string
  readonly agentRole: string
  readonly agentName: string
  readonly colorIndex: number
  readonly artifactName: string
  readonly action: 'wrote' | 'updated'
}

export type AgentsViewItem =
  | AgentsViewMessageItem
  | AgentsViewActivityItem
  | AgentsViewArtifactItem

export interface AgentsViewState {
  readonly items: readonly AgentsViewItem[]
  /** forkId -> active activity item id */
  readonly activeActivityIds: ReadonlyMap<string, string>
  /** toolCallId -> { toolKey, artifactName } buffered from ToolInputReady */
  readonly pendingToolInputs: ReadonlyMap<string, { toolKey: string; artifactName: string }>
}

// =============================================================================
// Helpers
// =============================================================================

/** Extract artifact wiki-link IDs from message content: [≡ name] patterns */
function extractArtifactRefs(content: string): string[] {
  const refs: string[] = []
  const regex = /\[≡\s+([^\]]+)\]/g
  let match: RegExpExecArray | null
  while ((match = regex.exec(content)) !== null) {
    if (match[1]) refs.push(match[1].trim())
  }
  return refs
}

// =============================================================================
// Projection
// =============================================================================

export const AgentsViewProjection = Projection.define<AppEvent, AgentsViewState>()(({
  name: 'AgentsView',

  reads: [AgentStatusProjection, AgentRoutingProjection, DisplayProjection] as const,

  initial: {
    items: [],
    activeActivityIds: new Map(),
    pendingToolInputs: new Map(),
  },

  signals: {},

  eventHandlers: {
    // Track artifact tool inputs and emit artifact items on completion
    tool_event: ({ event, state, read }) => {
      const inner = event.event
      if (event.forkId === null) return state  // root fork tools not shown in agents view

      if (inner._tag === 'ToolInputReady') {
        const toolKey = event.toolKey
        if (toolKey === 'artifactWrite' || toolKey === 'artifactUpdate') {
          const input = inner.input as { id?: string } | null
          const artifactName = input && typeof input === 'object' && typeof input.id === 'string'
            ? input.id
            : null
          if (artifactName) {
            const newPending = new Map(state.pendingToolInputs).set(event.toolCallId, {
              toolKey,
              artifactName,
            })
            return { ...state, pendingToolInputs: newPending }
          }
        }
        return state
      }

      if (inner._tag === 'ToolExecutionEnded') {
        const pending = state.pendingToolInputs.get(event.toolCallId)
        if (!pending) return state

        // Clean up pending entry
        const newPending = new Map(state.pendingToolInputs)
        newPending.delete(event.toolCallId)
        const nextState = { ...state, pendingToolInputs: newPending }

        // Only emit artifact item on success
        if (inner.result._tag !== 'Success') {
          return nextState
        }

        // Get agent info
        const agentState = read(AgentStatusProjection)
        const agent = getAgentByForkId(agentState, event.forkId)
        if (!agent) return nextState

        const artifactItem: AgentsViewArtifactItem = {
          id: generateId(),
          type: 'agents_view_artifact',
          timestamp: event.timestamp,
          forkId: event.forkId,
          agentRole: agent.role,
          agentName: agent.agentId,
          colorIndex: agent.colorIndex,
          artifactName: pending.artifactName,
          action: pending.toolKey === 'artifactWrite' ? 'wrote' : 'updated',
        }

        return { ...nextState, items: [...nextState.items, artifactItem] }
      }

      return state
    },
  },

  signalHandlers: (on) => [
    // Agent created: add active activity item
    on(AgentStatusProjection.signals.agentCreated, ({ value, state }) => {
      const activityId = generateId()
      const activityItem: AgentsViewActivityItem = {
        id: activityId,
        type: 'agents_view_activity',
        timestamp: value.timestamp,
        forkId: value.forkId,
        agentRole: value.role,
        agentName: value.agentId,
        colorIndex: value.colorIndex,
        status: 'active',
        toolCounts: EMPTY_TOOL_COUNTS,
        startedAt: value.timestamp,
      }

      const newItems: AgentsViewItem[] = [...state.items, activityItem]

      // Add initial orchestrator->agent message if present
      const initialContent = value.message?.trim()
      if (initialContent) {
        const msgItem: AgentsViewMessageItem = {
          id: generateId(),
          type: 'agents_view_message',
          timestamp: value.timestamp,
          direction: 'to_agent',
          fromColorIndex: null,
          fromRole: 'Orchestrator',
          fromName: 'Orchestrator',
          toColorIndex: value.colorIndex,
          toRole: value.role,
          toName: value.agentId,
          content: initialContent,
          attachedArtifacts: extractArtifactRefs(initialContent),
        }
        newItems.push(msgItem)
      }

      return {
        ...state,
        items: newItems,
        activeActivityIds: new Map(state.activeActivityIds).set(value.forkId, activityId),
      }
    }),

    // Agent became working (resume): add new active activity item
    on(AgentStatusProjection.signals.agentBecameWorking, ({ value, state }) => {
      // Only insert resume activity if there's no current active activity
      const existingActivityId = state.activeActivityIds.get(value.forkId)
      if (existingActivityId) return state  // already active (first run)

      const activityId = generateId()
      const item: AgentsViewActivityItem = {
        id: activityId,
        type: 'agents_view_activity',
        timestamp: value.timestamp,
        forkId: value.forkId,
        agentRole: value.role,
        agentName: value.agentId,
        colorIndex: value.colorIndex,
        status: 'active',
        toolCounts: EMPTY_TOOL_COUNTS,
        startedAt: value.timestamp,
      }
      return {
        ...state,
        items: [...state.items, item],
        activeActivityIds: new Map(state.activeActivityIds).set(value.forkId, activityId),
      }
    }),

    // Agent became idle: settle active activity item
    on(AgentStatusProjection.signals.agentBecameIdle, ({ value, state }) => {
      const activityId = state.activeActivityIds.get(value.forkId)
      if (!activityId) return state

      const newActiveIds = new Map(state.activeActivityIds)
      newActiveIds.delete(value.forkId)

      return {
        ...state,
        items: state.items.map(item =>
          item.id === activityId && item.type === 'agents_view_activity'
            ? { ...item, status: 'settled' as const, completedAt: value.timestamp }
            : item
        ),
        activeActivityIds: newActiveIds,
      }
    }),

    // Agent dismissed: settle active activity item
    on(AgentStatusProjection.signals.agentDismissed, ({ value, state }) => {
      const activityId = state.activeActivityIds.get(value.forkId)
      if (!activityId) return state

      const newActiveIds = new Map(state.activeActivityIds)
      newActiveIds.delete(value.forkId)

      return {
        ...state,
        items: state.items.map(item =>
          item.id === activityId && item.type === 'agents_view_activity'
            ? { ...item, status: 'settled' as const, completedAt: value.timestamp }
            : item
        ),
        activeActivityIds: newActiveIds,
      }
    }),

    // Orchestrator -> Agent message
    on(AgentRoutingProjection.signals.agentMessage, ({ value, state, read }) => {
      const { targetForkId, message, timestamp } = value
      const content = message.trim()
      if (!content) return state

      const agentState = read(AgentStatusProjection)
      const targetAgent = getAgentByForkId(agentState, targetForkId)
      if (!targetAgent) return state

      const attachedArtifacts = extractArtifactRefs(content)

      const item: AgentsViewMessageItem = {
        id: generateId(),
        type: 'agents_view_message',
        timestamp,
        direction: 'to_agent',
        fromColorIndex: null,
        fromRole: 'Orchestrator',
        fromName: 'Orchestrator',
        toColorIndex: targetAgent.colorIndex,
        toRole: targetAgent.role,
        toName: targetAgent.agentId,
        content,
        attachedArtifacts,
      }
      return { ...state, items: [...state.items, item] }
    }),

    // Agent -> Orchestrator message
    on(AgentRoutingProjection.signals.agentResponse, ({ value, state, read }) => {
      const { agentId, message, timestamp } = value
      const content = message.trim()
      if (!content) return state

      const agentState = read(AgentStatusProjection)
      const agent = agentState.agents.get(agentId)
      if (!agent) return state

      const item: AgentsViewMessageItem = {
        id: generateId(),
        type: 'agents_view_message',
        timestamp,
        direction: 'from_agent',
        fromColorIndex: agent.colorIndex,
        fromRole: agent.role,
        fromName: agent.agentId,
        toColorIndex: null,
        toRole: 'Orchestrator',
        toName: 'Orchestrator',
        content,
        attachedArtifacts: [],
      }
      return { ...state, items: [...state.items, item] }
    }),

    // Tool step: increment tool counts on active activity item
    on(forkToolStepSignal, ({ value, state }) => {
      const { forkId, toolKey } = value
      if (!forkId) return state

      const activityId = state.activeActivityIds.get(forkId)
      if (!activityId) return state

      return {
        ...state,
        items: state.items.map(item =>
          item.id === activityId && item.type === 'agents_view_activity'
            ? { ...item, toolCounts: incrementToolCount(item.toolCounts, toolKey) }
            : item
        ),
      }
    }),
  ],
}))