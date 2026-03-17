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
import { DisplayProjection, EMPTY_TOOL_COUNTS, incrementToolCount, addToolCounts, forkToolStepSignalExport as forkToolStepSignal } from './display'
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

export interface AgentsViewActivityStartItem {
  readonly id: string
  readonly type: 'agents_view_activity_start'
  readonly timestamp: number
  readonly forkId: string
  readonly agentRole: string
  readonly agentName: string
  readonly colorIndex: number
  readonly startedAt: number
  // resume context (set once at creation)
  readonly isResume: boolean
  readonly priorElapsedMs: number
  readonly priorToolCounts: ForkActivityToolCounts
  // completion data (written once when agent finishes)
  readonly completedAt?: number
  readonly finalToolCounts?: ForkActivityToolCounts
}

export interface AgentsViewActivityEndItem {
  readonly id: string
  readonly type: 'agents_view_activity_end'
  readonly timestamp: number
  readonly forkId: string
  readonly agentRole: string
  readonly agentName: string
  readonly colorIndex: number
  readonly startedAt: number
  readonly completedAt: number
  readonly toolCounts: ForkActivityToolCounts
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
  | AgentsViewActivityStartItem
  | AgentsViewActivityEndItem
  | AgentsViewArtifactItem

export interface ActiveActivityEntry {
  readonly itemId: string
  readonly startedAt: number
  readonly toolCounts: ForkActivityToolCounts
}

export interface ActivityAccumulator {
  readonly elapsedMs: number
  readonly toolCounts: ForkActivityToolCounts
}

export interface AgentsViewState {
  readonly items: readonly AgentsViewItem[]
  /** forkId -> active activity entry */
  readonly activeActivityIds: ReadonlyMap<string, ActiveActivityEntry>
  /** toolCallId -> { toolKey, artifactName } buffered from ToolInputReady */
  readonly pendingToolInputs: ReadonlyMap<string, { toolKey: string; artifactName: string }>
  /** forkId -> accumulated totals from completed sessions */
  readonly activityAccumulators: ReadonlyMap<string, ActivityAccumulator>
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

function handleAgentFinished(
  state: AgentsViewState,
  value: { forkId: string; role: string; agentId: string; colorIndex: number; timestamp: number }
): AgentsViewState {
  const entry = state.activeActivityIds.get(value.forkId)
  if (!entry) return state

  const completedAt = value.timestamp
  const finalToolCounts = entry.toolCounts
  const sessionElapsedMs = completedAt - entry.startedAt

  // Patch the start item with completion data
  const newItems = state.items.map(item => {
    if (item.type === 'agents_view_activity_start' && item.id === entry.itemId) {
      return { ...item, completedAt, finalToolCounts } as AgentsViewActivityStartItem
    }
    return item
  })

  // Append end item (kept for lane computation in computeLanes)
  const endItem: AgentsViewActivityEndItem = {
    id: generateId(),
    type: 'agents_view_activity_end',
    timestamp: value.timestamp,
    forkId: value.forkId,
    agentRole: value.role,
    agentName: value.agentId,
    colorIndex: value.colorIndex,
    startedAt: entry.startedAt,
    completedAt,
    toolCounts: finalToolCounts,
  }

  // Find the start item for its prior totals
  const startItem = state.items.find(
    i => i.type === 'agents_view_activity_start' && i.id === entry.itemId
  ) as AgentsViewActivityStartItem | undefined

  const newAccumulator: ActivityAccumulator = {
    elapsedMs: (startItem?.priorElapsedMs ?? 0) + sessionElapsedMs,
    toolCounts: addToolCounts(startItem?.priorToolCounts ?? EMPTY_TOOL_COUNTS, finalToolCounts),
  }

  const newActiveIds = new Map(state.activeActivityIds)
  newActiveIds.delete(value.forkId)
  const newAccumulators = new Map(state.activityAccumulators).set(value.forkId, newAccumulator)

  return {
    ...state,
    items: [...newItems, endItem],
    activeActivityIds: newActiveIds,
    activityAccumulators: newAccumulators,
  }
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
    activityAccumulators: new Map(),
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
    // Agent created: add initial message (outside lane) then start item
    on(AgentStatusProjection.signals.agentCreated, ({ value, state }) => {
      const startId = generateId()
      const newItems: AgentsViewItem[] = [...state.items]

      // Add initial orchestrator->agent message BEFORE start item (outside the lane)
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

      const startItem: AgentsViewActivityStartItem = {
        id: startId,
        type: 'agents_view_activity_start',
        timestamp: value.timestamp,
        forkId: value.forkId,
        agentRole: value.role,
        agentName: value.agentId,
        colorIndex: value.colorIndex,
        startedAt: value.timestamp,
        isResume: false,
        priorElapsedMs: 0,
        priorToolCounts: EMPTY_TOOL_COUNTS,
      }
      newItems.push(startItem)

      const entry: ActiveActivityEntry = {
        itemId: startId,
        startedAt: value.timestamp,
        toolCounts: EMPTY_TOOL_COUNTS,
      }

      return {
        ...state,
        items: newItems,
        activeActivityIds: new Map(state.activeActivityIds).set(value.forkId, entry),
      }
    }),

    // Agent became working (resume): add new start item if not already active
    on(AgentStatusProjection.signals.agentBecameWorking, ({ value, state }) => {
      const existing = state.activeActivityIds.get(value.forkId)
      if (existing) return state  // already active (first run)

      const accumulator = state.activityAccumulators.get(value.forkId)
      const startId = generateId()
      const startItem: AgentsViewActivityStartItem = {
        id: startId,
        type: 'agents_view_activity_start',
        timestamp: value.timestamp,
        forkId: value.forkId,
        agentRole: value.role,
        agentName: value.agentId,
        colorIndex: value.colorIndex,
        startedAt: value.timestamp,
        isResume: !!accumulator,
        priorElapsedMs: accumulator?.elapsedMs ?? 0,
        priorToolCounts: accumulator?.toolCounts ?? EMPTY_TOOL_COUNTS,
      }
      const entry: ActiveActivityEntry = {
        itemId: startId,
        startedAt: value.timestamp,
        toolCounts: EMPTY_TOOL_COUNTS,
      }
      return {
        ...state,
        items: [...state.items, startItem],
        activeActivityIds: new Map(state.activeActivityIds).set(value.forkId, entry),
      }
    }),

    // Agent became idle: patch start item, append end item
    on(AgentStatusProjection.signals.agentBecameIdle, ({ value, state }) => {
      return handleAgentFinished(state, value)
    }),

    // Agent dismissed: patch start item, append end item
    on(AgentStatusProjection.signals.agentDismissed, ({ value, state }) => {
      return handleAgentFinished(state, value)
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

    // Tool step: accumulate tool counts in active entry
    on(forkToolStepSignal, ({ value, state }) => {
      const { forkId, toolKey } = value
      if (!forkId) return state

      const entry = state.activeActivityIds.get(forkId)
      if (!entry) return state

      const newEntry: ActiveActivityEntry = {
        ...entry,
        toolCounts: incrementToolCount(entry.toolCounts, toolKey),
      }

      return {
        ...state,
        activeActivityIds: new Map(state.activeActivityIds).set(forkId, newEntry),
      }
    }),
  ],
}))