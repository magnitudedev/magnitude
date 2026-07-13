import { Option } from 'effect'

/**
 * AgentLifecycleProjection
 *
 * Tracks agent identity, metadata, and execution lifecycle status.
 * Also owns root work state (phase, chain timer, activity, child count).
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import { outcomeWillChainContinue } from '../events'
import type { AppEvent } from '../events'
import { ROLE_IDS, type RoleId } from '../agents/role-validation'
import { Schema } from 'effect'
import { DisplayActivity } from '@magnitudedev/protocol'
import type { ToolKey } from '../tools/toolkits'

export const AgentLifecycleSchema = Schema.Literal('working', 'idle', 'killed')
export type AgentLifecycleStatus = typeof AgentLifecycleSchema.Type

const RoleIdSchema = Schema.Literal(...ROLE_IDS)

export const AgentInfoSchema = Schema.Struct({
  agentId: Schema.String,
  forkId: Schema.String,
  parentForkId: Schema.NullOr(Schema.String),
  name: Schema.String,
  role: RoleIdSchema,
  context: Schema.String,
  mode: Schema.Literal('clone', 'spawn'),
  taskId: Schema.String,
  message: Schema.NullOr(Schema.String),
  status: AgentLifecycleSchema,
  lastIdleReason: Schema.NullOr(Schema.Literal('stable', 'interrupt', 'error')),
})
export type AgentInfo = typeof AgentInfoSchema.Type

// Root work state — merged from ActorWorkProjection (root only, no per-actor map)
const RootWorkStateSchema = Schema.Struct({
  phase: Schema.Literal('idle', 'working', 'worked', 'interrupted'),
  chainStartedAt: Schema.NullOr(Schema.Number),
  lastChainMs: Schema.Number,
  activity: Schema.NullOr(DisplayActivity),
  activeChildCount: Schema.Number,
  _currentTurnId: Schema.NullOr(Schema.String),
  _thinkingCharCount: Schema.NullOr(Schema.Number),
  _activeToolKey: Schema.NullOr(Schema.String),
})
type RootWorkState = typeof RootWorkStateSchema.Type

export const AgentLifecycleStateSchema = Schema.Struct({
  agents: Schema.ReadonlyMap({ key: Schema.String, value: AgentInfoSchema }),
  agentByForkId: Schema.ReadonlyMap({ key: Schema.String, value: Schema.String }),
  rootWork: RootWorkStateSchema,
})
export type AgentLifecycleState = typeof AgentLifecycleStateSchema.Type

export interface AgentCreatedSignal {
  readonly forkId: string
  readonly parentForkId: string | null
  readonly agentId: string
  readonly name: string
  readonly role: RoleId
  readonly taskId: string
  readonly mode: 'clone' | 'spawn'
  readonly context: string
  readonly timestamp: number
}

export interface AgentBecameIdleSignal {
  readonly agentId: string
  readonly forkId: string
  readonly role: RoleId
  readonly parentForkId: string | null
  readonly reason: 'stable' | 'interrupt' | 'error'
  readonly timestamp: number
}

export interface AgentBecameWorkingSignal {
  readonly agentId: string
  readonly forkId: string
  readonly role: RoleId
  readonly parentForkId: string | null
  readonly timestamp: number
}

export interface AgentKilledSignal {
  readonly agentId: string
  readonly forkId: string
  readonly parentForkId: string | null
  readonly role: RoleId
  readonly title: string
  readonly reason: string
  readonly timestamp: number
}

export interface SubagentUserKilledSignal {
  readonly agentId: string
  readonly forkId: string
  readonly parentForkId: string | null
  readonly role: RoleId
  readonly title: string
  readonly source: 'tab_close_confirm'
  readonly timestamp: number
}

export interface SubagentIdleClosedSignal {
  readonly agentId: string
  readonly forkId: string
  readonly parentForkId: string | null
  readonly role: RoleId
  readonly title: string
  readonly source: 'idle_tab_close'
  readonly timestamp: number
}

// --- Root work helpers ---

const STATUS_TOOL_ACTIVITIES: Partial<Record<string, DisplayActivity>> = {
  messageAdvisor: { kind: 'advisor', message: 'Asking advisor' },
  messageWorker: { kind: 'tool', message: 'Steering worker', decorator: Option.none() },
}

function thinkingMessage(charCount: number): string {
  if (charCount >= 2800) return 'Thinking very hard'
  if (charCount >= 1200) return 'Thinking hard'
  return 'Thinking'
}

function computeActivity(work: RootWorkState): DisplayActivity | null {
  if (work._activeToolKey) {
    const toolKey = work._activeToolKey
    const statusTool = STATUS_TOOL_ACTIVITIES[toolKey]
    if (statusTool) return statusTool
    if (toolKey === 'spawnWorker') {
      return { kind: 'tool', message: 'Starting worker', decorator: Option.some('spinner') }
    }
  }
  if (work._thinkingCharCount !== null) {
    return { kind: 'thinking', message: thinkingMessage(work._thinkingCharCount) }
  }
  return null
}

function resolveActivity(work: RootWorkState): RootWorkState {
  return { ...work, activity: computeActivity(work) }
}

function startRootWork(state: RootWorkState, timestamp: number): RootWorkState {
  return resolveActivity({
    ...state,
    phase: 'working',
    chainStartedAt: state.chainStartedAt ?? timestamp,
    _thinkingCharCount: null,
  })
}

function stopRootWork(
  state: RootWorkState,
  timestamp: number,
  phase: 'worked' | 'interrupted',
): RootWorkState {
  const chainMs = state.chainStartedAt === null
    ? 0
    : Math.max(0, timestamp - state.chainStartedAt)
  return {
    ...state,
    phase,
    chainStartedAt: null,
    lastChainMs: chainMs,
    activity: null,
    _currentTurnId: null,
    _thinkingCharCount: null,
    _activeToolKey: null,
  }
}

// --- Agent helpers ---

function removeKilledAgent(args: {
  forkId: string
  agentId: string
  timestamp: number
  state: AgentLifecycleState
}): { state: AgentLifecycleState; agent: AgentInfo | null } {
  const { forkId, agentId, timestamp, state } = args
  const agent = getAgentByForkId(state, forkId)
  if (!agent) return { state, agent: null }
  if (agent.agentId !== agentId) return { state, agent: null }

  const nextAgents = new Map(state.agents)
  nextAgents.delete(agent.agentId)
  const nextByFork = new Map(state.agentByForkId)
  nextByFork.delete(forkId)

  return {
    state: {
      ...state,
      agents: nextAgents,
      agentByForkId: nextByFork,
    },
    agent,
  }
}

export function getAgentByForkId(state: AgentLifecycleState, forkId: string): AgentInfo | undefined {
  const agentId = state.agentByForkId.get(forkId)
  if (!agentId) return undefined
  return state.agents.get(agentId)
}

export function getActiveAgent(state: AgentLifecycleState, agentId: string): AgentInfo | undefined {
  return state.agents.get(agentId)
}

export function hasActiveWorkers(state: AgentLifecycleState): boolean {
  return Array.from(state.agents.values()).some((agent) => agent.status === 'working')
}

export function countWorkingChildren(state: AgentLifecycleState, parentForkId: string | null): number {
  let count = 0
  for (const agent of state.agents.values()) {
    if (agent.parentForkId === parentForkId && agent.status === 'working') count++
  }
  return count
}

/**
 * Check if any agent was interrupted (not just went idle normally).
 * Used to determine root work phase on deferred close.
 */
function anyAgentInterrupted(state: AgentLifecycleState): boolean {
  return Array.from(state.agents.values()).some(
    (agent) => agent.lastIdleReason === 'interrupt',
  )
}

function updateChildCount(state: AgentLifecycleState): AgentLifecycleState {
  const childCount = countWorkingChildren(state, null)
  return {
    ...state,
    rootWork: { ...state.rootWork, activeChildCount: childCount },
  }
}

/**
 * Deferred root work close: if root's turn already ended but workers were
 * still running, close root work now that the last worker went idle.
 */
function maybeDeferCloseRootWork(
  state: AgentLifecycleState,
  timestamp: number,
): AgentLifecycleState {
  if (state.rootWork.phase !== 'working') return state
  if (state.rootWork._currentTurnId !== null) return state
  if (countWorkingChildren(state, null) > 0) return state
  const phase = anyAgentInterrupted(state) ? 'interrupted' : 'worked'
  return { ...state, rootWork: stopRootWork(state.rootWork, timestamp, phase) }
}

export const AgentLifecycleProjection = Projection.define<AppEvent>()(({
  name: 'AgentLifecycle',
  state: AgentLifecycleStateSchema,

  initial: {
    agents: new Map<string, AgentInfo>(),
    agentByForkId: new Map<string, string>(),
    rootWork: {
      phase: 'idle',
      chainStartedAt: null,
      lastChainMs: 0,
      activity: null,
      activeChildCount: 0,
      _currentTurnId: null,
      _thinkingCharCount: null,
      _activeToolKey: null,
    },
  },

  signals: {
    agentCreated: Signal.create<AgentCreatedSignal>('AgentLifecycle/created'),
    agentBecameIdle: Signal.create<AgentBecameIdleSignal>('AgentLifecycle/agentBecameIdle'),
    agentBecameWorking: Signal.create<AgentBecameWorkingSignal>('AgentLifecycle/agentBecameWorking'),
    agentKilled: Signal.create<AgentKilledSignal>('AgentLifecycle/agentKilled'),
    subagentUserKilled: Signal.create<SubagentUserKilledSignal>('AgentLifecycle/subagentUserKilled'),
    workerIdleClosed: Signal.create<SubagentIdleClosedSignal>('AgentLifecycle/workerIdleClosed'),
  },

  eventHandlers: {
    agent_created: ({ event, state, emit }) => {
      const normalizedMode: 'clone' | 'spawn' = event.mode === 'clone' ? 'clone' : 'spawn'
      const normalizedContext = typeof event.context === 'string' ? event.context : ''
      if (typeof event.taskId !== 'string' || event.taskId.trim().length === 0) {
        return state
      }
      const normalizedTaskId = event.taskId

      const existingAgent = state.agents.get(event.agentId)
      if (existingAgent) {
        throw new Error(`[AgentLifecycleProjection] Invalid state transition: agent_created for already existing agent ${event.agentId} (forkId: ${existingAgent.forkId})`)
      }

      const existingForkAgentId = state.agentByForkId.get(event.forkId)
      if (existingForkAgentId) {
        throw new Error(`[AgentLifecycleProjection] Invalid state transition: agent_created for already indexed fork ${event.forkId} (agentId: ${existingForkAgentId})`)
      }

      emit.agentCreated({
        forkId: event.forkId,
        parentForkId: event.parentForkId,
        agentId: event.agentId,
        name: event.name,
        role: event.role as RoleId,
        taskId: normalizedTaskId,
        mode: normalizedMode,
        context: normalizedContext,
        timestamp: event.timestamp,
      })

      const agent: AgentInfo = {
        agentId: event.agentId,
        forkId: event.forkId,
        parentForkId: event.parentForkId,
        name: event.name,
        role: event.role as RoleId,
        context: normalizedContext,
        mode: normalizedMode,
        taskId: normalizedTaskId,
        message: event.message ?? null,
        status: 'working',
        lastIdleReason: null,
      }

      let next: AgentLifecycleState = {
        ...state,
        agents: new Map(state.agents).set(event.agentId, agent),
        agentByForkId: new Map(state.agentByForkId).set(event.forkId, event.agentId),
      }
      next = updateChildCount(next)
      return next
    },

    turn_started: ({ event, state, emit }) => {
      if (event.forkId === null) {
        // Root turn starting — open chain (preserve chainStartedAt within chain, reset on new chain)
        const isNewChain = state.rootWork.chainStartedAt === null
        let next: AgentLifecycleState = {
          ...state,
          rootWork: startRootWork(state.rootWork, event.timestamp),
        }
        // Clear stale lastIdleReason on idle agents when a new chain begins,
        // so deferred close doesn't pick up interrupt reasons from previous chains
        if (isNewChain) {
          const agents = new Map(state.agents)
          for (const [agentId, agent] of agents) {
            if (agent.status === 'idle' && agent.lastIdleReason !== null) {
              agents.set(agentId, { ...agent, lastIdleReason: null })
            }
          }
          next = { ...next, agents }
        }
        next = {
          ...next,
          rootWork: { ...next.rootWork, _currentTurnId: event.turnId },
        }
        return next
      }

      // Worker turn starting
      const agent = getAgentByForkId(state, event.forkId)
      if (!agent) return state

      if (agent.status !== 'working') {
        emit.agentBecameWorking({
          agentId: agent.agentId,
          forkId: agent.forkId,
          role: agent.role,
          parentForkId: agent.parentForkId,
          timestamp: event.timestamp,
        })
      }

      let next: AgentLifecycleState = {
        ...state,
        agents: new Map(state.agents).set(agent.agentId, {
          ...agent,
          status: 'working',
          lastIdleReason: null,
        }),
      }
      next = updateChildCount(next)
      return next
    },

    turn_outcome: ({ event, state, emit }) => {
      if (event.forkId === null) {
        // Root turn ending — turn-aware close
        if (state.rootWork._currentTurnId !== event.turnId) return state

        // Chain continues — don't close, just clear turn id
        if (outcomeWillChainContinue(event.outcome)) {
          return { ...state, rootWork: { ...state.rootWork, _currentTurnId: null } }
        }

        if (countWorkingChildren(state, null) > 0) {
          // Root turn ended but workers still running — deferred close
          return { ...state, rootWork: { ...state.rootWork, _currentTurnId: null } }
        }

        // No workers active — close root work
        const phase = event.outcome._tag === 'Cancelled' ? 'interrupted' : 'worked'
        return { ...state, rootWork: stopRootWork(state.rootWork, event.timestamp, phase) }
      }

      // Worker turn ending
      if (outcomeWillChainContinue(event.outcome)) return state

      const agent = getAgentByForkId(state, event.forkId)
      if (!agent) return state

      const reason =
        event.outcome._tag === 'Cancelled'
          ? 'interrupt'
          : event.outcome._tag === 'Completed'
            ? 'stable'
            : 'error'

      if (agent.status !== 'idle') {
        emit.agentBecameIdle({
          agentId: agent.agentId,
          forkId: agent.forkId,
          role: agent.role,
          parentForkId: agent.parentForkId,
          reason,
          timestamp: event.timestamp,
        })
      }

      let next: AgentLifecycleState = {
        ...state,
        agents: new Map(state.agents).set(agent.agentId, {
          ...agent,
          status: 'idle',
          lastIdleReason: reason,
        }),
      }
      next = updateChildCount(next)
      // Check deferred root close
      next = maybeDeferCloseRootWork(next, event.timestamp)
      return next
    },

    interrupt: ({ event, state, emit }) => {
      if (event.forkId === null) {
        // Root interrupt — stop root work
        let next: AgentLifecycleState = {
          ...state,
          rootWork: stopRootWork(state.rootWork, event.timestamp, 'interrupted'),
        }
        next = updateChildCount(next)
        return next
      }

      // Worker interrupt
      const agent = getAgentByForkId(state, event.forkId)
      if (!agent) return state

      if (agent.status !== 'idle') {
        emit.agentBecameIdle({
          agentId: agent.agentId,
          forkId: agent.forkId,
          role: agent.role,
          parentForkId: agent.parentForkId,
          reason: 'interrupt',
          timestamp: event.timestamp,
        })
      }

      let nextState: AgentLifecycleState = {
        ...state,
        agents: new Map(state.agents).set(agent.agentId, {
          ...agent,
          status: 'idle',
          lastIdleReason: 'interrupt',
        }),
      }
      nextState = updateChildCount(nextState)
      nextState = maybeDeferCloseRootWork(nextState, event.timestamp)
      return nextState
    },

    agent_killed: ({ event, state, emit }) => {
      const removed = removeKilledAgent({
        forkId: event.forkId,
        agentId: event.agentId,
        timestamp: event.timestamp,
        state,
      })
      if (!removed.agent) return state

      emit.agentKilled({
        agentId: removed.agent.agentId,
        forkId: removed.agent.forkId,
        parentForkId: removed.agent.parentForkId,
        role: removed.agent.role,
        title: removed.agent.name,
        reason: event.reason,
        timestamp: event.timestamp,
      })

      let next = removed.state
      next = updateChildCount(next)
      next = maybeDeferCloseRootWork(next, event.timestamp)
      return next
    },

    worker_user_killed: ({ event, state, emit }) => {
      const removed = removeKilledAgent({
        forkId: event.forkId,
        agentId: event.agentId,
        timestamp: event.timestamp,
        state,
      })
      if (!removed.agent) return state

      emit.subagentUserKilled({
        agentId: removed.agent.agentId,
        forkId: removed.agent.forkId,
        parentForkId: removed.agent.parentForkId,
        role: removed.agent.role,
        title: removed.agent.name,
        source: event.source,
        timestamp: event.timestamp,
      })

      let next = removed.state
      next = updateChildCount(next)
      next = maybeDeferCloseRootWork(next, event.timestamp)
      return next
    },

    worker_idle_closed: ({ event, state, emit }) => {
      const removed = removeKilledAgent({
        forkId: event.forkId,
        agentId: event.agentId,
        timestamp: event.timestamp,
        state,
      })
      if (!removed.agent) return state

      emit.workerIdleClosed({
        agentId: removed.agent.agentId,
        forkId: removed.agent.forkId,
        parentForkId: removed.agent.parentForkId,
        role: removed.agent.role,
        title: removed.agent.name,
        source: event.source,
        timestamp: event.timestamp,
      })

      let next = removed.state
      next = updateChildCount(next)
      next = maybeDeferCloseRootWork(next, event.timestamp)
      return next
    },

    agent_task_changed: ({ event, state }) => {
      const agent = state.agents.get(event.agentId)
      if (!agent) return state

      return {
        ...state,
        agents: new Map(state.agents).set(event.agentId, { ...agent, taskId: event.newTaskId }),
      }
    },

    message_start: ({ event, state }) => {
      if (event.forkId !== null) return state
      if (state.rootWork._currentTurnId !== event.turnId) return state
      return { ...state, rootWork: { ...state.rootWork, _thinkingCharCount: null } }
    },

    thinking_chunk: ({ event, state }) => {
      if (event.forkId !== null) return state
      if (state.rootWork._currentTurnId !== event.turnId) return state
      const nextCount = (state.rootWork._thinkingCharCount ?? 0) + event.text.length
      return { ...state, rootWork: resolveActivity({ ...state.rootWork, _thinkingCharCount: nextCount }) }
    },

    tool_event: ({ event, state }) => {
      if (event.forkId !== null) return state
      if (state.rootWork._currentTurnId !== event.turnId) return state

      switch (event.event._tag) {
        case 'ToolInputStarted': {
          const toolKey = event.toolKey
          const statusTool = STATUS_TOOL_ACTIVITIES[toolKey]
          const activeToolKey = statusTool || toolKey === 'spawnWorker' ? toolKey : null
          return {
            ...state,
            rootWork: resolveActivity({
              ...state.rootWork,
              _activeToolKey: activeToolKey,
              _thinkingCharCount: null,
            }),
          }
        }
        case 'ToolExecutionEnded':
        case 'ToolInputRejected': {
          if (state.rootWork._activeToolKey !== event.toolKey) return state
          return {
            ...state,
            rootWork: resolveActivity({ ...state.rootWork, _activeToolKey: null }),
          }
        }
        default:
          return state
      }
    },
  },

}))
