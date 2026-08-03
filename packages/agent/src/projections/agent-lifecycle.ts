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
import { Option, Schema } from 'effect'
import type { WorkSummaryPerformance } from '@magnitudedev/acn-protocol'
import { ModelRequestActivityAmbient } from '../model/model-request-activity'

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
const RootGenerationAggregateSchema = Schema.Struct({
  modelDisplayName: Schema.String,
  generatedTokens: Schema.Number.pipe(Schema.int(), Schema.nonNegative()),
  decodeDurationMs: Schema.Number.pipe(Schema.finite(), Schema.nonNegative()),
  requestCount: Schema.Number.pipe(Schema.int(), Schema.positive()),
  measuredRequestCount: Schema.Number.pipe(Schema.int(), Schema.nonNegative()),
  singleRequestDecodeTokensPerSecond: Schema.Number.pipe(Schema.finite(), Schema.nonNegative()),
})
type RootGenerationAggregate = typeof RootGenerationAggregateSchema.Type

const RootCurrentTurnSchema = Schema.Struct({
  turnId: Schema.String,
  startedAt: Schema.Number,
  modelActivityStarted: Schema.Boolean,
})

const RootWorkStateSchema = Schema.Struct({
  phase: Schema.Literal('idle', 'active', 'worked', 'interrupted'),
  accumulatedProductiveMs: Schema.Number,
  productiveStartedAt: Schema.NullOr(Schema.Number),
  lastProductiveMs: Schema.Number,
  chainStartedAt: Schema.NullOr(Schema.Number),
  activeChildCount: Schema.Number,
  _currentTurn: Schema.NullOr(RootCurrentTurnSchema),
  _currentChainId: Schema.optionalWith(Schema.NullOr(Schema.String), {
    default: () => null,
  }),
  _isThinking: Schema.Boolean,
  _generation: Schema.optionalWith(Schema.NullOr(RootGenerationAggregateSchema), {
    default: () => null,
  }),
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

export interface RootWorkCompletedSignal {
  readonly chainId: string
  readonly completedAt: number
  readonly durationMs: number
  readonly phase: 'worked' | 'interrupted'
  readonly performance: WorkSummaryPerformance | null
}

// --- Root work helpers ---

export function isRootWorkActive(state: RootWorkState): boolean {
  return state.phase === 'active'
}

function isRootProductiveClockRunning(state: RootWorkState): boolean {
  return isRootWorkActive(state) && state.productiveStartedAt !== null
}

function accumulatedRootProductiveMs(state: RootWorkState, timestamp: number): number {
  return isRootProductiveClockRunning(state) && state.productiveStartedAt !== null
    ? state.accumulatedProductiveMs + Math.max(0, timestamp - state.productiveStartedAt)
    : state.accumulatedProductiveMs
}

function startRootTurn(
  state: RootWorkState,
  timestamp: number,
  chainId: string,
  turnId: string,
): RootWorkState {
  const isNewChain = !isRootWorkActive(state) || state._currentChainId !== chainId
  return {
    ...state,
    phase: 'active',
    accumulatedProductiveMs: isNewChain ? 0 : state.accumulatedProductiveMs,
    productiveStartedAt: isNewChain
      ? (state.activeChildCount > 0 ? timestamp : null)
      : state.productiveStartedAt,
    chainStartedAt: isNewChain ? timestamp : state.chainStartedAt,
    _currentTurn: { turnId, startedAt: timestamp, modelActivityStarted: false },
    _isThinking: false,
    _generation: isNewChain ? null : state._generation,
  }
}

function recordRootGeneration(
  state: RootWorkState,
  event: Extract<AppEvent, { type: 'turn_outcome' }>,
): RootWorkState {
  const measurement = event.generationPerformance
  const modelDisplayName = event.modelDisplayName ?? measurement?.modelDisplayName ?? null
  if (modelDisplayName === null) return state
  const current = state._generation
  const next: RootGenerationAggregate = current === null
    ? {
        modelDisplayName,
        generatedTokens: measurement?.generatedTokens ?? 0,
        decodeDurationMs: measurement?.decodeDurationMs ?? 0,
        requestCount: 1,
        measuredRequestCount: measurement === null || measurement === undefined ? 0 : 1,
        singleRequestDecodeTokensPerSecond: measurement?.decodeTokensPerSecond ?? 0,
      }
    : {
        ...current,
        generatedTokens: current.generatedTokens + (measurement?.generatedTokens ?? 0),
        decodeDurationMs: current.decodeDurationMs + (measurement?.decodeDurationMs ?? 0),
        requestCount: current.requestCount + 1,
        measuredRequestCount: current.measuredRequestCount
          + (measurement === null || measurement === undefined ? 0 : 1),
      }
  return { ...state, _generation: next }
}

function summarizeGeneration(aggregate: RootGenerationAggregate | null): WorkSummaryPerformance | null {
  if (aggregate === null) return null
  const hasCompleteDecodeMeasurement = aggregate.measuredRequestCount === aggregate.requestCount
    && aggregate.generatedTokens > 0
    && aggregate.decodeDurationMs > 0
  return {
    modelDisplayName: aggregate.modelDisplayName,
    decodeTokensPerSecond: hasCompleteDecodeMeasurement
      ? Option.some(aggregate.requestCount === 1
        ? aggregate.singleRequestDecodeTokensPerSecond
        : aggregate.generatedTokens * 1_000 / aggregate.decodeDurationMs)
      : Option.none(),
  }
}

function startRootGeneration(state: RootWorkState, timestamp: number): RootWorkState {
  if (!isRootWorkActive(state)) return state
  return {
    ...state,
    productiveStartedAt: state.productiveStartedAt ?? timestamp,
  }
}

function markRootModelActivityStarted(state: RootWorkState): RootWorkState {
  return state._currentTurn === null || state._currentTurn.modelActivityStarted
    ? state
    : {
        ...state,
        _currentTurn: { ...state._currentTurn, modelActivityStarted: true },
      }
}

function pauseRootForModel(state: RootWorkState, timestamp: number): RootWorkState {
  if (!isRootWorkActive(state)) return state
  const workersRemainActive = state.activeChildCount > 0
  return {
    ...state,
    accumulatedProductiveMs: workersRemainActive
      ? state.accumulatedProductiveMs
      : accumulatedRootProductiveMs(state, timestamp),
    productiveStartedAt: workersRemainActive
      ? (state.productiveStartedAt ?? timestamp)
      : null,
    _isThinking: false,
  }
}

function waitForRootWorkers(state: RootWorkState, timestamp: number): RootWorkState {
  if (!isRootWorkActive(state)) return state
  return isRootProductiveClockRunning(state)
    ? { ...state, _isThinking: false }
    : { ...state, productiveStartedAt: timestamp, _isThinking: false }
}

function stopRootWork(
  state: RootWorkState,
  timestamp: number,
  phase: 'worked' | 'interrupted',
): {
  readonly state: RootWorkState
  readonly completion: RootWorkCompletedSignal | null
} {
  const productiveMs = accumulatedRootProductiveMs(state, timestamp)
  return {
    state: {
      ...state,
      phase,
      accumulatedProductiveMs: 0,
      productiveStartedAt: null,
      lastProductiveMs: productiveMs,
      chainStartedAt: null,
      _currentTurn: null,
      _currentChainId: null,
      _isThinking: false,
      _generation: null,
    },
    completion: !isRootWorkActive(state) || state._currentChainId === null
      ? null
      : {
          chainId: state._currentChainId,
          completedAt: timestamp,
          durationMs: productiveMs,
          phase,
          performance: summarizeGeneration(state._generation),
        },
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

function updateChildCount(state: AgentLifecycleState, timestamp: number): AgentLifecycleState {
  const childCount = countWorkingChildren(state, null)
  const rootWork = state.rootWork
  const shouldClockFollowWorkers = isRootWorkActive(rootWork)
    && rootWork._currentTurn === null
  const nextRootWork = shouldClockFollowWorkers
    ? childCount > 0
      ? {
          ...rootWork,
          productiveStartedAt: rootWork.productiveStartedAt ?? timestamp,
        }
      : {
          ...rootWork,
          accumulatedProductiveMs: accumulatedRootProductiveMs(rootWork, timestamp),
          productiveStartedAt: null,
        }
    : rootWork
  return {
    ...state,
    rootWork: { ...nextRootWork, activeChildCount: childCount },
  }
}

/**
 * Deferred root work close: if root's turn already ended but workers were
 * still running, close root work now that the last worker went idle.
 */
function maybeDeferCloseRootWork(
  state: AgentLifecycleState,
  timestamp: number,
): {
  readonly state: AgentLifecycleState
  readonly completion: RootWorkCompletedSignal | null
} {
  if (!isRootWorkActive(state.rootWork)) return { state, completion: null }
  if (state.rootWork._currentTurn !== null) return { state, completion: null }
  if (countWorkingChildren(state, null) > 0) return { state, completion: null }
  const phase = anyAgentInterrupted(state) ? 'interrupted' : 'worked'
  const stopped = stopRootWork(state.rootWork, timestamp, phase)
  return {
    state: { ...state, rootWork: stopped.state },
    completion: stopped.completion,
  }
}

export const AgentLifecycleProjection = Projection.define<AppEvent>()(({
  name: 'AgentLifecycle',
  state: AgentLifecycleStateSchema,
  ambients: [ModelRequestActivityAmbient] as const,

  initial: {
    agents: new Map<string, AgentInfo>(),
    agentByForkId: new Map<string, string>(),
    rootWork: {
      phase: 'idle',
      accumulatedProductiveMs: 0,
      productiveStartedAt: null,
      lastProductiveMs: 0,
      chainStartedAt: null,
      activeChildCount: 0,
      _currentTurn: null,
      _currentChainId: null,
      _isThinking: false,
      _generation: null,
    },
  },

  signals: {
    agentCreated: Signal.create<AgentCreatedSignal>('AgentLifecycle/created'),
    agentBecameIdle: Signal.create<AgentBecameIdleSignal>('AgentLifecycle/agentBecameIdle'),
    agentBecameWorking: Signal.create<AgentBecameWorkingSignal>('AgentLifecycle/agentBecameWorking'),
    agentKilled: Signal.create<AgentKilledSignal>('AgentLifecycle/agentKilled'),
    subagentUserKilled: Signal.create<SubagentUserKilledSignal>('AgentLifecycle/subagentUserKilled'),
    workerIdleClosed: Signal.create<SubagentIdleClosedSignal>('AgentLifecycle/workerIdleClosed'),
    rootWorkCompleted: Signal.create<RootWorkCompletedSignal>('AgentLifecycle/rootWorkCompleted'),
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
      next = updateChildCount(next, event.timestamp)
      return next
    },

    turn_started: ({ event, state, emit }) => {
      if (event.forkId === null) {
        // Root turn starting — the live chain clock begins immediately; the
        // separate productive clock remains paused until generation begins.
        const isNewChain = !isRootWorkActive(state.rootWork)
          || state.rootWork._currentChainId !== event.chainId
        let next: AgentLifecycleState = {
          ...state,
          rootWork: startRootTurn(state.rootWork, event.timestamp, event.chainId, event.turnId),
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
        next = { ...next, rootWork: { ...next.rootWork, _currentChainId: event.chainId } }
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
      next = updateChildCount(next, event.timestamp)
      return next
    },

    model_generation_started: ({ event, state }) => {
      if (event.forkId !== null) return state
      if (state.rootWork._currentTurn?.turnId !== event.turnId) return state
      if (state.rootWork._currentChainId !== event.chainId) return state
      return {
        ...state,
        rootWork: markRootModelActivityStarted(startRootGeneration(state.rootWork, event.timestamp)),
      }
    },

    turn_outcome: ({ event, state, emit }) => {
      if (event.forkId === null) {
        // Root turn ending — turn-aware close
        if (state.rootWork._currentTurn?.turnId !== event.turnId) return state
        const nextState = {
          ...state,
          rootWork: recordRootGeneration(state.rootWork, event),
        }

        // Chain continues — don't close, just clear turn id
        if (outcomeWillChainContinue(event.outcome)) {
          return {
            ...nextState,
            rootWork: {
              ...pauseRootForModel(nextState.rootWork, event.timestamp),
              _currentTurn: null,
            },
          }
        }

        if (countWorkingChildren(state, null) > 0) {
          // Root turn ended but workers still running — deferred close
          return {
            ...nextState,
            rootWork: {
              ...waitForRootWorkers(nextState.rootWork, event.timestamp),
              _currentTurn: null,
            },
          }
        }

        // No workers active — close root work
        const phase = event.outcome._tag === 'Cancelled' ? 'interrupted' : 'worked'
        const stopped = stopRootWork(nextState.rootWork, event.timestamp, phase)
        if (stopped.completion !== null) emit.rootWorkCompleted(stopped.completion)
        return { ...nextState, rootWork: stopped.state }
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
      next = updateChildCount(next, event.timestamp)
      // Check deferred root close
      const deferred = maybeDeferCloseRootWork(next, event.timestamp)
      if (deferred.completion !== null) emit.rootWorkCompleted(deferred.completion)
      return deferred.state
    },

    interrupt: ({ event, state, emit }) => {
      if (event.forkId === null) {
        // Root interrupt — stop root work
        const stopped = stopRootWork(state.rootWork, event.timestamp, 'interrupted')
        if (stopped.completion !== null) emit.rootWorkCompleted(stopped.completion)
        let next: AgentLifecycleState = {
          ...state,
          rootWork: stopped.state,
        }
        next = updateChildCount(next, event.timestamp)
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
      nextState = updateChildCount(nextState, event.timestamp)
      const deferred = maybeDeferCloseRootWork(nextState, event.timestamp)
      if (deferred.completion !== null) emit.rootWorkCompleted(deferred.completion)
      return deferred.state
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
      next = updateChildCount(next, event.timestamp)
      const deferred = maybeDeferCloseRootWork(next, event.timestamp)
      if (deferred.completion !== null) emit.rootWorkCompleted(deferred.completion)
      return deferred.state
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
      next = updateChildCount(next, event.timestamp)
      const deferred = maybeDeferCloseRootWork(next, event.timestamp)
      if (deferred.completion !== null) emit.rootWorkCompleted(deferred.completion)
      return deferred.state
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
      next = updateChildCount(next, event.timestamp)
      const deferred = maybeDeferCloseRootWork(next, event.timestamp)
      if (deferred.completion !== null) emit.rootWorkCompleted(deferred.completion)
      return deferred.state
    },

    agent_task_changed: ({ event, state }) => {
      const agent = state.agents.get(event.agentId)
      if (!agent) return state

      return {
        ...state,
        agents: new Map(state.agents).set(event.agentId, { ...agent, taskId: event.newTaskId }),
      }
    },

    thinking_start: ({ event, state }) => {
      if (event.forkId !== null) return state
      if (state.rootWork._currentTurn?.turnId !== event.turnId) return state
      return {
        ...state,
        rootWork: {
          ...markRootModelActivityStarted(startRootGeneration(state.rootWork, event.timestamp)),
          _isThinking: true,
        },
      }
    },

    message_start: ({ event, state }) => {
      if (event.forkId !== null) return state
      if (state.rootWork._currentTurn?.turnId !== event.turnId) return state
      return {
        ...state,
        rootWork: {
          ...markRootModelActivityStarted(startRootGeneration(state.rootWork, event.timestamp)),
          _isThinking: false,
        },
      }
    },

    message_chunk: ({ event, state }) => {
      if (event.forkId !== null) return state
      if (state.rootWork._currentTurn?.turnId !== event.turnId) return state
      return {
        ...state,
        rootWork: {
          ...markRootModelActivityStarted(startRootGeneration(state.rootWork, event.timestamp)),
          _isThinking: false,
        },
      }
    },

    message_end: ({ event, state }) => {
      if (event.forkId !== null) return state
      if (state.rootWork._currentTurn?.turnId !== event.turnId) return state
      return {
        ...state,
        rootWork: { ...state.rootWork, _isThinking: false },
      }
    },

    thinking_end: ({ event, state }) => {
      if (event.forkId !== null) return state
      if (state.rootWork._currentTurn?.turnId !== event.turnId) return state
      return {
        ...state,
        rootWork: { ...state.rootWork, _isThinking: false },
      }
    },

    tool_event: ({ event, state }) => {
      if (event.forkId !== null) return state
      if (state.rootWork._currentTurn?.turnId !== event.turnId) return state

      switch (event.event._tag) {
        case 'ToolInputStarted': {
          return {
            ...state,
            rootWork: {
              ...markRootModelActivityStarted(startRootGeneration(state.rootWork, event.timestamp)),
              _isThinking: false,
            },
          }
        }
        case 'ToolExecutionEnded':
        case 'ToolInputRejected':
          return state
        default:
          return state
      }
    },
  },

  ambientHandlers: (on) => [
    on(ModelRequestActivityAmbient, ({ value, state }) => {
      if (value === null || value.turn.forkId !== null) return state
      if (value.turn.turnId !== state.rootWork._currentTurn?.turnId) return state
      if (value.turn.chainId !== state.rootWork._currentChainId) return state
      if (value.progress.phase !== 'prefill' && value.progress.phase !== 'generating') {
        return state
      }
      const rootWork = markRootModelActivityStarted(state.rootWork)
      return rootWork === state.rootWork ? state : { ...state, rootWork }
    }),
  ],

}))
