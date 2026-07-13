/**
 * Tests for the fixed display/actor-work projection system.
 * Originally repro tests for three regressions, now verifies the fixes:
 * 1. Working timer resets per chain (chainStartedAt, lastChainMs)
 * 2. Worker count only counts working workers
 * 3. Worker interrupt state is shown (phase = 'interrupted', deferred close)
 */

import { describe, it, expect } from 'vitest'
import { Effect, Layer } from 'effect'
import {
  Addressed,
  ProjectionBusTag,
  makeProjectionBusLayer,
  makeAmbientServiceLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import type { AppEvent } from '../src/events'
import { TurnProjection } from '../src/projections/turn'
import { AgentRoutingProjection } from '../src/projections/agent-routing'
import { AgentLifecycleProjection, type AgentLifecycleState } from '../src/projections/agent-lifecycle'
import { GoalProjection } from '../src/projections/goal'
import { HarnessStateProjection } from '../src/projections/harness-state'
import { UserMessageResolutionProjection } from '../src/projections/user-message-resolution'
import { DisplayTimelineProjection } from '../src/display/timeline-projection'
import { TaskAssignmentProjection } from '../src/projections/task-assignment'
import { TaskGraphProjection } from '../src/projections/task-graph'
import { SessionContextProjection } from '../src/projections/session-context'
import { ChatTitleProjection } from '../src/projections/chat-title'
import { WindowProjection } from '../src/window'
import { CompactionProjection } from '../src/projections/compaction'
import {
  materializeDisplayActors,
  materializeDisplayTasks,
} from '../src/display-view/semantic'

const ts = (n: number) => 1_700_100_000_000 + n
const InMemoryAddressedEntryStoreLive = Addressed.makeInMemoryAddressedEntryStoreLayer()

const runtimeLayer = Layer.provideMerge(
  Layer.mergeAll(
    GoalProjection.Layer,
    TurnProjection.Layer,
    AgentRoutingProjection.Layer,
    AgentLifecycleProjection.Layer,
    HarnessStateProjection.Layer,
    UserMessageResolutionProjection.Layer,
    TaskGraphProjection.Layer,
    TaskAssignmentProjection.Layer,
    SessionContextProjection.Layer,
    ChatTitleProjection.Layer,
    WindowProjection.Layer,
    CompactionProjection.Layer,
    Layer.provide(DisplayTimelineProjection.Layer, InMemoryAddressedEntryStoreLive),
  ),
  Layer.provideMerge(
    makeAmbientServiceLayer<AppEvent>(),
    Layer.provideMerge(
      makeProjectionBusLayer<AppEvent>(),
      Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    ),
  ),
)

// ── Event helpers ───────────────────────────────────────────────

const turnStarted = (forkId: string | null, turnId: string, chainId: string, timestamp: number): AppEvent => ({
  type: 'turn_started', timestamp, forkId, turnId, chainId,
} as AppEvent)

const completedTurnOutcome = (forkId: string | null, turnId: string, chainId: string, timestamp: number): AppEvent => ({
  type: 'turn_outcome', timestamp, forkId, turnId, chainId, strategyId: 'native',
  outcome: { _tag: 'Completed', requestId: null, completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [], yieldTarget: null } },
  inputTokens: null, outputTokens: null, cacheReadTokens: null, cacheWriteTokens: null, cost: null, providerId: null, modelId: null,
} as AppEvent)

const cancelledTurnOutcome = (forkId: string | null, turnId: string, chainId: string, timestamp: number): AppEvent => ({
  type: 'turn_outcome', timestamp, forkId, turnId, chainId, strategyId: 'native',
  outcome: { _tag: 'Cancelled', requestId: null, reason: { _tag: 'UserInterrupt' } },
  commitPolicy: { _tag: 'commitCleanTurn' },
  inputTokens: null, outputTokens: null, cacheReadTokens: null, cacheWriteTokens: null, cost: null, providerId: null, modelId: null,
} as AppEvent)

const agentCreated = (forkId: string, parentForkId: string | null, agentId: string, name: string, timestamp: number, taskId = `task-${forkId}`): AppEvent => ({
  type: 'agent_created', timestamp, forkId, parentForkId, agentId, name, role: 'engineer', context: 'worker context', mode: 'spawn', taskId, message: 'starting work',
} as AppEvent)

const interruptEvent = (forkId: string | null, timestamp: number, _isKill = false): AppEvent => ({
  type: 'interrupt', timestamp, forkId,
} as AppEvent)

// ── Run helper ──────────────────────────────────────────────────

async function runWithEvents(events: readonly AppEvent[]): Promise<{
  agentStatus: AgentLifecycleState
  displayActors: ReturnType<typeof materializeDisplayActors>
  displayTasks: ReturnType<typeof materializeDisplayTasks>
}> {
  const program = Effect.gen(function* () {
    const bus = yield* (ProjectionBusTag<AppEvent>())
    for (const event of events) {
      yield* bus.processEvent(event as never)
    }

    const agentStatusProj = yield* AgentLifecycleProjection.Tag
    const taskWorkerProj = yield* TaskAssignmentProjection.Tag
    const windowProj = yield* WindowProjection.Tag
    const compactionProj = yield* CompactionProjection.Tag

    const agentStatus = yield* agentStatusProj.get
    const taskWorker = yield* taskWorkerProj.get
    const windowForkState = yield* windowProj.getFork(null)
    const windowState = { forks: new Map([[null, windowForkState]]) }
    const compactionForkState = yield* compactionProj.getFork(null)
    const compactionState = { forks: new Map([[null, compactionForkState]]) }

    const displayActors = materializeDisplayActors(agentStatus, taskWorker, windowState, compactionState)
    const displayTasks = materializeDisplayTasks(taskWorker)

    return { agentStatus, displayActors, displayTasks }
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as never) as Promise<any>
}

const rootWork = (state: AgentLifecycleState) => state.rootWork

// ============================================================================
// FIX 1: Timer resets per chain
// ============================================================================

describe('FIX 1 — Working timer resets per chain', () => {
  it('chainStartedAt resets on each new chain', async () => {
    const result = await runWithEvents([
      turnStarted(null, 'turn-1', 'chain-1', ts(1000)),
      completedTurnOutcome(null, 'turn-1', 'chain-1', ts(5000)),
      turnStarted(null, 'turn-2', 'chain-2', ts(6000)),
    ])

    const work = rootWork(result.agentStatus)
    expect(work.phase).toBe('working')
    expect(work.chainStartedAt).toBe(ts(6000)) // reset to chain-2 start
    expect(work.lastChainMs).toBe(4000) // chain-1 duration saved
  })

  it('completed summary shows last chain duration, not accumulated total', async () => {
    const result = await runWithEvents([
      turnStarted(null, 'turn-1', 'chain-1', ts(1000)),
      completedTurnOutcome(null, 'turn-1', 'chain-1', ts(5000)),
      turnStarted(null, 'turn-2', 'chain-2', ts(6000)),
      completedTurnOutcome(null, 'turn-2', 'chain-2', ts(8000)),
    ])

    const work = rootWork(result.agentStatus)
    expect(work.phase).toBe('worked')
    expect(work.lastChainMs).toBe(2000) // chain-2 only, not 6000
  })

  it('display actor lastWorkMs matches lastChainMs', async () => {
    const result = await runWithEvents([
      turnStarted(null, 'turn-1', 'chain-1', ts(1000)),
      completedTurnOutcome(null, 'turn-1', 'chain-1', ts(5000)),
      turnStarted(null, 'turn-2', 'chain-2', ts(6000)),
      completedTurnOutcome(null, 'turn-2', 'chain-2', ts(8000)),
    ])

    const rootActor = result.displayActors['root']
    expect(rootActor.work.phase).toBe('worked')
    expect(rootActor.work.lastWorkMs).toBe(2000) // last chain only
  })
})

// ============================================================================
// FIX 2: Worker count only counts working workers
// ============================================================================

describe('FIX 2 — Worker count only counts working workers', () => {
  it('activeChildCount is 1 while worker is working, 0 after idle', async () => {
    const result1 = await runWithEvents([
      agentCreated('worker-1', null, 'agent-worker-1', 'Worker 1', ts(1000)),
      turnStarted(null, 'root-turn-1', 'chain-1', ts(1000)),
      turnStarted('worker-1', 'worker-turn-1', 'chain-1', ts(1100)),
    ])
    expect(rootWork(result1.agentStatus).activeChildCount).toBe(1)

    const result2 = await runWithEvents([
      agentCreated('worker-1', null, 'agent-worker-1', 'Worker 1', ts(1000)),
      turnStarted(null, 'root-turn-1', 'chain-1', ts(1000)),
      turnStarted('worker-1', 'worker-turn-1', 'chain-1', ts(1100)),
      completedTurnOutcome('worker-1', 'worker-turn-1', 'chain-1', ts(3000)),
    ])
    expect(rootWork(result2.agentStatus).activeChildCount).toBe(0)
  })
})

// ============================================================================
// FIX 3: Worker interrupt state shown + deferred close
// ============================================================================

describe('FIX 3a — Root work stays working while root is streaming', () => {
  it('worker going idle does NOT stop root work while root turn is still active', async () => {
    const result = await runWithEvents([
      agentCreated('worker-1', null, 'agent-worker-1', 'Worker 1', ts(1000)),
      turnStarted(null, 'root-turn-1', 'chain-1', ts(1000)),
      turnStarted('worker-1', 'worker-turn-1', 'chain-1', ts(1100)),
      completedTurnOutcome('worker-1', 'worker-turn-1', 'chain-1', ts(3000)),
    ])

    const work = rootWork(result.agentStatus)
    // Root turn is still active (_currentTurnId !== null) — root stays working
    expect(work.phase).toBe('working')
    expect(work._currentTurnId).toBe('root-turn-1')
  })

  it('worker interrupt does NOT stop root work while root turn is still active', async () => {
    const result = await runWithEvents([
      agentCreated('worker-1', null, 'agent-worker-1', 'Worker 1', ts(1000)),
      turnStarted(null, 'root-turn-1', 'chain-1', ts(1000)),
      turnStarted('worker-1', 'worker-turn-1', 'chain-1', ts(1100)),
      interruptEvent('worker-1', ts(2000), false),
    ])

    const work = rootWork(result.agentStatus)
    expect(work.phase).toBe('working')
    expect(work._currentTurnId).toBe('root-turn-1')
  })

  it('deferred close: root work closes when last worker goes idle AFTER root turn ended', async () => {
    const result = await runWithEvents([
      agentCreated('worker-1', null, 'agent-worker-1', 'Worker 1', ts(1000)),
      turnStarted(null, 'root-turn-1', 'chain-1', ts(1000)),
      turnStarted('worker-1', 'worker-turn-1', 'chain-1', ts(1100)),
      completedTurnOutcome(null, 'root-turn-1', 'chain-1', ts(2000)), // root turn ends
      completedTurnOutcome('worker-1', 'worker-turn-1', 'chain-1', ts(3000)), // worker goes idle
    ])

    const work = rootWork(result.agentStatus)
    expect(work.phase).toBe('worked') // deferred close fired
  })
})

describe('FIX 3b — Worker interrupt shows in display actors', () => {
  it('interrupted worker has phase "interrupted" in display actors', async () => {
    const result = await runWithEvents([
      agentCreated('worker-1', null, 'agent-worker-1', 'Worker 1', ts(1000)),
      turnStarted(null, 'root-turn-1', 'chain-1', ts(1000)),
      turnStarted('worker-1', 'worker-turn-1', 'chain-1', ts(1100)),
      interruptEvent('worker-1', ts(2000), false),
    ])

    const workerActor = result.displayActors['worker-1']
    expect(workerActor).toBeDefined()
    expect(workerActor.work.phase).toBe('interrupted')
  })

  it('idle worker has phase "worked" in display actors', async () => {
    const result = await runWithEvents([
      agentCreated('worker-1', null, 'agent-worker-1', 'Worker 1', ts(1000)),
      turnStarted(null, 'root-turn-1', 'chain-1', ts(1000)),
      turnStarted('worker-1', 'worker-turn-1', 'chain-1', ts(1100)),
      completedTurnOutcome('worker-1', 'worker-turn-1', 'chain-1', ts(3000)),
    ])

    const workerActor = result.displayActors['worker-1']
    expect(workerActor).toBeDefined()
    expect(workerActor.work.phase).toBe('worked')
  })
})

describe('FIX 3c — Root interrupt', () => {
  it('root interrupt sets rootWork phase to interrupted', async () => {
    const result = await runWithEvents([
      turnStarted(null, 'root-turn-1', 'chain-1', ts(1000)),
      interruptEvent(null, ts(2000), false),
    ])

    expect(rootWork(result.agentStatus).phase).toBe('interrupted')
  })

  it('interrupt-all (per-fork fan-out) sets root and workers to interrupted', async () => {
    const result = await runWithEvents([
      agentCreated('worker-1', null, 'agent-worker-1', 'Worker 1', ts(1000)),
      turnStarted(null, 'root-turn-1', 'chain-1', ts(1000)),
      turnStarted('worker-1', 'worker-turn-1', 'chain-1', ts(1100)),
      // ACN fans out: root interrupt + per-worker interrupt
      interruptEvent(null, ts(3000)),
      interruptEvent('worker-1', ts(3000)),
    ])

    expect(rootWork(result.agentStatus).phase).toBe('interrupted')
    // Worker agent goes idle with interrupt reason
    const workerActor = result.displayActors['worker-1']
    expect(workerActor.work.phase).toBe('interrupted')
  })
})
