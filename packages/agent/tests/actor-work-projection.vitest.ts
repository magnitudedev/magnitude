/**
 * Root work lifecycle — now owned by AgentLifecycleProjection.rootWork.
 *
 * Tests the invariant that root work phase tracks correctly when child
 * workers become active, go idle, and resume.
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
import {
  AgentLifecycleProjection,
  rootWorkActivity,
  type AgentLifecycleState,
} from '../src/projections/agent-lifecycle'
import { GoalProjection } from '../src/projections/goal'
import { HarnessStateProjection } from '../src/projections/harness-state'
import { UserMessageResolutionProjection } from '../src/projections/user-message-resolution'
import { ToolUniverseSourceLive } from '../src/tools/tool-universe-live'

const ts = (n: number) => 1_700_100_000_000 + n
const InMemoryAddressedEntryStoreLive = Addressed.makeInMemoryAddressedEntryStoreLayer()

const runWithEvents = async (events: AppEvent[]): Promise<AgentLifecycleState> => {
  const baseBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )
  const baseLayer = Layer.provideMerge(
    makeAmbientServiceLayer<AppEvent>(),
    baseBusLayer,
  )

  const runtimeLayer = Layer.provideMerge(
    Layer.mergeAll(
      GoalProjection.Layer,
      TurnProjection.Layer,
      AgentRoutingProjection.Layer,
      AgentLifecycleProjection.Layer,
      HarnessStateProjection.Layer,
      UserMessageResolutionProjection.Layer,
    ),
    Layer.merge(baseLayer, ToolUniverseSourceLive),
  )

  const program = Effect.gen(function* () {
    const bus = yield* (ProjectionBusTag<AppEvent>())

    for (const event of events) {
      yield* bus.processEvent(event as any)
    }

    const projection = yield* AgentLifecycleProjection.Tag
    return yield* projection.get
  })

  return Effect.runPromise(
    program.pipe(Effect.provide(runtimeLayer)) as unknown as Effect.Effect<AgentLifecycleState>,
  )
}

const rootWork = (state: AgentLifecycleState) => state.rootWork

const completedTurnOutcome = (
  forkId: string | null,
  turnId: string,
  chainId: string,
  timestamp: number,
): AppEvent => ({
  type: 'turn_outcome',
  timestamp,
  forkId,
  turnId,
  chainId,
  strategyId: 'native',
  outcome: {
    _tag: 'Completed',
    requestId: null,
    completion: {
      toolCallsCount: 0,
      finishReason: 'stop',
      feedback: [],
      yieldTarget: null,
    },
  },
  inputTokens: null,
  outputTokens: null,
  cacheReadTokens: null,
  cacheWriteTokens: null,
  cost: null,
  providerId: null,
  modelId: null,
} as AppEvent)

const rootTurnStarted = (turnId: string, timestamp: number): AppEvent => ({
  type: 'turn_started',
  timestamp,
  forkId: null,
  turnId,
  chainId: 'root-chain',
} as AppEvent)

const rootGenerationStarted = (turnId: string, timestamp: number): AppEvent => ({
  type: 'model_generation_started',
  timestamp,
  forkId: null,
  turnId,
  chainId: 'root-chain',
} as AppEvent)

const rootTurnOutcome = (turnId: string, timestamp: number): AppEvent =>
  completedTurnOutcome(null, turnId, 'root-chain', timestamp)

describe('AgentLifecycleProjection rootWork — worker lifecycle', () => {
  it('root work stays idle when only a child worker is created (no root turn)', async () => {
    const state = await runWithEvents([
      { type: 'agent_created', timestamp: ts(1), forkId: 'worker-a', parentForkId: null, agentId: 'agent-worker-a', name: 'Worker A', role: 'engineer', context: 'worker context', mode: 'spawn', taskId: 'task-worker-a', message: 'starting work' } as AppEvent,
    ])

    // Root work opens on turn_started(root), not on agent_created
    expect(rootWork(state).phase).toBe('idle')
  })

  it('root work is working while root turn is active', async () => {
    const state = await runWithEvents([
      rootTurnStarted('root-turn-1', ts(1)),
      rootGenerationStarted('root-turn-1', ts(2)),
      { type: 'agent_created', timestamp: ts(3), forkId: 'worker-a', parentForkId: null, agentId: 'agent-worker-a', name: 'Worker A', role: 'engineer', context: 'worker context', mode: 'spawn', taskId: 'task-worker-a', message: 'starting work' } as AppEvent,
    ])

    expect(rootWork(state).phase).toBe('working')
    expect(rootWork(state).activeChildCount).toBe(1)
  })

  it('root work transitions to worked when root turn ends and no workers active', async () => {
    const state = await runWithEvents([
      rootTurnStarted('root-turn-1', ts(1)),
      rootGenerationStarted('root-turn-1', ts(2)),
      rootTurnOutcome('root-turn-1', ts(4)),
    ])

    expect(rootWork(state).phase).toBe('worked')
    expect(rootWork(state).lastChainMs).toBe(ts(4) - ts(2))
  })

  it('root work stays working when root turn ends but workers still running (deferred close)', async () => {
    const state = await runWithEvents([
      rootTurnStarted('root-turn-1', ts(1)),
      { type: 'agent_created', timestamp: ts(2), forkId: 'worker-a', parentForkId: null, agentId: 'agent-worker-a', name: 'Worker A', role: 'engineer', context: 'worker context', mode: 'spawn', taskId: 'task-worker-a', message: 'starting work' } as AppEvent,
      { type: 'turn_started', timestamp: ts(3), forkId: 'worker-a', turnId: 'worker-turn-1', chainId: 'chain-1' } as AppEvent,
      rootGenerationStarted('root-turn-1', ts(3)),
      rootTurnOutcome('root-turn-1', ts(4)),
    ])

    // Root turn ended but worker still running — root remains active and counts worker time.
    expect(rootWork(state).phase).toBe('waiting_for_workers')
  })

  it('root work closes (worked) when last worker goes idle after root turn already ended', async () => {
    const state = await runWithEvents([
      rootTurnStarted('root-turn-1', ts(1)),
      { type: 'agent_created', timestamp: ts(2), forkId: 'worker-a', parentForkId: null, agentId: 'agent-worker-a', name: 'Worker A', role: 'engineer', context: 'worker context', mode: 'spawn', taskId: 'task-worker-a', message: 'starting work' } as AppEvent,
      { type: 'turn_started', timestamp: ts(3), forkId: 'worker-a', turnId: 'worker-turn-1', chainId: 'chain-1' } as AppEvent,
      rootGenerationStarted('root-turn-1', ts(3)),
      rootTurnOutcome('root-turn-1', ts(4)),
      completedTurnOutcome('worker-a', 'worker-turn-1', 'chain-1', ts(5)),
    ])

    // Root turn ended (ts(4)), worker went idle (ts(5)) — deferred close fires
    expect(rootWork(state).phase).toBe('worked')
  })

  it('root work transitions to interrupted on root interrupt', async () => {
    const state = await runWithEvents([
      rootTurnStarted('root-turn-1', ts(1)),
      { type: 'interrupt', timestamp: ts(2), forkId: null, allKilled: false } as AppEvent,
    ])

    expect(rootWork(state).phase).toBe('interrupted')
    expect(rootWork(state).lastChainMs).toBe(0)
  })

  it('does not count model wait before generation', async () => {
    const state = await runWithEvents([
      rootTurnStarted('root-turn-1', ts(1)),
      rootGenerationStarted('root-turn-1', ts(101)),
      rootTurnOutcome('root-turn-1', ts(151)),
    ])

    expect(rootWork(state).lastChainMs).toBe(50)
  })

  it('chain timer resets each chain', async () => {
    const state = await runWithEvents([
      rootTurnStarted('root-turn-1', ts(100)),
      rootGenerationStarted('root-turn-1', ts(120)),
      rootTurnOutcome('root-turn-1', ts(200)),
      rootTurnStarted('root-turn-2', ts(300)),
      rootGenerationStarted('root-turn-2', ts(300)),
      rootTurnOutcome('root-turn-2', ts(350)),
    ])

    // Chain 1 excludes 20ms of prefill. Chain 2 has no prefill delay.
    // lastChainMs should be chain 2 duration (50ms), not accumulated (150ms).
    expect(rootWork(state).phase).toBe('worked')
    expect(rootWork(state).lastChainMs).toBe(50)
  })
})

describe('AgentLifecycleProjection rootWork — thinking lifecycle', () => {
  it('opens thinking on start and closes it on end without stopping the work clock', async () => {
    const state = await runWithEvents([
      rootTurnStarted('root-turn-1', ts(1)),
      { type: 'thinking_start', timestamp: ts(2), forkId: null, turnId: 'root-turn-1' } as AppEvent,
      { type: 'thinking_chunk', timestamp: ts(3), forkId: null, turnId: 'root-turn-1', text: 'Considering options' } as AppEvent,
      { type: 'thinking_end', timestamp: ts(4), forkId: null, turnId: 'root-turn-1' } as AppEvent,
    ])

    expect(rootWork(state)).toMatchObject({
      phase: 'working',
      workingStartedAt: ts(2),
      _thinkingCharCount: null,
    })
    expect(rootWorkActivity(rootWork(state))).toBeNull()
  })

  it('represents an open empty thinking segment from thinking_start', async () => {
    const state = await runWithEvents([
      rootTurnStarted('root-turn-1', ts(1)),
      { type: 'thinking_start', timestamp: ts(2), forkId: null, turnId: 'root-turn-1' } as AppEvent,
    ])

    expect(rootWork(state)).toMatchObject({
      phase: 'working',
      _thinkingCharCount: 0,
    })
    expect(rootWorkActivity(rootWork(state))).toEqual({ kind: 'thinking', message: 'Thinking' })
  })

  it('clears thinking when answer text starts even if thinking-end is missing', async () => {
    const state = await runWithEvents([
      rootTurnStarted('root-turn-1', ts(1)),
      { type: 'thinking_start', timestamp: ts(2), forkId: null, turnId: 'root-turn-1' } as AppEvent,
      { type: 'thinking_chunk', timestamp: ts(3), forkId: null, turnId: 'root-turn-1', text: 'Considering options' } as AppEvent,
      {
        type: 'message_start',
        timestamp: ts(4),
        forkId: null,
        turnId: 'root-turn-1',
        id: 'message-1',
        destination: { kind: 'user' },
      } as AppEvent,
    ])

    expect(rootWork(state)).toMatchObject({
      phase: 'working',
      _thinkingCharCount: null,
    })
    expect(rootWorkActivity(rootWork(state))).toBeNull()
  })

  it('clears thinking for the cloud thought-end then message-start ordering', async () => {
    const state = await runWithEvents([
      rootTurnStarted('root-turn-1', ts(1)),
      { type: 'thinking_start', timestamp: ts(2), forkId: null, turnId: 'root-turn-1' } as AppEvent,
      { type: 'thinking_chunk', timestamp: ts(3), forkId: null, turnId: 'root-turn-1', text: 'Cloud thought' } as AppEvent,
      { type: 'thinking_end', timestamp: ts(4), forkId: null, turnId: 'root-turn-1' } as AppEvent,
      {
        type: 'message_start',
        timestamp: ts(4),
        forkId: null,
        turnId: 'root-turn-1',
        id: 'message-1',
        destination: { kind: 'user' },
      } as AppEvent,
      { type: 'message_chunk', timestamp: ts(5), forkId: null, turnId: 'root-turn-1', id: 'message-1', text: 'Answer' } as AppEvent,
    ])

    expect(rootWork(state)).toMatchObject({
      phase: 'working',
      workingStartedAt: ts(2),
      _thinkingCharCount: null,
    })
    expect(rootWorkActivity(rootWork(state))).toBeNull()
  })

  it('clears thinking when answer text resumes after an interleaved thinking segment', async () => {
    const state = await runWithEvents([
      rootTurnStarted('root-turn-1', ts(1)),
      {
        type: 'message_start',
        timestamp: ts(2),
        forkId: null,
        turnId: 'root-turn-1',
        id: 'message-1',
        destination: { kind: 'user' },
      } as AppEvent,
      { type: 'thinking_start', timestamp: ts(3), forkId: null, turnId: 'root-turn-1' } as AppEvent,
      { type: 'thinking_chunk', timestamp: ts(4), forkId: null, turnId: 'root-turn-1', text: 'Interleaved thought' } as AppEvent,
      { type: 'message_chunk', timestamp: ts(5), forkId: null, turnId: 'root-turn-1', id: 'message-1', text: 'Answer' } as AppEvent,
    ])

    expect(rootWork(state)).toMatchObject({
      phase: 'working',
      _thinkingCharCount: null,
    })
    expect(rootWorkActivity(rootWork(state))).toBeNull()
  })

  it('ignores thinking-end events that do not belong to the current root turn', async () => {
    const state = await runWithEvents([
      rootTurnStarted('root-turn-1', ts(1)),
      { type: 'thinking_start', timestamp: ts(2), forkId: null, turnId: 'root-turn-1' } as AppEvent,
      { type: 'thinking_chunk', timestamp: ts(3), forkId: null, turnId: 'root-turn-1', text: 'Still thinking' } as AppEvent,
      { type: 'thinking_end', timestamp: ts(4), forkId: null, turnId: 'stale-turn' } as AppEvent,
      { type: 'thinking_end', timestamp: ts(5), forkId: 'worker-a', turnId: 'root-turn-1' } as AppEvent,
    ])

    expect(rootWork(state)._thinkingCharCount).toBe('Still thinking'.length)
    expect(rootWorkActivity(rootWork(state))).toEqual({ kind: 'thinking', message: 'Thinking' })
  })
})
