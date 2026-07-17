import { describe, expect, it } from 'vitest'
import { Effect, Layer, Option } from 'effect'
import {
  Addressed,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
  ProjectionBusTag,
  makeAmbientServiceLayer,
  makeProjectionBusLayer,
} from '@magnitudedev/event-core'

import type { AppEvent, TurnOutcomeEvent } from '../src/events'
import { DisplayTimelineProjection, type DisplayTimeline } from '../src/display'
import { AgentLifecycleProjection } from '../src/projections/agent-lifecycle'
import { AgentRoutingProjection } from '../src/projections/agent-routing'
import { GoalProjection } from '../src/projections/goal'
import { HarnessStateProjection } from '../src/projections/harness-state'
import { TurnProjection, type ForkTurnState } from '../src/projections/turn'
import { UserMessageResolutionProjection } from '../src/projections/user-message-resolution'
import { ToolUniverseSourceLive } from '../src/tools/tool-universe-live'

// Materialize timeline messages for assertions — accepts the normalized
// byId/order display form or a plain array (addressed readAll results).
const listMessages = <M,>(
  m: readonly M[] | { readonly byId: { readonly [id: string]: M }; readonly order: readonly string[] },
): readonly M[] => ('order' in m ? m.order.map((id) => m.byId[id]!) : m)


const ts = (n: number) => 1_700_700_000_000 + n
const InMemoryAddressedEntryStoreLive = Addressed.makeInMemoryAddressedEntryStoreLayer()

function makeRuntimeLayer() {
  const projectionBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )
  const baseLayer = Layer.provideMerge(
    makeAmbientServiceLayer<AppEvent>(),
    projectionBusLayer,
  )

  return Layer.provideMerge(
    Layer.mergeAll(
      AgentLifecycleProjection.Layer,
      AgentRoutingProjection.Layer,
      UserMessageResolutionProjection.Layer,
      GoalProjection.Layer,
      HarnessStateProjection.Layer,
      TurnProjection.Layer,
      Layer.provide(DisplayTimelineProjection.Layer, InMemoryAddressedEntryStoreLive),
    ),
    Layer.merge(baseLayer, ToolUniverseSourceLive),
  )
}

function goalStarted(objective = 'finish the work'): AppEvent {
  return {
    type: 'goal_started',
    forkId: null,
    goalId: 'goal-1',
    objective,
  }
}

function goalFinished(): AppEvent {
  return {
    type: 'goal_finished',
    forkId: null,
    goalId: 'goal-1',
    evidence: 'done',
  }
}

function agentCreated(): AppEvent {
  return {
    type: 'agent_created',
    forkId: 'fork-worker-1',
    parentForkId: null,
    agentId: 'agent-worker-1',
    name: 'Worker',
    role: 'engineer',
    context: 'ctx',
    mode: 'spawn',
    taskId: 'task-1',
    message: 'work on the task',
  } as any
}

function turnStarted(turnId: string): AppEvent {
  return {
    type: 'turn_started',
    forkId: null,
    turnId,
    chainId: `chain-${turnId}`,
  } as any
}

function stopped(turnId: string): TurnOutcomeEvent {
  return {
    type: 'turn_outcome',
    forkId: null,
    turnId,
    chainId: `chain-${turnId}`,
    strategyId: 'native',
    outcome: {
      _tag: 'Completed',
      completion: {
        toolCallsCount: 0,
        finishReason: 'stop',
        feedback: [],
        yieldTarget: null,
      },
      requestId: null,
    },
    commitPolicy: { _tag: 'commitCleanTurn' },
    inputTokens: null,
    outputTokens: null,
    cacheReadTokens: null,
    cacheWriteTokens: null,
    cost: null,
    providerId: 'test',
    modelId: 'role/leader',
  } as any
}

async function rootAfter(events: readonly AppEvent[]): Promise<ForkTurnState> {
  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* TurnProjection.Tag
    for (const [index, event] of events.entries()) {
      yield* bus.processEvent({ ...event, timestamp: ts(index + 1) } as any)
    }
    return yield* projection.getFork(null)
  })

  return Effect.runPromise(program.pipe(Effect.provide(makeRuntimeLayer())) as any)
}

async function displayAfter(events: readonly AppEvent[]): Promise<DisplayTimeline> {
  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* DisplayTimelineProjection.Tag
    for (const [index, event] of events.entries()) {
      yield* bus.processEvent({ ...event, timestamp: ts(index + 1) } as any)
    }
    const fork = yield* projection.getFork(null)
    const messages = yield* projection.addressed.forFork(null).messages.readAll(fork.messages)
    return { ...fork, messages }
  })

  return Effect.runPromise(program.pipe(Effect.provide(makeRuntimeLayer())) as any)
}

describe('goal mode turn guard', () => {
  it('queues a wake trigger when a goal starts without an initial prompt', async () => {
    const state = await rootAfter([
      goalStarted(),
    ])

    expect(state._tag).toBe('idle')
    expect(state.triggers.map(trigger => trigger._tag)).toContain('wake')
  })

  it('queues a wake trigger when the root stops before finishing an active goal', async () => {
    const state = await rootAfter([
      goalStarted(),
      turnStarted('turn-1'),
      stopped('turn-1'),
    ])

    expect(state._tag).toBe('idle')
    expect(state.triggers.map(trigger => trigger._tag)).toContain('wake')
  })

  it('does not queue a goal reminder while a worker is still active', async () => {
    const state = await rootAfter([
      goalStarted(),
      agentCreated(),
      turnStarted('turn-1'),
      stopped('turn-1'),
    ])

    expect(state._tag).toBe('idle')
    expect(state.triggers).toHaveLength(0)
  })

  it('does not queue a wake trigger after the active goal is finished', async () => {
    const state = await rootAfter([
      goalStarted(),
      turnStarted('turn-1'),
      goalFinished(),
      stopped('turn-1'),
    ])

    expect(state._tag).toBe('idle')
    expect(state.triggers).toHaveLength(0)
  })

  it('adds timeline-visible goal status messages on start and finish', async () => {
    const display = await displayAfter([
      goalStarted('ship the fix'),
      goalFinished(),
    ])

    const goalMessages = listMessages(display.messages).filter((message) => message.type === 'goal_status')
    expect(goalMessages).toHaveLength(2)
    expect(goalMessages[0]).toMatchObject({ status: 'started', objective: Option.some('ship the fix') })
    expect(goalMessages[1]).toMatchObject({ status: 'finished', evidence: Option.some('done') })
  })
})
