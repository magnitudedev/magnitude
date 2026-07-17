import { describe, expect, it } from 'vitest'
import { Effect, Layer, Option } from 'effect'
import {
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
  ProjectionBusTag,
  makeAmbientServiceLayer,
  makeProjectionBusLayer,
} from '@magnitudedev/event-core'

import type { AppEvent, TurnOutcomeEvent } from '../src/events'
import { AgentLifecycleProjection } from '../src/projections/agent-lifecycle'
import { AgentRoutingProjection } from '../src/projections/agent-routing'
import {
  TurnActive,
  TurnProjection,
  hasPendingAdvisorRequirement,
  turnRequiresAdvisor,
  type ForkTurnState,
  type TurnTrigger,
} from '../src/projections/turn'
import { GoalProjection } from '../src/projections/goal'
import { UserMessageResolutionProjection } from '../src/projections/user-message-resolution'

const ts = (n: number) => 1_700_700_000_000 + n

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
      TurnProjection.Layer,
    ),
    baseLayer,
  )
}

async function rootSnapshots(
  events: readonly AppEvent[],
  captureAfterIndexes: readonly number[],
): Promise<readonly ForkTurnState[]> {
  const capture = new Set(captureAfterIndexes)

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* TurnProjection.Tag
    const snapshots: ForkTurnState[] = []

    for (let index = 0; index < events.length; index += 1) {
      yield* bus.processEvent(events[index] as any)
      if (capture.has(index)) {
        snapshots.push(yield* projection.getFork(null))
      }
    }

    return snapshots
  })

  return Effect.runPromise(program.pipe(Effect.provide(makeRuntimeLayer())) as any)
}

function turnStarted(turnId: string, n: number): AppEvent {
  return {
    type: 'turn_started',
    forkId: null,
    turnId,
    chainId: `chain-${turnId}`,
    timestamp: ts(n),
  } as any
}

function turnOutcome(turnId: string, n: number, toolCallsCount = 0): TurnOutcomeEvent {
  return {
    type: 'turn_outcome',
    forkId: null,
    turnId,
    chainId: `chain-${turnId}`,
    strategyId: 'native',
    outcome: {
      _tag: 'Completed',
      completion: {
        toolCallsCount,
        finishReason: toolCallsCount > 0 ? 'tool_calls' : 'stop',
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
    timestamp: ts(n),
  } as any
}

function interrupt(n: number): AppEvent {
  return {
    type: 'interrupt',
    forkId: null,
    timestamp: ts(n),
  } as any
}

function observerOutcome(args: {
  readonly observedTurnId: string
  readonly n: number
  readonly escalate: boolean
}): AppEvent {
  return {
    type: 'observer_outcome',
    forkId: null,
    observedTurnId: args.observedTurnId,
    observerTurnId: `observer-${args.n}`,
    chainId: `chain-${args.observedTurnId}`,
    escalate: args.escalate,
    justification: args.escalate ? 'churn' : null,
    reasoning: args.escalate ? 'stuck' : 'fine',
    timestamp: ts(args.n),
  } as any
}

function userMessage(messageId: string, n: number): AppEvent {
  return {
    type: 'user_message',
    forkId: null,
    messageId,
    timestamp: ts(n),
    text: 'please continue',
    mentions: [],
    attachments: [],
    mode: 'text',
    synthetic: false,
    taskMode: false,
  }
}

function userMessageReady(messageId: string, n: number): AppEvent {
  return {
    type: 'user_message_ready',
    forkId: null,
    messageId,
    mentionResolutions: [],
    timestamp: ts(n),
  } as any
}

function triggerTags(state: ForkTurnState): readonly TurnTrigger['_tag'][] {
  return state.triggers.map((trigger) => trigger._tag)
}

function expectNoAdvisorRequirement(state: ForkTurnState) {
  expect(hasPendingAdvisorRequirement(state)).toBe(false)
  expect(state.pendingInboundCommunications.some((message) => Option.getOrNull(message.requiresAdvisor) === true)).toBe(false)
}

function expectActiveTurnWithoutAdvisor(state: ForkTurnState, turnId: string) {
  expect(state._tag).toBe('active')
  expect(state._tag === 'active' ? state.turnId : null).toBe(turnId)
  expect(turnRequiresAdvisor(state, turnId)).toBe(false)
  expectNoAdvisorRequirement(state)
}

describe('advisor-required observer escalation while disabled', () => {
  it('ignores escalating observer outcomes before the next root turn', async () => {
    const [afterObserver, active] = await rootSnapshots([
      observerOutcome({ observedTurnId: 'turn-observed', n: 1, escalate: true }),
      turnStarted('turn-1', 2),
    ], [0, 1])

    expect(afterObserver!._tag).toBe('idle')
    expectNoAdvisorRequirement(afterObserver!)
    expect(triggerTags(afterObserver!)).not.toContain('communication')
    expectActiveTurnWithoutAdvisor(active!, 'turn-1')
  })

  it('does not attach escalation to an active turn or the following turn', async () => {
    const [afterEscalation, afterOutcome, nextTurn] = await rootSnapshots([
      turnStarted('turn-1', 1),
      observerOutcome({ observedTurnId: 'turn-0', n: 2, escalate: true }),
      turnOutcome('turn-1', 3),
      turnStarted('turn-2', 4),
    ], [1, 2, 3])

    expectActiveTurnWithoutAdvisor(afterEscalation!, 'turn-1')
    expect(afterOutcome!._tag).toBe('idle')
    expectNoAdvisorRequirement(afterOutcome!)
    expectActiveTurnWithoutAdvisor(nextTurn!, 'turn-2')
  })

  it('does not wake root from waiting_for_user', async () => {
    const [waiting, afterObserver, active] = await rootSnapshots([
      interrupt(1),
      observerOutcome({ observedTurnId: 'turn-0', n: 2, escalate: true }),
      userMessage('message-1', 3),
      userMessageReady('message-1', 4),
      turnStarted('turn-1', 4),
    ], [0, 1, 4])

    expect(waiting!._tag).toBe('waiting_for_user')
    expect(afterObserver!._tag).toBe('waiting_for_user')
    expectNoAdvisorRequirement(afterObserver!)
    expectActiveTurnWithoutAdvisor(active!, 'turn-1')
  })

  it('ignores non-escalating observer outcomes', async () => {
    const [afterObserver, active] = await rootSnapshots([
      observerOutcome({ observedTurnId: 'turn-0', n: 1, escalate: false }),
      turnStarted('turn-1', 2),
    ], [0, 1])

    expectNoAdvisorRequirement(afterObserver!)
    expect(triggerTags(afterObserver!)).not.toContain('communication')
    expectActiveTurnWithoutAdvisor(active!, 'turn-1')
  })

  it('keeps helper predicates false for legacy advisor-required turn state', () => {
    const legacyActive = new TurnActive({
      completedTurns: 0,
      triggers: [],
      pendingInboundCommunications: [],
      parentForkId: null,
      connectionRetryCount: 0,
      turnId: 'turn-1',
      chainId: 'chain-1',
      toolCalls: [],
      triggeredByUser: false,
      requiresAdvisor: true,
    })

    expect(turnRequiresAdvisor(legacyActive, 'turn-1')).toBe(false)
    expect(hasPendingAdvisorRequirement(legacyActive)).toBe(false)
  })
})
