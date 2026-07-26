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
import type { AppEvent } from '../src/events'
import type { DisplayMessage } from '../src/display'
import type { GenerationPerformance } from '@magnitudedev/ai'
import { DisplayTimelineProjection } from '../src/display'
import { AgentLifecycleProjection } from '../src/projections/agent-lifecycle'
import { AgentRoutingProjection } from '../src/projections/agent-routing'
import { GoalProjection } from '../src/projections/goal'
import { HarnessStateProjection } from '../src/projections/harness-state'
import { TurnProjection } from '../src/projections/turn'
import { UserMessageResolutionProjection } from '../src/projections/user-message-resolution'
import { ToolUniverseSourceLive } from '../src/tools/tool-universe-live'

const ts = (n: number) => 1_700_400_000_000 + n
const InMemoryAddressedEntryStoreLive = Addressed.makeInMemoryAddressedEntryStoreLayer()

const runtimeLayer = Layer.provideMerge(
  Layer.mergeAll(
    GoalProjection.Layer,
    TurnProjection.Layer,
    AgentRoutingProjection.Layer,
    AgentLifecycleProjection.Layer,
    HarnessStateProjection.Layer,
    UserMessageResolutionProjection.Layer,
    Layer.provide(DisplayTimelineProjection.Layer, InMemoryAddressedEntryStoreLive),
  ),
  Layer.merge(
    Layer.provideMerge(
      makeAmbientServiceLayer<AppEvent>(),
      Layer.provideMerge(
        makeProjectionBusLayer<AppEvent>(),
        Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
      ),
    ),
    ToolUniverseSourceLive,
  ),
)

const turnStarted = (
  turnId: string,
  chainId: string,
  timestamp: number,
  forkId: string | null = null,
): AppEvent => ({
  type: 'turn_started',
  timestamp,
  forkId,
  turnId,
  chainId,
} as AppEvent)

const generationStarted = (
  turnId: string,
  chainId: string,
  timestamp: number,
): AppEvent => ({
  type: 'model_generation_started',
  timestamp,
  forkId: null,
  turnId,
  chainId,
} as AppEvent)

const turnOutcome = (
  turnId: string,
  chainId: string,
  timestamp: number,
  toolCallsCount = 0,
  forkId: string | null = null,
  generationPerformance: GenerationPerformance | null = null,
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
      toolCallsCount,
      finishReason: toolCallsCount > 0 ? 'tool_calls' : 'stop',
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
  generationPerformance: generationPerformance
    ? { ...generationPerformance, modelDisplayName: 'Qwen3 Coder' }
    : null,
} as AppEvent)

const userMessage = (messageId: string, timestamp: number, text: string): AppEvent => ({
  type: 'user_message',
  messageId,
  timestamp,
  forkId: null,
  text,
  mentions: [],
  attachments: [],
  mode: 'text',
  synthetic: false,
  taskMode: false,
} as AppEvent)

async function runDisplay(events: readonly AppEvent[]): Promise<readonly DisplayMessage[]> {
  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const timeline = yield* DisplayTimelineProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event as never)
    }

    const root = yield* timeline.getFork(null)
    return yield* timeline.addressed.forFork(null).messages.readAll(root.messages)
  })

  return Effect.runPromise(
    program.pipe(Effect.provide(runtimeLayer)) as unknown as Effect.Effect<readonly DisplayMessage[]>,
  )
}

describe('persistent root work summaries', () => {
  it('records one total duration across continued turns in the same chain', async () => {
    const messages = await runDisplay([
      turnStarted('turn-1', 'chain-1', ts(1_000)),
      generationStarted('turn-1', 'chain-1', ts(1_500)),
      turnOutcome('turn-1', 'chain-1', ts(3_000), 1),
      turnStarted('turn-2', 'chain-1', ts(4_000)),
      generationStarted('turn-2', 'chain-1', ts(5_000)),
      turnOutcome('turn-2', 'chain-1', ts(6_000)),
    ])

    const summaries = messages.filter((message) => message.type === 'work_summary')
    expect(summaries).toEqual([{
      id: 'work_summary:chain-1',
      type: 'work_summary',
      chainId: 'chain-1',
      durationMs: 2_500,
      phase: 'worked',
      performance: Option.none(),
      timestamp: ts(6_000),
    }])
  })

  it('weights native decode throughput across continued turns', async () => {
    const messages = await runDisplay([
      turnStarted('turn-1', 'chain-1', ts(1_000)),
      generationStarted('turn-1', 'chain-1', ts(1_500)),
      turnOutcome('turn-1', 'chain-1', ts(3_000), 1, null, {
        generatedTokens: 10,
        decodeDurationMs: 500,
        decodeTokensPerSecond: 20,
        timeToFirstTokenMs: 80,
      }),
      turnStarted('turn-2', 'chain-1', ts(4_000)),
      generationStarted('turn-2', 'chain-1', ts(5_000)),
      turnOutcome('turn-2', 'chain-1', ts(6_000), 0, null, {
        generatedTokens: 30,
        decodeDurationMs: 1_000,
        decodeTokensPerSecond: 30,
        timeToFirstTokenMs: 120,
      }),
    ])

    const summary = messages.find((message) => message.type === 'work_summary')
    expect(summary?.type).toBe('work_summary')
    if (summary?.type !== 'work_summary') return
    expect(Option.getOrThrow(summary.performance)).toEqual({
      modelDisplayName: 'Qwen3 Coder',
      decodeTokensPerSecond: 40_000 / 1_500,
    })
  })

  it('preserves ICN native throughput unchanged for a single root request', async () => {
    const messages = await runDisplay([
      turnStarted('turn-1', 'chain-1', ts(1_000)),
      generationStarted('turn-1', 'chain-1', ts(1_500)),
      turnOutcome('turn-1', 'chain-1', ts(3_000), 0, null, {
        generatedTokens: 10,
        decodeDurationMs: 400,
        decodeTokensPerSecond: 24.5,
        timeToFirstTokenMs: 72,
      }),
    ])

    const summary = messages.find((message) => message.type === 'work_summary')
    expect(summary?.type).toBe('work_summary')
    if (summary?.type !== 'work_summary') return
    expect(Option.getOrThrow(summary.performance)).toEqual({
      modelDisplayName: 'Qwen3 Coder',
      decodeTokensPerSecond: 24.5,
    })
  })

  it('keeps counting an active worker during root prefill, then pauses when it settles', async () => {
    const messages = await runDisplay([
      turnStarted('turn-1', 'chain-1', ts(0)),
      generationStarted('turn-1', 'chain-1', ts(10)),
      {
        type: 'agent_created',
        timestamp: ts(12),
        forkId: 'worker-1',
        parentForkId: null,
        agentId: 'agent-1',
        name: 'Worker',
        role: 'engineer',
        context: 'context',
        mode: 'spawn',
        taskId: 'task-1',
        message: 'work',
      } as AppEvent,
      turnStarted('worker-turn', 'worker-chain', ts(13), 'worker-1'),
      turnOutcome('turn-1', 'chain-1', ts(20), 1),
      turnStarted('turn-2', 'chain-1', ts(21)),
      turnOutcome('worker-turn', 'worker-chain', ts(35), 0, 'worker-1'),
      generationStarted('turn-2', 'chain-1', ts(40)),
      turnOutcome('turn-2', 'chain-1', ts(50)),
    ])

    const summary = messages.find((message) => message.type === 'work_summary')
    expect(summary).toMatchObject({
      chainId: 'chain-1',
      durationMs: 35,
      phase: 'worked',
    })
  })

  it('places the completed summary before a queued follow-up message', async () => {
    const messages = await runDisplay([
      userMessage('user-1', ts(1), 'first'),
      turnStarted('turn-1', 'chain-1', ts(2)),
      userMessage('user-2', ts(3), 'follow-up'),
      turnOutcome('turn-1', 'chain-1', ts(7)),
    ])

    expect(messages.map((message) => message.type)).toEqual([
      'user_message',
      'work_summary',
      'queued_user_message',
    ])
  })

  it('waits for the last root worker and includes its time in the summary', async () => {
    const messages = await runDisplay([
      turnStarted('root-turn', 'root-chain', ts(1)),
      {
        type: 'agent_created',
        timestamp: ts(2),
        forkId: 'worker-1',
        parentForkId: null,
        agentId: 'agent-1',
        name: 'Worker',
        role: 'engineer',
        context: 'context',
        mode: 'spawn',
        taskId: 'task-1',
        message: 'work',
      } as AppEvent,
      turnStarted('worker-turn', 'worker-chain', ts(3), 'worker-1'),
      generationStarted('root-turn', 'root-chain', ts(3)),
      turnOutcome('root-turn', 'root-chain', ts(4)),
      turnOutcome('worker-turn', 'worker-chain', ts(9), 0, 'worker-1'),
    ])

    const summary = messages.find((message) => message.type === 'work_summary')
    expect(summary).toMatchObject({
      chainId: 'root-chain',
      durationMs: 7,
      phase: 'worked',
      timestamp: ts(9),
    })
    expect(messages.filter((message) => message.type === 'work_summary')).toHaveLength(1)
  })
})
