import { describe, expect, test } from 'vitest'
import { Effect, Layer, Option } from 'effect'
import {
  Addressed,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
  ProjectionBusTag,
  makeAmbientServiceLayer,
  makeProjectionBusLayer,
} from '@magnitudedev/event-core'
import type { ProviderToolCallId, ToolCallId } from '@magnitudedev/ai'
import type { AppEvent } from '../src/events'
import { DisplayTimelineProjection } from '../src/display'
import { AgentRoutingProjection } from '../src/projections/agent-routing'
import { AgentLifecycleProjection } from '../src/projections/agent-lifecycle'
import { GoalProjection } from '../src/projections/goal'
import { HarnessStateProjection } from '../src/projections/harness-state'
import { TurnProjection } from '../src/projections/turn'
import { UserMessageResolutionProjection } from '../src/projections/user-message-resolution'
import { toToolKeyErased } from '../src/tools/toolkits'
import type { DisplayMessage } from '../src/display/types'
import { ToolUniverseSourceLive } from '../src/tools/tool-universe-live'

const ts = (n: number) => 1_700_300_000_000 + n
const forkId = null
const turnId = 'turn-1'
const toolCallId = 'tool-1' as ToolCallId
const providerToolCallId = 'tool-1' as ProviderToolCallId
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

const startedShellEvents = (): AppEvent[] => [
  { type: 'turn_started', timestamp: ts(1), forkId, turnId, chainId: 'chain-1' } as AppEvent,
  {
    type: 'tool_event',
    timestamp: ts(2),
    forkId,
    turnId,
    toolCallId,
    providerToolCallId,
    toolKey: toToolKeyErased('shell'),
    event: { _tag: 'ToolInputStarted', toolCallId, providerToolCallId, toolName: 'shell', toolKey: toToolKeyErased('shell') },
  } as AppEvent,
  {
    type: 'tool_event',
    timestamp: ts(3),
    forkId,
    turnId,
    toolCallId,
    providerToolCallId,
    toolKey: toToolKeyErased('shell'),
    event: {
      _tag: 'ToolExecutionStarted',
      toolCallId,
      providerToolCallId,
      toolName: 'shell',
      toolKey: toToolKeyErased('shell'),
      input: { command: 'sleep 5' },
      cached: false,
    },
  } as AppEvent,
]

async function runDisplay(events: readonly AppEvent[]): Promise<readonly DisplayMessage[]> {
  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const timeline = yield* DisplayTimelineProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event as never)
    }

    const fork = yield* timeline.getFork(null)
    return yield* timeline.addressed.forFork(null).messages.readAll(fork.messages)
  })

  return Effect.runPromise(
    program.pipe(Effect.provide(runtimeLayer)) as unknown as Effect.Effect<readonly DisplayMessage[]>,
  )
}

function toolMessage(messages: readonly DisplayMessage[]) {
  const message = messages.find((candidate) => candidate.type === 'tool')
  if (!message || message.type !== 'tool') {
    throw new Error('Expected tool message')
  }
  return message
}

describe('display interrupt finalization', () => {
  test('interrupt event stores interrupted presentation from the typed handle', async () => {
    const messages = await runDisplay([
      ...startedShellEvents(),
      { type: 'interrupt', timestamp: ts(4), forkId } as AppEvent,
    ])

    const message = toolMessage(messages)
    expect(message.presentation._tag).toBe('Some')
    if (Option.isSome(message.presentation)) {
      expect(message.presentation.value).toMatchObject({
        toolKey: toToolKeyErased('shell'),
        phase: 'interrupted',
        command: 'sleep 5',
        running: false,
        failed: true,
      })
    }
  })

  test('cancelled turn finalizes only active tool messages as interrupted', async () => {
    const messages = await runDisplay([
      ...startedShellEvents(),
      {
        type: 'turn_outcome',
        timestamp: ts(4),
        forkId,
        turnId,
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: {
          _tag: 'Cancelled',
          reason: { _tag: 'UserInterrupt' },
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
      } as AppEvent,
    ])

    const message = toolMessage(messages)
    expect(message.presentation._tag).toBe('Some')
    if (Option.isSome(message.presentation)) {
      expect(message.presentation.value.phase).toBe('interrupted')
      expect(message.presentation.value.running).toBe(false)
      expect(message.presentation.value.failed).toBe(true)
    }
  })
})
