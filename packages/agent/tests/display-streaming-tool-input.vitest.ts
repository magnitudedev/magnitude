/**
 * Display projection — tool lifecycle event handling.
 *
 * Verifies that tool_event events correctly create and update ToolSteps
 * in the display projection's TurnBlock.
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
import type { ProviderToolCallId, ToolCallId } from '@magnitudedev/ai'
import type { AppEvent } from '../src/events'
import { TurnProjection } from '../src/projections/turn'
import { AgentRoutingProjection } from '../src/projections/agent-routing'
import { AgentLifecycleProjection, type AgentLifecycleState } from '../src/projections/agent-lifecycle'
import { GoalProjection } from '../src/projections/goal'
import { DisplayTimelineProjection } from '../src/display'
import { HarnessStateProjection } from '../src/projections/harness-state'
import { UserMessageResolutionProjection } from '../src/projections/user-message-resolution'
import { toToolKeyErased } from '../src/tools/toolkits'
import type { DisplayTimeline } from '../src/display'

// Materialize timeline messages for assertions — accepts the normalized
// byId/order display form or a plain array (addressed readAll results).
const listMessages = <M,>(
  m: readonly M[] | { readonly byId: { readonly [id: string]: M }; readonly order: readonly string[] },
): readonly M[] => ('order' in m ? m.order.map((id) => m.byId[id]!) : m)


const ts = (n: number) => 1_700_100_000_000 + n
const InMemoryAddressedEntryStoreLive = Addressed.makeInMemoryAddressedEntryStoreLayer()

const makeDisplay = async (events: AppEvent[]): Promise<DisplayTimeline> => {
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
      Layer.provide(DisplayTimelineProjection.Layer, InMemoryAddressedEntryStoreLive),
    ),
    baseLayer,
  )

  const program = Effect.gen(function* () {
    const bus = yield* (ProjectionBusTag<AppEvent>())
    const projection = yield* DisplayTimelineProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event as any)
    }

    const fork = yield* projection.getFork(null)
    const messages = yield* projection.addressed.forFork(null).messages.readAll(fork.messages)
    return { ...fork, messages }
  })

  return Effect.runPromise(
    program.pipe(Effect.provide(runtimeLayer)) as unknown as Effect.Effect<DisplayTimeline>,
  )
}

const makeDisplayAndRootWork = async (events: AppEvent[]): Promise<{
  readonly display: DisplayTimeline
  readonly rootWork: AgentLifecycleState['rootWork']
}> => {
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
      Layer.provide(DisplayTimelineProjection.Layer, InMemoryAddressedEntryStoreLive),
    ),
    baseLayer,
  )

  const program = Effect.gen(function* () {
    const bus = yield* (ProjectionBusTag<AppEvent>())
    const displayProjection = yield* DisplayTimelineProjection.Tag
    const agentLifecycleProjection = yield* AgentLifecycleProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event as any)
    }

    const fork = yield* displayProjection.getFork(null)
    const messages = yield* displayProjection.addressed.forFork(null).messages.readAll(fork.messages)
    const agentState = yield* agentLifecycleProjection.get
    return { display: { ...fork, messages }, rootWork: agentState.rootWork }
  })

  return Effect.runPromise(
    program.pipe(Effect.provide(runtimeLayer)) as unknown as Effect.Effect<{
      readonly display: DisplayTimeline
      readonly rootWork: AgentLifecycleState['rootWork']
    }>,
  )
}

const forkId = null
const turnId = 'turn-1'

describe('Display projection — tool lifecycle events', () => {
  it('creates a ToolStep on ToolInputStarted', async () => {
    const toolCallId = 'tc-1' as ToolCallId
    const providerToolCallId = 'tc-1' as ProviderToolCallId

    const state = await makeDisplay([
      { type: 'turn_started', timestamp: ts(1), forkId, turnId, chainId: 'chain-1' } as AppEvent,
      {
        type: 'tool_event', timestamp: ts(2), forkId, turnId, toolCallId, providerToolCallId, toolKey: toToolKeyErased('shell'),
        event: { _tag: 'ToolInputStarted', toolCallId, providerToolCallId, toolName: 'shell', toolKey: toToolKeyErased('shell') },
      } as AppEvent,
    ])

    const msg = listMessages(state.messages).find(m => m.type === 'tool')
    expect(msg).toBeDefined()
    if (!msg || msg.type !== 'tool') return

    const toolStep = listMessages(state.messages).find(s => s.id === toolCallId && s.type === 'tool')
    expect(toolStep).toBeDefined()
    if (!toolStep || toolStep.type !== 'tool') return
    expect(toolStep.toolKey).toBe('shell')
  })

  it('updates ToolStep state on ToolExecutionEnded', async () => {
    const toolCallId = 'tc-1' as ToolCallId
    const providerToolCallId = 'tc-1' as ProviderToolCallId

    const state = await makeDisplay([
      { type: 'turn_started', timestamp: ts(1), forkId, turnId, chainId: 'chain-1' } as AppEvent,
      {
        type: 'tool_event', timestamp: ts(2), forkId, turnId, toolCallId, providerToolCallId, toolKey: toToolKeyErased('shell'),
        event: { _tag: 'ToolInputStarted', toolCallId, providerToolCallId, toolName: 'shell', toolKey: toToolKeyErased('shell') },
      } as AppEvent,
      {
        type: 'tool_event', timestamp: ts(3), forkId, turnId, toolCallId, providerToolCallId, toolKey: toToolKeyErased('shell'),
        event: { _tag: 'ToolInputReady', toolCallId, providerToolCallId },
      } as AppEvent,
      {
        type: 'tool_event', timestamp: ts(4), forkId, turnId, toolCallId, providerToolCallId, toolKey: toToolKeyErased('shell'),
        event: {
          _tag: 'ToolExecutionStarted', toolCallId, providerToolCallId, toolName: 'shell', toolKey: toToolKeyErased('shell'),
          input: { command: 'ls' }, cached: false,
        },
      } as AppEvent,
      {
        type: 'tool_event', timestamp: ts(5), forkId, turnId, toolCallId, providerToolCallId, toolKey: toToolKeyErased('shell'),
        event: {
          _tag: 'ToolExecutionEnded', toolCallId, providerToolCallId, toolName: 'shell', toolKey: toToolKeyErased('shell'),
          result: { _tag: 'Success', output: 'file1.txt\nfile2.txt' },
        },
      } as AppEvent,
    ])

    const msg = listMessages(state.messages).find(m => m.type === 'tool')
    if (!msg || msg.type !== 'tool') {
      throw new Error('Expected turn block')
    }

    const toolStep = listMessages(state.messages).find(s => s.id === toolCallId && s.type === 'tool')
    expect(toolStep).toBeDefined()
    if (!toolStep || toolStep.type !== 'tool') return
    expect(toolStep.toolKey).toBe('shell')
  })

  it('multiple tool calls tracked independently', async () => {
    const state = await makeDisplay([
      { type: 'turn_started', timestamp: ts(1), forkId, turnId, chainId: 'chain-1' } as AppEvent,
      {
        type: 'tool_event', timestamp: ts(2), forkId, turnId, toolCallId: 'tc-1', providerToolCallId: 'tc-1' as ProviderToolCallId, toolKey: toToolKeyErased('shell'),
        event: { _tag: 'ToolInputStarted', toolCallId: 'tc-1' as ToolCallId, providerToolCallId: 'tc-1' as ProviderToolCallId, toolName: 'shell', toolKey: toToolKeyErased('shell') },
      } as AppEvent,
      {
        type: 'tool_event', timestamp: ts(3), forkId, turnId, toolCallId: 'tc-2', providerToolCallId: 'tc-2' as ProviderToolCallId, toolKey: toToolKeyErased('fileRead'),
        event: { _tag: 'ToolInputStarted', toolCallId: 'tc-2' as ToolCallId, providerToolCallId: 'tc-2' as ProviderToolCallId, toolName: 'read', toolKey: toToolKeyErased('fileRead') },
      } as AppEvent,
    ])

    const msg = listMessages(state.messages).find(m => m.type === 'tool')
    if (!msg || msg.type !== 'tool') throw new Error('Expected turn block')

    const step1 = listMessages(state.messages).find(s => s.id === 'tc-1' && s.type === 'tool')
    const step2 = listMessages(state.messages).find(s => s.id === 'tc-2' && s.type === 'tool')

    expect(step1).toBeDefined()
    if (!step1 || step1.type !== 'tool') return
    expect(step1.toolKey).toBe('shell')
    expect(step2).toBeDefined()
    if (!step2 || step2.type !== 'tool') return
    expect(step2.toolKey).toBe('fileRead')
  })

  it('shows advisor activity as status text while hidden advisor tool runs', async () => {
    const toolCallId = 'advisor-1' as ToolCallId
    const providerToolCallId = 'advisor-1' as ProviderToolCallId

    const active = await makeDisplayAndRootWork([
      { type: 'turn_started', timestamp: ts(1), forkId, turnId, chainId: 'chain-1' } as AppEvent,
      {
        type: 'thinking_chunk', timestamp: ts(2), forkId, turnId, text: 'Thinking before advisor.',
      } as AppEvent,
      {
        type: 'tool_event', timestamp: ts(3), forkId, turnId, toolCallId, providerToolCallId, toolKey: toToolKeyErased('messageAdvisor'),
        event: { _tag: 'ToolInputStarted', toolCallId, providerToolCallId, toolName: 'message_advisor', toolKey: toToolKeyErased('messageAdvisor') },
      } as AppEvent,
    ])

    expect(active.rootWork.activity?.message).toBe('Asking advisor')
    expect(listMessages(active.display.messages).some(m => m.type === 'tool' && m.toolKey === 'messageAdvisor')).toBe(false)

    const completed = await makeDisplayAndRootWork([
      { type: 'turn_started', timestamp: ts(1), forkId, turnId, chainId: 'chain-1' } as AppEvent,
      {
        type: 'tool_event', timestamp: ts(2), forkId, turnId, toolCallId, providerToolCallId, toolKey: toToolKeyErased('messageAdvisor'),
        event: { _tag: 'ToolInputStarted', toolCallId, providerToolCallId, toolName: 'message_advisor', toolKey: toToolKeyErased('messageAdvisor') },
      } as AppEvent,
      {
        type: 'tool_event', timestamp: ts(3), forkId, turnId, toolCallId, providerToolCallId, toolKey: toToolKeyErased('messageAdvisor'),
        event: {
          _tag: 'ToolExecutionEnded', toolCallId, providerToolCallId, toolName: 'message_advisor', toolKey: toToolKeyErased('messageAdvisor'),
          result: { _tag: 'Success', output: 'Keep going.' },
        },
      } as AppEvent,
    ])

    expect(completed.rootWork.activity).toBeNull()
  })
})
