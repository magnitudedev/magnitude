/**
 * Display projection — toolKey routing & no-duplicate steps.
 *
 * Asserts that:
 *  1. `tool_event ToolInputStarted` adds a ToolStep with the correct toolKey.
 *  2. `tool_event ToolExecutionStarted` does NOT add a duplicate step.
 *  3. The single step's toolKey equals the value carried on the event.
 */

import { describe, it, expect } from 'vitest'
import { Effect, Layer, Schema } from 'effect'
import {
  Addressed,
  ProjectionBusTag,
  makeProjectionBusLayer,
  makeAmbientServiceLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
  type Timestamped,
} from '@magnitudedev/event-core'
import { ProviderToolCallIdSchema, ToolCallIdSchema } from '@magnitudedev/ai'
import type { AppEvent } from '../src/events'
import type { ToolKey } from '../src/tools/toolkits'
import { toToolKeyErased } from '../src/tools/toolkits'
import { TurnProjection } from '../src/projections/turn'
import { AgentRoutingProjection } from '../src/projections/agent-routing'
import { AgentLifecycleProjection } from '../src/projections/agent-lifecycle'
import { GoalProjection } from '../src/projections/goal'
import { DisplayTimelineProjection } from '../src/display'
import { HarnessStateProjection } from '../src/projections/harness-state'
import { UserMessageResolutionProjection } from '../src/projections/user-message-resolution'
import type { DisplayMessage } from '../src/display'
import { ToolUniverseSourceLive } from '../src/tools/tool-universe-live'

// Materialize timeline messages for assertions — accepts the normalized
// byId/order display form or a plain array (addressed readAll results).
const listMessages = <M,>(
  m: readonly M[] | { readonly byId: { readonly [id: string]: M }; readonly order: readonly string[] },
): readonly M[] => ('order' in m ? m.order.map((id) => m.byId[id]!) : m)


const ts = (n: number) => 1_700_200_000_000 + n
const InMemoryAddressedEntryStoreLive = Addressed.makeInMemoryAddressedEntryStoreLayer()
type TimestampedAppEvent = Timestamped<AppEvent>

const runDisplay = async (events: readonly TimestampedAppEvent[]): Promise<readonly DisplayMessage[]> => {
  const frameworkErrorLayer = Layer.mergeAll(
    FrameworkErrorPubSubLive,
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )
  const baseBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    frameworkErrorLayer,
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
    Layer.merge(baseLayer, ToolUniverseSourceLive),
  )
  const program = Effect.gen(function* () {
    const bus = yield* (ProjectionBusTag<AppEvent>())
    const projection = yield* DisplayTimelineProjection.Tag
    for (const event of events) {
      yield* bus.processEvent(event)
    }
    const fork = yield* projection.getFork(null)
    return yield* projection.addressed.forFork(null).messages.readAll(fork.messages)
  })
  return Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)))
}

const turnId = 'turn-1'
const forkId = null
const toolCallId = Schema.decodeUnknownSync(ToolCallIdSchema)('call-1')
const providerToolCallId = Schema.decodeUnknownSync(ProviderToolCallIdSchema)('call-1')

const userMessage = (messageId: string, timestamp: number, text: string): TimestampedAppEvent => ({
  type: 'user_message',
  messageId,
  forkId,
  timestamp,
  text,
  mentions: [],
  attachments: [],
  mode: 'text',
  synthetic: false,
  taskMode: false,
})

const turnStarted = (timestamp: number, id = turnId): TimestampedAppEvent => ({
  type: 'turn_started',
  timestamp,
  forkId,
  turnId: id,
  chainId: 'c1',
})

const toolInputStarted = (
  timestamp: number,
  toolKey: ToolKey,
  toolName: string,
): TimestampedAppEvent => ({
  type: 'tool_event',
  timestamp,
  forkId,
  turnId,
  toolCallId,
  providerToolCallId,
  toolKey: toToolKeyErased(toolKey),
  event: { _tag: 'ToolInputStarted', toolCallId, providerToolCallId, toolName, toolKey: toToolKeyErased(toolKey) },
})

const toolInputReady = (timestamp: number, toolKey: ToolKey): TimestampedAppEvent => ({
  type: 'tool_event',
  timestamp,
  forkId,
  turnId,
  toolCallId,
  providerToolCallId,
  toolKey: toToolKeyErased(toolKey),
  event: { _tag: 'ToolInputReady', toolCallId, providerToolCallId },
})

const toolExecutionStarted = (
  timestamp: number,
  toolKey: ToolKey,
  toolName: string,
): TimestampedAppEvent => ({
  type: 'tool_event',
  timestamp,
  forkId,
  turnId,
  toolCallId,
  providerToolCallId,
  toolKey: toToolKeyErased(toolKey),
  event: {
    _tag: 'ToolExecutionStarted',
    toolCallId,
    providerToolCallId,
    toolName,
    toolKey: toToolKeyErased(toolKey),
    input: { path: '.' },
    cached: false,
  },
})

describe('Display — toolKey routing & no-duplicate steps', () => {
  it('uses the app event message id for user message display identity', async () => {
    const messages = await runDisplay([
      userMessage('client-message-1', ts(1), 'hello'),
    ])

    const msg = listMessages(messages).find(m => m.type === 'user_message')
    expect(msg?.id).toBe('client-message-1')
  })

  it('keeps the app event message id when rendering queued user messages', async () => {
    const messages = await runDisplay([
      turnStarted(ts(1)),
      userMessage('queued-client-message-1', ts(2), 'queued hello'),
    ])

    const msg = listMessages(messages).find(m => m.type === 'queued_user_message')
    expect(msg?.id).toBe('queued-client-message-1')
  })

  it("uses the toolKey carried on tool_event ToolInputStarted", async () => {
    const messages = await runDisplay([
      turnStarted(ts(1)),
      toolInputStarted(ts(2), 'fileTree', 'tree'),
    ])

    const msg = listMessages(messages).find(m => m.type === 'tool')
    if (!msg || msg.type !== 'tool') throw new Error('expected turn block')
    const step = listMessages(messages).find(s => s.id === toolCallId && s.type === 'tool')
    expect(step).toBeDefined()
    if (!step || step.type !== 'tool') return
    expect(step.toolKey).toBe('fileTree')
  })

  it('does NOT add a duplicate step on tool_event ToolExecutionStarted', async () => {
    const messages = await runDisplay([
      turnStarted(ts(1)),
      toolInputStarted(ts(2), 'fileTree', 'tree'),
      toolInputReady(ts(3), 'fileTree'),
      toolExecutionStarted(ts(4), 'fileTree', 'tree'),
    ])

    const msg = listMessages(messages).find(m => m.type === 'tool')
    if (!msg || msg.type !== 'tool') throw new Error('expected turn block')
    const toolSteps = listMessages(messages).filter(s => s.id === toolCallId && s.type === 'tool')
    expect(toolSteps.length).toBe(1)
  })

  it('skips rendering when toolKey is a hidden tool', async () => {
    const messages = await runDisplay([
      turnStarted(ts(1)),
      toolInputStarted(ts(2), 'createTask', 'createTask'),
    ])

    const msg = listMessages(messages).find(m => m.type === 'tool')
    if (!msg || msg.type !== 'tool') {
      // No think block at all is also acceptable
      return
    }
    const step = listMessages(messages).find(s => s.id === toolCallId)
    expect(step).toBeUndefined()
  })
})
