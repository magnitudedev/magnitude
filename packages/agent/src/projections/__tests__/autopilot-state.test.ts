import { describe, expect, it } from 'bun:test'
import { Effect, Layer } from 'effect'
import {
  ProjectionBusTag,
  makeProjectionBusLayer,
  makeAmbientServiceLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { AutopilotStateProjection, type AutopilotState } from '../autopilot-state'
import { UserMessageResolutionProjection } from '../user-message-resolution'

const ts = (n: number) => 1_700_300_000_000 + n

const makeState = async (events: AppEvent[]): Promise<AutopilotState> => {
  const baseBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )
  const baseLayer = Layer.provideMerge(
    makeAmbientServiceLayer<AppEvent>(),
    baseBusLayer,
  )

  // UserMessageResolutionProjection must be provided alongside AutopilotStateProjection
  // since AutopilotStateProjection reads from it via signal.
  const withUserMessageResolution = Layer.provideMerge(UserMessageResolutionProjection.Layer, baseLayer)
  const runtimeLayer = Layer.provideMerge(AutopilotStateProjection.Layer, withUserMessageResolution)

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* AutopilotStateProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event as any)
    }

    return yield* projection.get
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as any) as Promise<AutopilotState>
}

function expectAutopilotDisabled(state: AutopilotState) {
  expect(state.enabled).toBe(false)
  expect(state.pendingContent).toBeNull()
  expect(state.generating).toBe(false)
}

describe('AutopilotStateProjection', () => {
  it('initial state is disabled with no pending content', async () => {
    expectAutopilotDisabled(await makeState([]))
  })

  it('ignores legacy toggle-on events while temporarily disabled', async () => {
    const state = await makeState([
      { type: 'autopilot_toggled', timestamp: ts(1), forkId: null, enabled: true } as AppEvent,
    ])

    expectAutopilotDisabled(state)
  })

  it('ignores legacy generation-started events while temporarily disabled', async () => {
    const state = await makeState([
      { type: 'autopilot_toggled', timestamp: ts(1), forkId: null, enabled: true } as AppEvent,
      { type: 'autopilot_generation_started', timestamp: ts(2), forkId: null } as AppEvent,
    ])

    expectAutopilotDisabled(state)
  })

  it('ignores legacy success outcomes and does not create pending content', async () => {
    const state = await makeState([
      { type: 'autopilot_toggled', timestamp: ts(1), forkId: null, enabled: true } as AppEvent,
      { type: 'autopilot_generation_started', timestamp: ts(2), forkId: null } as AppEvent,
      { type: 'autopilot_outcome', timestamp: ts(3), forkId: null, result: { _tag: 'success', content: 'hello' } } as AppEvent,
    ])

    expectAutopilotDisabled(state)
  })

  it('ignores legacy error outcomes and keeps generation stopped', async () => {
    const state = await makeState([
      { type: 'autopilot_toggled', timestamp: ts(1), forkId: null, enabled: true } as AppEvent,
      { type: 'autopilot_generation_started', timestamp: ts(2), forkId: null } as AppEvent,
      { type: 'autopilot_outcome', timestamp: ts(3), forkId: null, result: { _tag: 'error', message: 'Connection failed' } } as AppEvent,
    ])

    expectAutopilotDisabled(state)
  })

  it('stays disabled across real and synthetic user messages', async () => {
    const state = await makeState([
      { type: 'autopilot_toggled', timestamp: ts(1), forkId: null, enabled: true } as AppEvent,
      { type: 'autopilot_outcome', timestamp: ts(2), forkId: null, result: { _tag: 'success', content: 'hello' } } as AppEvent,
      {
        type: 'user_message',
        timestamp: ts(3),
        forkId: null,
        messageId: 'msg-1',
        content: [{ _tag: 'TextPart', text: 'real message' }],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      } as AppEvent,
      {
        type: 'user_message_ready',
        timestamp: ts(4),
        forkId: null,
        messageId: 'msg-1',
        mentionResolutions: [],
      } as AppEvent,
      {
        type: 'user_message',
        timestamp: ts(5),
        forkId: null,
        messageId: 'auto-msg-1',
        content: [{ _tag: 'TextPart', text: 'synthetic message' }],
        attachments: [],
        mode: 'text',
        synthetic: true,
        taskMode: false,
      } as AppEvent,
      {
        type: 'user_message_ready',
        timestamp: ts(6),
        forkId: null,
        messageId: 'auto-msg-1',
        mentionResolutions: [],
      } as AppEvent,
    ])

    expectAutopilotDisabled(state)
  })
})
