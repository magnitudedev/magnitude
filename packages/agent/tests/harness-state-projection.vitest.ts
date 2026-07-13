import { describe, expect, it } from 'vitest'
import { Effect, Layer, Option } from 'effect'
import {
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
  ProjectionBusTag,
  type Timestamped,
  makeAmbientServiceLayer,
  makeProjectionBusLayer,
} from '@magnitudedev/event-core'
import type { ProviderToolCallId, ToolCallId } from '@magnitudedev/ai'
import type { AppEvent } from '../src/events'
import { publishToolkit } from '../src/ambient/toolkit-ambient'
import { HarnessStateProjection, type HarnessTurnState } from '../src/projections/harness-state'
import { shellToolkit, toToolKeyErased } from '../src/tools/toolkits'

const makeHarnessState = async (events: readonly Timestamped<AppEvent>[]): Promise<HarnessTurnState> => {
  const baseBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )
  const baseLayer = Layer.provideMerge(
    makeAmbientServiceLayer<AppEvent>(),
    baseBusLayer,
  )
  const runtimeLayer = Layer.provideMerge(HarnessStateProjection.Layer, baseLayer)

  const program = Effect.gen(function* () {
    yield* publishToolkit(shellToolkit)

    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* HarnessStateProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event)
    }

    return yield* projection.getFork(null)
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as Effect.Effect<HarnessTurnState>)
}

describe('HarnessStateProjection', () => {
  it('steps typed tool state without decoding it as wire JSON', async () => {
    const turnId = 'turn-1'
    const chainId = 'chain-1'
    const toolCallId = 'call-1' as ToolCallId
    const providerToolCallId = 'call-1' as ProviderToolCallId

    const state = await makeHarnessState([
      { type: 'turn_started', timestamp: 1, forkId: null, turnId, chainId },
      {
        type: 'tool_event',
        timestamp: 2,
        forkId: null,
        turnId,
        toolCallId,
        providerToolCallId,
        toolKey: toToolKeyErased('shell'),
        event: {
          _tag: 'ToolInputStarted',
          toolCallId,
          providerToolCallId,
          toolName: 'shell',
          toolKey: toToolKeyErased('shell'),
        },
      },
    ])

    const handle = state.handles.handles.get(toolCallId)
    expect(handle).toBeDefined()
    expect(handle?.toolKey).toBe('shell')
    expect(handle?.state.phase).toBe('streaming')
    expect(handle && Option.isNone(handle.state.errorMessage)).toBe(true)
  })
})
