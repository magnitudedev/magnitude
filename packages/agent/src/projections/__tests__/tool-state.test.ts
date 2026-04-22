import { describe, expect, it } from 'bun:test'
import { Effect, Layer } from 'effect'
import {
  ProjectionBusTag,
  makeProjectionBusLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { ToolStateProjection } from '../tool-state'

const ts = (n: number) => 1_700_200_000_000 + n

const makeToolState = async (events: AppEvent[], forkId: string | null = null) => {
  const projectionBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )

  const runtimeLayer = Layer.mergeAll(
    FrameworkErrorPubSubLive,
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    projectionBusLayer,
    Layer.provide(ToolStateProjection.Layer, projectionBusLayer),
  )

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* ToolStateProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event as any)
    }

    return yield* projection.getFork(forkId)
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as Effect.Effect<any>)
}

describe('ToolStateProjection', () => {
  it('tracks model-backed hidden spawn_worker state outside DisplayProjection', async () => {
    const state = await makeToolState([
      {
        type: 'tool_event',
        timestamp: ts(1),
        forkId: null,
        turnId: 't-1',
        toolCallId: 'spawn-1',
        toolKey: 'spawnWorker',
        event: { _tag: 'ToolInputStarted' },
      } as any,
      {
        type: 'tool_event',
        timestamp: ts(2),
        forkId: null,
        turnId: 't-1',
        toolCallId: 'spawn-1',
        toolKey: 'spawnWorker',
        event: {
          _tag: 'ToolInputReady',
          input: { id: 'task-1', role: 'builder', message: 'do it' },
        },
      } as any,
      {
        type: 'tool_event',
        timestamp: ts(3),
        forkId: null,
        turnId: 't-1',
        toolCallId: 'spawn-1',
        toolKey: 'spawnWorker',
        event: { _tag: 'ToolExecutionStarted' },
      } as any,
    ])

    const handle = state.toolHandles['spawn-1']
    expect(handle).toBeDefined()
    expect(handle.toolKey).toBe('spawnWorker')
    expect(handle.state).toMatchObject({
      id: 'task-1',
      role: 'builder',
      message: 'do it',
      phase: 'executing',
    })
  })

  it('tracks kill_worker parsed task id via existing tool model', async () => {
    const state = await makeToolState([
      {
        type: 'tool_event',
        timestamp: ts(1),
        forkId: null,
        turnId: 't-1',
        toolCallId: 'kill-1',
        toolKey: 'killWorker',
        event: { _tag: 'ToolInputStarted' },
      } as any,
      {
        type: 'tool_event',
        timestamp: ts(2),
        forkId: null,
        turnId: 't-1',
        toolCallId: 'kill-1',
        toolKey: 'killWorker',
        event: {
          _tag: 'ToolInputReady',
          input: { id: 'task-9' },
        },
      } as any,
    ])

    expect(state.toolHandles['kill-1']?.state).toMatchObject({
      id: 'task-9',
    })
  })
})
