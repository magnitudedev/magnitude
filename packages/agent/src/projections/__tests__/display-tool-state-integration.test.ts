import { describe, expect, it } from 'bun:test'
import { Effect, Layer } from 'effect'
import {
  ProjectionBusTag,
  makeProjectionBusLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { TurnProjection } from '../turn'
import { AgentRoutingProjection } from '../agent-routing'
import { AgentStatusProjection } from '../agent-status'
import { DisplayProjection } from '../display'
import { ToolStateProjection } from '../tool-state'

const ts = (n: number) => 1_700_300_000_000 + n

const makeRootDisplay = async (events: AppEvent[]) => {
  const projectionBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )

  const runtimeLayer = Layer.mergeAll(
    FrameworkErrorPubSubLive,
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    projectionBusLayer,
    Layer.provide(TurnProjection.Layer, projectionBusLayer),
    Layer.provide(AgentRoutingProjection.Layer, projectionBusLayer),
    Layer.provide(AgentStatusProjection.Layer, projectionBusLayer),
    Layer.provide(ToolStateProjection.Layer, projectionBusLayer),
    Layer.provide(DisplayProjection.Layer, projectionBusLayer),
  )

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* DisplayProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event as any)
    }

    return yield* projection.getFork(null)
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as Effect.Effect<any>)
}

describe('DisplayProjection tool state integration', () => {
  it('renders visible tools from ToolStateProjection state', async () => {
    const display = await makeRootDisplay([
      {
        type: 'turn_started',
        timestamp: ts(1),
        forkId: null,
        turnId: 'turn-1',
        chainId: 'chain-1',
      } as any,
      {
        type: 'tool_event',
        timestamp: ts(2),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'call-1',
        toolKey: 'shell',
        event: { _tag: 'ToolInputStarted' },
      } as any,
      {
        type: 'tool_event',
        timestamp: ts(3),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'call-1',
        toolKey: 'shell',
        event: {
          _tag: 'ToolInputReady',
          input: { command: 'echo hi' },
        },
      } as any,
    ])

    const thinkBlock = display.messages.find((message: any) => message.type === 'think_block')
    expect(thinkBlock?.type).toBe('think_block')
    const toolStep = thinkBlock?.steps.find((step: any) => step.type === 'tool')
    expect(toolStep?.type).toBe('tool')
    expect(toolStep).toMatchObject({
      id: 'call-1',
      toolKey: 'shell',
    })
    expect(toolStep?.state).toBeDefined()
  })

  it('keeps hidden tools out of chat while ToolStateProjection still tracks them', async () => {
    const display = await makeRootDisplay([
      {
        type: 'turn_started',
        timestamp: ts(1),
        forkId: null,
        turnId: 'turn-1',
        chainId: 'chain-1',
      } as any,
      {
        type: 'tool_event',
        timestamp: ts(2),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'spawn-1',
        toolKey: 'spawnWorker',
        event: { _tag: 'ToolInputStarted' },
      } as any,
      {
        type: 'tool_event',
        timestamp: ts(3),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'spawn-1',
        toolKey: 'spawnWorker',
        event: {
          _tag: 'ToolInputReady',
          input: { id: 'task-1', role: 'builder', message: 'do it' },
        },
      } as any,
    ])

    const thinkBlock = display.messages.find((message: any) => message.type === 'think_block')
    const toolSteps = thinkBlock?.type === 'think_block'
      ? thinkBlock.steps.filter((step: any) => step.type === 'tool')
      : []

    expect(toolSteps.length).toBe(0)
  })
})
