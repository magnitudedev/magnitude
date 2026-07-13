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
import { AgentLifecycleProjection, type AgentLifecycleState } from '../agent-lifecycle'

const ts = (n: number) => 1_700_100_000_000 + n

describe('AgentLifecycleProjection user kill semantics', () => {
  it('removes agent on worker_user_killed without requiring agent_killed', async () => {
    const projectionBusLayer = Layer.provideMerge(
      makeProjectionBusLayer<AppEvent>(),
      Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    )
    const baseLayer = Layer.provideMerge(
      makeAmbientServiceLayer<AppEvent>(),
      projectionBusLayer,
    )

    const runtimeLayer = Layer.provideMerge(AgentLifecycleProjection.Layer, baseLayer)

    const program = Effect.gen(function* () {
      const bus = yield* ProjectionBusTag<AppEvent>()
      const projection = yield* AgentLifecycleProjection.Tag

      yield* bus.processEvent({
        type: 'agent_created',
        timestamp: ts(1),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        role: 'engineer',
        name: 'Builder',
        context: 'ctx',
        mode: 'spawn',
        taskId: 'task-1',
        message: null,
      } as any)

      yield* bus.processEvent({
        type: 'worker_user_killed',
        timestamp: ts(2),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        source: 'tab_close_confirm',
      } as any)

      return yield* projection.get
    })

    const state = await Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as any) as AgentLifecycleState
    expect(state.agents.size).toBe(0)
    expect(state.agentByForkId.size).toBe(0)
  })

  it('removes agent on worker_idle_closed without requiring agent_killed', async () => {
    const projectionBusLayer = Layer.provideMerge(
      makeProjectionBusLayer<AppEvent>(),
      Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    )
    const baseLayer = Layer.provideMerge(
      makeAmbientServiceLayer<AppEvent>(),
      projectionBusLayer,
    )

    const runtimeLayer = Layer.provideMerge(AgentLifecycleProjection.Layer, baseLayer)

    const program = Effect.gen(function* () {
      const bus = yield* ProjectionBusTag<AppEvent>()
      const projection = yield* AgentLifecycleProjection.Tag

      yield* bus.processEvent({
        type: 'agent_created',
        timestamp: ts(1),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        role: 'engineer',
        name: 'Builder',
        context: 'ctx',
        mode: 'spawn',
        taskId: 'task-1',
        message: null,
      } as any)

      yield* bus.processEvent({
        type: 'worker_idle_closed',
        timestamp: ts(2),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        source: 'idle_tab_close',
      } as any)

      return yield* projection.get
    })

    const state = await Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as any) as AgentLifecycleState
    expect(state.agents.size).toBe(0)
    expect(state.agentByForkId.size).toBe(0)
  })
})
