import { describe, expect, it } from 'bun:test'
import { Effect, Layer } from 'effect'
import {
  ProjectionBusTag,
  makeProjectionBusLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { AgentStatusProjection, type AgentStatusState } from '../agent-status'

const ts = (n: number) => 1_700_100_000_000 + n

describe('AgentStatusProjection user kill semantics', () => {
  it('removes agent on subagent_user_killed without requiring agent_killed', async () => {
    const projectionBusLayer = Layer.provideMerge(
      makeProjectionBusLayer<AppEvent>(),
      Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    )

    const runtimeLayer = Layer.mergeAll(
      FrameworkErrorPubSubLive,
      Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
      projectionBusLayer,
      Layer.provide(AgentStatusProjection.Layer, projectionBusLayer),
    )

    const program = Effect.gen(function* () {
      const bus = yield* ProjectionBusTag<AppEvent>()
      const projection = yield* AgentStatusProjection.Tag

      yield* bus.processEvent({
        type: 'agent_created',
        timestamp: ts(1),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        role: 'builder',
        name: 'Builder',
        context: 'ctx',
        mode: 'spawn',
        taskId: 'task-1',
        message: '',
      } as any)

      yield* bus.processEvent({
        type: 'subagent_user_killed',
        timestamp: ts(2),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        source: 'tab_close_confirm',
      } as any)

      return yield* projection.get
    })

    const state = await Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as any) as AgentStatusState
    expect(state.agents.size).toBe(0)
    expect(state.agentByForkId.size).toBe(0)
  })

  it('removes agent on subagent_idle_closed without requiring agent_killed', async () => {
    const projectionBusLayer = Layer.provideMerge(
      makeProjectionBusLayer<AppEvent>(),
      Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    )

    const runtimeLayer = Layer.mergeAll(
      FrameworkErrorPubSubLive,
      Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
      projectionBusLayer,
      Layer.provide(AgentStatusProjection.Layer, projectionBusLayer),
    )

    const program = Effect.gen(function* () {
      const bus = yield* ProjectionBusTag<AppEvent>()
      const projection = yield* AgentStatusProjection.Tag

      yield* bus.processEvent({
        type: 'agent_created',
        timestamp: ts(1),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        role: 'builder',
        name: 'Builder',
        context: 'ctx',
        mode: 'spawn',
        taskId: 'task-1',
        message: '',
      } as any)

      yield* bus.processEvent({
        type: 'subagent_idle_closed',
        timestamp: ts(2),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        source: 'idle_tab_close',
      } as any)

      return yield* projection.get
    })

    const state = await Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as any) as AgentStatusState
    expect(state.agents.size).toBe(0)
    expect(state.agentByForkId.size).toBe(0)
  })
})
