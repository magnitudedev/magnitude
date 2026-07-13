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
import { AgentLifecycleProjection } from '../agent-lifecycle'
import { GoalProjection } from '../goal'
import { WindowProjection, type ForkWindowState } from '../../window'

import { WorkerActivityProjection } from '../worker-activity'
import { OutboundMessagesProjection } from '../outbound-messages'
import { UserMessageResolutionProjection } from '../user-message-resolution'
import { TaskGraphProjection } from '../task-graph'
import { HarnessStateProjection } from '../harness-state'
import { TaskAssignmentProjection } from '../task-assignment'

const ts = (n: number) => 1_700_100_000_000 + n

describe('WindowProjection worker_user_killed awareness', () => {
  it('queues coordinator system notification for worker_user_killed and flushes to system_inbox on next turn', async () => {
    const projectionBusLayer = Layer.provideMerge(
      makeProjectionBusLayer<AppEvent>(),
      Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    )
    const baseLayer = Layer.provideMerge(
      makeAmbientServiceLayer<AppEvent>(),
      projectionBusLayer,
    )

    const withAgentLifecycle = Layer.provideMerge(AgentLifecycleProjection.Layer, baseLayer)
    const withGoal = Layer.provideMerge(GoalProjection.Layer, withAgentLifecycle)
    const withWorkerActivity = Layer.provideMerge(WorkerActivityProjection.Layer, withGoal)
    const withOutboundMessages = Layer.provideMerge(OutboundMessagesProjection.Layer, withWorkerActivity)
    const withUserMessageResolution = Layer.provideMerge(UserMessageResolutionProjection.Layer, withOutboundMessages)
    const withHarnessState = Layer.provideMerge(HarnessStateProjection.Layer, withUserMessageResolution)
    const withTaskGraph = Layer.provideMerge(TaskGraphProjection.Layer, withHarnessState)
    const withTaskWorker = Layer.provideMerge(TaskAssignmentProjection.Layer, withTaskGraph)
    const runtimeLayer = Layer.provideMerge(WindowProjection.Layer, withTaskWorker)

    const program = Effect.gen(function* () {
      const bus = yield* ProjectionBusTag<AppEvent>()
      const projection = yield* WindowProjection.Tag

      yield* bus.processEvent({
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any)

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

      yield* bus.processEvent({
        type: 'turn_started',
        timestamp: ts(3),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any)

      return yield* projection.getFork(null)
    })

    const rootFork = await Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as any) as ForkWindowState
    expect(rootFork).toBeTruthy()

    const ctx = rootFork!.messages.findLast((m: any) => m.type === 'context') as any
    expect(ctx).toBeTruthy()

    const userKilled = ctx.timeline.find((e: any) => e.kind === 'worker_user_killed')
    expect(userKilled).toBeTruthy()
    expect(userKilled.agentId).toBe('agent-sub')
    expect(userKilled.agentType).toBe('engineer')
  })

  it('does not queue coordinator worker_user_killed notification for worker_idle_closed', async () => {
    const projectionBusLayer = Layer.provideMerge(
      makeProjectionBusLayer<AppEvent>(),
      Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    )
    const baseLayer = Layer.provideMerge(
      makeAmbientServiceLayer<AppEvent>(),
      projectionBusLayer,
    )

    const withAgentLifecycle = Layer.provideMerge(AgentLifecycleProjection.Layer, baseLayer)
    const withGoal = Layer.provideMerge(GoalProjection.Layer, withAgentLifecycle)
    const withWorkerActivity = Layer.provideMerge(WorkerActivityProjection.Layer, withGoal)
    const withOutboundMessages = Layer.provideMerge(OutboundMessagesProjection.Layer, withWorkerActivity)
    const withUserMessageResolution = Layer.provideMerge(UserMessageResolutionProjection.Layer, withOutboundMessages)
    const withHarnessState = Layer.provideMerge(HarnessStateProjection.Layer, withUserMessageResolution)
    const withTaskGraph = Layer.provideMerge(TaskGraphProjection.Layer, withHarnessState)
    const withTaskWorker = Layer.provideMerge(TaskAssignmentProjection.Layer, withTaskGraph)
    const runtimeLayer = Layer.provideMerge(WindowProjection.Layer, withTaskWorker)

    const program = Effect.gen(function* () {
      const bus = yield* ProjectionBusTag<AppEvent>()
      const projection = yield* WindowProjection.Tag

      yield* bus.processEvent({
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any)

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

      yield* bus.processEvent({
        type: 'turn_started',
        timestamp: ts(3),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any)

      return yield* projection.getFork(null)
    })

    const rootFork = await Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as any) as ForkWindowState
    expect(rootFork).toBeTruthy()

    const ctx = rootFork!.messages.findLast((m: any) => m.type === 'context') as any
    if (!ctx) {
      expect(ctx).toBeFalsy()
      return
    }

    const userKilled = ctx.timeline.find((e: any) => e.kind === 'worker_user_killed')
    expect(userKilled).toBeUndefined()
  })
})
