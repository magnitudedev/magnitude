import { describe, expect, it } from 'bun:test'
import { Effect, Layer } from 'effect'
import {
  ProjectionBusTag,
  makeProjectionBusLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { AgentStatusProjection } from '../agent-status'
import { MemoryProjection, type ForkMemoryState } from '../memory'

import { SubagentActivityProjection } from '../subagent-activity'
import { CanonicalTurnProjection } from '../canonical-turn'
import { UserPresenceProjection } from '../user-presence'
import { OutboundMessagesProjection } from '../outbound-messages'
import { UserMessageResolutionProjection } from '../user-message-resolution'
import { TaskGraphProjection } from '../task-graph'

const ts = (n: number) => 1_700_100_000_000 + n

describe('MemoryProjection subagent_user_killed awareness', () => {
  it('queues parent system notification for subagent_user_killed and flushes to system_inbox on next turn', async () => {
    const projectionBusLayer = Layer.provideMerge(
      makeProjectionBusLayer<AppEvent>(),
      Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    )

    const runtimeLayer = Layer.mergeAll(
      FrameworkErrorPubSubLive,
      Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
      projectionBusLayer,
      Layer.provide(AgentStatusProjection.Layer, projectionBusLayer),
      Layer.provide(SubagentActivityProjection.Layer, projectionBusLayer),
      Layer.provide(CanonicalTurnProjection.Layer, projectionBusLayer),
      Layer.provide(UserPresenceProjection.Layer, projectionBusLayer),
      Layer.provide(OutboundMessagesProjection.Layer, projectionBusLayer),
      Layer.provide(UserMessageResolutionProjection.Layer, projectionBusLayer),
      Layer.provide(TaskGraphProjection.Layer, projectionBusLayer),
      Layer.provide(MemoryProjection.Layer, projectionBusLayer),
    )

    const program = Effect.gen(function* () {
      const bus = yield* ProjectionBusTag<AppEvent>()
      const projection = yield* MemoryProjection.Tag

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

    const rootFork = await Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as any) as ForkMemoryState
    expect(rootFork).toBeTruthy()

    const inbox = rootFork!.messages.findLast((m: any) => m.type === 'inbox') as any
    expect(inbox).toBeTruthy()

    const userKilled = inbox.timeline.find((e: any) => e.kind === 'subagent_user_killed')
    expect(userKilled).toBeTruthy()
    expect(userKilled.agentId).toBe('agent-sub')
    expect(userKilled.agentType).toBe('builder')
  })

  it('does not queue parent subagent_user_killed notification for subagent_idle_closed', async () => {
    const projectionBusLayer = Layer.provideMerge(
      makeProjectionBusLayer<AppEvent>(),
      Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    )

    const runtimeLayer = Layer.mergeAll(
      FrameworkErrorPubSubLive,
      Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
      projectionBusLayer,
      Layer.provide(AgentStatusProjection.Layer, projectionBusLayer),
      Layer.provide(SubagentActivityProjection.Layer, projectionBusLayer),
      Layer.provide(CanonicalTurnProjection.Layer, projectionBusLayer),
      Layer.provide(UserPresenceProjection.Layer, projectionBusLayer),
      Layer.provide(OutboundMessagesProjection.Layer, projectionBusLayer),
      Layer.provide(UserMessageResolutionProjection.Layer, projectionBusLayer),
      Layer.provide(TaskGraphProjection.Layer, projectionBusLayer),
      Layer.provide(MemoryProjection.Layer, projectionBusLayer),
    )

    const program = Effect.gen(function* () {
      const bus = yield* ProjectionBusTag<AppEvent>()
      const projection = yield* MemoryProjection.Tag

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

    const rootFork = await Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as any) as ForkMemoryState
    expect(rootFork).toBeTruthy()

    const inbox = rootFork!.messages.findLast((m: any) => m.type === 'inbox') as any
    if (!inbox) {
      expect(inbox).toBeFalsy()
      return
    }

    const userKilled = inbox.timeline.find((e: any) => e.kind === 'subagent_user_killed')
    expect(userKilled).toBeUndefined()
  })
})
