import { describe, expect, it } from 'vitest'
import { Addressed, EventEngine } from '@magnitudedev/event-core'
import { Duration, Effect, Fiber, Layer, Queue, Stream } from 'effect'
import type { DisplayViewShape } from '@magnitudedev/protocol'
import type { AppEvent } from '../../src/events'
import { AgentRoutingProjection } from '../../src/projections/agent-routing'
import { AgentLifecycleProjection } from '../../src/projections/agent-lifecycle'
import { ChatTitleProjection } from '../../src/projections/chat-title'
import { CompactionProjection } from '../../src/projections/compaction'
import { DetachedProcessProjection } from '../../src/projections/detached-process'
import { GoalProjection } from '../../src/projections/goal'
import { HarnessStateProjection } from '../../src/projections/harness-state'
import { OutboundMessagesProjection } from '../../src/projections/outbound-messages'
import { SessionContextProjection } from '../../src/projections/session-context'
import { TaskGraphProjection } from '../../src/projections/task-graph'
import { TaskAssignmentProjection } from '../../src/projections/task-assignment'
import { TurnProjection } from '../../src/projections/turn'
import { UserMessageResolutionProjection } from '../../src/projections/user-message-resolution'
import { WorkerActivityProjection } from '../../src/projections/worker-activity'
import { DisplayTimelineProjection } from '../../src/display'
import { WindowProjection } from '../../src/window'
import {
  DisplayViewNotFoundError,
  DisplayViewRuntime,
  DisplayViewRuntimeLive,
  defaultDisplayViewShape,
} from '../../src/display-view'
import { makeCountingAddressedEntryStore } from '../helpers/counting-addressed-store'

const TestAgent = EventEngine.make<AppEvent>()({
  name: 'DisplayViewRuntimeTestAgent',
  schemaVersion: 'test',
  projections: [
    SessionContextProjection,
    AgentRoutingProjection,
    AgentLifecycleProjection,
    GoalProjection,
    TaskGraphProjection,
    TurnProjection,
    HarnessStateProjection,
    DetachedProcessProjection,
    WorkerActivityProjection,
    OutboundMessagesProjection,
    UserMessageResolutionProjection,
    TaskAssignmentProjection,
    WindowProjection,
    CompactionProjection,
    ChatTitleProjection,
    DisplayTimelineProjection,
  ],
  workers: [],
})

const listMessages = <M,>(
  m: { readonly byId: { readonly [id: string]: M }; readonly order: readonly string[] },
): readonly M[] => m.order.map((id) => m.byId[id]!)

const rootSmallShape: DisplayViewShape = {
  timelines: {
    root: { kind: 'tail', limit: 25, live: true, presentation: 'default' },
  },
}

const provideRuntime = (storeLayer: Layer.Layer<Addressed.AddressedEntryStore>) =>
  Layer.provideMerge(
    DisplayViewRuntimeLive,
    Layer.provideMerge(TestAgent.EngineLayer, storeLayer)
  )

describe('display view runtime', () => {
  it('materializes a shape without display view app events', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)

    const snapshot = await Effect.runPromise(Effect.gen(function* () {
      const engine = (yield* EventEngine.Service) as EventEngine.Shape<AppEvent, unknown>
      const runtime = yield* DisplayViewRuntime

      yield* engine.send({ type: 'turn_started', forkId: null, turnId: 'turn-1', chainId: 'chain-1' })
      yield* engine.send({ type: 'message_start', forkId: null, turnId: 'turn-1', id: 'msg-1', destination: { kind: 'user' } })
      yield* engine.send({ type: 'message_chunk', forkId: null, turnId: 'turn-1', id: 'msg-1', text: 'hello' })
      yield* engine.send({ type: 'message_end', forkId: null, turnId: 'turn-1', id: 'msg-1' })

      yield* runtime.setShape('view-1', defaultDisplayViewShape)
      return yield* runtime.snapshot('view-1')
    }).pipe(
      Effect.scoped,
      Effect.provide(provideRuntime(Layer.succeed(Addressed.AddressedEntryStore, fixture.store))),
      Effect.orDie
    ))

    expect(listMessages(snapshot.state.timelines.root.messages)).toMatchObject([
      { type: 'assistant_message', content: 'hello' },
    ])
  })

  it('streams updates when tracked addressed timeline content changes', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)

    const snapshots = await Effect.runPromise(Effect.gen(function* () {
      const engine = (yield* EventEngine.Service) as EventEngine.Shape<AppEvent, unknown>
      const runtime = yield* DisplayViewRuntime
      const queue = yield* Queue.unbounded<unknown>()

      yield* runtime.setShape('view-stream', rootSmallShape)
      const fiber = yield* runtime.stream('view-stream').pipe(
        Stream.tap((snapshot) => Queue.offer(queue, snapshot)),
        Stream.runDrain,
        Effect.fork
      )

      const first = yield* Queue.take(queue)
      yield* engine.send({ type: 'turn_started', forkId: null, turnId: 'turn-1', chainId: 'chain-1' })
      yield* engine.send({ type: 'message_start', forkId: null, turnId: 'turn-1', id: 'msg-1', destination: { kind: 'user' } })
      yield* engine.send({ type: 'message_chunk', forkId: null, turnId: 'turn-1', id: 'msg-1', text: 'updated' })
      let second = yield* Queue.take(queue).pipe(
        Effect.timeoutFail({
          duration: Duration.millis(500),
          onTimeout: () => 'display view stream did not update',
        })
      )
      const hasUpdatedMessage = (snapshot: any): boolean =>
        listMessages(snapshot.state.timelines.root.messages).some((message: any) => message.content === 'updated')
      for (let i = 0; i < 5 && !hasUpdatedMessage(second); i += 1) {
        second = yield* Queue.take(queue).pipe(
          Effect.timeoutFail({
            duration: Duration.millis(500),
            onTimeout: () => 'display view stream did not publish addressed content',
          })
        )
      }

      yield* Fiber.interrupt(fiber)
      return [first, second] as const
    }).pipe(
      Effect.scoped,
      Effect.provide(provideRuntime(Layer.succeed(Addressed.AddressedEntryStore, fixture.store))),
      Effect.orDie
    ))

    expect(listMessages((snapshots[0] as any).state.timelines.root.messages)).toEqual([])
    expect(listMessages((snapshots[1] as any).state.timelines.root.messages)).toMatchObject([
      { type: 'assistant_message', content: 'updated' },
    ])
  })

  it('closes a runtime view explicitly', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)

    const result = await Effect.runPromise(Effect.gen(function* () {
      const runtime = yield* DisplayViewRuntime
      yield* runtime.setShape('view-close', defaultDisplayViewShape)
      yield* runtime.close('view-close')
      return yield* runtime.snapshot('view-close').pipe(Effect.either)
    }).pipe(
      Effect.scoped,
      Effect.provide(provideRuntime(Layer.succeed(Addressed.AddressedEntryStore, fixture.store))),
      Effect.orDie
    ))

    expect(result._tag).toBe('Left')
    if (result._tag === 'Left') {
      expect(result.left).toBeInstanceOf(DisplayViewNotFoundError)
    }
  })
})
