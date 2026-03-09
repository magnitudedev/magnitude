/**
 * Tests for ephemeral event support in EventBusCore.
 *
 * Ephemeral events flow through the EventBus and trigger projections/signals,
 * but are NOT persisted to the EventSink and NOT replayed during hydration.
 */
import { describe, test, expect } from 'bun:test'
import { Effect, Layer, Ref } from 'effect'
import {
  EventBusCoreTag,
  makeEventBusCoreLayer,
} from './event-bus-core'
import { EventSinkTag, makeEventSinkLayer } from './event-sink'
import { makeProjectionBusLayer, ProjectionBusTag } from './projection-bus'
import { HydrationContext } from './hydration-context'
import { InterruptPubSubLive } from './interrupt-pubsub'
import {
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from './framework-error'

// ---------------------------------------------------------------------------
// Test event types
// ---------------------------------------------------------------------------

type TestEvent =
  | { readonly type: 'normal_event'; readonly data: string }
  | { readonly type: 'ephemeral_event'; readonly ephemeral: true; readonly data: string }

// ---------------------------------------------------------------------------
// Layer setup
// ---------------------------------------------------------------------------

const BusTag = EventBusCoreTag<TestEvent>()
const SinkTag = EventSinkTag<TestEvent>()
const ProjBusTag = ProjectionBusTag<TestEvent>()

function makeTestLayers() {
  const frameworkErrorLayer = Layer.provide(
    FrameworkErrorReporterLive,
    FrameworkErrorPubSubLive,
  )

  const coreDeps = Layer.mergeAll(
    HydrationContext.Default,
    makeEventSinkLayer<TestEvent>(),
    InterruptPubSubLive,
    FrameworkErrorPubSubLive,
    frameworkErrorLayer,
  )

  const withProjBus = Layer.provideMerge(
    makeProjectionBusLayer<TestEvent>(),
    coreDeps,
  )

  return Layer.provideMerge(
    makeEventBusCoreLayer<TestEvent>(),
    withProjBus,
  )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('ephemeral events', () => {
  test('normal event is persisted to EventSink', async () => {
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const bus = yield* BusTag
          const sink = yield* SinkTag

          yield* bus.publish({ type: 'normal_event', data: 'hello' })

          const pending = yield* sink.readPending()
          expect(pending).toHaveLength(1)
          expect(pending[0].type).toBe('normal_event')
        }),
      ).pipe(Effect.provide(makeTestLayers())),
    )
  })

  test('ephemeral event is NOT persisted to EventSink', async () => {
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const bus = yield* BusTag
          const sink = yield* SinkTag

          yield* bus.publish({ type: 'ephemeral_event', ephemeral: true, data: 'transient' })

          const pending = yield* sink.readPending()
          expect(pending).toHaveLength(0)
        }),
      ).pipe(Effect.provide(makeTestLayers())),
    )
  })

  test('ephemeral event is processed by projections', async () => {
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const bus = yield* BusTag
          const projBus = yield* ProjBusTag

          const handledRef = yield* Ref.make<string[]>([])
          yield* projBus.register(
            (event) => Ref.update(handledRef, (arr) => [...arr, event.type]),
            ['ephemeral_event', 'normal_event'],
            'TestProjection',
          )

          yield* bus.publish({ type: 'ephemeral_event', ephemeral: true, data: 'transient' })
          yield* bus.publish({ type: 'normal_event', data: 'persistent' })

          const handled = yield* Ref.get(handledRef)
          expect(handled).toEqual(['ephemeral_event', 'normal_event'])
        }),
      ).pipe(Effect.provide(makeTestLayers())),
    )
  })

  test('mix of normal and ephemeral — only normal events persisted', async () => {
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const bus = yield* BusTag
          const sink = yield* SinkTag

          yield* bus.publish({ type: 'normal_event', data: 'first' })
          yield* bus.publish({ type: 'ephemeral_event', ephemeral: true, data: 'skip' })
          yield* bus.publish({ type: 'normal_event', data: 'second' })
          yield* bus.publish({ type: 'ephemeral_event', ephemeral: true, data: 'skip2' })
          yield* bus.publish({ type: 'normal_event', data: 'third' })

          const pending = yield* sink.readPending()
          expect(pending).toHaveLength(3)
          expect(pending.map((e) => (e as { data: string }).data)).toEqual([
            'first',
            'second',
            'third',
          ])
        }),
      ).pipe(Effect.provide(makeTestLayers())),
    )
  })
})
