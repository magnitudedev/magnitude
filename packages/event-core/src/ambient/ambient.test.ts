import { describe, expect, it, vi } from 'vitest'
import { Cause, Deferred, Effect, Exit, Fiber, Option, Schema, Stream } from 'effect'
import * as EventEngine from '../event-engine'
import * as Ambient from './index'
import * as Projection from '../projection'
import * as Signal from '../signal'
import { AmbientServiceTag } from '../core/ambient-service'
import { ProjectionBusTag } from '../core/projection-bus'

type TestEvent =
  | { type: 'set'; value: number }
  | { type: 'bump'; forkId: string | null; amount: number }

const TotalStateSchema = Schema.Struct({ total: Schema.Number })
const SeenStateSchema = Schema.Struct({ seen: Schema.Array(Schema.Number) })
const LatestStateSchema = Schema.Struct({ latest: Schema.NullOr(Schema.Number) })
const ValuesStateSchema = Schema.Struct({ values: Schema.Array(Schema.Number) })
const CountStateSchema = Schema.Struct({ count: Schema.Number })

describe('Ambient primitive', () => {
  it('defines, registers, and reads an ambient value via AmbientService.getValue()', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'TestNumber', initial: 0 })

    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [],
      workers: []
    })

    const client = await TestAgent.createClient()

    try {
      const value = await client.runEffect(
        Effect.gen(function* () {
          const ambients = yield* AmbientServiceTag
          yield* ambients.register(NumberAmbient)
          yield* ambients.update(NumberAmbient, 42)
          return ambients.getValue(NumberAmbient)
        })
      )

      expect(value).toBe(42)
    } finally {
      await client.dispose()
    }
  })

  it('supports sync ambient reads in projection event handlers', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'EventReadNumber', initial: 10 })

    const ReaderProjection = Projection.define<TestEvent>()({
      name: 'Reader',
      state: TotalStateSchema,
      initial: { total: 0 },
      ambients: [NumberAmbient],
      eventHandlers: {
        set: ({ event, state, ambient }) => ({
          total: state.total + event.value + ambient.get(NumberAmbient)
        })
      }
    })

    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [ReaderProjection],
      workers: [],
      expose: {
        state: {
          reader: ReaderProjection
        }
      }
    })

    const client = await TestAgent.createClient()

    try {
      await client.send({ type: 'set', value: 5 })

      expect(await client.state.reader.get()).toEqual({ total: 15 })
    } finally {
      await client.dispose()
    }
  })

  it('triggers ambientHandlers when AmbientService.update() is called', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'ReactiveNumber', initial: 1 })

    const ReactiveProjection = Projection.define<TestEvent>()({
      name: 'Reactive',
      state: SeenStateSchema,
      initial: { seen: [] },
      ambients: [NumberAmbient],
      ambientHandlers: (on) => [
        on(NumberAmbient, ({ value, state }) => ({
          seen: [...state.seen, value]
        }))
      ]
    })

    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [ReactiveProjection],
      workers: [],
      expose: {
        state: {
          reactive: ReactiveProjection
        }
      }
    })

    const client = await TestAgent.createClient()

    try {
      await client.runEffect(
        Effect.flatMap(AmbientServiceTag, (ambients) => ambients.update(NumberAmbient, 2))
      )

      expect(await client.state.reactive.get()).toEqual({ seen: [2] })
    } finally {
      await client.dispose()
    }
  })

  it('processes an ambient update started by another ambient handler', async () => {
    const SourceAmbient = Ambient.define<number>({ name: 'NestedSource', initial: 0 })
    const TargetAmbient = Ambient.define<number>({ name: 'NestedTarget', initial: 0 })

    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [],
      workers: [],
    })

    const client = await TestAgent.createClient()

    try {
      const target = await client.runEffect(Effect.gen(function* () {
        const ambients = yield* AmbientServiceTag
        const bus = yield* ProjectionBusTag<TestEvent>()
        yield* ambients.register(SourceAmbient)
        yield* ambients.register(TargetAmbient)
        yield* bus.registerAmbientHandler(
          SourceAmbient.name,
          (value) => ambients.update(TargetAmbient, (value as number) + 1),
          'NestedAmbientHandler',
        )
        yield* ambients.update(SourceAmbient, 4)
        return ambients.getValue(TargetAmbient)
      }))

      expect(target).toBe(5)
    } finally {
      await client.dispose()
    }
  })

  it('processes an ambient update started by an event handler', async () => {
    const TargetAmbient = Ambient.define<number>({ name: 'EventNestedTarget', initial: 0 })
    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [],
      workers: [],
    })
    const client = await TestAgent.createClient()

    try {
      const target = await client.runEffect(Effect.gen(function* () {
        const ambients = yield* AmbientServiceTag
        const bus = yield* ProjectionBusTag<TestEvent>()
        yield* ambients.register(TargetAmbient)
        yield* bus.register(
          () => ambients.update(TargetAmbient, 7),
          ['set'],
          'EventAmbientHandler',
        )
        yield* bus.processEvent({ type: 'set', value: 1, timestamp: Date.now() })
        return ambients.getValue(TargetAmbient)
      }))

      expect(target).toBe(7)
    } finally {
      await client.dispose()
    }
  })

  it('defects when reading an unregistered ambient', async () => {
    const UnregisteredAmbient = Ambient.define<number>({ name: 'UnregisteredRead', initial: 0 })
    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [],
      workers: [],
    })
    const client = await TestAgent.createClient()

    try {
      const exit = await client.runEffect(Effect.exit(Effect.gen(function* () {
        const ambients = yield* AmbientServiceTag
        return ambients.getValue(UnregisteredAmbient)
      })))
      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) {
        expect(Cause.pretty(exit.cause)).toContain('UnregisteredAmbientDefect')
      }
    } finally {
      await client.dispose()
    }
  })

  it('does not expose a projection read while an ambient transaction is in flight', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'TransactionalNumber', initial: 1 })
    const handlerStarted = await Effect.runPromise(Deferred.make<void>())
    const releaseHandler = await Effect.runPromise(Deferred.make<void>())

    const ReactiveProjection = Projection.define<TestEvent>()({
      name: 'TransactionalReactive',
      state: LatestStateSchema,
      initial: { latest: null },
      ambients: [NumberAmbient],
      ambientHandlers: (on) => [
        on(NumberAmbient, ({ value }) => Effect.gen(function* () {
          yield* Deferred.succeed(handlerStarted, undefined)
          yield* Deferred.await(releaseHandler)
          return { latest: value }
        }))
      ]
    })

    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [ReactiveProjection],
      workers: []
    })

    const client = await TestAgent.createClient()

    try {
      const result = await client.runEffect(Effect.gen(function* () {
        const ambients = yield* AmbientServiceTag
        const projection = yield* ReactiveProjection.Tag
        const update = yield* Effect.fork(ambients.update(NumberAmbient, 2))
        yield* Deferred.await(handlerStarted)

        const read = yield* Effect.fork(projection.get)
        yield* Effect.yieldNow()
        const whileUpdating = yield* Fiber.poll(read)

        yield* Deferred.succeed(releaseHandler, undefined)
        yield* Fiber.join(update)
        const after = yield* Fiber.join(read)
        return { whileUpdating, after }
      }))

      expect(Option.isNone(result.whileUpdating)).toBe(true)
      expect(result.after).toEqual({ latest: 2 })
    } finally {
      await client.dispose()
    }
  })

  it('does not expose a runtime consumer snapshot while an ambient transaction is in flight', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'ConsumerTransactionalNumber', initial: 1 })
    const handlerStarted = await Effect.runPromise(Deferred.make<void>())
    const releaseHandler = await Effect.runPromise(Deferred.make<void>())

    const ReactiveProjection = Projection.define<TestEvent>()({
      name: 'ConsumerTransactionalReactive',
      state: LatestStateSchema,
      initial: { latest: null },
      ambients: [NumberAmbient],
      ambientHandlers: (on) => [
        on(NumberAmbient, ({ value }) => Effect.gen(function* () {
          yield* Deferred.succeed(handlerStarted, undefined)
          yield* Deferred.await(releaseHandler)
          return { latest: value }
        }))
      ]
    })

    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [ReactiveProjection],
      workers: []
    })

    const client = await TestAgent.createClient()

    try {
      const result = await client.runEffect(Effect.scoped(Effect.gen(function* () {
        const ambients = yield* AmbientServiceTag
        const consumer = yield* Projection.consumer.acquire('ambient-transaction-test')
        const update = yield* Effect.fork(ambients.update(NumberAmbient, 2))
        yield* Deferred.await(handlerStarted)

        const read = yield* Effect.fork(
          Projection.consumer.provide(consumer)(
            Projection.consumer.read(ReactiveProjection)
          )
        )
        yield* Effect.yieldNow()
        const whileUpdating = yield* Fiber.poll(read)

        yield* Deferred.succeed(releaseHandler, undefined)
        yield* Fiber.join(update)
        const after = yield* Fiber.join(read)
        return { whileUpdating, after }
      })))

      expect(Option.isNone(result.whileUpdating)).toBe(true)
      expect(result.after.state).toEqual({ latest: 2 })
    } finally {
      await client.dispose()
    }
  })

  it('does not miss an ambient invalidation before the consumer change stream starts', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'ConsumerSubscriptionNumber', initial: 1 })

    const ReactiveProjection = Projection.define<TestEvent>()({
      name: 'ConsumerSubscriptionReactive',
      state: LatestStateSchema,
      initial: { latest: null },
      ambients: [NumberAmbient],
      ambientHandlers: (on) => [
        on(NumberAmbient, ({ value }) => ({ latest: value }))
      ]
    })

    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [ReactiveProjection],
      workers: []
    })

    const client = await TestAgent.createClient()

    try {
      const snapshots = await client.runEffect(Effect.scoped(Effect.gen(function* () {
        const ambients = yield* AmbientServiceTag
        const consumer = yield* Projection.consumer.acquire('ambient-subscription-test')
        const firstDelivered = yield* Deferred.make<void>()
        const releaseFirst = yield* Deferred.make<void>()
        let delivered = 0

        const snapshots = yield* Projection.consumer.stream(consumer)(
          Projection.consumer.read(ReactiveProjection)
        ).pipe(
          Stream.tap(() => Effect.gen(function* () {
            delivered += 1
            if (delivered !== 1) return
            yield* Deferred.succeed(firstDelivered, undefined)
            yield* Deferred.await(releaseFirst)
          })),
          Stream.take(2),
          Stream.runCollect,
          Effect.fork,
        )

        yield* Deferred.await(firstDelivered)
        yield* ambients.update(NumberAmbient, 2)
        yield* Deferred.succeed(releaseFirst, undefined)

        return yield* Fiber.join(snapshots)
      })))

      expect(Array.from(snapshots, ({ state }) => state)).toEqual([
        { latest: null },
        { latest: 2 },
      ])
    } finally {
      await client.dispose()
    }
  })

  it('finishes an admitted ambient transaction when its caller is interrupted', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'InterruptedCallerNumber', initial: 1 })
    const handlerStarted = await Effect.runPromise(Deferred.make<void>())
    const releaseHandler = await Effect.runPromise(Deferred.make<void>())

    const ReactiveProjection = Projection.define<TestEvent>()({
      name: 'InterruptedCallerReactive',
      state: LatestStateSchema,
      initial: { latest: null },
      ambients: [NumberAmbient],
      ambientHandlers: (on) => [
        on(NumberAmbient, ({ value }) => Effect.gen(function* () {
          yield* Deferred.succeed(handlerStarted, undefined)
          yield* Deferred.await(releaseHandler)
          return { latest: value }
        }))
      ]
    })

    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [ReactiveProjection],
      workers: []
    })

    const client = await TestAgent.createClient()

    try {
      const after = await client.runEffect(Effect.gen(function* () {
        const ambients = yield* AmbientServiceTag
        const projection = yield* ReactiveProjection.Tag
        const caller = yield* Effect.fork(ambients.update(NumberAmbient, 2))
        yield* Deferred.await(handlerStarted)
        yield* Fiber.interrupt(caller)
        yield* Deferred.succeed(releaseHandler, undefined)
        return yield* projection.get
      }))

      expect(after).toEqual({ latest: 2 })
    } finally {
      await client.dispose()
    }
  })

  it('supports forked projection ambient reads in event handlers', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'ForkReadNumber', initial: 3 })

    const ForkReaderProjection = Projection.defineForked<Extract<TestEvent, { type: 'bump' }>>()({
      name: 'ForkReader',
      forkState: TotalStateSchema,
      initialFork: { total: 0 },
      ambients: [NumberAmbient],
      eventHandlers: {
        bump: ({ event, fork, ambient }) => ({
          total: fork.total + event.amount + ambient.get(NumberAmbient)
        })
      }
    })

    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [ForkReaderProjection],
      workers: [],
      expose: {
        state: {
          forkReader: ForkReaderProjection
        }
      }
    })

    const client = await TestAgent.createClient()

    try {
      await client.send({ type: 'bump', forkId: 'fork-a', amount: 4 })

      expect(await client.state.forkReader.getFork('fork-a')).toEqual({ total: 7 })
    } finally {
      await client.dispose()
    }
  })

  it('supports forked projection ambientHandlers over full ForkedState', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'ForkReactiveNumber', initial: 1 })

    const ForkReactiveProjection = Projection.defineForked<Extract<TestEvent, { type: 'bump' }>>()({
      name: 'ForkReactive',
      forkState: TotalStateSchema,
      initialFork: { total: 0 },
      ambients: [NumberAmbient],
      eventHandlers: {
        bump: ({ event, fork }) => ({
          total: fork.total + event.amount
        })
      },
      ambientHandlers: (on) => [
        on(NumberAmbient, ({ value, state }) => ({
          forks: new Map(
            [...state.forks.entries()].map(([forkId, forkState]) => [
              forkId,
              { total: forkState.total + value }
            ])
          )
        }))
      ]
    })

    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [ForkReactiveProjection],
      workers: [],
      expose: {
        state: {
          forkReactive: ForkReactiveProjection
        }
      }
    })

    const client = await TestAgent.createClient()

    try {
      await client.send({ type: 'bump', forkId: null, amount: 2 })
      await client.send({ type: 'bump', forkId: 'fork-b', amount: 5 })

      await client.runEffect(
        Effect.flatMap(AmbientServiceTag, (ambients) => ambients.update(NumberAmbient, 10))
      )

      expect(await client.state.forkReactive.getFork(null)).toEqual({ total: 12 })
      expect(await client.state.forkReactive.getFork('fork-b')).toEqual({ total: 15 })
    } finally {
      await client.dispose()
    }
  })

  it('flushes signals emitted from ambientHandlers to other projections', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'SignalAmbient', initial: 0 })

    const SourceProjection = Projection.define<TestEvent>()({
      name: 'Source',
      state: LatestStateSchema,
      initial: { latest: null },
      ambients: [NumberAmbient],
      signals: {
        changed: Signal.create<{ value: number }>('Source/changed')
      },
      ambientHandlers: (on) => [
        on(NumberAmbient, ({ value, state, emit }) => {
          emit.changed({ value })
          return { latest: value ?? state.latest }
        })
      ]
    })

    const ListenerProjection = Projection.define<TestEvent>()({
      name: 'Listener',
      state: ValuesStateSchema,
      initial: { values: [] },
      signalHandlers: (on) => [
        on(SourceProjection.signals.changed, ({ value, state }) => ({
          values: [...state.values, value.value]
        }))
      ]
    })

    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [SourceProjection, ListenerProjection],
      workers: [],
      expose: {
        state: {
          listener: ListenerProjection
        }
      }
    })

    const client = await TestAgent.createClient()

    try {
      await client.runEffect(
        Effect.flatMap(AmbientServiceTag, (ambients) => ambients.update(NumberAmbient, 9))
      )

      expect(await client.state.listener.get()).toEqual({ values: [9] })
    } finally {
      await client.dispose()
    }
  })

  it('does not create events when ambients update', async () => {
    const NumberAmbient = Ambient.define<number>({ name: 'NoEventAmbient', initial: 0 })
    const onEvent = vi.fn()

    const PassiveProjection = Projection.define<TestEvent>()({
      name: 'Passive',
      state: CountStateSchema,
      initial: { count: 0 },
      ambients: [NumberAmbient],
      ambientHandlers: (on) => [
        on(NumberAmbient, ({ state }) => ({ count: state.count + 1 }))
      ]
    })

    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [PassiveProjection],
      workers: []
    })

    const client = await TestAgent.createClient()

    try {
      const unsubscribe = client.onEvent(onEvent)

      await client.runEffect(
        Effect.flatMap(AmbientServiceTag, (ambients) => ambients.update(NumberAmbient, 1))
      )

      expect(onEvent).not.toHaveBeenCalled()

      unsubscribe()
    } finally {
      await client.dispose()
    }
  })

  it('reads the declared initial value without manual registration', async () => {
    const InitialAmbient = Ambient.define<number>({ name: 'InitialAmbient', initial: 7 })

    const InitialProjection = Projection.define<TestEvent>()({
      name: 'Initial',
      state: TotalStateSchema,
      initial: { total: 0 },
      ambients: [InitialAmbient],
      eventHandlers: {
        set: ({ event, ambient }) => ({
          total: event.value + ambient.get(InitialAmbient)
        })
      }
    })

    const TestAgent = EventEngine.make<TestEvent>()({
      name: 'TestAgent',
      schemaVersion: 'test',
      projections: [InitialProjection],
      workers: [],
      expose: {
        state: {
          initial: InitialProjection
        }
      }
    })

    const client = await TestAgent.createClient()

    try {
      await client.send({ type: 'set', value: 1 })

      expect(await client.state.initial.get()).toEqual({ total: 8 })
    } finally {
      await client.dispose()
    }
  })
})
