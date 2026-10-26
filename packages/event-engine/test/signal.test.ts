import { describe, expect, it } from "vitest"
import { Deferred, Effect, Fiber, Option, Ref, Schema, Stream } from "effect"
import {
  Address,
  AddressTarget,
  CurrentAddressUnavailable,
  DuplicateSignalName,
  EventEngine,
  ForeignEngineSignal,
  MissingSignalPublisher,
  Projection,
  ProjectionDependencyCycle,
  SignalCascadeExceeded,
  Worker as WorkerControl,
} from "../src"

type SignalEvent = {
  readonly type: "value_observed"
  readonly itemId: string
  readonly value: number
}

const Items = Address.space({ name: "signal-item", key: Schema.String })
const Signals = EventEngine.define<SignalEvent>({ discriminator: "type" })
const ValueChanged = Signals.Signal.define<{ readonly itemId: string; readonly value: number }>({
  name: "value-changed",
})
const ValueReady = Signals.Signal.define<{ readonly itemId: string; readonly label: string }>({
  name: "value-ready",
})

const byEventItem = (event: SignalEvent) => AddressTarget.one(Items.at(event.itemId))
const byChangedItem = (signal: { readonly payload: { readonly itemId: string } }) =>
  AddressTarget.one(Items.at(signal.payload.itemId))

describe("EventEngine signals", () => {
  it("cascades projection signals before releasing workers and external observers", async () => {
    const changedSources: unknown[] = []
    const workerReads = await Effect.runPromise(Ref.make<readonly string[]>([]))

    const Source = Signals.Projection.partitioned({
      name: "SignalSource",
      by: Items,
      state: Schema.Number,
      initial: () => 0,
      emits: [ValueChanged],
      handlers: (on) => [
        on.event("value_observed", { route: byEventItem }, ({ event, emit }) => {
          emit(ValueChanged, { itemId: event.itemId, value: event.value })
          return Projection.set(event.value)
        }),
      ],
    })

    const Labels = Signals.Projection.singleton({
      name: "SignalLabels",
      state: Schema.Array(Schema.String),
      initial: [],
      emits: [ValueReady],
      handlers: (on) => [
        on.signal(ValueChanged, ({ signal, state, emit }) => {
          changedSources.push(signal)
          const label = `${signal.payload.itemId}:${signal.payload.value}`
          emit(ValueReady, { itemId: signal.payload.itemId, label })
          return Projection.set([...state, label])
        }),
      ],
    })

    const Ready = Signals.Projection.partitioned({
      name: "SignalReady",
      by: Items,
      state: Schema.String,
      initial: () => "missing",
      handlers: (on) => [
        on.signal(ValueReady, { route: byChangedItem }, ({ signal }) =>
          Projection.set(signal.payload.label)),
      ],
    })

    const Observer = Signals.Worker.define({
      name: "SignalObserver",
      reads: [Ready],
      handlers: (on) => [
        on.signal(ValueReady, ({ signal, read }) => Effect.gen(function* () {
          const ready = yield* read.at(Ready, Items.at(signal.payload.itemId))
          yield* Ref.update(workerReads, (values) => [
            ...values,
            Option.getOrElse(ready, () => "unsettled"),
          ])
        })),
      ],
    })

    const definition = Signals.make({
      name: "signal-cascade",
      projections: [Ready, Labels, Source],
      workers: [Observer],
    })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* definition.build
      yield* runtime.activate()
      const readySignals = yield* runtime.signals.subscribe(ValueReady)
      const observed = yield* readySignals.pipe(
        Stream.runHead,
        Effect.fork,
      )

      yield* runtime.accept({ type: "value_observed", itemId: "a", value: 7 })
      yield* runtime.awaitIdle

      expect(yield* runtime.read.at(Source, Items.at("a"))).toEqual(Option.some(7))
      expect(yield* runtime.read.get(Labels)).toEqual(["a:7"])
      expect(yield* runtime.read.at(Ready, Items.at("a"))).toEqual(Option.some("a:7"))
      expect(yield* Ref.get(workerReads)).toEqual(["a:7"])

      const occurrence = Option.getOrThrow(yield* Fiber.join(observed))
      expect(occurrence.payload).toEqual({ itemId: "a", label: "a:7" })
      expect("timestamp" in occurrence.payload).toBe(false)
      expect(occurrence.source.projection).toBe(Labels)
      expect(occurrence.source.address).toEqual(Option.none())

      const changed = changedSources[0] as {
        readonly source: { readonly projection: unknown; readonly address: Option.Option<unknown> }
      }
      expect(changed.source.projection).toBe(Source)
      expect(changed.source.address).toEqual(Option.some(Items.at("a")))
    })))
  })

  it("reconstructs signal-derived projections while workers are inactive", async () => {
    const workerCalls = await Effect.runPromise(Ref.make(0))
    const Derived = Signals.Projection.singleton({
      name: "ReplayDerived",
      state: Schema.Number,
      initial: 0,
      handlers: (on) => [
        on.signal(ValueChanged, ({ signal }) => Projection.set(signal.payload.value)),
      ],
    })
    const Source = Signals.Projection.partitioned({
      name: "ReplaySource",
      by: Items,
      state: Schema.Number,
      initial: () => 0,
      emits: [ValueChanged],
      handlers: (on) => [
        on.event("value_observed", { route: byEventItem }, ({ event, emit }) => {
          emit(ValueChanged, { itemId: event.itemId, value: event.value })
          return Projection.set(event.value)
        }),
      ],
    })
    const Worker = Signals.Worker.define({
      name: "ReplayWorker",
      handlers: (on) => [
        on.signal(ValueChanged, () => Ref.update(workerCalls, (count) => count + 1)),
      ],
    })
    const definition = Signals.make({
      name: "signal-replay",
      projections: [Derived, Source],
      workers: [Worker],
    })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* definition.build
      yield* runtime.accept({ type: "value_observed", itemId: "a", value: 3 })
      expect(yield* runtime.read.get(Derived)).toBe(3)
      expect(yield* Ref.get(workerCalls)).toBe(0)

      yield* runtime.activate()
      yield* runtime.accept({ type: "value_observed", itemId: "a", value: 4 })
      yield* runtime.awaitIdle
      expect(yield* runtime.read.get(Derived)).toBe(4)
      expect(yield* Ref.get(workerCalls)).toBe(1)
    })))
  })

  it("admits and runs partitioned worker lanes from signal triggers independently", async () => {
    const firstStarted = await Effect.runPromise(Deferred.make<void>())
    const releaseFirst = await Effect.runPromise(Deferred.make<void>())
    const secondCompleted = await Effect.runPromise(Deferred.make<void>())
    const Source = Signals.Projection.partitioned({
      name: "LaneSignalSource",
      by: Items,
      state: Schema.Number,
      initial: () => 0,
      emits: [ValueChanged],
      handlers: (on) => [on.event("value_observed", { route: byEventItem }, ({ event, emit }) => {
        emit(ValueChanged, { itemId: event.itemId, value: event.value })
        return Projection.set(event.value)
      })],
    })
    const Runner = Signals.Worker.define({
      name: "SignalLaneRunner",
      partition: {
        by: Items,
        admission: (on) => [
          on.signal(ValueChanged, { route: byChangedItem }, WorkerControl.admit),
        ],
      },
      handlers: (on) => [
        on.signal(ValueChanged, { route: byChangedItem }, ({ signal }) =>
          signal.payload.itemId === "a"
            ? Deferred.succeed(firstStarted, undefined).pipe(Effect.zipRight(Deferred.await(releaseFirst)))
            : Deferred.succeed(secondCompleted, undefined)),
      ],
    })
    const definition = Signals.make({
      name: "signal-lanes",
      projections: [Source],
      workers: [Runner],
    })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* definition.build
      yield* runtime.activate()
      yield* runtime.accept({ type: "value_observed", itemId: "a", value: 1 })
      yield* Deferred.await(firstStarted)
      yield* runtime.accept({ type: "value_observed", itemId: "b", value: 2 })
      yield* Deferred.await(secondCompleted)
      yield* Deferred.succeed(releaseFirst, undefined)
      yield* runtime.awaitIdle
    })))
  })

  it("rejects missing publishers and combined signal dependency cycles", () => {
    const Missing = Signals.Signal.define<void>({ name: "missing" })
    const Subscriber = Signals.Projection.singleton({
      name: "MissingSubscriber",
      state: Schema.Void,
      initial: undefined,
      handlers: (on) => [on.signal(Missing, () => Projection.keep)],
    })
    expect(() => Signals.make({
      name: "missing-publisher",
      projections: [Subscriber],
      workers: [],
    })).toThrow(MissingSignalPublisher)

    const LeftSignal = Signals.Signal.define<void>({ name: "left" })
    const RightSignal = Signals.Signal.define<void>({ name: "right" })
    const Left = Signals.Projection.singleton({
      name: "SignalCycleLeft",
      state: Schema.Void,
      initial: undefined,
      emits: [LeftSignal],
      handlers: (on) => [on.signal(RightSignal, ({ emit }) => {
        emit(LeftSignal, undefined)
        return Projection.keep
      })],
    })
    const Right = Signals.Projection.singleton({
      name: "SignalCycleRight",
      state: Schema.Void,
      initial: undefined,
      emits: [RightSignal],
      handlers: (on) => [on.signal(LeftSignal, ({ emit }) => {
        emit(RightSignal, undefined)
        return Projection.keep
      })],
    })
    expect(() => Signals.make({
      name: "signal-cycle",
      projections: [Left, Right],
      workers: [],
    })).toThrow(ProjectionDependencyCycle)

    const DuplicateLeft = Signals.Signal.define<void>({ name: "duplicate" })
    const DuplicateRight = Signals.Signal.define<void>({ name: "duplicate" })
    const Duplicates = Signals.Projection.singleton({
      name: "DuplicateSignals",
      state: Schema.Void,
      initial: undefined,
      emits: [DuplicateLeft, DuplicateRight],
      handlers: () => [],
    })
    expect(() => Signals.make({
      name: "duplicate-signals",
      projections: [Duplicates],
      workers: [],
    })).toThrow(DuplicateSignalName)

    const OtherSignals = EventEngine.define<SignalEvent>({ discriminator: "type" })
    const Foreign = OtherSignals.Signal.define<void>({ name: "foreign" })
    const ForeignPublisher = Signals.Projection.singleton({
      name: "ForeignSignalPublisher",
      state: Schema.Void,
      initial: undefined,
      emits: [Foreign],
      handlers: () => [],
    })
    expect(() => Signals.make({
      name: "foreign-signal",
      projections: [ForeignPublisher],
      workers: [],
    })).toThrow(ForeignEngineSignal)
  })

  it("does not release signals buffered by a failed handler", async () => {
    const Leaked = Signals.Signal.define<void>({ name: "leaked" })
    const Failing = Signals.Projection.singleton({
      name: "FailingSignalSource",
      state: Schema.Number,
      initial: 0,
      emits: [Leaked],
      handlers: (on) => [
        on.event("value_observed", ({ emit }) => {
          emit(Leaked, undefined)
          return Effect.fail(new CurrentAddressUnavailable())
        }),
      ],
    })
    const Subscriber = Signals.Projection.singleton({
      name: "FailedSignalSubscriber",
      state: Schema.Number,
      initial: 0,
      handlers: (on) => [on.signal(Leaked, ({ state }) => Projection.set(state + 1))],
    })
    const definition = Signals.make({
      name: "failed-signal",
      projections: [Subscriber, Failing],
      workers: [],
    })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* definition.build
      expect((yield* Effect.either(runtime.accept({ type: "value_observed", itemId: "a", value: 1 })))._tag)
        .toBe("Left")
      expect(yield* runtime.read.get(Subscriber)).toBe(0)
    })))
  })

  it("fails cascade exhaustion explicitly", async () => {
    const Burst = Signals.Signal.define<number>({ name: "burst" })
    const Source = Signals.Projection.singleton({
      name: "BurstSource",
      state: Schema.Void,
      initial: undefined,
      emits: [Burst],
      handlers: (on) => [on.event("value_observed", ({ emit }) => {
        for (let index = 0; index <= 1_000; index += 1) emit(Burst, index)
        return Projection.keep
      })],
    })
    const definition = Signals.make({ name: "burst", projections: [Source], workers: [] })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* definition.build
      const result = yield* Effect.either(runtime.accept({
        type: "value_observed",
        itemId: "a",
        value: 1,
      }))
      expect(result._tag).toBe("Left")
      if (result._tag === "Left") expect(result.left).toBeInstanceOf(SignalCascadeExceeded)
    })))
  })
})
