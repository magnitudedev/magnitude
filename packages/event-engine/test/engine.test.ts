import { describe, expect, it } from "vitest"
import { Deferred, Effect, Fiber, Option, Schema } from "effect"
import {
  Address,
  AddressTarget,
  EventEngine,
  ForeignEngineComponent,
  MissingProjectionDependency,
  Projection,
} from "../src"

type CounterEvent =
  | { readonly kind: "counter_opened"; readonly counterId: string }
  | { readonly kind: "counter_incremented"; readonly counterId: string; readonly amount: number }

const Counters = Address.space({ name: "counter", key: Schema.String })
const CounterEngine = EventEngine.define<CounterEvent>({ discriminator: "kind" })

const byCounter = (event: CounterEvent) => AddressTarget.one(Counters.at(event.counterId))

const Counter = CounterEngine.Projection.partitioned({
  name: "Counter",
  by: Counters,
  state: Schema.Number,
  initial: () => 0,
  handlers: (on) => [
    on.event("counter_incremented", { route: byCounter }, ({ event, state }) =>
      Projection.set(state + event.amount)),
  ],
})

const Total = CounterEngine.Projection.singleton({
  name: "Total",
  state: Schema.Number,
  initial: 0,
  reads: [Counter],
  handlers: (on) => [
    on.event("counter_incremented", ({ event, read }) => Effect.gen(function* () {
      const counter = yield* read.at(Counter, Counters.at(event.counterId))
      return Projection.set(Option.getOrElse(counter, () => 0))
    })),
  ],
})

describe("EventEngine projections", () => {
  it("settles projections in dependency order regardless of declaration order", async () => {
    const definition = CounterEngine.make({
      name: "counter-test",
      projections: [Total, Counter],
      workers: [],
    })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* definition.build
      yield* runtime.accept({ kind: "counter_incremented", counterId: "a", amount: 3 })
      yield* runtime.accept({ kind: "counter_incremented", counterId: "a", amount: 2 })

      expect(yield* runtime.read.get(Total)).toBe(5)
      expect(yield* runtime.read.at(Counter, Counters.at("a"))).toEqual(Option.some(5))
      expect(yield* runtime.read.at(Counter, Counters.at("missing"))).toEqual(Option.none())
    })))
  })

  it("supports named refinements without a shared discriminator", async () => {
    type UntaggedEvent = { readonly message: string } | { readonly code: number }
    const Untagged = EventEngine.define<UntaggedEvent>()
    const IsCode = {
      name: "code",
      refinement: (event: UntaggedEvent): event is { readonly code: number } => "code" in event,
    }
    const Codes = Untagged.Projection.singleton({
      name: "Codes",
      state: Schema.Array(Schema.Number),
      initial: [],
      handlers: (on) => [
        on.event(IsCode, ({ event, state }) => Projection.set([...state, event.code])),
      ],
    })
    const definition = Untagged.make({ name: "untagged", projections: [Codes], workers: [] })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* definition.build
      yield* runtime.accept({ message: "ignored" })
      yield* runtime.accept({ code: 7 })
      expect(yield* runtime.read.get(Codes)).toEqual([7])
    })))
  })

  it("rejects missing projection dependencies", () => {
    expect(() => CounterEngine.make({
      name: "invalid",
      projections: [Total],
      workers: [],
    })).toThrow(MissingProjectionDependency)
  })

  it("rejects components from another same-shaped engine family", () => {
    const OtherFamily = EventEngine.define<CounterEvent>({ discriminator: "kind" })
    const Foreign = OtherFamily.Projection.singleton({
      name: "Foreign",
      state: Schema.Number,
      initial: 0,
      handlers: () => [],
    })

    expect(() => CounterEngine.make({
      name: "foreign",
      projections: [Foreign],
      workers: [],
    })).toThrow(ForeignEngineComponent)
  })

  it("does not expose partially settled projection state", async () => {
    const entered = await Effect.runPromise(Deferred.make<void>())
    const release = await Effect.runPromise(Deferred.make<void>())
    const readCompleted = await Effect.runPromise(Deferred.make<void>())
    const Blocking = CounterEngine.Projection.singleton({
      name: "Blocking",
      state: Schema.Number,
      initial: 0,
      handlers: (on) => [
        on.event("counter_incremented", () => Deferred.succeed(entered, undefined).pipe(
          Effect.zipRight(Deferred.await(release)),
          Effect.as(Projection.set(1)),
        )),
      ],
    })
    const definition = CounterEngine.make({ name: "atomic-read", projections: [Blocking], workers: [] })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* definition.build
      const accepting = yield* runtime.accept({ kind: "counter_incremented", counterId: "a", amount: 1 }).pipe(Effect.fork)
      yield* Deferred.await(entered)
      const reading = yield* runtime.read.get(Blocking).pipe(
        Effect.tap(() => Deferred.succeed(readCompleted, undefined)),
        Effect.fork,
      )
      expect(Option.isNone(yield* Deferred.poll(readCompleted))).toBe(true)
      yield* Deferred.succeed(release, undefined)
      yield* Fiber.join(accepting)
      expect(yield* Fiber.join(reading)).toBe(1)
    })))
  })
})
