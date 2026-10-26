import { describe, expect, it } from "vitest"
import { Data, Effect, Exit, Ref, Schema } from "effect"
import { Address, AddressTarget, Ambient, EventEngine, EventStore, Projection } from "../src"

type LogEvent =
  | { readonly type: "added"; readonly amount: number }
  | { readonly type: "named"; readonly name: string }
  | { readonly type: "pinged" }

class StoreDown extends Data.TaggedError("StoreDown")<{}> {}

const Log = EventEngine.define<LogEvent, "type", StoreDown>({ discriminator: "type" })

const Total = Log.Projection.singleton({
  name: "Total",
  state: Schema.Number,
  initial: 0,
  handlers: (on) => [on.event("added", ({ event, state }) => Projection.set(state + event.amount))],
})

const Name = Log.Projection.singleton({
  name: "Name",
  state: Schema.String,
  initial: "",
  handlers: (on) => [on.event("named", ({ event }) => Projection.set(event.name))],
})

const Pings = Log.Worker.define({
  name: "Pings",
  reads: [Total],
  handlers: (on) => [on.event("pinged", ({ emit }) => emit({ type: "added", amount: 100 }))],
})

const Definition = Log.make({ name: "log", projections: [Total, Name], workers: [Pings] })

// A memory store whose commits can be made to fail, recording every commit it is handed.
const testStore = Effect.gen(function* () {
  const memory = yield* EventStore.memory<LogEvent>()
  const failing = yield* Ref.make(false)
  const commits = yield* Ref.make<readonly (readonly LogEvent[])[]>([])
  return {
    ...memory,
    failing,
    commits: Ref.get(commits),
    commit: (events: readonly LogEvent[]) => Effect.flatMap(Ref.get(failing), (fail) => fail
      ? Effect.fail(new StoreDown())
      : Effect.zipRight(Ref.update(commits, (all) => [...all, events]), memory.commit(events))),
  }
})

const run = <A, E>(effect: Effect.Effect<A, E, import("effect").Scope.Scope>) => Effect.runPromise(Effect.scoped(effect))

describe("EventEngine persistence", () => {
  it("commits each accepted event", () => run(Effect.gen(function* () {
    const store = yield* testStore
    const runtime = yield* Definition.open({ store })
    yield* runtime.accept({ type: "added", amount: 1 })
    yield* runtime.accept({ type: "named", name: "a" })
    expect(yield* store.commits).toEqual([[{ type: "added", amount: 1 }], [{ type: "named", name: "a" }]])
  })))

  it("commits nothing for an ambient update", () => run(Effect.gen(function* () {
    const Level = Ambient.define<number>({ name: "Level", initial: 0 })
    const Live = Log.Projection.singleton({
      name: "Live",
      state: Schema.Number,
      initial: 0,
      handlers: (on) => [on.ambient(Level, ({ value }) => Projection.set(value))],
    })
    const store = yield* testStore
    const runtime = yield* Log.make({ name: "ambient", projections: [Live, Name], workers: [] }).open({ store })
    yield* runtime.accept({ type: "named", name: "a" })
    yield* runtime.update(Level, 3)
    expect(yield* runtime.read.get(Live)).toBe(3)
    expect(yield* store.commits).toEqual([[{ type: "named", name: "a" }]])
  })))

  it("replays the store on open with workers inactive, then recovers", () => run(Effect.gen(function* () {
    const store = yield* testStore
    yield* store.commit([{ type: "added", amount: 2 }, { type: "pinged" }, { type: "named", name: "b" }])
    const recovered = yield* Ref.make<number>(-1)
    const runtime = yield* Definition.open({
      store,
      recover: ({ read, emit }) => Effect.gen(function* () {
        yield* Ref.set(recovered, yield* read.get(Total))
        // Workers are inactive during recovery: this ping releases nothing.
        yield* emit({ type: "pinged" })
      }),
    })
    yield* runtime.awaitIdle
    expect(yield* Ref.get(recovered)).toBe(2)
    expect(yield* runtime.read.get(Total)).toBe(2)
    expect(yield* runtime.read.get(Name)).toBe("b")
    // Recovery's emit commits; opening commits nothing else.
    expect((yield* store.commits).slice(1)).toEqual([[{ type: "pinged" }]])
    yield* runtime.accept({ type: "pinged" })
    yield* runtime.awaitIdle
    expect(yield* runtime.read.get(Total)).toBe(102)
  })))

  it("releases no signals for replayed events", () => run(Effect.gen(function* () {
    const Changed = Log.Signal.define<number>({ name: "TotalChanged" })
    const Emitting = Log.Projection.singleton({
      name: "Emitting",
      state: Schema.Number,
      initial: 0,
      emits: [Changed],
      handlers: (on) => [on.event("added", ({ event, state, emit }) => {
        emit(Changed, state + event.amount)
        return Projection.set(state + event.amount)
      })],
    })
    const seen = yield* Ref.make<readonly number[]>([])
    const Watcher = Log.Worker.define({
      name: "Watcher",
      handlers: (on) => [on.signal(Changed, ({ signal }) => Ref.update(seen, (all) => [...all, signal.payload]))],
    })
    const store = yield* testStore
    yield* store.commit([{ type: "added", amount: 5 }])
    const runtime = yield* Log.make({ name: "emitting", projections: [Emitting], workers: [Watcher] }).open({ store })
    yield* runtime.awaitIdle
    expect(yield* Ref.get(seen)).toEqual([])
    yield* runtime.accept({ type: "added", amount: 1 })
    yield* runtime.awaitIdle
    expect(yield* runtime.read.get(Emitting)).toBe(6)
    expect(yield* Ref.get(seen)).toEqual([6])
  })))

  it("installs nothing and releases nothing when the commit fails", () => run(Effect.gen(function* () {
    const store = yield* testStore
    const runtime = yield* Definition.open({ store })
    yield* runtime.accept({ type: "added", amount: 1 })
    yield* Ref.set(store.failing, true)
    expect(yield* Effect.flip(runtime.accept({ type: "pinged" }))).toEqual(new StoreDown())
    expect(yield* Effect.flip(runtime.accept({ type: "named", name: "x" }))).toEqual(new StoreDown())
    yield* Ref.set(store.failing, false)
    yield* runtime.awaitIdle
    expect(yield* runtime.read.get(Total)).toBe(1)
    expect(yield* runtime.read.get(Name)).toBe("")
    expect(yield* store.events).toEqual([{ type: "added", amount: 1 }])
  })))

  it("resolves every worker route before committing", () => run(Effect.gen(function* () {
    const Rows = Address.space({ name: "row", key: Schema.String })
    const Others = Address.space({ name: "other", key: Schema.String })
    const Misrouted = Log.Worker.define({
      name: "Misrouted",
      partition: { by: Rows, initial: [Rows.at("r")], admission: () => [] },
      handlers: (on) => [
        // A route into another space, which only an erased route can produce.
        on.event("pinged", { route: () => AddressTarget.one(Others.at("o")) as never }, () => Effect.void),
      ],
    })
    const store = yield* testStore
    const runtime = yield* Log.make({ name: "misrouted", projections: [Total], workers: [Misrouted] }).open({ store })
    const exit = yield* Effect.exit(runtime.accept({ type: "pinged" }))
    expect(Exit.isFailure(exit)).toBe(true)
    expect(yield* store.events).toEqual([])
  })))

  it("restores every projection when a settlement fails partway", () => run(Effect.gen(function* () {
    const Unknown = Log.Projection.singleton({ name: "Unknown", state: Schema.Number, initial: 0, handlers: () => [] })
    const Reading = Log.Projection.singleton({
      name: "Reading",
      state: Schema.Number,
      initial: 0,
      reads: [Total],
      // Reads a projection it did not declare, which fails the settlement after Total has folded.
      handlers: (on) => [on.event("added", ({ read }) => Effect.map(read.get(Unknown), (n) => Projection.set(n)))],
    })
    const store = yield* testStore
    const runtime = yield* Log.make({ name: "partial", projections: [Total, Reading], workers: [] }).open({ store })
    const failure = yield* Effect.flip(runtime.accept({ type: "added", amount: 3 }))
    expect(failure._tag).toBe("UndeclaredProjectionRead")
    expect(yield* runtime.read.get(Total)).toBe(0)
    expect(yield* store.events).toEqual([])
  })))
})
