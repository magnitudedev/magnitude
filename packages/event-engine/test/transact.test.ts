import { describe, expect, it } from "vitest"
import { Context, Data, Deferred, Effect, Option, Ref, Schema, Scope } from "effect"
import { EventEngine, EventStore, Projection, type WorkerInterruption } from "../src"

type CommandEvent =
  | { readonly type: "asked"; readonly id: string }
  | { readonly type: "answered"; readonly id: string }
  | { readonly type: "work"; readonly id: string }
  | { readonly type: "recorded"; readonly line: string }

class StoreDown extends Data.TaggedError("StoreDown")<{}> {}

const Commands = EventEngine.define<CommandEvent, "type", StoreDown>({ discriminator: "type" })

const Asked = Commands.Projection.singleton({
  name: "Asked",
  state: Schema.Array(Schema.String),
  initial: [] as readonly string[],
  handlers: (on) => [
    on.event("asked", ({ event, state }) => Projection.set([...state, `asked:${event.id}`])),
    on.event("answered", ({ event, state }) => Projection.set([...state, `answered:${event.id}`])),
  ],
})

const Log = Commands.Projection.singleton({
  name: "Log",
  state: Schema.Array(Schema.String),
  initial: [] as readonly string[],
  handlers: (on) => [on.event("recorded", ({ event, state }) => Projection.set([...state, event.line]))],
})

const label = (observed: Option.Option<WorkerInterruption>) => Option.match(observed, { onNone: () => "none", onSome: ({ _tag }) => _tag })

// Unpartitioned: the first `work` waits until it is interrupted; every task records what it observed.
const gate = { started: Effect.runSync(Deferred.make<void>()) }
const Worker = Commands.Worker.define({
  name: "Worker",
  handlers: (on) => [
    on.event("work", ({ event, emit, interruption }) => Effect.gen(function* () {
      if (event.id === "blocking") {
        yield* Deferred.succeed(gate.started, undefined)
        yield* Effect.exit(Effect.never)
      }
      const observed = yield* interruption
      yield* emit({ type: "recorded", line: `${event.id}:${label(observed)}` })
    }).pipe(Effect.onInterrupt(() => Effect.flatMap(interruption, (observed) =>
      emit({ type: "recorded", line: `${event.id}:${label(observed)}` }).pipe(Effect.orDie))))),
  ],
})

const testStore = Effect.gen(function* () {
  const memory = yield* EventStore.memory<CommandEvent>()
  const failing = yield* Ref.make(false)
  const commits = yield* Ref.make<readonly (readonly CommandEvent[])[]>([])
  return {
    ...memory,
    failing,
    commits: Ref.get(commits),
    commit: (events: readonly CommandEvent[]) => Effect.flatMap(Ref.get(failing), (fail) => fail
      ? Effect.fail(new StoreDown())
      : Effect.zipRight(Ref.update(commits, (all) => [...all, events]), memory.commit(events))),
  }
})

const Definition = Commands.make({ name: "commands", projections: [Asked, Log], workers: [Worker] })

const run = <A, E>(effect: Effect.Effect<A, E, Scope.Scope>) => Effect.runPromise(Effect.scoped(effect))

// A facade whose mutation is a transaction.
const Asking = Commands.Facade.define({
  name: "Asking",
  mutations: ({ transact }) => ({
    ask: (id: string) => transact(({ read }) => Effect.map(read.get(Asked), (asked) =>
      asked.includes(`asked:${id}`)
        ? { events: [], result: false }
        : { events: [{ type: "asked", id }, { type: "answered", id }] as const, interrupts: [Worker], result: true })),
  }),
  queries: ({ read }) => ({ asked: () => read.get(Asked) }),
})

describe("EventEngine transactions", () => {
  it("runs a facade mutation as a transaction of its runtime", () => run(Effect.gen(function* () {
    const store = yield* testStore
    const runtime = yield* Definition.open({ store })
    const { mutations, queries } = Asking.make({ runtime })
    gate.started = yield* Deferred.make<void>()
    yield* runtime.accept({ type: "work", id: "blocking" })
    yield* Deferred.await(gate.started)
    expect(yield* mutations.ask("q")).toBe(true)
    expect(yield* mutations.ask("q")).toBe(false)
    yield* runtime.awaitIdle
    expect(yield* queries.asked.get()).toEqual(["asked:q", "answered:q"])
    expect(yield* runtime.read.get(Log)).toEqual(["blocking:Interrupted"])
    expect(yield* store.commits).toEqual([
      [{ type: "work", id: "blocking" }],
      [{ type: "asked", id: "q" }, { type: "answered", id: "q" }],
      [{ type: "recorded", line: "blocking:Interrupted" }],
    ])
  })))


  it("decides on settled state and accepts its events in one commit", () => run(Effect.gen(function* () {
    const store = yield* testStore
    const runtime = yield* Definition.open({ store })
    const seen = yield* runtime.transact(({ read }) => Effect.map(read.get(Asked), (asked) => ({
      events: [{ type: "asked", id: "q" }, { type: "answered", id: "q" }] as const,
      result: asked.length,
    })))
    expect(seen).toBe(0)
    expect(yield* runtime.read.get(Asked)).toEqual(["asked:q", "answered:q"])
    expect(yield* store.commits).toEqual([[{ type: "asked", id: "q" }, { type: "answered", id: "q" }]])
  })))

  it("accepts nothing when the body fails or the commit fails, and then applies no interrupt", () => run(Effect.gen(function* () {
    const store = yield* testStore
    const runtime = yield* Definition.open({ store })
    expect(yield* Effect.flip(runtime.transact(() => Effect.fail("refused" as const)))).toBe("refused")
    gate.started = yield* Deferred.make<void>()
    yield* runtime.accept({ type: "work", id: "blocking" })
    yield* Deferred.await(gate.started)
    yield* Ref.set(store.failing, true)
    const failure = yield* Effect.flip(runtime.transact(() => Effect.succeed({
      events: [{ type: "asked", id: "x" }, { type: "answered", id: "x" }],
      interrupts: [Worker],
      result: undefined,
    })))
    expect(failure).toEqual(new StoreDown())
    expect(yield* runtime.read.get(Asked)).toEqual([])
    yield* Ref.set(store.failing, false)
    yield* runtime.interrupt(Worker)
    yield* runtime.awaitIdle
    expect(yield* runtime.read.get(Log)).toEqual(["blocking:Interrupted"])
  })))

  it("interrupts a worker's running and queued work from before it, which still runs, and not the work it releases", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* testStore })
    gate.started = yield* Deferred.make<void>()
    yield* runtime.accept({ type: "work", id: "blocking" })
    yield* Deferred.await(gate.started)
    yield* runtime.accept({ type: "work", id: "queued" })
    // One transaction stops the running work and starts new work.
    yield* runtime.transact(() => Effect.succeed({
      events: [{ type: "work", id: "released" }],
      interrupts: [Worker],
      result: undefined,
    }))
    yield* runtime.awaitIdle
    expect(yield* runtime.read.get(Log)).toEqual(["blocking:Interrupted", "queued:Interrupted", "released:none"])
  })))

  it("keeps the first interruption a queued task receives", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* testStore })
    gate.started = yield* Deferred.make<void>()
    yield* runtime.accept({ type: "work", id: "blocking" })
    yield* Deferred.await(gate.started)
    yield* runtime.accept({ type: "work", id: "queued" })
    yield* runtime.transact(() => Effect.succeed({ events: [], interrupts: [Worker], result: undefined }))
    yield* runtime.close("closed")
    expect(yield* runtime.read.get(Log)).toEqual(["blocking:Interrupted", "queued:Interrupted"])
  })))
})

describe("EventEngine worker requirements", () => {
  class Clock extends Context.Tag("test/Clock")<Clock, { readonly now: string }>() {}
  const Stamper = Commands.Worker.define({
    name: "Stamper",
    handlers: (on) => [
      on.event("asked", ({ emit }) => Effect.flatMap(Clock, ({ now }) => emit({ type: "recorded", line: now }))),
    ],
  })

  it("provides handlers' services where the engine is opened", () => run(Effect.gen(function* () {
    const definition = Commands.make({ name: "stamped", projections: [Log], workers: [Stamper] })
    const runtime = yield* definition.open({ store: yield* testStore }).pipe(Effect.provideService(Clock, { now: "noon" }))
    yield* runtime.accept({ type: "asked", id: "a" })
    yield* runtime.awaitIdle
    expect(yield* runtime.read.get(Log)).toEqual(["noon"])
  })))
})
