import { describe, expect, it } from "vitest"
import { Cause, Chunk, Data, Deferred, Effect, Exit, Fiber, Option, Ref, Schema, Scope, Stream } from "effect"
import {
  Address,
  AddressTarget,
  Ambient,
  EngineClosed,
  EventEngine,
  EventStore,
  Projection,
  type WorkerInterruption,
} from "../src"

type E =
  | { readonly type: "added"; readonly amount: number }
  | { readonly type: "set"; readonly value: number }
  | { readonly type: "job"; readonly id: number; readonly block: boolean }
  | { readonly type: "done"; readonly id: number; readonly seen: string }
  | { readonly type: "chain"; readonly lane: string; readonly remaining: number }
  | { readonly type: "named"; readonly name: string }

class StoreDown extends Data.TaggedError("StoreDown")<{}> {}

const F = EventEngine.define<E, "type", StoreDown>({ discriminator: "type" })
const Lanes = Address.space({ name: "lane", key: Schema.String })
const Level = Ambient.define<number>({ name: "Level", initial: 0 })
const Tick = F.Signal.define<number>({ name: "Tick" })

const Total = F.Projection.singleton({
  name: "Total",
  state: Schema.Number,
  initial: 0,
  handlers: (on) => [
    on.event("added", ({ event, state }) => Projection.set(state + event.amount)),
    on.event("set", ({ event }) => Projection.set(event.value)),
  ],
})

const Order = F.Projection.singleton({
  name: "Order",
  state: Schema.Array(Schema.Number),
  initial: [] as readonly number[],
  handlers: (on) => [on.event("added", ({ event, state }) => Projection.set([...state, event.amount]))],
})

const Done = F.Projection.singleton({
  name: "Done",
  state: Schema.Array(Schema.String),
  initial: [] as readonly string[],
  handlers: (on) => [on.event("done", ({ event, state }) => Projection.set([...state, `${event.id}:${event.seen}`]))],
})

const Chains = F.Projection.singleton({
  name: "Chains",
  state: Schema.Number,
  initial: 0,
  emits: [Tick],
  handlers: (on) => [on.event("chain", ({ event, state, emit }) => {
    emit(Tick, event.remaining)
    return Projection.set(state + 1)
  })],
})

const LiveLevel = F.Projection.singleton({
  name: "LiveLevel",
  state: Schema.Number,
  initial: 0,
  handlers: (on) => [on.ambient(Level, ({ value }) => Projection.set(value))],
})

const Name = F.Projection.singleton({
  name: "Name",
  state: Schema.Option(Schema.String),
  initial: Option.none<string>(),
  handlers: (on) => [on.event("named", ({ event }) => Projection.set(Option.some(event.name)))],
})

const label = (o: Option.Option<WorkerInterruption>) => Option.match(o, { onNone: () => "none", onSome: (i) => i._tag })

// Jobs that block wait forever until interrupted; every job records what it observed, exactly once.
const blocking = { started: new Map<number, Deferred.Deferred<void>>() }
const Jobs = F.Worker.define({
  name: "Jobs",
  handlers: (on) => [
    on.event("job", ({ event, emit, interruption }) => Effect.gen(function* () {
      if (event.block) {
        const gate = blocking.started.get(event.id)
        if (gate !== undefined) yield* Deferred.succeed(gate, undefined)
        return yield* Effect.never
      }
      const observed = yield* interruption
      yield* emit({ type: "done", id: event.id, seen: label(observed) })
    }).pipe(Effect.onInterrupt(() => Effect.flatMap(interruption, (observed) =>
      emit({ type: "done", id: event.id, seen: label(observed) }).pipe(Effect.orDie))))),
  ],
})

// A partitioned chain: each signal tick re-emits on the same lane until zero, with a delay.
const Chainer = F.Worker.define({
  name: "Chainer",
  partition: { by: Lanes, initial: [Lanes.at("a"), Lanes.at("b"), Lanes.at("c")], admission: () => [] },
  handlers: (on) => [
    on.event("chain", { route: (event) => AddressTarget.one(Lanes.at(event.lane)) }, ({ event, emit }) =>
      event.remaining === 0
        ? Effect.void
        : Effect.zipRight(Effect.sleep("1 millis"), emit({ type: "chain", lane: event.lane, remaining: event.remaining - 1 }))),
  ],
})

const TickCounter = { count: 0 }
const Ticker = F.Worker.define({
  name: "Ticker",
  handlers: (on) => [on.signal(Tick, ({ update }) => Effect.suspend(() => { TickCounter.count += 1; return update(Level, TickCounter.count) }))],
})

const Definition = F.make({
  name: "adversarial",
  projections: [Total, Order, Done, Chains, LiveLevel, Name],
  workers: [Jobs, Chainer, Ticker],
})

const testStore = Effect.gen(function* () {
  const memory = yield* EventStore.memory<E>()
  const failing = yield* Ref.make(false)
  const commits = yield* Ref.make<readonly (readonly E[])[]>([])
  return {
    ...memory,
    failing,
    commits: Ref.get(commits),
    commit: (events: readonly E[]) => Effect.flatMap(Ref.get(failing), (fail) => fail
      ? Effect.fail(new StoreDown())
      : Effect.zipRight(Ref.update(commits, (all) => [...all, events]), memory.commit(events))),
  }
})

const run = <A, Err>(effect: Effect.Effect<A, Err, Scope.Scope>) => Effect.runPromise(Effect.scoped(effect))
const within = <A, Err, R>(effect: Effect.Effect<A, Err, R>, ms = 2000) =>
  Effect.timeoutFail(effect, { duration: `${ms} millis`, onTimeout: () => "timed out" as const })

describe("adversarial: concurrency", () => {
  it("serializes 200 concurrent accepts: no lost update, store order equals settle order", () => run(Effect.gen(function* () {
    const store = yield* testStore
    const runtime = yield* Definition.open({ store })
    yield* Effect.forEach(Array.from({ length: 200 }, (_, i) => i + 1), (amount) => runtime.accept({ type: "added", amount }), { concurrency: "unbounded", discard: true })
    expect(yield* runtime.read.get(Total)).toBe(200 * 201 / 2)
    const order = yield* runtime.read.get(Order)
    const stored = (yield* store.events).flatMap((event) => event.type === "added" ? [event.amount] : [])
    expect(stored).toEqual(order)
    expect(new Set(order).size).toBe(200)
  })))

  it("read-modify-write transacts from 100 fibers never lose an update", () => run(Effect.gen(function* () {
    const store = yield* testStore
    const runtime = yield* Definition.open({ store })
    const results = yield* Effect.forEach(Array.from({ length: 100 }), () => runtime.transact(({ read }) => Effect.gen(function* () {
      const current = yield* read.get(Total)
      yield* Effect.yieldNow()
      return { events: [{ type: "set", value: current + 1 }] as const, result: current }
    })), { concurrency: "unbounded" })
    expect(yield* runtime.read.get(Total)).toBe(100)
    expect([...results].sort((a, b) => a - b)).toEqual(Array.from({ length: 100 }, (_, i) => i))
    expect((yield* store.events).length).toBe(100)
  })))

  it("mixes accept, transact, update and worker emits from many fibers consistently", () => run(Effect.gen(function* () {
    const store = yield* testStore
    const runtime = yield* Definition.open({ store })
    yield* Effect.all([
      Effect.forEach(Array.from({ length: 50 }), () => runtime.accept({ type: "added", amount: 1 }), { concurrency: "unbounded", discard: true }),
      Effect.forEach(Array.from({ length: 50 }), () => runtime.transact(() => Effect.succeed({ events: [{ type: "added", amount: 1 }, { type: "added", amount: 1 }] as const, result: undefined })), { concurrency: "unbounded", discard: true }),
      Effect.forEach(Array.from({ length: 50 }, (_, i) => i), (i) => runtime.update(Level, i), { concurrency: "unbounded", discard: true }),
      Effect.forEach(["a", "b", "c"], (lane) => runtime.accept({ type: "chain", lane, remaining: 5 }), { concurrency: "unbounded", discard: true }),
      Effect.forEach(Array.from({ length: 30 }, (_, i) => i), (id) => runtime.accept({ type: "job", id, block: false }), { concurrency: "unbounded", discard: true }),
    ], { concurrency: "unbounded" })
    yield* within(runtime.awaitIdle)
    expect(yield* runtime.read.get(Total)).toBe(150)
    expect(yield* runtime.read.get(Chains)).toBe(18)
    expect((yield* runtime.read.get(Done)).length).toBe(30)
    // Transactions commit in one store commit each.
    const commits = yield* store.commits
    expect(commits.filter((c) => c.length === 2).length).toBe(50)
  })))
})

describe("adversarial: interruption races", () => {
  it("every task interrupted at any point observes exactly one outcome, and nothing is lost", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* testStore })
    const n = 200
    yield* Effect.all([
      Effect.forEach(Array.from({ length: n }, (_, i) => i), (id) => runtime.accept({ type: "job", id, block: false }), { discard: true }),
      Effect.forEach(Array.from({ length: n }), () => Effect.zipRight(Effect.yieldNow(), runtime.interrupt(Jobs)), { discard: true }),
    ], { concurrency: "unbounded" })
    yield* within(runtime.awaitIdle)
    const done = yield* runtime.read.get(Done)
    expect(done.length).toBe(n)
    expect(new Set(done.map((line) => line.split(":")[0])).size).toBe(n)
    for (const line of done) expect(["none", "Interrupted"]).toContain(line.split(":")[1])
    // The engine is still open.
    yield* runtime.accept({ type: "added", amount: 1 })
  })))

  it("interrupting a blocking task as it starts still records it once", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* testStore })
    for (let id = 1000; id < 1050; id += 1) {
      yield* Effect.all([
        runtime.accept({ type: "job", id, block: true }),
        Effect.zipRight(Effect.yieldNow(), runtime.interrupt(Jobs)),
      ], { concurrency: "unbounded" })
      // A blocking task marked before it started still blocks; interrupt until it has recorded.
      const recorded = Effect.map(runtime.read.get(Done), (all) => all.some((l) => l.startsWith(`${id}:`)))
      yield* within(Effect.repeat(Effect.zipRight(runtime.interrupt(Jobs), Effect.zipRight(Effect.yieldNow(), recorded)), { until: (r) => r }))
    }
    yield* within(runtime.awaitIdle)
    const done = (yield* runtime.read.get(Done)).filter((line) => Number(line.split(":")[0]) >= 1000)
    expect(done.length).toBe(50)
    expect(new Set(done).size).toBe(50)
  })))
})

describe("adversarial: closing", () => {
  it("close racing transacts and accepts: every call either commits or fails EngineClosed", () => run(Effect.gen(function* () {
    const store = yield* testStore
    const runtime = yield* Definition.open({ store })
    const calls = Array.from({ length: 100 }, (_, i) => i % 2 === 0
      ? runtime.accept({ type: "added", amount: 1 })
      : runtime.transact(() => Effect.succeed({ events: [{ type: "added", amount: 1 }] as const, result: undefined })))
    const [exits] = yield* Effect.all([
      Effect.forEach(calls, (call) => Effect.exit(call), { concurrency: "unbounded" }),
      Effect.zipRight(Effect.yieldNow(), runtime.close("closed")),
    ], { concurrency: "unbounded" })
    const succeeded = exits.filter(Exit.isSuccess).length
    for (const exit of exits) {
      if (Exit.isFailure(exit)) expect(Cause.squash(exit.cause)).toBeInstanceOf(EngineClosed)
    }
    expect((yield* store.events).length).toBe(succeeded)
    expect(yield* runtime.closed).toEqual({ _tag: "Closed" })
  })))

  it("awaitIdle racing close never hangs", () => run(Effect.gen(function* () {
    for (let round = 0; round < 100; round += 1) {
      yield* Effect.scoped(Effect.gen(function* () {
        const runtime = yield* Definition.open({ store: yield* testStore })
        yield* runtime.accept({ type: "chain", lane: "a", remaining: 3 })
        const idle = yield* Effect.fork(runtime.awaitIdle)
        yield* Effect.repeatN(Effect.yieldNow(), round % 5)
        yield* runtime.close("closed")
        const exit = yield* Fiber.await(idle).pipe(within)
        expect(Exit.isSuccess(exit)).toBe(true)
      }))
    }
  })), 20_000)

  it("awaitIdle that took a barrier before close is not left waiting on the ended lane", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* testStore })
    const gate = yield* Deferred.make<void>()
    blocking.started.set(5000, gate)
    yield* runtime.accept({ type: "job", id: 5000, block: true })
    yield* Deferred.await(gate)
    // The barrier is queued behind the blocked task; close interrupts it and ends the lane.
    const idle = yield* Effect.fork(runtime.awaitIdle)
    yield* Effect.sleep("5 millis")
    yield* runtime.close("removed")
    expect(Exit.isSuccess(yield* within(Fiber.await(idle)))).toBe(true)
  })))

  it("an ambient update after close fails typed or succeeds, it does not interrupt the caller", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* testStore })
    yield* runtime.close("closed")
    const exit = yield* Effect.exit(runtime.update(Level, 3))
    expect(Exit.isFailure(exit) && Cause.isInterruptedOnly(exit.cause)).toBe(false)
  })))

  it("a query's changes() started after close ends instead of interrupting", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* testStore })
    const Q = F.Facade.define({ name: "Q", mutations: () => ({}), queries: ({ read }) => ({ total: () => read.get(Total).pipe(Effect.orDie) }) })
    const { queries } = Q.make({ runtime })
    yield* runtime.close("closed")
    const exit = yield* Effect.exit(Stream.runCollect(queries.total.changes()).pipe(within))
    expect(Exit.isFailure(exit) && Cause.isInterruptedOnly(exit.cause)).toBe(false)
  })))

  it("closing the scope ends running handlers and change streams (no leaked fibers)", () => Effect.runPromise(Effect.gen(function* () {
    const scope = yield* Scope.make()
    const runtime = yield* Scope.extend(Definition.open({ store: yield* testStore }), scope)
    const gate = yield* Deferred.make<void>()
    blocking.started.set(6000, gate)
    yield* runtime.accept({ type: "job", id: 6000, block: true })
    yield* Deferred.await(gate)
    const Q = F.Facade.define({ name: "Q2", mutations: () => ({}), queries: ({ read }) => ({ total: () => read.get(Total).pipe(Effect.orDie) }) })
    const watcher = yield* Effect.fork(Stream.runCollect(Q.make({ runtime }).queries.total.changes()))
    const signals = yield* Effect.fork(Stream.runCollect(runtime.signals.stream(Tick)))
    yield* Effect.sleep("5 millis")
    yield* Scope.close(scope, Exit.void)
    expect(Exit.isSuccess(yield* within(Fiber.await(watcher)))).toBe(true)
    expect(Exit.isSuccess(yield* within(Fiber.await(signals)))).toBe(true)
    expect(yield* runtime.closed).toEqual({ _tag: "Closed" })
    const line = (yield* runtime.read.get(Done)).find((l) => l.startsWith("6000:"))
    expect(line).toBe("6000:Closed")
  })))
})

describe("adversarial: awaitIdle with cascades", () => {
  it("waits for event chains on several lanes, their signals, and the ambient updates their workers make", () => run(Effect.gen(function* () {
    TickCounter.count = 0
    const runtime = yield* Definition.open({ store: yield* testStore })
    yield* Effect.forEach(["a", "b", "c"], (lane) => runtime.accept({ type: "chain", lane, remaining: 10 }), { concurrency: "unbounded", discard: true })
    yield* within(runtime.awaitIdle)
    expect(yield* runtime.read.get(Chains)).toBe(33)
    expect(TickCounter.count).toBe(33)
    expect(yield* runtime.read.get(LiveLevel)).toBe(33)
  })))
})

describe("adversarial: open and recover", () => {
  it("fails open when the store read fails mid-replay, without committing anything, and closes cleanly", () => Effect.runPromise(Effect.gen(function* () {
    const memory = yield* testStore
    yield* memory.commit([{ type: "named", name: "x" }, { type: "added", amount: 1 }])
    const commitsBefore = (yield* memory.commits).length
    const store = {
      ...memory,
      read: Stream.concat(Stream.fromIterable<E>([{ type: "named", name: "y" }]), Stream.fail(new StoreDown())),
    }
    const scope = yield* Scope.make()
    const exit = yield* Effect.exit(Scope.extend(Definition.open({ store }), scope))
    expect(Exit.isFailure(exit) && Cause.squash(exit.cause)).toEqual(new StoreDown())
    expect((yield* memory.commits).length).toBe(commitsBefore)
    yield* within(Scope.close(scope, Exit.void))
  })))

  it("recover emits commit, release to no worker, and are visible after open", () => run(Effect.gen(function* () {
    const store = yield* testStore
    yield* store.commit([{ type: "named", name: "a" }])
    const runtime = yield* Definition.open({
      store,
      recover: ({ emit, read }) => Effect.gen(function* () {
        expect(yield* read.get(Name)).toEqual(Option.some("a"))
        yield* emit({ type: "named", name: "b" })
        // Workers are inactive: the job runs nowhere.
        yield* emit({ type: "job", id: 7000, block: false })
      }),
    })
    yield* runtime.awaitIdle
    expect(yield* runtime.read.get(Name)).toEqual(Option.some("b"))
    expect((yield* runtime.read.get(Done)).some((l) => l.startsWith("7000:"))).toBe(false)
    expect((yield* store.commits).slice(1)).toEqual([[{ type: "named", name: "b" }], [{ type: "job", id: 7000, block: false }]])
  })))

  it("a failing recover emit fails open with the store error and leaves the state unchanged", () => Effect.runPromise(Effect.gen(function* () {
    const store = yield* testStore
    yield* store.commit([{ type: "added", amount: 4 }])
    yield* Ref.set(store.failing, true)
    const scope = yield* Scope.make()
    const exit = yield* Effect.exit(Scope.extend(Definition.open({ store, recover: ({ emit }) => emit({ type: "added", amount: 1 }) }), scope))
    expect(Exit.isFailure(exit) && Cause.squash(exit.cause)).toEqual(new StoreDown())
    yield* Scope.close(scope, Exit.void)
    expect(yield* store.events).toEqual([{ type: "added", amount: 4 }])
  })))
})

describe("adversarial: changes()", () => {
  const Q = F.Facade.define({
    name: "Changes",
    mutations: () => ({}),
    queries: ({ read }) => ({
      total: () => read.get(Total).pipe(Effect.orDie),
      parity: () => read.get(Total).pipe(Effect.map((n) => n % 2), Effect.orDie),
    }),
  })

  it("emits no consecutive duplicates (Equal) and ends on the final state after many fast accepts", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* testStore })
    const { queries } = Q.make({ runtime })
    const parity = yield* Effect.fork(Stream.runCollect(queries.parity.changes()))
    const total = yield* Effect.fork(Stream.runCollect(queries.total.changes()))
    yield* Effect.sleep("2 millis")
    yield* Effect.forEach(Array.from({ length: 50 }), () => runtime.accept({ type: "added", amount: 2 }), { discard: true })
    yield* Effect.forEach(Array.from({ length: 10 }), (_, i) => runtime.update(Level, i), { discard: true })
    yield* runtime.accept({ type: "added", amount: 1 })
    yield* Effect.sleep("10 millis")
    yield* runtime.close("closed")
    const p = Chunk.toReadonlyArray(yield* Fiber.join(parity))
    const t = Chunk.toReadonlyArray(yield* Fiber.join(total))
    expect(p).toEqual([0, 1])
    expect(t.at(-1)).toBe(101)
    for (let i = 1; i < t.length; i += 1) expect(t[i]).not.toBe(t[i - 1])
  })))

  it("a slow subscriber still sees the final state when the engine closes right after the last accept", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* testStore })
    const { queries } = Q.make({ runtime })
    const first = yield* Deferred.make<void>()
    const total = yield* Effect.fork(Stream.runCollect(queries.total.changes().pipe(
      Stream.tap(() => Effect.zipRight(Deferred.succeed(first, undefined), Effect.sleep("20 millis"))),
    )))
    yield* Deferred.await(first)
    yield* runtime.accept({ type: "added", amount: 7 })
    yield* runtime.close("closed")
    const t = Chunk.toReadonlyArray(yield* within(Fiber.join(total)))
    expect(t.at(-1)).toBe(7)
  })))
})

describe("adversarial: query consistency", () => {
  // Two projections that fold the same events, so in any settled state they are equal.
  const A = F.Projection.singleton({ name: "A", state: Schema.Number, initial: 0, handlers: (on) => [on.event("added", ({ state }) => Projection.set(state + 1))] })
  const B = F.Projection.singleton({ name: "B", state: Schema.Number, initial: 0, handlers: (on) => [on.event("added", ({ state }) => Projection.set(state + 1))] })
  const Pair = F.Facade.define({
    name: "Pair",
    mutations: () => ({}),
    queries: ({ read }) => ({
      pair: () => Effect.all([read.get(A), read.get(B)]).pipe(Effect.orDie),
      // The same reads with a yield between them, as the scheduler's op budget may insert anywhere.
      yielding: () => Effect.all([read.get(A), Effect.yieldNow(), read.get(B)]).pipe(Effect.map(([a, , b]) => [a, b] as const), Effect.orDie),
    }),
  })

  it("a query that reads several projections reads one settled state, never a torn one", () => run(Effect.gen(function* () {
    const runtime = yield* F.make({ name: "pair", projections: [A, B], workers: [] }).open({ store: yield* testStore })
    const { queries } = Pair.make({ runtime })
    const torn = yield* Ref.make(0)
    yield* Effect.all([
      Effect.forEach(Array.from({ length: 300 }), () => runtime.accept({ type: "added", amount: 1 }), { concurrency: 4, discard: true }),
      Effect.forEach(Array.from({ length: 300 }), () => Effect.flatMap(queries.pair.get(), ([a, b]) => a === b ? Effect.void : Ref.update(torn, (n) => n + 1)), { concurrency: 4, discard: true }),
    ], { concurrency: "unbounded" })
    expect(yield* Ref.get(torn)).toBe(0)
  })))

  it("a query whose reads are separated by a yield still reads one settled state", () => run(Effect.gen(function* () {
    const runtime = yield* F.make({ name: "pair-yield", projections: [A, B], workers: [] }).open({ store: yield* testStore })
    const { queries } = Pair.make({ runtime })
    const torn = yield* Ref.make(0)
    yield* Effect.all([
      Effect.forEach(Array.from({ length: 300 }), () => runtime.accept({ type: "added", amount: 1 }), { concurrency: 4, discard: true }),
      Effect.forEach(Array.from({ length: 300 }), () => Effect.flatMap(queries.yielding.get(), ([a, b]) => a === b ? Effect.void : Ref.update(torn, (n) => n + 1)), { concurrency: 4, discard: true }),
    ], { concurrency: "unbounded" })
    expect(yield* Ref.get(torn)).toBe(0)
  })))
})

describe("adversarial: calls made while holding the acceptance lock", () => {
  it("a transaction body reads through the runtime's reader directly, and an accept in it dies instead of waiting", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* testStore })
    yield* runtime.accept({ type: "added", amount: 2 })
    const read = yield* within(runtime.transact(() => Effect.map(runtime.read.get(Total), (total) => ({ events: [], result: total }))))
    expect(read).toBe(2)
    const accepted = yield* Effect.exit(within(runtime.transact(() => Effect.as(runtime.accept({ type: "added", amount: 1 }), { events: [], result: undefined }))))
    expect(Exit.isFailure(accepted) && Cause.isDie(accepted.cause)).toBe(true)
    expect(yield* runtime.read.get(Total)).toBe(2)
  })))

  it("a mutation's emit, interrupt and transact die inside its own transaction, and a query reads directly", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* testStore })
    const M = F.Facade.define({
      name: "Reentrant",
      mutations: ({ emit, interrupt, transact }) => ({
        emit: () => transact(() => Effect.as(emit({ type: "added", amount: 1 }), { events: [], result: undefined })),
        interrupt: () => transact(() => Effect.as(interrupt(Jobs), { events: [], result: undefined })),
        nested: () => transact(() => Effect.as(transact(() => Effect.succeed({ events: [], result: undefined })), { events: [], result: undefined })),
      }),
      queries: ({ read }) => ({ total: () => read.get(Total).pipe(Effect.orDie) }),
    })
    const { mutations, queries } = M.make({ runtime })
    for (const mutation of [mutations.emit, mutations.interrupt, mutations.nested]) {
      const exit = yield* Effect.exit(within(mutation()))
      expect(Exit.isFailure(exit) && Cause.isDie(exit.cause)).toBe(true)
    }
    expect(yield* within(runtime.transact(() => Effect.map(queries.total.get(), (total) => ({ events: [], result: total }))))).toBe(0)
  })))
})

describe("adversarial: closing, continued", () => {
  it("rejects a public ambient update with EngineClosed once closing has begun", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* testStore })
    yield* runtime.close("closed")
    expect(yield* Effect.flip(runtime.update(Level, 3))).toBeInstanceOf(EngineClosed)
  })))

  it("a handler failing while the engine's scope closes closes it as Failed", () => Effect.runPromise(Effect.gen(function* () {
    const Failing = F.Worker.define({
      name: "FailsOnClose",
      handlers: (on) => [on.event("job", () => Effect.never.pipe(Effect.onInterrupt(() => Effect.die("failed on close"))))],
    })
    const scope = yield* Scope.make()
    const runtime = yield* Scope.extend(F.make({ name: "fails-on-close", projections: [Total], workers: [Failing] }).open({ store: yield* testStore }), scope)
    yield* runtime.accept({ type: "job", id: 1, block: true })
    yield* Effect.sleep("5 millis")
    yield* Scope.close(scope, Exit.void)
    expect((yield* within(runtime.closed))._tag).toBe("Failed")
  })))

  it("a transaction interrupted during its commit keeps both the commit and its interrupts", () => run(Effect.gen(function* () {
    const memory = yield* testStore
    const committing = yield* Deferred.make<void>()
    const gate = yield* Deferred.make<void>()
    const gated = yield* Ref.make(false)
    const store = {
      ...memory,
      commit: (events: readonly E[]) => Effect.flatMap(Ref.get(gated), (wait) => wait
        ? Effect.zipRight(Effect.zipRight(Deferred.succeed(committing, undefined), Deferred.await(gate)), memory.commit(events))
        : memory.commit(events)),
    }
    const runtime = yield* Definition.open({ store })
    const started = yield* Deferred.make<void>()
    blocking.started.set(8000, started)
    yield* runtime.accept({ type: "job", id: 8000, block: true })
    yield* Deferred.await(started)
    yield* Ref.set(gated, true)
    const transaction = yield* Effect.fork(runtime.transact(() => Effect.succeed({ events: [{ type: "added", amount: 1 }] as const, interrupts: [Jobs], result: undefined })))
    yield* Deferred.await(committing)
    const interrupting = yield* Effect.fork(Fiber.interrupt(transaction))
    yield* Effect.sleep("2 millis")
    yield* Deferred.succeed(gate, undefined)
    yield* Fiber.join(interrupting)
    yield* within(runtime.awaitIdle)
    expect(yield* runtime.read.get(Total)).toBe(1)
    expect((yield* runtime.read.get(Done)).find((line) => line.startsWith("8000:"))).toBe("8000:Interrupted")
  })))
})

describe("adversarial: slow observers", () => {
  it("a slow subscriber reads the latest state, not every settlement, and ends on the final one", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* testStore })
    const Q = F.Facade.define({ name: "Slow", mutations: () => ({}), queries: ({ read }) => ({ total: () => read.get(Total).pipe(Effect.orDie) }) })
    const first = yield* Deferred.make<void>()
    const total = yield* Effect.fork(Stream.runCollect(Q.make({ runtime }).queries.total.changes().pipe(
      Stream.tap(() => Effect.zipRight(Deferred.succeed(first, undefined), Effect.sleep("20 millis"))),
    )))
    yield* Deferred.await(first)
    yield* Effect.forEach(Array.from({ length: 100 }), () => runtime.accept({ type: "added", amount: 1 }), { discard: true })
    yield* runtime.close("closed")
    const t = Chunk.toReadonlyArray(yield* within(Fiber.join(total)))
    expect(t).toEqual([0, 100])
  })))
})
