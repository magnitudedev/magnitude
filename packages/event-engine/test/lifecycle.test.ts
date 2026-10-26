import { describe, expect, it } from "vitest"
import { Cause, Deferred, Effect, Exit, Fiber, Option, Ref, Schema, Scope, Stream } from "effect"
import {
  Address,
  Ambient,
  AddressTarget,
  EngineClosed,
  EventEngine,
  EventStore,
  Projection,
  type WorkerInterruption,
} from "../src"

type JobEvent =
  | { readonly type: "started"; readonly id: string }
  | { readonly type: "ended"; readonly id: string; readonly outcome: string }
  | { readonly type: "chained"; readonly remaining: number }
  | { readonly type: "broken" }

const Jobs = EventEngine.define<JobEvent>({ discriminator: "type" })
const Lanes = Address.space({ name: "job", key: Schema.String })

const Outcomes = Jobs.Projection.singleton({
  name: "Outcomes",
  state: Schema.Array(Schema.String),
  initial: [] as readonly string[],
  handlers: (on) => [on.event("ended", ({ event, state }) => Projection.set([...state, `${event.id}:${event.outcome}`]))],
})

const Chain = Jobs.Projection.singleton({
  name: "Chain",
  state: Schema.Number,
  initial: 0,
  handlers: (on) => [on.event("chained", ({ state }) => Projection.set(state + 1))],
})

const label = (interruption: Option.Option<WorkerInterruption>) =>
  Option.match(interruption, { onNone: () => "none", onSome: (i) => i._tag })

// Each `started` job blocks until interrupted, then records the interruption it observed.
const started = new Map<string, Deferred.Deferred<void>>()
const Runner = Jobs.Worker.define({
  name: "Runner",
  partition: { by: Lanes, initial: [Lanes.at("job")], admission: () => [] },
  handlers: (on) => [
    on.event("started", { route: () => AddressTarget.one(Lanes.at("job")) }, ({ event, emit, interruption }) =>
      Effect.flatMap(Effect.sync(() => started.get(event.id)), (signal) => signal === undefined
        ? Effect.void
        : Deferred.succeed(signal, undefined).pipe(
          Effect.zipRight(Effect.never),
          Effect.onInterrupt(() => Effect.flatMap(interruption, (observed) =>
            emit({ type: "ended", id: event.id, outcome: label(observed) }).pipe(Effect.orDie))),
        ))),
  ],
})

const Chainer = Jobs.Worker.define({
  name: "Chainer",
  handlers: (on) => [
    on.event("chained", ({ event, emit }) => event.remaining === 0
      ? Effect.void
      : Effect.zipRight(Effect.sleep("2 millis"), emit({ type: "chained", remaining: event.remaining - 1 }))),
    on.event("broken", () => Effect.die("broken handler")),
  ],
})

const Definition = Jobs.make({ name: "jobs", projections: [Outcomes, Chain], workers: [Runner, Chainer] })

const startJob = (id: string) => Effect.gen(function* () {
  const signal = yield* Deferred.make<void>()
  started.set(id, signal)
  return signal
})

const run = <A, E>(effect: Effect.Effect<A, E, Scope.Scope>) => Effect.runPromise(Effect.scoped(effect))

describe("EventEngine lifecycle", () => {
  it("lets an interrupted handler emit while its interrupter holds the acceptance lock", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* EventStore.memory<JobEvent>() })
    const job = yield* startJob("a")
    yield* runtime.accept({ type: "started", id: "a" })
    yield* Deferred.await(job)
    yield* runtime.interrupt(Lanes.at("job"))
    yield* runtime.awaitIdle
    expect(yield* runtime.read.get(Outcomes)).toEqual(["a:Interrupted"])
  })))

  it("waits for work that work admitted", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* EventStore.memory<JobEvent>() })
    yield* runtime.accept({ type: "chained", remaining: 3 })
    yield* runtime.awaitIdle
    expect(yield* runtime.read.get(Chain)).toBe(4)
  })))

  it("closes with a reason its handlers observe, rejects accepts, and ends observation", () => run(Effect.gen(function* () {
    const store = yield* EventStore.memory<JobEvent>()
    const runtime = yield* Definition.open({ store })
    const job = yield* startJob("b")
    yield* runtime.accept({ type: "started", id: "b" })
    yield* Deferred.await(job)
    const signals = yield* runtime.signals.subscribe(Jobs.Signal.define<void>({ name: "unused" }))
    yield* runtime.close("removed")
    expect(yield* runtime.closed).toEqual({ _tag: "Removed" })
    expect((yield* store.events).at(-1)).toEqual({ type: "ended", id: "b", outcome: "Removed" })
    expect(yield* Effect.flip(runtime.accept({ type: "chained", remaining: 0 })))
      .toEqual(new EngineClosed({ closure: { _tag: "Removed" } }))
    expect(yield* Stream.runCollect(signals).pipe(Effect.map((chunk) => chunk.length))).toBe(0)
    yield* runtime.awaitIdle
  })))

  it("closes as Closed when its scope closes", () => Effect.runPromise(Effect.gen(function* () {
    const store = yield* EventStore.memory<JobEvent>()
    const scope = yield* Scope.make()
    const runtime = yield* Scope.extend(Definition.open({ store }), scope)
    const job = yield* startJob("c")
    yield* runtime.accept({ type: "started", id: "c" })
    yield* Deferred.await(job)
    yield* Scope.close(scope, Exit.void)
    expect(yield* runtime.closed).toEqual({ _tag: "Closed" })
    expect((yield* store.events).at(-1)).toEqual({ type: "ended", id: "c", outcome: "Closed" })
  })))

  it("closes as Failed when a handler fails, rather than logging and continuing", () => run(Effect.gen(function* () {
    const runtime = yield* Definition.open({ store: yield* EventStore.memory<JobEvent>() })
    yield* runtime.accept({ type: "broken" })
    const closure = yield* runtime.closed
    expect(closure._tag === "Failed" && Cause.squash(closure.cause)).toBe("broken handler")
    expect((yield* Effect.flip(runtime.accept({ type: "chained", remaining: 0 })))._tag).toBe("EngineClosed")
  })))
})

describe("EventEngine close", () => {
  // A `started` job blocks until interrupted, then finishes once `finish` completes.
  const lingering = (finish: Deferred.Deferred<void>, running: Deferred.Deferred<void>) => Jobs.Worker.define({
    name: "Lingering",
    handlers: (on) => [
      on.event("started", ({ event, emit }) => Deferred.succeed(running, undefined).pipe(
        Effect.zipRight(Effect.never),
        Effect.onInterrupt(() => Deferred.await(finish).pipe(
          Effect.zipRight(emit({ type: "ended", id: event.id, outcome: "finished" })),
          Effect.orDie,
        )),
      )),
    ],
  })

  it("completes a close whose caller is interrupted while it waits for the lanes", () => run(Effect.gen(function* () {
    const finish = yield* Deferred.make<void>()
    const running = yield* Deferred.make<void>()
    const store = yield* EventStore.memory<JobEvent>()
    const definition = Jobs.make({ name: "interrupted-close", projections: [Outcomes], workers: [lingering(finish, running)] })
    const runtime = yield* definition.open({ store })
    const watcher = yield* Effect.fork(Stream.runCollect(Jobs.Facade.define({
      name: "Watch",
      mutations: () => ({}),
      queries: ({ read }) => ({ outcomes: () => read.get(Outcomes) }),
    }).make({ runtime }).queries.outcomes.changes()))
    yield* runtime.accept({ type: "started", id: "a" })
    yield* Deferred.await(running)
    const closer = yield* Effect.fork(runtime.close("removed"))
    yield* Effect.yieldNow()
    yield* Fiber.interrupt(closer)
    yield* Deferred.succeed(finish, undefined)
    expect(yield* runtime.closed).toEqual({ _tag: "Removed" })
    expect((yield* Effect.flip(runtime.accept({ type: "chained", remaining: 0 })))._tag).toBe("EngineClosed")
    expect(Array.from(yield* Fiber.join(watcher)).at(-1)).toEqual(["a:finished"])
    // A second close shares the first.
    yield* runtime.close("closed")
    expect(yield* runtime.closed).toEqual({ _tag: "Removed" })
  })))

  it("closes when a handler closes its own engine", () => run(Effect.gen(function* () {
    const engine: { close?: Effect.Effect<void> } = {}
    const Closer = Jobs.Worker.define({
      name: "Closer",
      handlers: (on) => [on.event("chained", () => Effect.suspend(() => engine.close!))],
    })
    const runtime = yield* Jobs.make({ name: "self-close", projections: [], workers: [Closer] })
      .open({ store: yield* EventStore.memory<JobEvent>() })
    engine.close = runtime.close("removed")
    yield* runtime.accept({ type: "chained", remaining: 0 })
    expect(yield* runtime.closed).toEqual({ _tag: "Removed" })
  })))

  it("rejects an accept once a handler has failed, before the close it starts takes the lock", () => run(Effect.gen(function* () {
    const failing = yield* Deferred.make<void>()
    const Breaker = Jobs.Worker.define({
      name: "Breaker",
      handlers: (on) => [on.event("broken", () => Effect.zipRight(Deferred.await(failing), Effect.die("broken handler")))],
    })
    const store = yield* EventStore.memory<JobEvent>()
    const runtime = yield* Jobs.make({ name: "failed-accept", projections: [Chain], workers: [Breaker] }).open({ store })
    yield* runtime.accept({ type: "broken" })
    // A transaction holds the lock while the handler fails; an accept waits on the lock ahead of the close.
    const holding = yield* Deferred.make<void>()
    const held = yield* Deferred.make<void>()
    const transaction = yield* Effect.fork(runtime.transact(() => Effect.zipRight(
      Deferred.succeed(held, undefined),
      Effect.as(Deferred.await(holding), { events: [], result: undefined }),
    )))
    yield* Deferred.await(held)
    const accept = yield* Effect.fork(runtime.accept({ type: "chained", remaining: 0 }))
    yield* Deferred.succeed(failing, undefined)
    yield* Effect.sleep("20 millis")
    yield* Deferred.succeed(holding, undefined)
    yield* Fiber.join(transaction)
    const rejected = yield* Effect.flip(Fiber.join(accept))
    expect(rejected._tag === "EngineClosed" && rejected.closure._tag).toBe("Failed")
    expect(yield* runtime.read.get(Chain)).toBe(0)
    expect((yield* runtime.closed)._tag).toBe("Failed")
  })))

  it("dies on a handler's emit and update from a fiber that outlives the engine, and stores nothing", () => run(Effect.gen(function* () {
    const Progress = Ambient.define<number>({ name: "LateProgress", initial: 0 })
    const late = yield* Deferred.make<void>()
    const outcome = yield* Deferred.make<readonly [Exit.Exit<void, unknown>, Exit.Exit<void, unknown>]>()
    const Forker = Jobs.Worker.define({
      name: "Forker",
      handlers: (on) => [on.event("chained", ({ emit, update }) => Effect.asVoid(Effect.forkDaemon(Deferred.await(late).pipe(
        Effect.zipRight(Effect.all([Effect.exit(emit({ type: "chained", remaining: 1 })), Effect.exit(update(Progress, 1))])),
        Effect.flatMap((exits) => Deferred.succeed(outcome, exits)),
      ))))],
    })
    const store = yield* EventStore.memory<JobEvent>()
    const runtime = yield* Jobs.make({ name: "late-emit", projections: [Chain], workers: [Forker] }).open({ store })
    yield* runtime.accept({ type: "chained", remaining: 0 })
    yield* runtime.awaitIdle
    yield* runtime.close("closed")
    yield* Deferred.succeed(late, undefined)
    const [emitted, updated] = yield* Deferred.await(outcome)
    const defect = (exit: Exit.Exit<void, unknown>) => Exit.isFailure(exit) && Cause.isDie(exit.cause)
      ? String(Cause.squash(exit.cause))
      : "no defect"
    expect(defect(emitted)).toContain("emitted after the engine ended, from a fiber that outlived its handler")
    expect(defect(updated)).toContain("updated an ambient after the engine ended, from a fiber that outlived its handler")
    expect(yield* store.events).toEqual([{ type: "chained", remaining: 0 }])
    expect(yield* runtime.read.get(Chain)).toBe(1)
  })))
})

describe("EventEngine ambients", () => {
  const Progress = Ambient.define<number>({ name: "Progress", initial: 0 })
  const Reached = Jobs.Signal.define<number>({ name: "Reached" })

  const Seen = Jobs.Projection.singleton({
    name: "Seen",
    state: Schema.Array(Schema.Number),
    initial: [] as readonly number[],
    emits: [Reached],
    handlers: (on) => [
      on.ambient(Progress, ({ value, state, emit }) => {
        emit(Reached, value)
        return Projection.set([...state, value])
      }),
      // Event handlers read an ambient's current value synchronously.
      on.event("chained", ({ state, ambient }) => Projection.set([...state, -ambient.get(Progress)])),
    ],
  })

  it("folds updates, releases their signals, and stores nothing", () => run(Effect.gen(function* () {
    const reached = yield* Ref.make<readonly number[]>([])
    const Reporter = Jobs.Worker.define({
      name: "Reporter",
      handlers: (on) => [
        on.event("started", ({ update }) => Effect.zipRight(update(Progress, 1), update(Progress, 2))),
        on.signal(Reached, ({ signal }) => Ref.update(reached, (all) => [...all, signal.payload])),
      ],
    })
    const store = yield* EventStore.memory<JobEvent>()
    const runtime = yield* Jobs.make({ name: "ambient", projections: [Seen], workers: [Reporter] }).open({ store })
    yield* runtime.accept({ type: "started", id: "x" })
    yield* runtime.awaitIdle
    yield* runtime.accept({ type: "chained", remaining: 0 })
    expect(yield* runtime.read.get(Seen)).toEqual([1, 2, -2])
    expect(yield* Ref.get(reached)).toEqual([1, 2])
    expect(yield* store.events).toEqual([{ type: "started", id: "x" }, { type: "chained", remaining: 0 }])
  })))

  it("starts from the initial value on every open", () => run(Effect.gen(function* () {
    const store = yield* EventStore.memory<JobEvent>()
    yield* store.commit([{ type: "chained", remaining: 0 }])
    const runtime = yield* Jobs.make({ name: "ambient-replay", projections: [Seen], workers: [] }).open({ store })
    expect(yield* runtime.read.get(Seen)).toEqual([-0])
    yield* runtime.update(Progress, 7)
    expect(yield* runtime.read.get(Seen)).toEqual([-0, 7])
  })))
})
