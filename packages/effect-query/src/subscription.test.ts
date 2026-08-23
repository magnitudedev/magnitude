import * as Registry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import * as Atom from "@effect-atom/atom/Atom"
import * as Data from "effect/Data"
import * as Deferred from "effect/Deferred"
import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
import * as Option from "effect/Option"
import * as Queue from "effect/Queue"
import * as Schedule from "effect/Schedule"
import * as Stream from "effect/Stream"
import { describe, expect, it } from "vitest"
import { Client, Key, Query, QueryClient, Subscription } from "./index.js"

const sleep = (millis: number) => Effect.runPromise(Effect.sleep(`${millis} millis`))

/** The QueryClient owned by this Client's runtime in this registry. */
const queryClientFor = (client: Client.Client<never, never>, registry: Registry.Registry) =>
  Effect.runPromise(Registry.getResult(registry, client.runtime.atom(QueryClient.QueryClient)))

describe("Key", () => {
  it("canonicalizes structurally equal values to one string", () => {
    expect(Key.canonical({ b: 1, a: [1, { d: 2, c: 3 }] })).toBe(Key.canonical({ a: [1, { c: 3, d: 2 }], b: 1 }))
    expect(Key.canonical({ a: 1, b: undefined })).toBe(Key.canonical({ a: 1 }))
    expect(Key.canonical(Option.some("x"))).toBe(Key.canonical(Option.some("x")))
    expect(Key.canonical(Option.some("x"))).not.toBe(Key.canonical(Option.none()))
    expect(Key.canonical(1n)).toBe(Key.canonical(1n))
  })
})

describe("Subscription", () => {
  it("shares one open stream per key and delivers events to every consumer", async () => {
    const registry = Registry.make()
    const client = Client.make(Layer.empty)
    let opened = 0
    const ticks = Subscription.make("Ticks", {
      key: (topic: string) => topic,
      stream: () => Stream.suspend(() => {
        opened += 1
        return Stream.fromSchedule(Schedule.spaced("5 millis")).pipe(Stream.take(3))
      })
    })
    const atom = client.subscription(ticks, "a")
    expect(client.subscription(ticks, "a")).toBe(atom)

    const consume = Subscription.events(atom).pipe(
      Stream.take(2),
      Stream.runCollect,
      Effect.provideService(Registry.AtomRegistry, registry)
    )
    const [first, second] = await Effect.runPromise(Effect.all([consume, consume], { concurrency: "unbounded" }))

    expect(Array.from(first)).toEqual([0, 1])
    expect(Array.from(second)).toEqual([0, 1])
    expect(opened).toBe(1)
  })

  it("reports connecting, active, and completed status", async () => {
    const registry = Registry.make()
    const client = Client.make(Layer.empty)
    const gate = Effect.runSync(Deferred.make<void>())
    const ticks = Subscription.make("Gated", {
      key: (_: void) => "gated",
      stream: () => Stream.fromEffect(Deferred.await(gate)).pipe(Stream.as(1))
    })
    const atom = client.subscription(ticks, undefined)
    const unmount = registry.mount(atom)
    await sleep(1)
    expect(registry.get(atom).status).toBe("connecting")
    Effect.runSync(Deferred.succeed(gate, undefined))
    await sleep(5)
    expect(registry.get(atom).status).toBe("completed")
    expect(registry.get(atom).latest).toEqual(Option.some(1))
    unmount()
    registry.dispose()
  })

  it("reconnects per schedule and retains the failure while reconnecting", async () => {
    const registry = Registry.make()
    const client = Client.make(Layer.empty)
    let attempts = 0
    const flaky = Subscription.make("Flaky", {
      key: (_: void) => "flaky",
      stream: () => Stream.suspend(() => {
        attempts += 1
        return attempts < 3
          ? Stream.fail(new Error(`attempt ${attempts}`))
          : Stream.make("ok").pipe(Stream.concat(Stream.never))
      }),
      reconnect: Schedule.spaced("1 millis")
    })
    const atom = client.subscription(flaky, undefined)
    const unmount = registry.mount(atom)
    await sleep(20)
    const state = registry.get(atom)
    expect(attempts).toBe(3)
    expect(state.status).toBe("active")
    expect(state.latest).toEqual(Option.some("ok"))
    expect(state.attempt).toBe(3)
    unmount()
    registry.dispose()
  })

  it("fails when the reconnect schedule is exhausted", async () => {
    const registry = Registry.make()
    const client = Client.make(Layer.empty)
    const broken = Subscription.make("Broken", {
      key: (_: void) => "broken",
      stream: () => Stream.fail(new Error("down")),
      reconnect: Schedule.recurs(1)
    })
    const atom = client.subscription(broken, undefined)
    const unmount = registry.mount(atom)
    await sleep(10)
    const state = registry.get(atom)
    expect(state.status).toBe("failed")
    expect(Option.isSome(state.failure)).toBe(true)
    unmount()
    registry.dispose()
  })

  it("reopens on Reset and closes on Interrupt", async () => {
    const registry = Registry.make()
    const client = Client.make(Layer.empty)
    let opened = 0
    const ticks = Subscription.make("Resettable", {
      key: (_: void) => "resettable",
      stream: () => Stream.suspend(() => {
        opened += 1
        return Stream.make(opened).pipe(Stream.concat(Stream.never))
      })
    })
    const atom = client.subscription(ticks, undefined)
    const unmount = registry.mount(atom)
    await sleep(5)
    expect(opened).toBe(1)
    registry.set(atom, Atom.Reset)
    await sleep(5)
    expect(opened).toBe(2)
    expect(registry.get(atom).latest).toEqual(Option.some(2))
    registry.set(atom, Atom.Interrupt)
    await sleep(5)
    expect(registry.get(atom).status).toBe("idle")
    expect(opened).toBe(2)
    unmount()
    registry.dispose()
  })

  it("is reconnected through the QueryClient by filter", async () => {
    const registry = Registry.make()
    const client = Client.make(Layer.empty)
    const queryClient = await queryClientFor(client, registry)
    let opened = 0
    const ticks = Subscription.make("ClientReconnect", {
      key: (_: void) => "client-reconnect",
      stream: () => Stream.suspend(() => {
        opened += 1
        return Stream.never
      })
    })
    const atom = client.subscription(ticks, undefined)
    const unmount = registry.mount(atom)
    await sleep(2)
    await Effect.runPromise(queryClient.reconnect(ticks.match()))
    await sleep(2)
    expect(opened).toBe(2)
    unmount()
    registry.dispose()
  })
})

describe("Query.fromStream", () => {
  it("succeeds on the first element and folds later elements into the data", async () => {
    const registry = Registry.make()
    const client = Client.make(Layer.empty)
    const queue = Effect.runSync(Queue.unbounded<number>())
    const totals = Query.fromStream("Totals", {
      key: (_: void) => "totals",
      stream: () => Stream.fromQueue(queue),
      reduce: (previous: Option.Option<number>, event: number) => Option.getOrElse(previous, () => 0) + event
    })
    const atom = client.query(totals, undefined)
    const unmount = registry.mount(atom)
    await sleep(2)
    expect(registry.get(atom).fetchStatus).toBe("fetching")

    Effect.runSync(Queue.offer(queue, 5))
    await sleep(2)
    expect(AtomResult.value(registry.get(atom).result)).toEqual(Option.some(5))
    expect(registry.get(atom).fetchStatus).toBe("idle")
    expect(registry.get(atom).isStale).toBe(false)

    Effect.runSync(Queue.offer(queue, 7))
    await sleep(2)
    expect(AtomResult.value(registry.get(atom).result)).toEqual(Option.some(12))
    unmount()
    registry.dispose()
  })

  it("surfaces a terminal stream failure as a failure carrying the previous data", async () => {
    const registry = Registry.make()
    const client = Client.make(Layer.empty)
    const queue = Effect.runSync(Queue.unbounded<number>())
    const totals = Query.fromStream("FailingTotals", {
      key: (_: void) => "failing",
      stream: () => Stream.fromQueue(queue).pipe(
        Stream.mapEffect((value) => value < 0 ? Effect.fail(new Error("negative")) : Effect.succeed(value))
      ),
      reduce: (previous: Option.Option<number>, event: number) => Option.getOrElse(previous, () => 0) + event
    })
    const atom = client.query(totals, undefined)
    const unmount = registry.mount(atom)
    Effect.runSync(Queue.offer(queue, 1))
    await sleep(2)
    Effect.runSync(Queue.offer(queue, -1))
    await sleep(5)
    const result = registry.get(atom).result
    expect(result._tag).toBe("Failure")
    expect(AtomResult.value(result)).toEqual(Option.some(1))
    unmount()
    registry.dispose()
  })

  it("reopens the stream on invalidation", async () => {
    const registry = Registry.make()
    const client = Client.make(Layer.empty)
    const queryClient = await queryClientFor(client, registry)
    let opened = 0
    const snapshots = Query.fromStream("Snapshots", {
      key: (_: void) => "snapshots",
      stream: () => Stream.suspend(() => {
        opened += 1
        return Stream.make(opened).pipe(Stream.concat(Stream.never))
      }),
      reduce: (_: Option.Option<number>, event: number) => event
    })
    const atom = client.query(snapshots, undefined)
    const unmount = registry.mount(atom)
    await sleep(3)
    expect(AtomResult.value(registry.get(atom).result)).toEqual(Option.some(1))
    await Effect.runPromise(queryClient.invalidate(snapshots.match()))
    await sleep(5)
    expect(opened).toBe(2)
    expect(AtomResult.value(registry.get(atom).result)).toEqual(Option.some(2))
    unmount()
    registry.dispose()
  })
})

describe("QueryClient", () => {
  it("invalidates by definition name", async () => {
    const registry = Registry.make()
    const client = Client.make(Layer.empty)
    const queryClient = await queryClientFor(client, registry)
    let fetches = 0
    const users = Query.make("NamedUsers", {
      key: ({ id }: { readonly id: string }) => Data.struct({ id }),
      staleTime: Infinity,
      effect: ({ id }) => Effect.sync(() => {
        fetches += 1
        return { id }
      })
    })
    const atom = client.query(users, { id: "1" })
    const unmount = registry.mount(atom)
    await sleep(2)
    expect(fetches).toBe(1)
    await Effect.runPromise(queryClient.invalidate({ name: "NamedUsers" }))
    await sleep(2)
    expect(fetches).toBe(2)
    await Effect.runPromise(queryClient.invalidate({ name: "Other" }))
    await sleep(2)
    expect(fetches).toBe(2)
    unmount()
    registry.dispose()
  })
})
