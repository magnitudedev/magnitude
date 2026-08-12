import * as Atom from "@effect-atom/atom/Atom"
import * as Registry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import * as Cause from "effect/Cause"
import * as Data from "effect/Data"
import * as Deferred from "effect/Deferred"
import * as Duration from "effect/Duration"
import * as Effect from "effect/Effect"
import * as Fiber from "effect/Fiber"
import * as Context from "effect/Context"
import * as Equal from "effect/Equal"
import * as Hash from "effect/Hash"
import * as Layer from "effect/Layer"
import * as Option from "effect/Option"
import * as Schedule from "effect/Schedule"
import { describe, expect, expectTypeOf, it } from "vitest"
import { Client, Mutation, Query, QueryClient } from "./index.js"

const clientFor = (registry: Registry.Registry): QueryClient.Service => Effect.runSync(
  QueryClient.QueryClient.pipe(
    Effect.provide(
      QueryClient.makeLayer(() => {
        throw new Error("not used by low-level service tests")
      }).pipe(
        Layer.provide(Layer.succeed(Registry.AtomRegistry, registry))
      )
    )
  )
)

describe("Query", () => {
  it("runs definition-based cache operations through its connection client", async () => {
    const registry = Registry.make()
    const effectQuery = Client.make(Layer.empty)
    const query = Query.make("DefinitionFetch", {
      key: () => Data.tuple("definition-fetch"),
      effect: () => Effect.succeed(42),
    })
    const queryAtom = effectQuery.query(query, undefined)
    const fetchAtom = effectQuery.runtime.atom(QueryClient.fetch(query, undefined))

    const unmount = registry.mount(fetchAtom)
    await Effect.runPromise(Effect.sleep("1 millis"))

    expect(AtomResult.value(registry.get(queryAtom).result)).toEqual(Option.some(42))
    unmount()
    registry.dispose()
  })

  it("rejects structured keys without Effect equality", () => {
    const effectQuery = Client.make(Layer.empty)
    const query = Query.make("InvalidKey", {
      key: (id: string) => ({ id }),
      effect: Effect.succeed
    })

    expect(() => effectQuery.query(query, "1")).toThrow("without Effect Equal semantics")
  })

  it("uses one canonical atom and shares its in-flight fetch", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    let calls = 0
    const users = Query.make("User", {
      key: ({ id }: { readonly id: string }) => Data.struct({ id }),
      effect: ({ id }) => Effect.sync(() => calls++).pipe(
        Effect.zipRight(Effect.sleep("10 millis")),
        Effect.as({ id, name: "Ada" })
      )
    })

    const first = effectQuery.query(users, { id: "1" })
    const same = effectQuery.query(users, { id: "1" })
    expect(first).toBe(same)

    const values = await Effect.runPromise(Effect.all([
      client.fetch(first),
      client.fetch(same)
    ], { concurrency: "unbounded" }))

    expect(values).toEqual([{ id: "1", name: "Ada" }, { id: "1", name: "Ada" }])
    expect(calls).toBe(1)
    registry.dispose()
  })

  it("retains the complete query entry across observer-free remounts", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    let calls = 0
    const query = Query.make("Remounted", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Effect.sync(() => ++calls).pipe(Effect.delay("1 millis")),
      staleTime: Duration.infinity,
      gcTime: Duration.infinity
    })
    const atom = effectQuery.query(query, undefined)

    const unmount = registry.mount(atom)
    expect(await Effect.runPromise(client.fetch(atom))).toBe(1)
    unmount()
    await Effect.runPromise(Effect.sleep("1 millis"))

    const remount = registry.mount(atom)
    const state = registry.get(atom)
    expect(AtomResult.value(state.result)).toEqual(Option.some(1))
    expect(state.fetchStatus).toBe("idle")
    expect(calls).toBe(1)
    remount()
    registry.dispose()
  })

  it("returns retained fresh data from fetch without executing again", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    let calls = 0
    const query = Query.make("FreshFetch", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Effect.sync(() => ++calls),
      staleTime: Duration.infinity,
      gcTime: Duration.infinity
    })
    const atom = effectQuery.query(query, undefined)

    expect(await Effect.runPromise(client.fetch(atom))).toBe(1)
    expect(await Effect.runPromise(client.fetch(atom))).toBe(1)
    expect(calls).toBe(1)
    registry.dispose()
  })

  it("does not prefetch retained fresh data again", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    let calls = 0
    const query = Query.make("FreshPrefetch", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Effect.sync(() => ++calls),
      staleTime: Duration.infinity,
      gcTime: Duration.infinity
    })
    const atom = effectQuery.query(query, undefined)

    await Effect.runPromise(client.prefetch(atom))
    await Effect.runPromise(client.prefetch(atom))
    expect(calls).toBe(1)
    registry.dispose()
  })

  it("does not execute again when switching between selector observers", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    let calls = 0
    const query = Query.make("SelectorSwitch", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Effect.sync(() => ({ value: ++calls })).pipe(Effect.delay("1 millis")),
      staleTime: Duration.infinity,
      gcTime: Duration.infinity
    })
    const source = effectQuery.query(query, undefined)
    const modelsSelector = Query.select(source, (data) => `models:${data.value}`)

    const unmountModels = registry.mount(modelsSelector)
    expect(await Effect.runPromise(client.fetch(source))).toEqual({ value: 1 })
    unmountModels()
    const catalogSelector = Query.select(source, (data) => `catalog:${data.value}`)
    const unmountCatalog = registry.mount(catalogSelector)

    expect(AtomResult.value(registry.get(catalogSelector).result)).toEqual(Option.some("catalog:1"))
    expect(calls).toBe(1)
    unmountCatalog()
    registry.dispose()
  })

  it("does not refetch fresh data when a fetch bridge unmounts and remounts", async () => {
    const registry = Registry.make()
    const effectQuery = Client.make(Layer.empty)
    const runtime = effectQuery.runtime
    let calls = 0
    const query = Query.make("BridgeRemount", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Effect.sync(() => ++calls),
      staleTime: Duration.infinity,
      gcTime: Duration.infinity
    })
    const atom = effectQuery.query(query, undefined)
    const bridge = Atom.setIdleTTL(runtime.atom(
      Effect.flatMap(QueryClient.QueryClient, (client) => client.fetch(atom))
    ), 0)

    const firstUnmount = registry.mount(bridge)
    await Effect.runPromise(Effect.sleep("1 millis"))
    firstUnmount()
    await Effect.runPromise(Effect.sleep("1 millis"))
    const secondUnmount = registry.mount(bridge)
    await Effect.runPromise(Effect.sleep("1 millis"))

    expect(calls).toBe(1)
    secondUnmount()
    registry.dispose()
  })

  it("collects the complete query entry when gcTime expires", async () => {
    const registry = Registry.make({ timeoutResolution: 1 })
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    let calls = 0
    const query = Query.make("Expired", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Effect.sync(() => ++calls),
      staleTime: Duration.infinity,
      gcTime: "50 millis"
    })
    const atom = effectQuery.query(query, undefined)

    expect(await Effect.runPromise(client.fetch(atom))).toBe(1)
    await Effect.runPromise(Effect.sleep("2 millis"))
    const remount = registry.mount(atom)
    expect(AtomResult.value(registry.get(atom).result)).toEqual(Option.some(1))
    expect(calls).toBe(1)
    remount()
    await Effect.runPromise(Effect.sleep("75 millis"))
    expect(await Effect.runPromise(client.fetch(atom))).toBe(2)
    expect(calls).toBe(2)
    registry.dispose()
  })

  it("uses equality, not hash identity, for canonical keys", () => {
    class CollidingKey implements Equal.Equal {
      constructor(readonly value: string) {}
      [Equal.symbol](other: Equal.Equal): boolean {
        return other instanceof CollidingKey && other.value === this.value
      }
      [Hash.symbol](): number {
        return 1
      }
    }
    const effectQuery = Client.make(Layer.empty)
    const query = Query.make("Collision", {
      key: (value: string) => new CollidingKey(value),
      effect: Effect.succeed
    })

    const firstA = effectQuery.query(query, "a")
    expect(firstA).toBe(effectQuery.query(query, "a"))
    expect(firstA.input).toBe("a")
    expect(effectQuery.query(query, "a")).not.toBe(effectQuery.query(query, "b"))
  })

  it("keeps canonical query identity while the returned atom is strongly reachable", () => {
    const effectQuery = Client.make(Layer.empty)
    const query = Query.make("StrongCanonicalIdentity", {
      key: () => Data.tuple("singleton"),
      effect: () => Effect.succeed(1)
    })

    const first = effectQuery.query(query, undefined)
    Bun.gc(true)

    expect(effectQuery.query(query, undefined)).toBe(first)
  })

  it("retains successful data during background refetch failure", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    let succeeds = true
    const query = Query.make("Retained", {
      key: () => Data.struct({ singleton: true }),
      effect: () => succeeds ? Effect.succeed(1) : Effect.fail("offline" as const)
    })
    const atom = effectQuery.query(query, undefined)

    expect(await Effect.runPromise(client.fetch(atom))).toBe(1)
    succeeds = false
    await Effect.runPromise(client.prefetch(atom))

    const state = registry.get(atom)
    expect(state.result._tag).toBe("Failure")
    expect(Option.getOrThrow(AtomResult.value(state.result))).toBe(1)
    expect(state.failureCount).toBe(1)
    registry.dispose()
  })

  it("uses the definition retry schedule without widening errors", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    let calls = 0
    const query = Query.make("Retry", {
      key: () => Data.struct({ singleton: true }),
      effect: () => ++calls === 1 ? Effect.fail("transient" as const) : Effect.succeed(2),
      retry: Schedule.recurs(1)
    })

    expect(await Effect.runPromise(client.fetch(effectQuery.query(query, undefined)))).toBe(2)
    expect(calls).toBe(2)
    registry.dispose()
  })

  it("select derives state without creating another query entry", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    const query = Query.make("Selectable", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Effect.succeed({ title: "hello", ignored: true })
    })
    const source = effectQuery.query(query, undefined)
    const selected = Query.select(source, (data) => data.title)
    await Effect.runPromise(client.fetch(source))
    expect(AtomResult.value(registry.get(selected).result)).toEqual(Option.some("hello"))
    registry.dispose()
  })

  it("does not materialize a query for Option.none", () => {
    const registry = Registry.make()
    const effectQuery = Client.make(Layer.empty)
    let calls = 0
    const query = Query.make("Optional", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Effect.sync(() => ++calls)
    })

    expect(registry.get(Query.when(Option.none<ReturnType<typeof effectQuery.query>>())))
      .toEqual(Option.none())
    expect(calls).toBe(0)
    registry.dispose()
  })

  it("lets authoritative fetches replace data written through setData", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    let server = 1
    const query = Query.make("SetData", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Effect.succeed(server)
    })
    const atom = effectQuery.query(query, undefined)
    await Effect.runPromise(client.fetch(atom))
    await Effect.runPromise(client.setData(atom, () => 2))
    expect(AtomResult.value(registry.get(atom).result)).toEqual(Option.some(2))

    server = 3
    await Effect.runPromise(client.fetch(atom))
    expect(AtomResult.value(registry.get(atom).result)).toEqual(Option.some(3))
    registry.dispose()
  })

  it("supersedes a fetch invalidated while it is in flight", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    const first = Effect.runSync(Deferred.make<number>())
    const second = Effect.runSync(Deferred.make<number>())
    let calls = 0
    const query = Query.make("Generation", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Deferred.await(calls++ === 0 ? first : second),
      staleTime: Duration.infinity
    })
    const atom = effectQuery.query(query, undefined)

    const waiter = Effect.runFork(client.fetch(atom))
    await Effect.runPromise(Effect.sleep("1 millis"))
    await Effect.runPromise(client.invalidate(query.match()))
    await Effect.runPromise(Deferred.succeed(first, 1))
    await Effect.runPromise(Deferred.succeed(second, 2))

    expect(await Effect.runPromise(Fiber.join(waiter))).toBe(2)
    expect(AtomResult.value(registry.get(atom).result)).toEqual(Option.some(2))
    expect(registry.get(atom).isStale).toBe(false)
    registry.dispose()
  })

  it("coalesces repeated invalidations while a replacement fetch is active", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    const first = Effect.runSync(Deferred.make<number>())
    const replacement = Effect.runSync(Deferred.make<number>())
    let calls = 0
    const query = Query.make("CoalescedInvalidation", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Deferred.await(calls++ === 0 ? first : replacement),
      staleTime: Duration.infinity
    })
    const atom = effectQuery.query(query, undefined)

    const waiter = Effect.runFork(client.fetch(atom))
    await Effect.runPromise(Effect.yieldNow())
    await Effect.runPromise(Effect.all(
      Array.from({ length: 25 }, () => client.invalidate(query.match())),
      { concurrency: "unbounded" }
    ))
    await Effect.runPromise(Deferred.succeed(first, 1))
    await Effect.runPromise(Deferred.succeed(replacement, 2))

    expect(await Effect.runPromise(Fiber.join(waiter))).toBe(2)
    expect(calls).toBe(2)
    expect(AtomResult.value(registry.get(atom).result)).toEqual(Option.some(2))
    registry.dispose()
  })

  it("awaits the replacement when fetching an invalidated active query", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    const first = Effect.runSync(Deferred.make<number>())
    const replacement = Effect.runSync(Deferred.make<number>())
    let calls = 0
    const query = Query.make("ActiveRefetch", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Deferred.await(calls++ === 0 ? first : replacement),
      staleTime: Duration.infinity
    })
    const atom = effectQuery.query(query, undefined)
    const unmount = registry.mount(atom)
    await Effect.runPromise(Effect.yieldNow())

    await Effect.runPromise(client.invalidate(query.match()))
    const fetch = Effect.runFork(client.fetch(atom))
    await Effect.runPromise(Deferred.succeed(first, 1))
    await Effect.runPromise(Effect.yieldNow())

    expect(Option.isNone(await Effect.runPromise(Fiber.poll(fetch)))).toBe(true)
    expect(calls).toBe(2)

    await Effect.runPromise(Deferred.succeed(replacement, 2))
    expect(await Effect.runPromise(Fiber.join(fetch))).toBe(2)
    expect(AtomResult.value(registry.get(atom).result)).toEqual(Option.some(2))
    unmount()
    registry.dispose()
  })

  it("fetches an unobserved query once after invalidation", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    let calls = 0
    const query = Query.make("UnobservedRefetch", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Effect.sync(() => ++calls),
      staleTime: Duration.infinity
    })

    await Effect.runPromise(client.invalidate(query.match()))
    expect(await Effect.runPromise(client.fetch(effectQuery.query(query, undefined)))).toBe(1)
    expect(calls).toBe(1)
    registry.dispose()
  })

  it("does not materialize an unobserved query when a notification invalidates its key", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    let calls = 0
    const query = Query.make("RemoteNotification", {
      key: (id: string) => Data.tuple(id),
      effect: (id) => Effect.sync(() => {
        calls++
        return id
      })
    })
    const unobserved = effectQuery.query(query, "unobserved")

    await Effect.runPromise(client.invalidate(query.match("unobserved")))

    expect(calls).toBe(0)
    expect(await Effect.runPromise(client.getState(unobserved))).toEqual(Option.none())
    registry.dispose()
  })

  it("exposes reactive aggregate fetch state", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    const pending = Effect.runSync(Deferred.make<number>())
    const query = Query.make("Fetching", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Deferred.await(pending)
    })
    const atom = effectQuery.query(query, undefined)
    const fetching = client.isFetching(query.match())
    const fiber = Effect.runFork(client.fetch(atom))
    await Effect.runPromise(Effect.yieldNow())
    expect(registry.get(fetching)).toBe(1)

    await Effect.runPromise(Deferred.succeed(pending, 1))
    await Effect.runPromise(Fiber.join(fiber))
    expect(registry.get(fetching)).toBe(0)
    registry.dispose()
  })

  it("cancellation restores retained data and pauses the entry", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    const pending = Effect.runSync(Deferred.make<number>())
    let calls = 0
    const query = Query.make("Cancellation", {
      key: () => Data.struct({ singleton: true }),
      effect: () => calls++ === 0 ? Effect.succeed(1) : Deferred.await(pending),
      staleTime: Duration.infinity
    })
    const atom = effectQuery.query(query, undefined)

    expect(await Effect.runPromise(client.fetch(atom))).toBe(1)
    await Effect.runPromise(client.invalidate(query.match()))
    await Effect.runPromise(client.cancel(query.match()))
    await Effect.runPromise(Effect.yieldNow())

    const state = registry.get(atom)
    expect(AtomResult.value(state.result)).toEqual(Option.some(1))
    expect(state.fetchStatus).toBe("paused")
    registry.dispose()
  })

  it("installs one definition-owned refresh schedule while mounted", async () => {
    const registry = Registry.make()
    const effectQuery = Client.make(Layer.empty)
    let calls = 0
    const query = Query.make("Scheduled", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Effect.sync(() => ++calls),
      refresh: Schedule.spaced("2 millis").pipe(Schedule.intersect(Schedule.recurs(1)))
    })
    const unmount = registry.mount(effectQuery.query(query, undefined))
    await Effect.runPromise(Effect.sleep("10 millis"))
    unmount()

    expect(calls).toBe(2)
    registry.dispose()
  })

  it("lets Atom registry collection remove the client index entry", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    const query = Query.make("Collected", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Effect.succeed(1),
      gcTime: 0
    })
    const atom = effectQuery.query(query, undefined)
    await Effect.runPromise(client.fetch(atom))
    await Effect.runPromise(Effect.sleep("1 millis"))

    expect(await Effect.runPromise(client.getState(atom))).toEqual(Option.none())
    registry.dispose()
  })
})

describe("Mutation", () => {
  it("indexes mutation states and distinguishes synchronization failure", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    const mutation = Mutation.make("Rename", {
      effect: (name: string) => Effect.succeed(name.toUpperCase()),
      synchronize: () => Effect.fail("not-visible" as const)
    })
    const mutationAtom = effectQuery.mutation(mutation)

    const exit = await Effect.runPromiseExit(
      Mutation.execute(mutationAtom, "ada").pipe(
        Effect.provideService(Registry.AtomRegistry, registry)
      )
    )
    expect(exit._tag).toBe("Failure")
    const state = registry.get(mutationAtom)
    expect(state._tag).toBe("Failure")
    if (state._tag === "Failure") {
      const failure = Option.getOrThrow(Cause.failureOption(state.cause))
      expect(failure).toBeInstanceOf(Mutation.MutationSynchronizationError)
      expect((failure as Mutation.MutationSynchronizationError<string, string>).output).toBe("ADA")
    }

    const mutationStatesAtom = client.mutationState(mutation.match())
    const mutationStates = registry.get(mutationStatesAtom)
    expect(mutationStates).toHaveLength(1)
    expect(mutationStates[0].result._tag).toBe("Failure")
    registry.dispose()
  })

  it("keeps mutation state pending until synchronization completes", async () => {
    const registry = Registry.make()
    const effectQuery = Client.make(Layer.empty)
    const synchronized = Effect.runSync(Deferred.make<void>())
    const mutation = Mutation.make("SynchronizedMutation", {
      effect: (input: string) => Effect.succeed(input.toUpperCase()),
      synchronize: () => Deferred.await(synchronized),
    })
    const mutationAtom = effectQuery.mutation(mutation)
    const mutatingState = Mutation.isMutating({ mutation })
    const fiber = Effect.runFork(Mutation.execute(mutationAtom, "ready").pipe(
      Effect.provideService(Registry.AtomRegistry, registry),
    ))
    await Effect.runPromise(Effect.yieldNow())

    expect(registry.get(mutatingState)).toBe(1)
    Effect.runSync(Deferred.succeed(synchronized, undefined))
    expect(await Effect.runPromise(Fiber.join(fiber))).toBe("READY")
    expect(registry.get(mutatingState)).toBe(0)
    registry.dispose()
  })

  it("serializes equal scopes", async () => {
    const registry = Registry.make()
    const effectQuery = Client.make(Layer.empty)
    let running = 0
    let maximum = 0
    const mutation = Mutation.make("Scoped", {
      effect: (value: number) => Effect.sync(() => {
        running++
        maximum = Math.max(maximum, running)
      }).pipe(
        Effect.zipRight(Effect.sleep("5 millis")),
        Effect.ensuring(Effect.sync(() => running--)),
        Effect.as(value)
      ),
      scope: () => Mutation.MutationScope("shared")
    })
    const mutationAtom = effectQuery.mutation(mutation)

    const exit = await Effect.runPromiseExit(
      Effect.all([
        Mutation.execute(mutationAtom, 1),
        Mutation.execute(mutationAtom, 2)
      ], { concurrency: "unbounded" }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry)
      )
    )
    expect(exit._tag, exit._tag === "Failure" ? Cause.pretty(exit.cause) : undefined).toBe("Success")
    const values = exit._tag === "Success" ? exit.value : []
    expect(values).toEqual([1, 2])
    expect(maximum).toBe(1)
    registry.dispose()
  })

  it("runs unscoped invocations concurrently", async () => {
    const registry = Registry.make()
    const effectQuery = Client.make(Layer.empty)
    let running = 0
    let maximum = 0
    const mutation = Mutation.make("Concurrent", {
      effect: (value: number) => Effect.sync(() => {
        running++
        maximum = Math.max(maximum, running)
      }).pipe(
        Effect.zipRight(Effect.sleep("5 millis")),
        Effect.ensuring(Effect.sync(() => running--)),
        Effect.as(value)
      )
    })
    const mutationAtom = effectQuery.mutation(mutation)

    await Effect.runPromise(
      Effect.all([
        Mutation.execute(mutationAtom, 1),
        Mutation.execute(mutationAtom, 2)
      ], { concurrency: "unbounded" }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry)
      )
    )
    expect(maximum).toBe(2)
    registry.dispose()
  })

  it("serializes equal scopes across mutation definitions", async () => {
    const registry = Registry.make()
    const effectQuery = Client.make(Layer.empty)
    let running = 0
    let maximum = 0
    const makeMutation = (name: string) => Mutation.make(name, {
      effect: (value: number) => Effect.sync(() => {
        running++
        maximum = Math.max(maximum, running)
      }).pipe(
        Effect.zipRight(Effect.sleep("2 millis")),
        Effect.ensuring(Effect.sync(() => running--)),
        Effect.as(value)
      ),
      scope: () => Mutation.MutationScope("shared")
    })
    const first = effectQuery.mutation(makeMutation("FirstScopedDefinition"))
    const second = effectQuery.mutation(makeMutation("SecondScopedDefinition"))

    await Effect.runPromise(
      Effect.all([
        Mutation.execute(first, 1),
        Mutation.execute(second, 2)
      ], { concurrency: "unbounded" }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry)
      )
    )

    expect(maximum).toBe(1)
    registry.dispose()
  })

  it("selects typed mutation states by semantic scope and pending status", async () => {
    const registry = Registry.make()
    const effectQuery = Client.make(Layer.empty)
    const first = Effect.runSync(Deferred.make<string>())
    const second = Effect.runSync(Deferred.make<string>())
    const mutation = Mutation.make("ScopedSelectors", {
      effect: ({ id }: { readonly id: "first" | "second" }) =>
        Deferred.await(id === "first" ? first : second),
      scope: ({ id }) => Mutation.MutationScope(`item:${id}`),
    })
    const mutationAtom = effectQuery.mutation(mutation)
    const firstScope = Mutation.MutationScope("item:first")
    const firstMutationStates = Mutation.state({
      filters: { mutation, scope: firstScope },
    })
    const firstInputsState = Mutation.state({
      filters: { mutation, scope: firstScope },
      select: ({ input }) => input.id,
    })
    const firstMutatingState = Mutation.isMutating({ mutation, scope: firstScope })

    const firstInput: Mutation.Input<typeof mutation> = { id: "first" }
    const secondInput: Mutation.Input<typeof mutation> = { id: "second" }
    const firstFiber = Effect.runFork(Mutation.execute(mutationAtom, firstInput).pipe(
      Effect.provideService(Registry.AtomRegistry, registry),
    ))
    const secondFiber = Effect.runFork(Mutation.execute(mutationAtom, secondInput).pipe(
      Effect.provideService(Registry.AtomRegistry, registry),
    ))
    await Effect.runPromise(Effect.yieldNow())

    expect(registry.get(firstMutatingState)).toBe(1)
    expect(registry.get(firstInputsState)).toEqual(["first"])
    expect(registry.get(firstMutationStates).at(-1)?.input.id).toBe("first")

    Effect.runSync(Deferred.succeed(first, "FIRST"))
    await Effect.runPromise(Fiber.join(firstFiber))
    expect(registry.get(firstMutatingState)).toBe(0)
    expect(registry.get(Mutation.isMutating({ mutation }))).toBe(1)

    Effect.runSync(Deferred.succeed(second, "SECOND"))
    await Effect.runPromise(Fiber.join(secondFiber))
    registry.dispose()
  })

  it("interrupts active invocations and resets only the public latest result", async () => {
    const registry = Registry.make()
    const effectQuery = Client.make(Layer.empty)
    const pending = Effect.runSync(Deferred.make<string>())
    const mutation = Mutation.make("Interruptible", {
      effect: () => Deferred.await(pending)
    })
    const mutationAtom = effectQuery.mutation(mutation)
    const fiber = Effect.runFork(
      Mutation.execute(mutationAtom, undefined).pipe(
        Effect.provideService(Registry.AtomRegistry, registry)
      )
    )
    await Effect.runPromise(Effect.yieldNow())
    registry.set(mutationAtom, Atom.Interrupt)

    const exit = await Effect.runPromise(Fiber.await(fiber))
    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure") expect(Cause.isInterruptedOnly(exit.cause)).toBe(true)

    registry.set(mutationAtom, Atom.Reset)
    expect(registry.get(mutationAtom)._tag).toBe("Initial")
    registry.dispose()
  })

  it("retries commands and collects settled mutation states", async () => {
    const registry = Registry.make()
    const client = clientFor(registry)
    const effectQuery = Client.make(Layer.empty)
    let calls = 0
    const mutation = Mutation.make("RetryingMutation", {
      effect: () => ++calls === 1 ? Effect.fail("transient" as const) : Effect.succeed("done"),
      retry: Schedule.recurs(1),
      gcTime: 0
    })
    const mutationAtom = effectQuery.mutation(mutation)

    expect(await Effect.runPromise(
      Mutation.execute(mutationAtom, undefined).pipe(
        Effect.provideService(Registry.AtomRegistry, registry)
      )
    )).toBe("done")
    expect(calls).toBe(2)
    await Effect.runPromise(Effect.sleep("1 millis"))
    expect(registry.get(client.mutationState(mutation.match()))).toEqual([])
    registry.dispose()
  })
})

describe("type propagation", () => {
  it("preserves associated query and mutation types", () => {
    const query = Query.make("Typed", {
      key: (input: { readonly id: number }) => Data.struct(input),
      effect: () => Effect.fail("query-error" as const) as Effect.Effect<string, "query-error">
    })
    const mutation = Mutation.make("TypedMutation", {
      effect: (input: number) => Effect.succeed(String(input)),
      synchronize: () => Effect.fail("sync-error" as const)
    })

    expectTypeOf<Query.Input<typeof query>>().toEqualTypeOf<{ readonly id: number }>()
    expectTypeOf<Query.Data<typeof query>>().toEqualTypeOf<string>()
    expectTypeOf<Query.Error<typeof query>>().toEqualTypeOf<"query-error">()
    expectTypeOf<Mutation.Input<typeof mutation>>().toEqualTypeOf<number>()
    expectTypeOf<Mutation.Output<typeof mutation>>().toEqualTypeOf<string>()
    expectTypeOf<Mutation.SynchronizationError<typeof mutation>>().toEqualTypeOf<"sync-error">()
    expectTypeOf<Mutation.State<typeof mutation>["input"]>().toEqualTypeOf<number>()
    expectTypeOf<Mutation.State<typeof mutation>["result"]>().toEqualTypeOf<
      AtomResult.Result<string, Mutation.MutationSynchronizationError<string, "sync-error">>
    >()
  })

  it("discharges only requirements supplied by the bound runtime", () => {
    class RequiredService extends Context.Tag("test/RequiredService")<RequiredService, {
      readonly value: string
    }>() {}
    const client = Client.make(Layer.succeed(RequiredService, { value: "ok" }))
    const query = Query.make("Required", {
      key: () => Data.struct({ singleton: true }),
      effect: () => Effect.map(RequiredService, (service) => service.value)
    })

    expectTypeOf<Query.Requirements<typeof query>>().toEqualTypeOf<RequiredService>()

    class RuntimeLayerError extends Data.TaggedError("RuntimeLayerError")<{}> {}
    const failingClient = Client.make(Layer.effect(RequiredService, Effect.fail(new RuntimeLayerError())))
    const failedQuery = Query.make("LayerFailure", {
      key: () => Data.struct({ singleton: true }),
      effect: () => RequiredService.pipe(
        Effect.zipRight(Effect.fail("domain-error" as const))
      )
    })
    const failedAtom = failingClient.query(failedQuery, undefined)
    expectTypeOf<Query.Error<typeof failedQuery>>().toEqualTypeOf<"domain-error">()
    expectTypeOf<Query.QueryAtom.Error<typeof failedAtom>>()
      .toEqualTypeOf<"domain-error" | RuntimeLayerError>()

    class SynchronizationService extends Context.Tag("test/SynchronizationService")<
      SynchronizationService,
      { readonly synchronize: Effect.Effect<void, "sync-error"> }
    >() {}
    const mutationClient = Client.make(Layer.merge(
      Layer.succeed(RequiredService, { value: "ok" }),
      Layer.succeed(SynchronizationService, { synchronize: Effect.fail("sync-error" as const) })
    ))
    const mutation = Mutation.make("RequiredMutation", {
      effect: () => Effect.map(RequiredService, (service) => service.value),
      synchronize: () => Effect.flatMap(SynchronizationService, (service) => service.synchronize)
    })
    expectTypeOf<Mutation.Requirements<typeof mutation>>().toEqualTypeOf<
      RequiredService | SynchronizationService
    >()
    expectTypeOf<Mutation.CommandError<typeof mutation>>().toEqualTypeOf<never>()
    expectTypeOf<Mutation.SynchronizationError<typeof mutation>>().toEqualTypeOf<"sync-error">()
    void client
    void mutationClient
  })
})
