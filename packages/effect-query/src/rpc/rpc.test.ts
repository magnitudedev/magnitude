import * as Registry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import { RpcClientError, RpcTest } from "@effect/rpc"
import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
import * as Option from "effect/Option"
import * as Schema from "effect/Schema"
import * as Stream from "effect/Stream"
import { describe, expect, expectTypeOf, it } from "vitest"
import { Client, Group, Mutation, Operation, Query, QueryClient, Subscription } from "../index.js"
import * as RpcAdapter from "./index.js"

class NotFound extends Schema.TaggedError<NotFound>()("NotFound", { id: Schema.String }) {}

const TestRpc = RpcAdapter.make()

const GetUser = Query.make("GetUser", {
  payload: { id: Schema.String },
  success: Schema.Struct({ id: Schema.String, name: Schema.String }),
  error: NotFound,
  staleTime: Infinity,
})

const Rename = Mutation.make("Rename", {
  payload: { id: Schema.String, name: Schema.String },
  success: Schema.Struct({ id: Schema.String, name: Schema.String }),
  error: NotFound,
  scope: ({ id }) => Mutation.MutationScope(`user:${id}`),
  synchronize: (_, { id }) => QueryClient.invalidate(GetUser.match({ id })).pipe(
    Effect.zipRight(QueryClient.fetch(GetUser, { id })),
    Effect.asVoid,
  ),
})

const Changes = Subscription.make("Changes", {
  payload: {},
  success: Schema.Struct({ query: Schema.String }),
})

const Counter = Query.fromStream("Counter", {
  payload: { start: Schema.Number },
  success: Schema.Number,
  reduce: (previous: Option.Option<number>, event: number) => Option.getOrElse(previous, () => 0) + event,
})

const TestOperations = Group.make({ GetUser, Rename, Changes, Counter })
const RpcOperations = TestRpc.toRpcGroup(TestOperations)

const users = new Map<string, string>([["1", "Ada"]])

const handlers = TestRpc.toLayer(TestOperations, {
  GetUser: ({ id }) => {
    const name = users.get(id)
    return name === undefined ? Effect.fail(new NotFound({ id })) : Effect.succeed({ id, name })
  },
  Rename: ({ id, name }) => Effect.sync(() => {
    users.set(id, name)
    return { id, name }
  }),
  Changes: () => Stream.make({ query: "GetUser" }, { query: "Other" }),
  Counter: ({ start }) => Stream.make(start, 1, 2),
})

const implementationLayer = Layer.scoped(
  Operation.implementationsTag<Group.Declaration<typeof TestOperations>, RpcClientError.RpcClientError>(),
  Effect.map(
    RpcTest.makeClient(RpcOperations, { flatten: true }),
    (client) => TestRpc.implementations(TestOperations, client),
  ),
).pipe(Layer.provide(handlers))

const sleep = (millis: number) => Effect.runPromise(Effect.sleep(`${millis} millis`))

describe("effect-query/rpc", () => {
  it("rejects Effect-backed definitions at the transport projection", () => {
    const Local = Mutation.make("Local", { effect: () => Effect.void })
    const mixed = Group.make({ GetUser, Local })

    // @ts-expect-error Mixed application graphs are not transport boundaries.
    expect(() => TestRpc.toRpcGroup(mixed)).toThrow("only declared operations")
  })

  it("recursively composes base primitives and exposes derived metadata", () => {
    const Reads = Group.make({ GetUser, Counter })
    const Writes = Group.make({ Rename })
    const Nested = Group.make({ Reads, Writes, Changes })

    expect(TestRpc.operations(Nested).map(({ name, kind }) => [name, kind])).toEqual([
      ["GetUser", "query"],
      ["Counter", "queryFromStream"],
      ["Rename", "mutation"],
      ["Changes", "subscription"],
    ])
    expect(TestRpc.operation(Nested, "Changes")?.stream).toBe(true)
  })

  it("rejects duplicate wire names across nested groups", () => {
    const AlsoGetUser = Query.make("GetUser", {
      payload: { id: Schema.String },
      success: Schema.String,
    })
    const Reads = Group.make({ GetUser })
    expect(() => Group.make({ Reads, AlsoGetUser })).toThrow("Duplicate operation GetUser")
  })

  it("executes a base Query through the derived Rpc transport", async () => {
    const registry = Registry.make()
    const client = Client.make(implementationLayer)
    const atom = client.query(GetUser, { id: "1" })
    expect(client.query(GetUser, { id: "1" })).toBe(atom)
    const unmount = registry.mount(atom)
    await sleep(10)
    expect(AtomResult.value(registry.get(atom).result)).toEqual(Option.some({ id: "1", name: "Ada" }))
    unmount()
    registry.dispose()
  })

  it("surfaces typed Rpc errors", async () => {
    const registry = Registry.make()
    const client = Client.make(implementationLayer)
    const atom = client.query(GetUser, { id: "missing" })
    const unmount = registry.mount(atom)
    await sleep(10)
    const result = registry.get(atom).result
    expect(result._tag).toBe("Failure")
    if (result._tag === "Failure") expect(result.cause.toString()).toContain("NotFound")
    unmount()
    registry.dispose()
  })

  it("runs base Mutations with synchronization", async () => {
    const registry = Registry.make()
    const client = Client.make(implementationLayer)
    const query = client.query(GetUser, { id: "1" })
    const unmount = registry.mount(query)
    await sleep(10)
    const renamed = await Effect.runPromise(
      Mutation.execute(client.mutation(Rename), { id: "1", name: "Grace" }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry),
      ),
    )
    expect(renamed).toEqual({ id: "1", name: "Grace" })
    await sleep(10)
    expect(AtomResult.value(registry.get(query).result)).toEqual(Option.some({ id: "1", name: "Grace" }))
    unmount()
    registry.dispose()
  })

  it("materializes base Subscriptions over derived stream Rpcs", async () => {
    const registry = Registry.make()
    const client = Client.make(implementationLayer)
    const events = await Effect.runPromise(
      Subscription.events(client.subscription(Changes, {})).pipe(
        Stream.take(2),
        Stream.runCollect,
        Effect.provideService(Registry.AtomRegistry, registry),
      ),
    )
    expect(Array.from(events)).toEqual([{ query: "GetUser" }, { query: "Other" }])
    registry.dispose()
  })

  it("folds a derived stream Rpc through a base Query", async () => {
    const registry = Registry.make()
    const client = Client.make(implementationLayer)
    const atom = client.query(Counter, { start: 10 })
    const unmount = registry.mount(atom)
    await sleep(20)
    expect(AtomResult.value(registry.get(atom).result)).toEqual(Option.some(13))
    unmount()
    registry.dispose()
  })

  it("keeps primitive kinds apart at the type level", () => {
    const client = Client.make(implementationLayer)
    const typeOnly = () => {
      // @ts-expect-error a mutation is not a query
      client.query(Rename, { id: "1", name: "x" })
      // @ts-expect-error a query is not a mutation
      client.mutation(GetUser)
      // @ts-expect-error a query is not a subscription
      client.subscription(GetUser, { id: "1" })
    }
    expectTypeOf(typeOnly).toBeFunction()
    expectTypeOf<RpcAdapter.GroupRpcs<typeof TestOperations>["_tag"]>()
      .toEqualTypeOf<"GetUser" | "Rename" | "Changes" | "Counter">()
  })
})

describe("effect-query/rpc group client", () => {
  const Reads = Group.make({ GetUser, Counter })
  const Writes = Group.make({ Rename })
  const Boundary = Group.make({ Reads, Writes, Changes })

  it("materializes every member at its name with the materializer's exact type", () => {
    const client = Client.make(Boundary, implementationLayer)
    expectTypeOf(client.Reads.GetUser).parameter(0).toEqualTypeOf<{ readonly id: string }>()
    expectTypeOf(client.Reads.GetUser({ id: "1" })).toEqualTypeOf(client.query(GetUser, { id: "1" }))
    expectTypeOf(client.Reads.Counter({ start: 1 })).toEqualTypeOf(client.query(Counter, { start: 1 }))
    expectTypeOf(client.Writes.Rename).toEqualTypeOf(client.mutation(Rename))
    expectTypeOf(client.Changes({})).toEqualTypeOf(client.subscription(Changes, {}))
    const typeOnly = () => {
      // @ts-expect-error wrong payload
      client.Reads.GetUser({ id: 1 })
      // @ts-expect-error a mutation member is the atom, not a function
      client.Writes.Rename()
      // @ts-expect-error a member named like a materializer does not compile
      Client.make(Group.make({ query: GetUser }), implementationLayer)
      // @ts-expect-error an operation requiring an unprovided service does not compile
      Client.make(Boundary, Layer.empty)
    }
    expectTypeOf(typeOnly).toBeFunction()
  })

  it("returns the canonical atom through members and materializers alike", () => {
    const client = Client.make(Boundary, implementationLayer)
    expect(client.Reads.GetUser({ id: "1" })).toBe(client.query(GetUser, { id: "1" }))
    expect(client.Reads.Counter({ start: 1 })).toBe(client.query(Counter, { start: 1 }))
    expect(client.Writes.Rename).toBe(client.mutation(Rename))
    expect(client.Writes.Rename).toBe(client.Writes.Rename)
    expect(client.Changes({})).toBe(client.subscription(Changes, {}))
    expect(Object.isFrozen(client.Reads)).toBe(true)
  })

  it("rejects a member named like a client property at runtime", () => {
    const Colliding = Group.make({ mutation: GetUser })
    expect(() => Client.make(Colliding as never, implementationLayer)).toThrow("collides with a client property")
  })

  it("executes through members", async () => {
    const registry = Registry.make()
    const client = Client.make(Boundary, implementationLayer)
    const user = client.Reads.GetUser({ id: "1" })
    const unmount = registry.mount(user)
    await sleep(10)
    expect(Option.isSome(AtomResult.value(registry.get(user).result))).toBe(true)
    const renamed = await Effect.runPromise(
      Mutation.execute(client.Writes.Rename, { id: "1", name: "Linus" }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry),
      ),
    )
    expect(renamed).toEqual({ id: "1", name: "Linus" })
    await sleep(10)
    expect(AtomResult.value(registry.get(user).result)).toEqual(Option.some({ id: "1", name: "Linus" }))
    const events = await Effect.runPromise(
      Subscription.events(client.Changes({})).pipe(
        Stream.take(2),
        Stream.runCollect,
        Effect.provideService(Registry.AtomRegistry, registry),
      ),
    )
    expect(Array.from(events)).toEqual([{ query: "GetUser" }, { query: "Other" }])
    unmount()
    registry.dispose()
  })

  it("hands the group client to the additional Layer", async () => {
    class Seen extends Effect.Tag("Seen")<Seen, { readonly rename: unknown }>() {}
    const client = Client.make<typeof Boundary, Operation.Implementations<Operation.Name<Group.Declaration<typeof Boundary>>, RpcClientError.RpcClientError>, never, Seen>(
      Boundary,
      implementationLayer,
      (client) => Layer.effect(Seen, Effect.sync(() => ({ rename: client.Writes.Rename }))),
    )
    const registry = Registry.make()
    const seen = await Effect.runPromise(Registry.getResult(registry, client.runtime.atom(Seen)))
    expect(seen.rename).toBe(client.mutation(Rename))
    registry.dispose()
  })
})
