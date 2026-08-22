import * as Registry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import { RpcGroup, RpcTest } from "@effect/rpc"
import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
import * as Option from "effect/Option"
import * as Schema from "effect/Schema"
import * as Stream from "effect/Stream"
import { describe, expect, expectTypeOf, it } from "vitest"
import { Client, Mutation, QueryClient, Subscription } from "../index.js"
import * as Boundary from "./index.js"

class NotFound extends Schema.TaggedError<NotFound>()("NotFound", { id: Schema.String }) {}

const Test = Boundary.make("Test")

const GetUser = Test.query("GetUser", {
  payload: { id: Schema.String },
  success: Schema.Struct({ id: Schema.String, name: Schema.String }),
  error: NotFound,
  staleTime: Infinity
})

const Rename = Test.mutation("Rename", {
  payload: { id: Schema.String, name: Schema.String },
  success: Schema.Struct({ id: Schema.String, name: Schema.String }),
  error: NotFound,
  scope: ({ id }) => Mutation.MutationScope(`user:${id}`),
  synchronize: (_, { id }) => QueryClient.invalidate(GetUser.match({ id })).pipe(
    Effect.zipRight(QueryClient.fetch(GetUser, { id })),
    Effect.asVoid
  )
})

const Changes = Test.subscription("Changes", {
  payload: {},
  success: Schema.Struct({ query: Schema.String })
})

const Counter = Test.queryFromStream("Counter", {
  payload: { start: Schema.Number },
  success: Schema.Number,
  reduce: (previous: Option.Option<number>, event: number) => Option.getOrElse(previous, () => 0) + event
})

const Group = RpcGroup.make(GetUser.rpc, Rename.rpc, Changes.rpc, Counter.rpc)

const users = new Map<string, string>([["1", "Ada"]])

const handlers = Group.toLayer({
  GetUser: ({ id }) => {
    const name = users.get(id)
    return name === undefined ? Effect.fail(new NotFound({ id })) : Effect.succeed({ id, name })
  },
  Rename: ({ id, name }) => Effect.sync(() => {
    users.set(id, name)
    return { id, name }
  }),
  Changes: () => Stream.make({ query: "GetUser" }, { query: "Other" }),
  Counter: ({ start }) => Stream.make(start, 1, 2)
})

const transportLayer = Layer.scoped(
  Test.Client,
  Effect.map(RpcTest.makeClient(Group, { flatten: true }), (client) => Test.transport(client))
).pipe(Layer.provide(handlers))

const sleep = (millis: number) => Effect.runPromise(Effect.sleep(`${millis} millis`))

describe("effect-query/rpc", () => {
  it("defines queries whose identity and execution derive from the Rpc", async () => {
    const registry = Registry.make()
    const client = Client.make(transportLayer)
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
    const client = Client.make(transportLayer)
    const atom = client.query(GetUser, { id: "missing" })
    const unmount = registry.mount(atom)
    await sleep(10)
    const result = registry.get(atom).result
    expect(result._tag).toBe("Failure")
    if (result._tag === "Failure") {
      expect(result.cause.toString()).toContain("NotFound")
    }
    unmount()
    registry.dispose()
  })

  it("runs mutations with contract-level scope and synchronization", async () => {
    const registry = Registry.make()
    const client = Client.make(transportLayer)
    const query = client.query(GetUser, { id: "1" })
    const unmount = registry.mount(query)
    await sleep(10)
    const renamed = await Effect.runPromise(
      Mutation.execute(client.mutation(Rename), { id: "1", name: "Grace" }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry)
      )
    )
    expect(renamed).toEqual({ id: "1", name: "Grace" })
    await sleep(10)
    expect(AtomResult.value(registry.get(query).result)).toEqual(Option.some({ id: "1", name: "Grace" }))
    unmount()
    registry.dispose()
  })

  it("materializes subscriptions over stream Rpcs", async () => {
    const registry = Registry.make()
    const client = Client.make(transportLayer)
    const events = await Effect.runPromise(
      Subscription.events(client.subscription(Changes, {})).pipe(
        Stream.take(2),
        Stream.runCollect,
        Effect.provideService(Registry.AtomRegistry, registry)
      )
    )
    expect(Array.from(events)).toEqual([{ query: "GetUser" }, { query: "Other" }])
    registry.dispose()
  })

  it("folds stream Rpcs into query data", async () => {
    const registry = Registry.make()
    const client = Client.make(transportLayer)
    const atom = client.query(Counter, { start: 10 })
    const unmount = registry.mount(atom)
    await sleep(20)
    expect(AtomResult.value(registry.get(atom).result)).toEqual(Option.some(13))
    unmount()
    registry.dispose()
  })

  it("keeps kinds apart at the type level", () => {
    const client = Client.make(transportLayer)
    const typeOnly = () => {
      // @ts-expect-error a mutation is not a query
      client.query(Rename, { id: "1", name: "x" })
      // @ts-expect-error a query is not a mutation
      client.mutation(GetUser)
      // @ts-expect-error a query is not a subscription
      client.subscription(GetUser, { id: "1" })
    }
    expectTypeOf(typeOnly).toBeFunction()
    expectTypeOf(GetUser.rpc._tag).toEqualTypeOf<"GetUser">()
  })
})
