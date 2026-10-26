import { FetchHttpClient, HttpServer } from "@effect/platform"
import * as HttpLayerRouter from "@effect/platform/HttpLayerRouter"
import { BunHttpServer } from "@effect/platform-bun"
import { RpcClient, RpcGroup, RpcSerialization, RpcServer } from "@effect/rpc"
import { Context, Effect, Fiber, Layer, Option, Queue, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { AcnRpcRecoveryPolicyTag, Chat, type ChatUpdate, type SessionId } from "@magnitudedev/acn-protocol"
import { acnSubscriptionProtocol } from "../../../sdk/src/acn-jit/acn-subscription-protocol"
import { isInterruptedExit, makeRecoveringProtocol } from "../../../sdk/src/jit-rpc"
import { makeAcnSubscriptionProtocol } from "../acn-subscription-protocol"
import { AcnSubscriptionsLive } from "../acn-subscriptions"
import { ACN_INSTANCE_ID } from "../identity"
import { acnRpcServerOptions } from "../server"
import { newChat, payload, serviceFixture, text } from "./chat.test-support"
import { ChatService } from "./service"

// A chat watch over the real wire: ACN's subscription protocol on the server, the SDK's recovering
// protocol on the client. A subscription never ends on its own there; only the server's terminal
// control at shutdown ends one, and any other clean exit is a protocol violation.
//
// The client half is imported from the SDK's sources by path, in this test only. The SDK is ACN's
// client, so ACN never depends on it, not even for tests: a dependency would invert the layering,
// and the SDK keeps these protocol internals private. Neither package can hold a test of both
// halves through the other's public surface, and the pairing is what is tested, so the path
// reaches the exact code the SDK runs, and nothing shipped imports it.

const WatchRpcs = RpcGroup.make(Chat.watch, Chat.remove)

const isRemoved = (update: ChatUpdate) => update._tag === "removed"

/** Serves the chat watch and removal through ACN's subscription protocol, and returns a client of them. */
const connect = (watch: (chats: ChatService, chatId: SessionId) => ReturnType<ChatService["watch"]>) => Effect.gen(function* () {
  const chats = yield* ChatService
  const router = yield* HttpLayerRouter.make
  const raw = yield* RpcServer.makeProtocolHttpRouter({ path: "/rpc" }).pipe(
    Effect.provideService(HttpLayerRouter.HttpRouter, router),
    Effect.provide(RpcSerialization.layerNdjson),
  )
  const protocol = yield* makeAcnSubscriptionProtocol(raw)
  yield* RpcServer.make(WatchRpcs, acnRpcServerOptions).pipe(
    Effect.provide(WatchRpcs.toLayer({
      WatchChat: ({ chatId }) => watch(chats, chatId),
      RemoveChat: ({ chatId }) => chats.remove(chatId),
    })),
    Effect.provideService(RpcServer.Protocol, protocol),
    Effect.forkScoped,
  )
  const infrastructure = yield* Layer.build(BunHttpServer.layer({ hostname: "127.0.0.1", port: 0, idleTimeout: 0 }))
  const server = Context.get(infrastructure, HttpServer.HttpServer)
  yield* server.serve(router.asHttpEffect()).pipe(Effect.provide(infrastructure))
  if (server.address._tag !== "TcpAddress") return yield* Effect.dieMessage("Expected TCP")
  const endpoint = { id: ACN_INSTANCE_ID, url: `http://127.0.0.1:${server.address.port}` }

  const client = yield* makeRecoveringProtocol({
    endpoint: Effect.succeed(endpoint),
    recover: () => Effect.succeed(endpoint),
    rpcPath: "/rpc",
    streamProtocol: acnSubscriptionProtocol,
    isEndpointRetirementExit: isInterruptedExit,
    recoveryPolicy: (tag) => Option.getOrThrow(Context.getOption(WatchRpcs.requests.get(tag)!.annotations, AcnRpcRecoveryPolicyTag)),
    classifyInfraError: (error: never) => error,
  }).pipe(Effect.provide([RpcSerialization.layerNdjson, FetchHttpClient.layer]))
  return yield* RpcClient.make(WatchRpcs).pipe(Effect.provideService(RpcClient.Protocol, client))
}).pipe(Effect.provide(AcnSubscriptionsLive))

/** Starts a chat whose response runs until it is removed, and follows its watch over the wire. */
const watchedOverTheWire = (watch: Parameters<typeof connect>[0]) => Effect.gen(function* () {
  const service = yield* ChatService
  const client = yield* connect(watch)
  const chatId = newChat()
  yield* service.start(payload(chatId))
  const received = yield* Queue.unbounded<ChatUpdate>()
  const watcher = yield* client.WatchChat({ chatId }).pipe(
    Stream.runForEach((update) => Queue.offer(received, update)),
    Effect.forkScoped,
  )
  const next = Queue.take(received).pipe(Effect.timeout("5 seconds"))
  expect((yield* next)._tag).toBe("snapshot")
  yield* client.RemoveChat({ chatId })
  return { watcher, next }
})

const endless = () => Stream.concat(Stream.make(text("Partial")), Stream.never)

describe("Chat watch over the ACN subscription protocol", () => {
  it("tells the client its chat was removed, and stays open rather than exiting without the terminal control", async () => {
    const f = await Effect.runPromise(serviceFixture(endless))
    await f.run(Effect.gen(function* () {
      const { watcher, next } = yield* watchedOverTheWire((chats, chatId) => chats.watch(chatId))
      let update = yield* next
      while (!isRemoved(update)) update = yield* next
      // An exit that broke the protocol would follow `removed` at once and fail the client.
      yield* Effect.sleep("250 millis")
      expect(yield* Fiber.poll(watcher)).toEqual(Option.none())
      yield* Fiber.interrupt(watcher)
    }))
  })

  it("fails the client with a protocol violation for a watch that completes after `removed`", async () => {
    const f = await Effect.runPromise(serviceFixture(endless))
    await f.run(Effect.gen(function* () {
      const { watcher } = yield* watchedOverTheWire((chats, chatId) => chats.watch(chatId).pipe(Stream.takeUntil(isRemoved)))
      const failure = yield* Fiber.join(watcher).pipe(Effect.flip, Effect.timeout("5 seconds"))
      expect(failure).toMatchObject({ _tag: "RpcClientError", cause: { _tag: "SubscriptionProtocolViolation" } })
    }))
  })
})
