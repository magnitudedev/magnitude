import { describe, expect, it } from "vitest"
import {
  AcnInstanceIdSchema,
  AcnRpc,
  type AcnInstance,
  AcnReady,
  AcnBoundary,
  ModelSlotUnassigned,
  PRIMARY_SLOT_ID,
  ProcessStartIdentitySchema,
  AcnRevisionSchema,
  SECONDARY_SLOT_ID,
} from "@magnitudedev/acn-protocol"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientError from "@effect/platform/HttpClientError"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import {
  Deferred,
  Duration,
  Effect,
  Exit,
  Fiber,
  Layer,
  Option,
  Schema,
  Stream,
  TestClock,
  TestContext,
} from "effect"
import { AcnEnsuranceFailed } from "./errors"
import { AcnInstanceManager } from "./acn-instance-manager"
import { makeAcnConnection } from "./acn-recovering-client"
import { SDK_VERSION } from "../version"

type ReadyInstance = AcnInstance<AcnReady>

const ready: ReadyInstance = {
  revision: AcnRevisionSchema.make(1_000_000),
  id: AcnInstanceIdSchema.make("ready-acn"),
  identity: SDK_VERSION,
  url: "http://ready-acn",
  pid: 123,
  processStartIdentity: ProcessStartIdentitySchema.make("ready-process"),
  lifecycle: new AcnReady({}),
}

const bodyText = (request: Parameters<typeof HttpClientResponse.fromWeb>[0]): string => {
  const body = request.body
  if (body._tag === "Uint8Array") return new TextDecoder().decode(body.body)
  if (body._tag === "Raw" && typeof body.body === "string") return body.body
  throw new Error(`Unexpected request body ${body._tag}`)
}

const rpcClient = (
  tags: string[],
  instances: ReadonlyArray<ReadyInstance> = [ready],
  refuseHealthFor: ReadonlySet<string> = new Set(),
  failRpcTags: ReadonlySet<string> = new Set(),
) => HttpClient.make((request) => Effect.suspend(() => {
  const message = JSON.parse(bodyText(request).split("\n")[0]!) as { id: string; tag: string }
  tags.push(message.tag)
  const instance = instances.find((candidate) => request.url.startsWith(candidate.url)) ?? instances[0]!
  if (message.tag === "Health" && refuseHealthFor.has(instance.id)) {
    return Effect.fail(new HttpClientError.RequestError({
      request,
      reason: "Transport",
      cause: new Error("connection refused"),
    }))
  }
  if (AcnRpc.operation(AcnBoundary, message.tag) === undefined) throw new Error(`Unknown RPC ${message.tag}`)
  const success = message.tag === "Health"
    ? {
        service: "magnitude-acn" as const,
        revision: instance.revision,
        version: instance.identity,
        id: instance.id,
        pid: instance.pid,
        state: instance.lifecycle,
      }
    : message.tag === "GetModelSlots"
          ? {
              revision: 0,
              state: {
                slots: {
                  primary: new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID }),
                  secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
                },
                recentModels: { primary: [], secondary: [] },
                favoriteModels: [],
              },
            }
          : undefined
  if (success === undefined) throw new Error(`No response for ${message.tag}`)
  const rpcExit = failRpcTags.has(message.tag)
    ? Exit.die(`Simulated ${message.tag} failure`)
    : Exit.succeed(success)
  const exit = Schema.encodeUnknownSync(Schema.Exit({
    success: Schema.Unknown,
    failure: Schema.Unknown,
    defect: Schema.Defect,
  }))(rpcExit)
  return Effect.succeed(HttpClientResponse.fromWeb(request, new Response(`${JSON.stringify({
    _tag: "Exit",
    requestId: message.id,
    exit,
  })}\n`, { status: 200 })))
}))

describe("AcnConnection", () => {
  it("publishes no visible startup observation before an already-ready selection", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const connection = yield* makeAcnConnection().pipe(
        Effect.provideService(AcnInstanceManager, AcnInstanceManager.of({
          ensure: () => Stream.succeed({ _tag: "Ready" as const, instance: ready }),
          stop: Effect.void,
        })),
        Effect.provideService(HttpClient.HttpClient, rpcClient([])),
      )
      yield* connection.startup.awaitReady
      expect((yield* connection.startup.state.get)._tag).toBe("Ready")
      const states = yield* connection.startup.state.changes.pipe(
        Stream.take(1),
        Stream.runCollect,
      )
      expect(Array.from(states).map((state) => state._tag)).toEqual(["Ready"])
    })))
  })

  it("single-flights bootstrap and retry", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const release = yield* Deferred.make<void>()
      let calls = 0
      const manager = AcnInstanceManager.of({
        ensure: () => Stream.unwrap(Effect.gen(function* () {
          calls += 1
          yield* Deferred.await(release)
          return Stream.succeed({ _tag: "Ready" as const, instance: ready })
        })),
        stop: Effect.void,
      })
      const tags: string[] = []
      const connection = yield* makeAcnConnection().pipe(
        Effect.provideService(AcnInstanceManager, manager),
        Effect.provideService(HttpClient.HttpClient, rpcClient(tags)),
      )
      const retry = yield* connection.startup.retry.pipe(Effect.fork)
      while (calls === 0) yield* Effect.sleep(Duration.millis(1))
      expect(calls).toBe(1)
      yield* Deferred.succeed(release, undefined)
      const joined = yield* Fiber.join(retry).pipe(Effect.timeoutOption("1 second"))
      expect(Option.isSome(joined), `calls=${calls} tags=${tags.join(",")}`).toBe(true)
      expect(calls).toBe(1)
    })))
  })

  it("scope finalization interrupts selection before readiness", async () => {
    const tags: string[] = []
    await Effect.runPromise(Effect.scoped(makeAcnConnection().pipe(
      Effect.provideService(AcnInstanceManager, AcnInstanceManager.of({
        ensure: () => Stream.never,
        stop: Effect.void,
      })),
      Effect.provideService(HttpClient.HttpClient, rpcClient(tags)),
      Effect.asVoid,
    )))
    expect(tags).toHaveLength(0)
  })

  it("closes an admitted connection from its owning scope finalizer", async () => {
    const tags: string[] = []
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const connection = yield* makeAcnConnection().pipe(
        Effect.provideService(AcnInstanceManager, AcnInstanceManager.of({
          ensure: () => Stream.succeed({ _tag: "Ready" as const, instance: ready }),
          stop: Effect.void,
        })),
        Effect.provideService(HttpClient.HttpClient, rpcClient(tags)),
      )

      // The CLI startup scope owns the connection this way. Accepting an update exits
      // its scope without calling close first, so this finalizer performs the
      // first close after the owning scope has begun finalization.
      yield* Effect.addFinalizer(() => connection.close.pipe(Effect.asVoid))
      yield* connection.startup.retry
    })))

    expect(tags.filter((tag) => tag === "GetModelSlots")).toHaveLength(0)
  })

  it("close interrupts initial selection", async () => {
    const tags: string[] = []
    let entered = 0
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const connection = yield* makeAcnConnection().pipe(
        Effect.provideService(AcnInstanceManager, AcnInstanceManager.of({
          ensure: () => Stream.unwrap(Effect.sync(() => {
            entered += 1
            return Stream.never
          })),
          stop: Effect.void,
        })),
        Effect.provideService(HttpClient.HttpClient, rpcClient(tags)),
      )
      while (entered === 0) yield* Effect.sleep(Duration.millis(1))
      yield* connection.close
      expect(tags).toHaveLength(0)
    })))
  })

  it("shares one deterministic ensurance failure with every concurrent waiter", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const release = yield* Deferred.make<void>()
      let calls = 0
      const manager = AcnInstanceManager.of({
        ensure: () => Stream.unwrap(Effect.gen(function* () {
          calls += 1
          yield* Deferred.await(release)
          return Stream.fail(new AcnEnsuranceFailed({ reason: "deterministic failure" }))
        })),
        stop: Effect.void,
      })
      const connection = yield* makeAcnConnection().pipe(
        Effect.provideService(AcnInstanceManager, manager),
        Effect.provideService(HttpClient.HttpClient, rpcClient([])),
      )
      const waiters = yield* Effect.all([
        connection.startup.retry.pipe(Effect.exit),
        connection.startup.retry.pipe(Effect.exit),
        connection.startup.retry.pipe(Effect.exit),
      ], { concurrency: "unbounded" }).pipe(Effect.fork)
      while (calls === 0) yield* Effect.sleep(Duration.millis(1))
      yield* Effect.yieldNow()
      yield* Deferred.succeed(release, undefined)
      const exits = yield* Fiber.join(waiters)
      expect(exits.every(Exit.isFailure)).toBe(true)
      expect(calls).toBe(1)
    })))
  })

  it("terminalizes one selection when its manager never resolves", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const entered = yield* Deferred.make<void>()
      const manager = AcnInstanceManager.of({
        ensure: () => Stream.unwrap(
          Deferred.succeed(entered, undefined).pipe(Effect.as(Stream.never)),
        ),
        stop: Effect.void,
      })
      const connection = yield* makeAcnConnection().pipe(
        Effect.provideService(AcnInstanceManager, manager),
        Effect.provideService(HttpClient.HttpClient, rpcClient([])),
      )
      const waiting = yield* connection.startup.retry.pipe(Effect.exit, Effect.fork)
      yield* Deferred.await(entered)
      yield* TestClock.adjust(Duration.minutes(10))
      expect(Exit.isFailure(yield* Fiber.join(waiting))).toBe(true)
      expect((yield* connection.startup.state.get)._tag).toBe("Failed")
    })).pipe(Effect.provide(TestContext.TestContext)))
  })

  it("recovers an application RPC by replacing only its failed exact endpoint", async () => {
    const successor: ReadyInstance = {
      ...ready,
      id: AcnInstanceIdSchema.make("successor-acn"),
      url: "http://successor-acn",
      pid: 456,
      processStartIdentity: ProcessStartIdentitySchema.make("successor-process"),
    }
    let ensures = 0
    const manager = AcnInstanceManager.of({
      ensure: () => Stream.sync(() => ({
        _tag: "Ready" as const,
        instance: ensures++ === 0 ? ready : successor,
      })),
      stop: Effect.void,
    })
    const tags: string[] = []
    const http = rpcClient(tags, [ready, successor], new Set([ready.id]))
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const connection = yield* makeAcnConnection().pipe(
        Effect.provideService(AcnInstanceManager, manager),
        Effect.provideService(HttpClient.HttpClient, http),
      )
      yield* connection.startup.retry
      expect((yield* connection.startup.state.get)._tag).toBe("Ready")
      const client = yield* AcnRpc.makeRpcClient(AcnBoundary).pipe(
        Effect.provide(connection.protocolLayer.pipe(
          Layer.provide(Layer.succeed(HttpClient.HttpClient, http)),
        )),
      )
      const recoveryStates = yield* connection.startup.recovery.changes.pipe(
        Stream.take(3),
        Stream.runCollect,
        Effect.fork,
      )
      yield* Effect.yieldNow()
      const health = yield* client.Health({})
      expect(health.id).toBe(successor.id)
      expect(ensures).toBe(2)
      expect(tags.filter((tag) => tag === "Health")).toHaveLength(2)
      expect((yield* connection.startup.state.get)._tag).toBe("Ready")
      expect(Array.from(yield* Fiber.join(recoveryStates)).map((state) => state._tag)).toEqual([
        "Inactive",
        "Recovering",
        "Recovered",
      ])
    })))
  })

  it("closes idempotently without an RPC and never ensures again", async () => {
    const tags: string[] = []
    let ensures = 0
    const manager = AcnInstanceManager.of({
      ensure: () => Stream.sync(() => {
        ensures += 1
        return { _tag: "Ready" as const, instance: ready }
      }),
      stop: Effect.void,
    })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const connection = yield* makeAcnConnection().pipe(
        Effect.provideService(AcnInstanceManager, manager),
        Effect.provideService(HttpClient.HttpClient, rpcClient(
          tags,
          [ready],
          new Set(),
          new Set(["GetModelSlots"]),
        )),
      )
      yield* connection.startup.retry
      yield* connection.close
      yield* connection.close
      expect(tags).toHaveLength(0)
      expect(Exit.isFailure(yield* Effect.exit(connection.startup.retry))).toBe(true)
      expect(ensures).toBe(1)
    })))
  })
})
