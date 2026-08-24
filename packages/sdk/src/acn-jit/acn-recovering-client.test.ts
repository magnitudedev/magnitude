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
import { makeAcnJitRuntime } from "./acn-recovering-client"
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
    : message.tag === "RenewClientLease"
      ? { connectedClientCount: 1 }
      : message.tag === "ReleaseClientLease"
        ? { connectedClientCount: 0 }
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

describe("AcnJitRuntime", () => {
  it("single-flights bootstrap and retry, then starts one lease", async () => {
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
      const runtime = yield* makeAcnJitRuntime().pipe(
        Effect.provideService(AcnInstanceManager, manager),
        Effect.provideService(HttpClient.HttpClient, rpcClient(tags)),
      )
      const retry = yield* runtime.startup.retry.pipe(Effect.fork)
      while (calls === 0) yield* Effect.sleep(Duration.millis(1))
      expect(calls).toBe(1)
      expect(tags).not.toContain("RenewClientLease")
      yield* Deferred.succeed(release, undefined)
      const joined = yield* Fiber.join(retry).pipe(Effect.timeoutOption("1 second"))
      expect(Option.isSome(joined), `calls=${calls} tags=${tags.join(",")}`).toBe(true)
      yield* Effect.gen(function* () {
        while (!tags.includes("RenewClientLease")) yield* Effect.sleep(Duration.millis(1))
      }).pipe(Effect.timeout(Duration.seconds(1)))
      expect(calls).toBe(1)
      expect(tags.filter((tag) => tag === "RenewClientLease")).toHaveLength(1)
    })))
  })

  it("scope finalization before readiness does not create a lease", async () => {
    const tags: string[] = []
    await Effect.runPromise(Effect.scoped(makeAcnJitRuntime().pipe(
      Effect.provideService(AcnInstanceManager, AcnInstanceManager.of({
        ensure: () => Stream.never,
        stop: Effect.void,
      })),
      Effect.provideService(HttpClient.HttpClient, rpcClient(tags)),
      Effect.asVoid,
    )))
    expect(tags).not.toContain("RenewClientLease")
  })

  it("closes an admitted runtime from its owning scope finalizer", async () => {
    const tags: string[] = []
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* makeAcnJitRuntime().pipe(
        Effect.provideService(AcnInstanceManager, AcnInstanceManager.of({
          ensure: () => Stream.succeed({ _tag: "Ready" as const, instance: ready }),
          stop: Effect.void,
        })),
        Effect.provideService(HttpClient.HttpClient, rpcClient(tags)),
      )

      // TerminalPlatform owns the runtime this way. Accepting an update exits
      // its scope without calling close first, so this finalizer performs the
      // first close after the owning scope has begun finalization.
      yield* Effect.addFinalizer(() => runtime.close.pipe(Effect.asVoid))
      yield* runtime.startup.retry
    })))

    expect(tags.filter((tag) => tag === "RenewClientLease")).toHaveLength(1)
    expect(tags.filter((tag) => tag === "GetModelSlots")).toHaveLength(0)
    expect(tags.filter((tag) => tag === "ReleaseClientLease")).toHaveLength(1)
  })

  it("close interrupts initial selection without starting a lease", async () => {
    const tags: string[] = []
    let entered = 0
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* makeAcnJitRuntime().pipe(
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
      expect(Option.isNone(yield* runtime.close)).toBe(true)
      expect(tags).not.toContain("RenewClientLease")
    })))
  })

  it("serializes close with lease establishment and releases the admitted lease", async () => {
    const tags: string[] = []
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const leaseEntered = yield* Deferred.make<void>()
      const releaseRenewal = yield* Deferred.make<void>()
      const base = rpcClient(tags)
      const http = HttpClient.make((request) => {
        const message = JSON.parse(bodyText(request).split("\n")[0]!) as { tag: string }
        return message.tag === "RenewClientLease"
          ? Deferred.succeed(leaseEntered, undefined).pipe(
              Effect.zipRight(Deferred.await(releaseRenewal)),
              Effect.zipRight(base.execute(request)),
            )
          : base.execute(request)
      })
      const runtime = yield* makeAcnJitRuntime().pipe(
        Effect.provideService(AcnInstanceManager, AcnInstanceManager.of({
          ensure: () => Stream.succeed({ _tag: "Ready" as const, instance: ready }),
          stop: Effect.void,
        })),
        Effect.provideService(HttpClient.HttpClient, http),
      )
      yield* Deferred.await(leaseEntered)
      const closing = yield* runtime.close.pipe(Effect.fork)
      yield* Effect.yieldNow()
      expect(Option.isNone(yield* Fiber.poll(closing))).toBe(true)
      yield* Deferred.succeed(releaseRenewal, undefined)
      yield* Fiber.join(closing)
      expect(tags.filter((tag) => tag === "RenewClientLease")).toHaveLength(1)
      expect(tags.filter((tag) => tag === "ReleaseClientLease")).toHaveLength(1)
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
      const runtime = yield* makeAcnJitRuntime().pipe(
        Effect.provideService(AcnInstanceManager, manager),
        Effect.provideService(HttpClient.HttpClient, rpcClient([])),
      )
      const waiters = yield* Effect.all([
        runtime.startup.retry.pipe(Effect.exit),
        runtime.startup.retry.pipe(Effect.exit),
        runtime.startup.retry.pipe(Effect.exit),
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
      const runtime = yield* makeAcnJitRuntime().pipe(
        Effect.provideService(AcnInstanceManager, manager),
        Effect.provideService(HttpClient.HttpClient, rpcClient([])),
      )
      const waiting = yield* runtime.startup.retry.pipe(Effect.exit, Effect.fork)
      yield* Deferred.await(entered)
      yield* TestClock.adjust(Duration.minutes(10))
      expect(Exit.isFailure(yield* Fiber.join(waiting))).toBe(true)
      expect((yield* runtime.startup.state.get)._tag).toBe("Failed")
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
      const runtime = yield* makeAcnJitRuntime().pipe(
        Effect.provideService(AcnInstanceManager, manager),
        Effect.provideService(HttpClient.HttpClient, http),
      )
      yield* runtime.startup.retry
      expect((yield* runtime.startup.state.get)._tag).toBe("Ready")
      const client = yield* AcnRpc.makeRpcClient(AcnBoundary).pipe(
        Effect.provide(runtime.protocolLayer.pipe(
          Layer.provide(Layer.succeed(HttpClient.HttpClient, http)),
        )),
      )
      const health = yield* client.Health({})
      expect(health.id).toBe(successor.id)
      expect(ensures).toBe(2)
      expect(tags.filter((tag) => tag === "Health")).toHaveLength(2)
      expect(tags.filter((tag) => tag === "RenewClientLease")).toHaveLength(2)
      expect((yield* runtime.startup.state.get)._tag).toBe("Ready")
    })))
  })

  it("treats lease loss after readiness as convergence instead of startup failure", async () => {
    const successor: ReadyInstance = {
      ...ready,
      id: AcnInstanceIdSchema.make("lease-successor-acn"),
      url: "http://lease-successor-acn",
      pid: 789,
      processStartIdentity: ProcessStartIdentitySchema.make("lease-successor-process"),
    }
    let ensures = 0
    let refusedFirstLease = false
    const manager = AcnInstanceManager.of({
      ensure: () => Stream.succeed({
        _tag: "Ready" as const,
        instance: ensures++ === 0 ? ready : successor,
      }),
      stop: Effect.void,
    })
    const tags: string[] = []
    const healthy = rpcClient(tags, [ready, successor])
    const http = HttpClient.make((request) => {
      const message = JSON.parse(bodyText(request).split("\n")[0]!) as { tag: string }
      if (message.tag === "RenewClientLease" && request.url.startsWith(ready.url) && !refusedFirstLease) {
        refusedFirstLease = true
        return Effect.fail(new HttpClientError.RequestError({
          request,
          reason: "Transport",
          cause: new Error("owner retired before lease establishment"),
        }))
      }
      return healthy.execute(request)
    })

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* makeAcnJitRuntime().pipe(
        Effect.provideService(AcnInstanceManager, manager),
        Effect.provideService(HttpClient.HttpClient, http),
      )
      yield* runtime.startup.retry.pipe(Effect.timeout(Duration.seconds(2)))
      expect((yield* runtime.startup.state.get)._tag).toBe("Ready")
      expect(ensures).toBe(2)
      expect(refusedFirstLease).toBe(true)
    })))
  })

  it("releases one lease without a slot observation and never ensures again", async () => {
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
      const runtime = yield* makeAcnJitRuntime().pipe(
        Effect.provideService(AcnInstanceManager, manager),
        Effect.provideService(HttpClient.HttpClient, rpcClient(
          tags,
          [ready],
          new Set(),
          new Set(["GetModelSlots"]),
        )),
      )
      yield* runtime.startup.retry
      expect(Option.isSome(yield* runtime.close)).toBe(true)
      expect(Option.isSome(yield* runtime.close)).toBe(true)
      expect(tags.filter((tag) => tag === "ReleaseClientLease")).toHaveLength(1)
      expect(Exit.isFailure(yield* Effect.exit(runtime.startup.retry))).toBe(true)
      expect(ensures).toBe(1)
    })))
  })
})
