import { describe, expect, it } from "vitest"
import {
  AcnInstanceIdSchema,
  AcnReady,
  MagnitudeRpcs,
  ModelSlotUnassigned,
  PRIMARY_SLOT_ID,
  ProcessStartIdentitySchema,
  SECONDARY_SLOT_ID,
} from "@magnitudedev/acn-protocol"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientError from "@effect/platform/HttpClientError"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { Rpc, RpcClient } from "@effect/rpc"
import { Deferred, Duration, Effect, Exit, Fiber, Layer, Option, Schema, Stream } from "effect"
import { AcnEnsuranceFailed } from "./errors"
import { AcnEnsurer, type ReadyAcn } from "./acn-ensurer"
import { makeAcnJitRuntime } from "./acn-recovering-client"
import { SDK_VERSION } from "../version"

const ready: ReadyAcn = {
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
  instances: ReadonlyArray<ReadyAcn> = [ready],
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
  const rpc = MagnitudeRpcs.requests.get(message.tag)
  if (rpc === undefined) throw new Error(`Unknown RPC ${message.tag}`)
  const success = message.tag === "Health"
    ? {
        service: "magnitude-acn" as const,
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
                recentModelIds: { primary: [], secondary: [] },
                favoriteModels: [],
              },
            }
          : undefined
  if (success === undefined) throw new Error(`No response for ${message.tag}`)
  const rpcExit = failRpcTags.has(message.tag)
    ? Exit.die(`Simulated ${message.tag} failure`)
    : Exit.succeed(success)
  const exit = Schema.encodeUnknownSync(Rpc.exitSchema(rpc))(rpcExit)
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
      const ensurer = AcnEnsurer.of({
        ensure: () => Stream.unwrap(Effect.gen(function* () {
          calls += 1
          yield* Deferred.await(release)
          return Stream.succeed({ _tag: "Ready" as const, instance: ready })
        })),
      })
      const tags: string[] = []
      const runtime = yield* makeAcnJitRuntime().pipe(
        Effect.provideService(AcnEnsurer, ensurer),
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
      Effect.provideService(AcnEnsurer, AcnEnsurer.of({ ensure: () => Stream.never })),
      Effect.provideService(HttpClient.HttpClient, rpcClient(tags)),
      Effect.asVoid,
    )))
    expect(tags).not.toContain("RenewClientLease")
  })

  it("close interrupts initial selection without starting a lease", async () => {
    const tags: string[] = []
    let entered = 0
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* makeAcnJitRuntime().pipe(
        Effect.provideService(AcnEnsurer, AcnEnsurer.of({
          ensure: () => Stream.unwrap(Effect.sync(() => {
            entered += 1
            return Stream.never
          })),
        })),
        Effect.provideService(HttpClient.HttpClient, rpcClient(tags)),
      )
      while (entered === 0) yield* Effect.sleep(Duration.millis(1))
      expect(Option.isNone(yield* runtime.close)).toBe(true)
      expect(tags).not.toContain("RenewClientLease")
    })))
  })

  it("shares one deterministic ensurance failure with every concurrent waiter", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const release = yield* Deferred.make<void>()
      let calls = 0
      const ensurer = AcnEnsurer.of({
        ensure: () => Stream.unwrap(Effect.gen(function* () {
          calls += 1
          yield* Deferred.await(release)
          return Stream.fail(new AcnEnsuranceFailed({ reason: "deterministic failure" }))
        })),
      })
      const runtime = yield* makeAcnJitRuntime().pipe(
        Effect.provideService(AcnEnsurer, ensurer),
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

  it("recovers an application RPC by replacing only its failed exact endpoint", async () => {
    const successor: ReadyAcn = {
      ...ready,
      id: AcnInstanceIdSchema.make("successor-acn"),
      url: "http://successor-acn",
      pid: 456,
      processStartIdentity: ProcessStartIdentitySchema.make("successor-process"),
    }
    let ensures = 0
    const ensurer = AcnEnsurer.of({
      ensure: () => Stream.sync(() => ({
        _tag: "Ready" as const,
        instance: ensures++ === 0 ? ready : successor,
      })),
    })
    const tags: string[] = []
    const http = rpcClient(tags, [ready, successor], new Set([ready.id]))
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* makeAcnJitRuntime().pipe(
        Effect.provideService(AcnEnsurer, ensurer),
        Effect.provideService(HttpClient.HttpClient, http),
      )
      yield* runtime.startup.retry
      const client = yield* RpcClient.make(MagnitudeRpcs).pipe(
        Effect.provide(runtime.protocolLayer.pipe(
          Layer.provide(Layer.succeed(HttpClient.HttpClient, http)),
        )),
      )
      const health = yield* client.Health({})
      expect(health.id).toBe(successor.id)
      expect(ensures).toBe(2)
      expect(tags.filter((tag) => tag === "Health")).toHaveLength(2)
    })))
  })

  it("releases one lease when close observation fails and never ensures again", async () => {
    const tags: string[] = []
    let ensures = 0
    const ensurer = AcnEnsurer.of({
      ensure: () => Stream.sync(() => {
        ensures += 1
        return { _tag: "Ready" as const, instance: ready }
      }),
    })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const runtime = yield* makeAcnJitRuntime().pipe(
        Effect.provideService(AcnEnsurer, ensurer),
        Effect.provideService(HttpClient.HttpClient, rpcClient(
          tags,
          [ready],
          new Set(),
          new Set(["GetModelSlots"]),
        )),
      )
      yield* runtime.startup.retry
      expect(Option.isNone(yield* runtime.close)).toBe(true)
      expect(Option.isNone(yield* runtime.close)).toBe(true)
      expect(tags.filter((tag) => tag === "ReleaseClientLease")).toHaveLength(1)
      expect(Exit.isFailure(yield* Effect.exit(runtime.startup.retry))).toBe(true)
      expect(ensures).toBe(1)
    })))
  })
})
