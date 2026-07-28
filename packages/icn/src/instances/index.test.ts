import { describe, expect, it } from "vitest"
import {
  Data,
  Deferred,
  Effect,
  Layer,
  PubSub,
  Ref,
  Stream,
} from "effect"
import { IcnClient, type IcnClientService } from "../client.js"
import type {
  ModelInstancesInvalidation,
  ModelInstancesSnapshot,
} from "../generated/schemas.js"
import { IcnInstances, makeIcnInstances } from "./index.js"

class TestFailure extends Data.TaggedError("TestFailure")<{
  readonly message: string
}> {}

const instances = (revision: number): ModelInstancesSnapshot => ({
  revision,
  instances: [{
    id: "instance-test",
    configurationId: "configuration-test",
    lifecycle: {
      _tag: "Ready",
      allocation: {
        contextWindowTokens: 8_192,
        parallelSequences: 1,
        physicalContextTokens: 8_192,
        memoryDomains: [],
      },
    },
  }],
})

const response = (events: Stream.Stream<ModelInstancesInvalidation, TestFailure>) => ({
  status: 200,
  headers: {},
  events,
})

describe("ICN model-instance observation", () => {
  it("retries a failed snapshot refresh without abandoning the invalidation", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const invalidations = yield* PubSub.unbounded<ModelInstancesInvalidation>()
      const reads = yield* Ref.make(0)
      const watching = yield* Deferred.make<void>()
      const refreshed = yield* Deferred.make<void>()
      const client = {
        models: {
          watchModelInstances: () =>
            Effect.succeed(response(
              Stream.fromEffect(Deferred.succeed(watching, undefined)).pipe(
                Stream.drain,
                Stream.concat(Stream.fromPubSub(invalidations)),
              ),
            )),
          getModelInstances: () => Ref.getAndUpdate(reads, (count) => count + 1).pipe(
            Effect.flatMap((call) => {
              if (call === 0) return Effect.succeed(instances(0))
              if (call === 1) {
                return Effect.fail(new TestFailure({
                  message: "transient refresh failure",
                }))
              }
              return Deferred.succeed(refreshed, undefined).pipe(
                Effect.as(instances(1)),
              )
            }),
          ),
        },
      } as unknown as IcnClientService

      const snapshot = yield* Effect.gen(function* () {
        const observed = yield* IcnInstances
        yield* Deferred.await(watching)
        yield* PubSub.publish(invalidations, { revision: 1 })
        yield* Deferred.await(refreshed)
        return yield* observed.get
      }).pipe(
        Effect.provide(makeIcnInstances({ retryInterval: "1 millis" }).pipe(
          Layer.provide(Layer.succeed(IcnClient, client)),
        )),
      )

      expect(snapshot).toEqual(instances(1))
      expect(yield* Ref.get(reads)).toBe(3)
    })))
  })

  it("re-admits a terminated watch and refreshes after reconnect", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const watchCalls = yield* Ref.make(0)
      const reads = yield* Ref.make(0)
      const refreshed = yield* Deferred.make<void>()
      const client = {
        models: {
          watchModelInstances: () => Ref.getAndUpdate(watchCalls, (count) => count + 1).pipe(
            Effect.flatMap((call) => {
              if (call === 1) {
                return Effect.fail(new TestFailure({
                  message: "watch re-admission failed",
                }))
              }
              return Effect.succeed(response(
                call === 0
                  ? Stream.fail(new TestFailure({ message: "watch failed" }))
                  : Stream.never,
              ))
            }),
          ),
          getModelInstances: () => Ref.getAndUpdate(reads, (count) => count + 1).pipe(
            Effect.flatMap((call) => call === 0
              ? Effect.succeed(instances(0))
              : Deferred.succeed(refreshed, undefined).pipe(
                  Effect.as(instances(1)),
                )),
          ),
        },
      } as unknown as IcnClientService

      const snapshot = yield* Effect.gen(function* () {
        const observed = yield* IcnInstances
        yield* Deferred.await(refreshed)
        return yield* observed.get
      }).pipe(
        Effect.provide(makeIcnInstances({ retryInterval: "1 millis" }).pipe(
          Layer.provide(Layer.succeed(IcnClient, client)),
        )),
      )

      expect(snapshot).toEqual(instances(1))
      expect(yield* Ref.get(watchCalls)).toBe(3)
    })))
  })
})
