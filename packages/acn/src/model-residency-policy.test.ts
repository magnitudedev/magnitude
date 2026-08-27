import { IcnClient, type IcnClientService } from "@magnitudedev/icn"
import { Deferred, Effect } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelResidencyPolicy,
  ModelResidencyPolicyLive,
} from "./model-residency-policy"

describe("ModelResidencyPolicy", () => {
  it("publishes monotonic connected and disconnected idle policies", async () => {
    const requests: Array<{
      readonly generation: number
      readonly idleTimeoutSeconds: number
    }> = []
    let current = { generation: 40, idleTimeoutSeconds: 600 }
    await Effect.runPromise(Effect.gen(function* () {
      const firstPublished = yield* Deferred.make<void>()
      const secondPublished = yield* Deferred.make<void>()
      const client = {
        models: {
          getModelResidencyPolicy: () => Effect.succeed(current),
          setModelResidencyPolicy: ({ payload }: {
            payload: { generation: number; idleTimeoutSeconds: number }
          }) => Effect.sync(() => {
            requests.push(payload)
            if (payload.generation > current.generation) current = payload
            return requests.length
          }).pipe(Effect.flatMap((count) => count === 1
            ? Deferred.succeed(firstPublished, undefined)
            : Deferred.succeed(secondPublished, undefined))),
        },
      } as unknown as IcnClientService
      yield* Effect.gen(function* () {
        const policy = yield* ModelResidencyPolicy
        yield* policy.setConnected(true)
        yield* Deferred.await(firstPublished)
        yield* policy.setConnected(false)
        yield* Deferred.await(secondPublished)
      }).pipe(
        Effect.scoped,
        Effect.provide(ModelResidencyPolicyLive),
        Effect.provideService(IcnClient, client),
      )
    }))

    expect(requests).toEqual([
      { generation: 41, idleTimeoutSeconds: 60 * 60 },
      { generation: 42, idleTimeoutSeconds: 10 * 60 },
    ])
  })

  it("accepts desired policy immediately while ICN convergence is blocked", async () => {
    const attempted = await Effect.runPromise(Effect.gen(function* () {
      const entered = yield* Deferred.make<void>()
      const client = {
        models: {
          getModelResidencyPolicy: () => Deferred.succeed(entered, undefined).pipe(
            Effect.zipRight(Effect.never),
          ),
        },
      } as unknown as IcnClientService
      return yield* Effect.gen(function* () {
        const policy = yield* ModelResidencyPolicy
        yield* policy.setConnected(true)
        yield* Deferred.await(entered)
        return true
      }).pipe(
        Effect.scoped,
        Effect.provide(ModelResidencyPolicyLive),
        Effect.provideService(IcnClient, client),
      )
    }))

    expect(attempted).toBe(true)
  })
})
