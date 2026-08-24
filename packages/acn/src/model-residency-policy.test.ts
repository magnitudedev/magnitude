import { IcnClient, type IcnClientService } from "@magnitudedev/icn"
import { Effect } from "effect"
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
    const client = {
      models: {
        getModelResidencyPolicy: () => Effect.succeed(current),
        setModelResidencyPolicy: ({ payload }: {
          payload: { generation: number; idleTimeoutSeconds: number }
        }) => Effect.sync(() => {
          requests.push(payload)
          if (payload.generation > current.generation) current = payload
        }),
      },
    } as unknown as IcnClientService

    await Effect.runPromise(Effect.gen(function* () {
      const policy = yield* ModelResidencyPolicy
      yield* policy.setConnected(true)
      yield* policy.setConnected(false)
    }).pipe(
      Effect.provide(ModelResidencyPolicyLive),
      Effect.provideService(IcnClient, client),
    ))

    expect(requests).toEqual([
      { generation: 41, idleTimeoutSeconds: 60 * 60 },
      { generation: 42, idleTimeoutSeconds: 10 * 60 },
    ])
  })
})
