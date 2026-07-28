import { describe, expect, it } from "vitest"
import { Effect } from "effect"
import {
  PRIMARY_SLOT_ID,
  LocalModelMutationFailed,
} from "@magnitudedev/protocol"
import {
  ProviderIdSchema,
  ProviderModelIdSchema,
} from "@magnitudedev/sdk"
import type { ModelRequestProgress } from "@magnitudedev/ai"
import { makeModelRequestPreparation } from "./model-request-preparation"

const localProviderId = ProviderIdSchema.make("local")
const remoteProviderId = ProviderIdSchema.make("magnitude")
const modelId = ProviderModelIdSchema.make("model")

describe("model request preparation", () => {
  it("acquires the local model without fabricating request progress", async () => {
    const order: string[] = []
    const progress: ModelRequestProgress[] = []
    const prepare = makeModelRequestPreparation({
      acquireLocalModel: () => Effect.acquireRelease(
        Effect.sync(() => {
          order.push("acquire")
        }),
        () => Effect.sync(() => {
          order.push("release")
        }),
      ),
    })

    await Effect.runPromise(Effect.scoped(prepare({
      slotId: PRIMARY_SLOT_ID,
      providerId: localProviderId,
      providerModelId: modelId,
      reportProgress: (update) => Effect.sync(() => {
        progress.push(update)
        order.push(update.phase)
      }),
    })))

    expect(order).toEqual(["acquire", "release"])
    expect(progress).toEqual([])
  })

  it("does not prepare remote provider requests", async () => {
    let acquired = false
    const prepare = makeModelRequestPreparation({
      acquireLocalModel: () => Effect.sync(() => {
        acquired = true
      }),
    })

    await Effect.runPromise(Effect.scoped(prepare({
      slotId: PRIMARY_SLOT_ID,
      providerId: remoteProviderId,
      providerModelId: modelId,
      reportProgress: () => Effect.die("remote requests must not report local preparation"),
    })))

    expect(acquired).toBe(false)
  })

  it("preserves admission failure details without constructing a provider error", async () => {
    const prepare = makeModelRequestPreparation({
      acquireLocalModel: () => Effect.fail(new LocalModelMutationFailed({
        code: "low_memory",
        message: "Not enough memory",
        retryable: true,
      })),
    })

    const failure = await Effect.runPromise(Effect.flip(Effect.scoped(prepare({
      slotId: PRIMARY_SLOT_ID,
      providerId: localProviderId,
      providerModelId: modelId,
      reportProgress: () => Effect.void,
    }))))

    expect(failure).toMatchObject({
      _tag: "ModelRequestPreparationFailed",
      code: "low_memory",
      message: "Not enough memory",
      retryable: true,
    })
  })
})
