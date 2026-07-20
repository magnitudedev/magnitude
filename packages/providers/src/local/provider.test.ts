import { describe, expect, it } from "vitest"
import { Effect } from "effect"
import { ProviderModelIdSchema } from "@magnitudedev/ai"
import { createLocalProvider, PROVIDER_ID, type LocalProviderSource } from "./provider"

describe("local provider", () => {
  it("uses the one public local provider identity and delegates binding to its ICN-backed source", async () => {
    const modelId = ProviderModelIdSchema.make("model-1")
    let bound: string | undefined
    const source = {
      catalog: {
        list: Effect.succeed([]),
        refresh: Effect.succeed([]),
        get: () => Effect.dieMessage("unused"),
      },
      discoverModelProperties: () => Effect.dieMessage("unused"),
      bindModel: (requested: typeof modelId) => {
        bound = requested
        return Effect.succeed({ stream: () => { throw new Error("unused") } })
      },
      status: Effect.succeed({ status: "ok" as const }),
    } as unknown as LocalProviderSource

    const local = createLocalProvider(source)
    expect(PROVIDER_ID).toBe("local")
    expect(local.provider.id).toBe("local")
    await Effect.runPromise(local.provider.bindModel(modelId))
    expect(bound).toBe(modelId)
  })
})
