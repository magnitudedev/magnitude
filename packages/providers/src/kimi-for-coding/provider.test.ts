import { describe, expect, it } from "vitest"
import { FetchHttpClient } from "@effect/platform"
import { Effect } from "effect"
import { createKimiForCodingProvider } from "./provider"
import { KIMI_FOR_CODING_MODEL_ID } from "./catalog"
import { createKimiForCodingCompatibleSpec } from "./models"

describe("Kimi Code", () => {
  it("exposes and binds the official stable callable model ID", async () => {
    const instance = createKimiForCodingProvider({ apiKey: "test" })
    const models = await Effect.runPromise(instance.catalog.list.pipe(Effect.provide(FetchHttpClient.layer)))
    const spec = createKimiForCodingCompatibleSpec({
      modelId: KIMI_FOR_CODING_MODEL_ID,
      endpoint: "https://api.kimi.com/coding/v1",
    })

    expect(models.map((model) => model.providerModelId)).toEqual(["kimi-for-coding"])
    expect(spec.modelId).toBe("kimi-for-coding")
    expect(spec.endpoint).toBe("https://api.kimi.com/coding/v1")
  })
})
