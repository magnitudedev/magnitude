import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { FetchHttpClient } from "@effect/platform"
import { describe, expect, it } from "vitest"
import { Effect } from "effect"
import type { ModelCatalog } from "../catalog"
import { makeFileBackedModelCatalog } from "../file-catalog"
import type { ProviderModel } from "../model"

const model = (providerId: string, displayName: string): ProviderModel => ({
  providerId,
  providerModelId: "shared-model-id",
  modelFamilyId: "unknown",
  displayName,
  contextWindow: 8_192,
  maxOutputTokens: 1_024,
  capabilities: { vision: false, toolCalls: true, structuredOutput: true, grammar: true, toolChoiceModes: ["auto", "none", "required", "named"] },
  pricing: { input: 0, output: 0, cached_input: null },
  reasoningEfforts: ["none"],
})

describe("file-backed model catalog", () => {
  it("uses provider ID and provider model ID as the cache key", async () => {
    const directory = await mkdtemp(join(tmpdir(), "magnitude-model-cache-"))
    const models = [
      model("magnitude", "Hosted model"),
      model("llamacpp", "Local model"),
    ]
    const inner: ModelCatalog<ProviderModel> = {
      list: Effect.succeed(models),
      refresh: Effect.succeed(models),
      get: (providerId, providerModelId) => {
        const found = models.find((entry) =>
          entry.providerId === providerId && entry.providerModelId === providerModelId
        )
        return found
          ? Effect.succeed(found)
          : Effect.die("test model not found")
      },
    }

    try {
      const catalog = makeFileBackedModelCatalog(inner, join(directory, "models.json"))
      const local = await Effect.runPromise(
        catalog.get("llamacpp", "shared-model-id").pipe(
          Effect.provide(FetchHttpClient.layer),
        ),
      )

      expect(local.displayName).toBe("Local model")
    } finally {
      await rm(directory, { recursive: true, force: true })
    }
  })

  it("preserves stale models only for providers that remain configured", async () => {
    const directory = await mkdtemp(join(tmpdir(), "magnitude-model-cache-"))
    const cachePath = join(directory, "models.json")
    const stale = model("openrouter", "Stale router model")
    const fresh = model("magnitude", "Fresh hosted model")
    const catalogFor = (models: readonly ProviderModel[], preserveProviderIds: readonly string[]) =>
      makeFileBackedModelCatalog<ProviderModel>({
        list: Effect.succeed(models),
        refresh: Effect.succeed(models),
        get: () => Effect.die("not used"),
      }, cachePath, undefined, undefined, { preserveProviderIds })

    try {
      await Effect.runPromise(catalogFor([stale], []).refresh.pipe(Effect.provide(FetchHttpClient.layer)))
      const partial = await Effect.runPromise(
        catalogFor([fresh], ["openrouter", "magnitude"]).refresh.pipe(Effect.provide(FetchHttpClient.layer)),
      )
      const disconnected = await Effect.runPromise(
        catalogFor([fresh], ["magnitude"]).refresh.pipe(Effect.provide(FetchHttpClient.layer)),
      )

      expect(partial.map((entry) => entry.providerId)).toEqual(["magnitude", "openrouter"])
      expect(disconnected.map((entry) => entry.providerId)).toEqual(["magnitude"])
    } finally {
      await rm(directory, { recursive: true, force: true })
    }
  })
})
