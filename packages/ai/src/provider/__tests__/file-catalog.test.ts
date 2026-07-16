import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { FetchHttpClient } from "@effect/platform"
import { describe, expect, it } from "vitest"
import { Effect } from "effect"
import type { ModelCatalog } from "../catalog"
import { makeFileBackedModelCatalog } from "../file-catalog"
import { ProviderIdSchema, ProviderModelIdSchema, type ProviderId, type ProviderModel } from "../model"

const model = (providerId: ProviderId, displayName: string): ProviderModel => ({
  providerId,
  providerModelId: ProviderModelIdSchema.make("shared-model-id"),
  displayName,
  contextWindow: 8_192,
  maxOutputTokens: 1_024,
  capabilities: { vision: false },
  availability: { _tag: "Available" },
  pricing: { input: 0, output: 0, cached_input: null },
  reasoningEfforts: ["none"],
})

describe("file-backed model catalog", () => {
  it("uses provider ID and provider model ID as the cache key", async () => {
    const directory = await mkdtemp(join(tmpdir(), "magnitude-model-cache-"))
    const models = [
      model(ProviderIdSchema.make("magnitude"), "Hosted model"),
      model(ProviderIdSchema.make("llamacpp"), "Local model"),
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
        catalog.get(ProviderIdSchema.make("llamacpp"), ProviderModelIdSchema.make("shared-model-id")).pipe(
          Effect.provide(FetchHttpClient.layer),
        ),
      )

      expect(local.displayName).toBe("Local model")
    } finally {
      await rm(directory, { recursive: true, force: true })
    }
  })
})
