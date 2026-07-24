import { Effect, Option, Ref } from "effect"
import { describe, expect, it } from "vitest"
import { ProviderIdSchema, ProviderModelIdSchema, ReasoningEffortSchema } from "@magnitudedev/ai"
import { PRIMARY_SLOT_ID } from "@magnitudedev/protocol"
import type { MagnitudeConfig } from "@magnitudedev/storage"
import { makeModelConfiguration } from "./model-configuration"

const selection = (model: string) => ({
  providerId: ProviderIdSchema.make("magnitude"),
  providerModelId: ProviderModelIdSchema.make(model),
  reasoningEffort: ReasoningEffortSchema.make("high"),
})

const localSelection = (model: string) => ({
  providerId: ProviderIdSchema.make("local"),
  providerModelId: ProviderModelIdSchema.make(model),
  reasoningEffort: ReasoningEffortSchema.make("high"),
})

const modelIdentity = (provider: string, model: string) => ({
  providerId: ProviderIdSchema.make(provider),
  providerModelId: ProviderModelIdSchema.make(model),
})

const updateStorage = (state: Ref.Ref<MagnitudeConfig>) => ({
  load: () => Ref.get(state),
  update: (update: (current: MagnitudeConfig) => MagnitudeConfig) => Ref.modify(state, (current) => {
    const next = update(current)
    return [next, next]
  }),
})

describe("model configuration ownership", () => {
  it("updates one addressed slot without replacing its sibling", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const state = yield* Ref.make<MagnitudeConfig>({
        models: {
          slots: {
            primary: Option.some(selection("primary-old")),
            secondary: Option.some(selection("secondary-old")),
          },
          localModelRecency: { primary: [], secondary: [] },
          favoriteModels: [],
          localProviderOfferings: [],
          dismissedDownloadFailures: [],
        },
      })
      const configuration = yield* makeModelConfiguration(updateStorage(state))
      yield* configuration.updateSlot(PRIMARY_SLOT_ID, Option.some(selection("primary-new")))
      return yield* Ref.get(state)
    }))

    if (!result.models) return expect.fail("model configuration was not persisted")
    expect(Option.getOrThrow(result.models.slots.primary).providerModelId).toBe("primary-new")
    expect(Option.getOrThrow(result.models.slots.secondary).providerModelId).toBe("secondary-old")
  })

  it("records local model use as bounded per-slot recency", async () => {
    const state = await Effect.runPromise(Effect.gen(function* () {
      const stored = yield* Ref.make<MagnitudeConfig>({
        models: {
          slots: {
            primary: Option.some(localSelection("local-a")),
            secondary: Option.none(),
          },
          localModelRecency: {
            primary: [ProviderModelIdSchema.make("local-b"), ProviderModelIdSchema.make("local-a")],
            secondary: [],
          },
          favoriteModels: [],
          localProviderOfferings: [],
          dismissedDownloadFailures: [],
        },
      })
      const configuration = yield* makeModelConfiguration(updateStorage(stored))
      yield* configuration.recordUse(PRIMARY_SLOT_ID, ProviderModelIdSchema.make("local-a"))
      return yield* configuration.get
    }))

    expect(state.localModelRecency.primary).toEqual(["local-a", "local-b"])
    expect(state.localModelRecency.secondary).toEqual([])
  })

  it("persists provider-qualified model favorites", async () => {
    const state = await Effect.runPromise(Effect.gen(function* () {
      const stored = yield* Ref.make<MagnitudeConfig>({})
      const configuration = yield* makeModelConfiguration(updateStorage(stored))
      yield* configuration.setFavorite(modelIdentity("local", "shared"), true)
      yield* configuration.setFavorite(modelIdentity("magnitude", "shared"), true)
      yield* configuration.setFavorite(modelIdentity("local", "shared"), false)
      return yield* configuration.get
    }))

    expect(state.favoriteModels).toEqual([modelIdentity("magnitude", "shared")])
  })
})
