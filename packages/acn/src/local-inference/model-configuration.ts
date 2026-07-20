import { Context, Data, Effect, Layer, PubSub, Stream } from "effect"
import type { ProviderModelId } from "@magnitudedev/sdk"
import {
  MagnitudeStorage,
  type ConfigStorageShape,
  type LocalInferenceConfig,
  type MagnitudeConfig,
  type SelectedLocalModelProfile,
  type SlotId,
  type SlotModelConfig,
} from "@magnitudedev/storage"

export class ModelConfigurationError extends Data.TaggedError("ModelConfigurationError")<{
  readonly operation: string
  readonly reason: string
  readonly cause?: unknown
}> {}

export interface LocalModelConfigurationApi {
  readonly get: Effect.Effect<LocalInferenceConfig, ModelConfigurationError>
  readonly getModels: Effect.Effect<MagnitudeConfig["models"], ModelConfigurationError>
  readonly selectProfile: (profile: SelectedLocalModelProfile) => Effect.Effect<void, ModelConfigurationError>
  readonly updateSlots: (slots: Partial<Record<SlotId, SlotModelConfig>>) => Effect.Effect<void, ModelConfigurationError>
  readonly recordUse: (slotId: SlotId, providerModelId: ProviderModelId) => Effect.Effect<void>
  readonly changes: Stream.Stream<void>
}

export class LocalModelConfiguration extends Context.Tag("LocalModelConfiguration")<
  LocalModelConfiguration,
  LocalModelConfigurationApi
>() {}

const failure = (operation: string, cause: unknown) => new ModelConfigurationError({
  operation,
  reason: cause instanceof Error ? cause.message : String(cause),
  cause,
})

type Storage = Pick<ConfigStorageShape, "getLocalInferenceConfig" | "getModelConfig" | "update">

export const makeLocalModelConfiguration = (storage: Storage): Effect.Effect<LocalModelConfigurationApi> =>
  Effect.gen(function* () {
    const changes = yield* PubSub.unbounded<void>()
    const publish = PubSub.publish(changes, undefined).pipe(Effect.asVoid)
    const mutate = (operation: string, update: (current: MagnitudeConfig) => MagnitudeConfig) =>
      storage.update(update).pipe(
        Effect.mapError((cause) => failure(operation, cause)),
        Effect.asVoid,
        Effect.zipRight(publish),
      )
    return LocalModelConfiguration.of({
      get: storage.getLocalInferenceConfig().pipe(
        Effect.map((value) => value ?? {}),
        Effect.mapError((cause) => failure("read local inference configuration", cause)),
      ),
      getModels: storage.getModelConfig().pipe(
        Effect.map((value) => value ?? {}),
        Effect.mapError((cause) => failure("read model configuration", cause)),
      ),
      selectProfile: (selectedProfile) => mutate("select local model profile", (current) => ({
        ...current,
        localInference: { ...current.localInference, selectedProfile },
      })),
      updateSlots: (updates) => mutate("update model slots", (current) => {
        const models = current.models ?? {}
        const slots = { ...(models.slots ?? {}) }
        const localSlotIntent = { ...(models.localSlotIntent ?? {}) }
        for (const slotId of ["primary", "secondary"] as const) {
          const update = updates[slotId]
          if (!update) continue
          if (update.providerId || update.providerModelId || update.reasoningEffort) slots[slotId] = update
          else delete slots[slotId]
          if (update.providerId === "local") localSlotIntent[slotId] = "local"
          else if (update.providerId) localSlotIntent[slotId] = "cloud"
        }
        return { ...current, models: { ...models, slots, localSlotIntent } }
      }),
      recordUse: () => Effect.void,
      changes: Stream.fromPubSub(changes),
    })
  })

export const LocalModelConfigurationLive: Layer.Layer<LocalModelConfiguration, never, MagnitudeStorage> =
  Layer.effect(LocalModelConfiguration, Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    return yield* makeLocalModelConfiguration(storage.config)
  }))
