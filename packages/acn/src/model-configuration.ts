import { Context, Data, Effect, Layer, Option, PubSub, Ref, Stream } from "effect"
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
  readonly cause: unknown
}> {}

type ModelConfiguration = NonNullable<MagnitudeConfig["models"]>
export type ModelSlotsConfiguration = NonNullable<ModelConfiguration["slots"]>

export interface LocalModelConfigurationApi {
  readonly get: Effect.Effect<LocalInferenceConfig, ModelConfigurationError>
  readonly getModels: Effect.Effect<ModelConfiguration, ModelConfigurationError>
  readonly selectProfile: (profile: SelectedLocalModelProfile) => Effect.Effect<void, ModelConfigurationError>
  readonly updateSlots: (slots: Partial<Record<SlotId, SlotModelConfig>>) => Effect.Effect<void, ModelConfigurationError>
  readonly recordUse: (slotId: SlotId, providerModelId: ProviderModelId) => Effect.Effect<void>
  readonly revision: Effect.Effect<number>
  readonly changes: Stream.Stream<ModelConfigurationChange>
}

export interface ModelConfigurationChange {
  readonly revision: number
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
const EMPTY_MODEL_CONFIGURATION: ModelConfiguration = {}
const EMPTY_MODEL_SLOTS: ModelSlotsConfiguration = {}

export const makeLocalModelConfiguration = (storage: Storage): Effect.Effect<LocalModelConfigurationApi> =>
  Effect.gen(function* () {
    const changes = yield* PubSub.unbounded<ModelConfigurationChange>()
    const revision = yield* Ref.make(0)
    const publish = Ref.updateAndGet(revision, (value) => value + 1).pipe(
      Effect.flatMap((nextRevision) => PubSub.publish(changes, { revision: nextRevision })),
      Effect.asVoid,
    )
    const mutate = (operation: string, update: (current: MagnitudeConfig) => MagnitudeConfig) =>
      storage.update(update).pipe(
        Effect.mapError((cause) => failure(operation, cause)),
        Effect.asVoid,
        Effect.zipRight(publish),
      )
    return LocalModelConfiguration.of({
      get: storage.getLocalInferenceConfig().pipe(
        Effect.map((value) => Option.getOrElse(Option.fromNullable(value), () => ({}))),
        Effect.mapError((cause) => failure("read local inference configuration", cause)),
      ),
      getModels: storage.getModelConfig().pipe(
        Effect.map((value) => Option.getOrElse(
          Option.fromNullable(value),
          () => EMPTY_MODEL_CONFIGURATION,
        )),
        Effect.mapError((cause) => failure("read model configuration", cause)),
      ),
      selectProfile: (selectedProfile) => mutate("select local model profile", (current) => ({
        ...current,
        localInference: { ...current.localInference, selectedProfile },
      })),
      updateSlots: (updates) => mutate("update model slots", (current) => {
        const models = Option.getOrElse(
          Option.fromNullable(current.models),
          () => EMPTY_MODEL_CONFIGURATION,
        )
        const slots = {
          ...Option.getOrElse(Option.fromNullable(models.slots), () => EMPTY_MODEL_SLOTS),
        }
        const localSlotIntent = {
          ...Option.getOrElse(Option.fromNullable(models.localSlotIntent), () => ({})),
        }
        for (const slotId of ["primary", "secondary"] as const) {
          const update = Option.fromNullable(updates[slotId])
          if (Option.isNone(update)) continue
          const providerId = Option.fromNullable(update.value.providerId)
          const hasConfiguration = Option.isSome(providerId)
            || Option.isSome(Option.fromNullable(update.value.providerModelId))
            || Option.isSome(Option.fromNullable(update.value.reasoningEffort))
          if (hasConfiguration) slots[slotId] = update.value
          else delete slots[slotId]
          Option.match(providerId, {
            onNone: () => { delete localSlotIntent[slotId] },
            onSome: (value) => { localSlotIntent[slotId] = value === "local" ? "local" : "cloud" },
          })
        }
        return { ...current, models: { ...models, slots, localSlotIntent } }
      }),
      recordUse: () => Effect.void,
      revision: Ref.get(revision),
      changes: Stream.fromPubSub(changes),
    })
  })

export const makeLocalModelConfigurationLayer = (): Layer.Layer<LocalModelConfiguration, never, MagnitudeStorage> =>
  Layer.effect(LocalModelConfiguration, Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    return yield* makeLocalModelConfiguration(storage.config)
  }))
