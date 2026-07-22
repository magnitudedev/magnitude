import { Context, Effect, Layer, Option, PubSub, Stream } from "effect"
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

type ModelConfiguration = NonNullable<MagnitudeConfig["models"]>
export type ModelSlotsConfiguration = NonNullable<ModelConfiguration["slots"]>
type Storage = Pick<ConfigStorageShape, "getLocalInferenceConfig" | "getModelConfig" | "update">
export type ModelConfigurationError = Effect.Effect.Error<ReturnType<Storage["update"]>>

export interface LocalModelConfigurationApi {
  readonly get: Effect.Effect<LocalInferenceConfig, ModelConfigurationError>
  readonly getModels: Effect.Effect<ModelConfiguration, ModelConfigurationError>
  readonly selectProfile: (profile: SelectedLocalModelProfile) => Effect.Effect<void, ModelConfigurationError>
  readonly updateSlots: (slots: Partial<Record<SlotId, SlotModelConfig>>) => Effect.Effect<void, ModelConfigurationError>
  readonly recordUse: (slotId: SlotId, providerModelId: ProviderModelId) => Effect.Effect<void>
  readonly changes: Stream.Stream<true>
}

export class LocalModelConfiguration extends Context.Tag("LocalModelConfiguration")<
  LocalModelConfiguration,
  LocalModelConfigurationApi
>() {}

const EMPTY_MODEL_CONFIGURATION: ModelConfiguration = {}
const EMPTY_MODEL_SLOTS: ModelSlotsConfiguration = {}

export const makeLocalModelConfiguration = (storage: Storage): Effect.Effect<LocalModelConfigurationApi> =>
  Effect.gen(function* () {
    const changes = yield* PubSub.unbounded<true>()
    const mutate = (update: (current: MagnitudeConfig) => MagnitudeConfig) =>
      storage.update(update).pipe(
        Effect.asVoid,
        Effect.tap(() => PubSub.publish(changes, true)),
      )
    return LocalModelConfiguration.of({
      get: storage.getLocalInferenceConfig().pipe(
        Effect.map((value) => Option.getOrElse(Option.fromNullable(value), () => ({}))),
      ),
      getModels: storage.getModelConfig().pipe(
        Effect.map((value) => Option.getOrElse(
          Option.fromNullable(value),
          () => EMPTY_MODEL_CONFIGURATION,
        )),
      ),
      selectProfile: (selectedProfile) => mutate((current) => ({
        ...current,
        localInference: { ...current.localInference, selectedProfile },
      })),
      updateSlots: (updates) => mutate((current) => {
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
      changes: Stream.fromPubSub(changes),
    })
  })

export const makeLocalModelConfigurationLayer = (): Layer.Layer<LocalModelConfiguration, never, MagnitudeStorage> =>
  Layer.effect(LocalModelConfiguration, Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    return yield* makeLocalModelConfiguration(storage.config)
  }))
