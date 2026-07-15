import { Context, Data, Effect, Layer, PubSub, Stream } from "effect"
import type { LocalInferenceUsageSelection } from "@magnitudedev/protocol"
import {
  MagnitudeStorage,
  type ConfigStorageShape,
  type DurableLocalModelBinding,
  type LocalInferenceConfig,
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
  readonly updateUsage: (usage: LocalInferenceUsageSelection) => Effect.Effect<void, ModelConfigurationError>
  readonly updateSlots: (
    slots: Partial<Record<SlotId, SlotModelConfig>>,
  ) => Effect.Effect<void, ModelConfigurationError>
  readonly activateLocal: (binding: DurableLocalModelBinding) => Effect.Effect<void, ModelConfigurationError>
  readonly disableLocal: Effect.Effect<void, ModelConfigurationError>
  readonly changes: Stream.Stream<void>
}

export class LocalModelConfiguration extends Context.Tag("LocalModelConfiguration")<
  LocalModelConfiguration,
  LocalModelConfigurationApi
>() {}

const configurationError = (operation: string, cause: unknown): ModelConfigurationError =>
  new ModelConfigurationError({
    operation,
    reason: cause instanceof Error ? cause.message : String(cause),
    cause,
  })

const localSlot = (binding: DurableLocalModelBinding): SlotModelConfig => ({
  providerId: "llamacpp",
  providerModelId: binding.providerModelId,
})

type LocalModelConfigurationStorage = Pick<
  ConfigStorageShape,
  "getLocalInferenceConfig" | "updateModelConfig" | "update"
>

export const makeLocalModelConfiguration = (
  storage: LocalModelConfigurationStorage,
): Effect.Effect<LocalModelConfigurationApi> => Effect.gen(function* () {
  const changes = yield* PubSub.unbounded<void>()
  const publish = PubSub.publish(changes, undefined).pipe(Effect.asVoid)

  const get = storage.getLocalInferenceConfig().pipe(
    Effect.map((config) => config ?? {}),
    Effect.mapError((cause) => configurationError("read local model configuration", cause)),
  )

  return LocalModelConfiguration.of({
    get,

    updateSlots: (slots) => storage.updateModelConfig(slots).pipe(
      Effect.mapError((cause) => configurationError("update model slots", cause)),
      Effect.zipRight(publish),
    ),

    updateUsage: (usage) => storage.update((current) => {
      const previous = current.localInference?.usage
      const changed = previous?.localModelRole !== usage.localModelRole
        || previous.sessionConcurrency !== usage.sessionConcurrency
      if (!changed) return current
      const existing = current.models?.slots ?? {}
      const primary = existing.primary?.providerId === "llamacpp" ? undefined : existing.primary
      const secondary = existing.secondary?.providerId === "llamacpp" ? undefined : existing.secondary
      return {
        ...current,
        localInference: { usage },
        models: {
          ...current.models,
          slots: {
            ...(primary ? { primary } : {}),
            ...(secondary ? { secondary } : {}),
          },
        },
      }
    }).pipe(
      Effect.mapError((cause) => configurationError("update local inference usage", cause)),
      Effect.zipRight(publish),
    ),

    activateLocal: (binding) => storage.update((current) => {
      const usage = current.localInference?.usage
      if (!usage) return current
      const existingSlots = current.models?.slots ?? {}
      const slots = usage.localModelRole === "main"
        ? { ...existingSlots, primary: localSlot(binding) }
        : { ...existingSlots, secondary: localSlot(binding) }
      return {
        ...current,
        localInference: { ...current.localInference, binding },
        models: { ...current.models, slots },
      }
    }).pipe(
      Effect.mapError((cause) => configurationError("activate local model", cause)),
      Effect.flatMap((config) => config.localInference?.usage && config.localInference.binding
        ? Effect.void
        : Effect.fail(new ModelConfigurationError({
          operation: "activate local model",
          reason: "Choose local inference usage before activating a model.",
        }))),
      Effect.zipRight(publish),
    ),

    disableLocal: storage.update((current) => {
      const existing = current.models?.slots ?? {}
      const primary = existing.primary?.providerId === "llamacpp" ? undefined : existing.primary
      const secondary = existing.secondary?.providerId === "llamacpp" ? undefined : existing.secondary
      const slots = {
        ...(primary ? { primary } : {}),
        ...(secondary ? { secondary } : {}),
      }
      const { binding: _, ...localInference } = current.localInference ?? {}
      return {
        ...current,
        localInference,
        models: { ...current.models, slots },
      }
    }).pipe(
      Effect.mapError((cause) => configurationError("disable local model", cause)),
      Effect.zipRight(publish),
    ),

    changes: Stream.fromPubSub(changes),
  })
})

export const LocalModelConfigurationLive: Layer.Layer<
  LocalModelConfiguration,
  never,
  MagnitudeStorage
> = Layer.effect(
  LocalModelConfiguration,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    return yield* makeLocalModelConfiguration(storage.config)
  }),
)
