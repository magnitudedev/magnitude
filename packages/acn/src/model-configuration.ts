import { Context, Effect, Layer, Option, Stream, SubscriptionRef } from "effect"
import type { ProviderModelId, SlotId, SlotSelection } from "@magnitudedev/sdk"
import {
  MagnitudeStorage,
  type ConfigStorageShape,
  type MagnitudeConfig,
  type ResolvedContextLimitPolicy,
  resolveContextLimitPolicy,
} from "@magnitudedev/storage"

type StoredModelConfiguration = NonNullable<MagnitudeConfig["models"]>
export type ModelSlotsConfiguration = StoredModelConfiguration["slots"]
type Storage = Pick<
  ConfigStorageShape,
  "load" | "update"
>
export type ModelConfigurationError = Effect.Effect.Error<ReturnType<Storage["load"]>>

export interface ModelConfigurationState extends StoredModelConfiguration {
  readonly contextLimits: ResolvedContextLimitPolicy
}

export interface ModelConfigurationApi {
  readonly get: Effect.Effect<ModelConfigurationState>
  readonly changes: Stream.Stream<ModelConfigurationState>
  readonly updateSlot: (
    slotId: SlotId,
    selection: Option.Option<SlotSelection>,
  ) => Effect.Effect<void, ModelConfigurationError>
  readonly recordUse: (
    slotId: SlotId,
    providerModelId: ProviderModelId,
  ) => Effect.Effect<void, ModelConfigurationError>
}

export class ModelConfiguration extends Context.Tag("ModelConfiguration")<
  ModelConfiguration,
  ModelConfigurationApi
>() {}

const EMPTY_MODEL_CONFIGURATION: StoredModelConfiguration = {
  slots: { primary: Option.none(), secondary: Option.none() },
  localModelRecency: { primary: [], secondary: [] },
  localProviderOfferings: [],
  dismissedDownloadFailures: [],
}

const RECENCY_LIMIT = 32
const moveToFront = (
  items: readonly ProviderModelId[],
  providerModelId: ProviderModelId,
): readonly ProviderModelId[] => [
  providerModelId,
  ...items.filter((item) => item !== providerModelId),
].slice(0, RECENCY_LIMIT)

const recencyFor = (
  recency: StoredModelConfiguration["localModelRecency"],
  slotId: SlotId,
) => slotId === "primary" ? recency.primary : recency.secondary

export const makeModelConfiguration = (
  storage: Storage,
): Effect.Effect<ModelConfigurationApi, ModelConfigurationError> => Effect.gen(function* () {
  const loaded = yield* storage.load()
  const initial: ModelConfigurationState = {
    ...Option.getOrElse(Option.fromNullable(loaded.models), () => EMPTY_MODEL_CONFIGURATION),
    contextLimits: resolveContextLimitPolicy(loaded),
  }
  const state = yield* SubscriptionRef.make(initial)
  const lock = yield* Effect.makeSemaphore(1)

  const persist = (
    update: (current: StoredModelConfiguration) => StoredModelConfiguration,
  ) => storage.update((current) => ({
    ...current,
    models: update(Option.getOrElse(
      Option.fromNullable(current.models),
      () => EMPTY_MODEL_CONFIGURATION,
    )),
  })).pipe(Effect.map((updated) => updated.models ?? EMPTY_MODEL_CONFIGURATION))

  const updateSlot: ModelConfigurationApi["updateSlot"] = (slotId, selection) =>
    lock.withPermits(1)(Effect.uninterruptible(Effect.gen(function* () {
      const persisted = yield* persist((current) => ({
        ...current,
        slots: { ...current.slots, [slotId]: selection },
        localModelRecency: Option.match(selection, {
          onNone: () => current.localModelRecency,
          onSome: (selected) => selected.providerId !== "local"
            ? current.localModelRecency
            : {
                ...current.localModelRecency,
                [slotId]: moveToFront(recencyFor(current.localModelRecency, slotId), selected.providerModelId),
              },
        }),
      }))
      const current = yield* SubscriptionRef.get(state)
      yield* SubscriptionRef.set(state, {
        ...persisted,
        contextLimits: current.contextLimits,
      })
    })))

  const recordUse: ModelConfigurationApi["recordUse"] = (slotId, providerModelId) =>
    lock.withPermits(1)(Effect.uninterruptible(Effect.gen(function* () {
      const observed = yield* SubscriptionRef.get(state)
      if (recencyFor(observed.localModelRecency, slotId)[0] === providerModelId) return
      const persisted = yield* persist((current) => ({
        ...current,
        localModelRecency: {
          ...current.localModelRecency,
          [slotId]: moveToFront(recencyFor(current.localModelRecency, slotId), providerModelId),
        },
      }))
      yield* SubscriptionRef.set(state, {
        ...persisted,
        contextLimits: observed.contextLimits,
      })
    })))

  return ModelConfiguration.of({
    get: SubscriptionRef.get(state),
    changes: state.changes,
    updateSlot,
    recordUse,
  })
})

export const makeModelConfigurationLayer = (): Layer.Layer<
  ModelConfiguration,
  ModelConfigurationError,
  MagnitudeStorage
> => Layer.effect(ModelConfiguration, Effect.gen(function* () {
  const storage = yield* MagnitudeStorage
  return yield* makeModelConfiguration(storage.config)
}))
