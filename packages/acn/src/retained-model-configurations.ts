import { Context, Effect, Layer, Option, Stream } from 'effect'
import {
  sameServableModelBundleIdentity,
  type ModelServingConfiguration,
  type ModelServingConfigurationId,
} from '@magnitudedev/acn-protocol'
import {
  MagnitudeStorage,
  type ModelState,
  type StateDocumentError,
  type StateHandle,
} from '@magnitudedev/storage'

export interface RetainedModelConfigurationsApi {
  readonly get: Effect.Effect<readonly ModelServingConfiguration[]>
  readonly recoveryCompleted: Effect.Effect<boolean>
  readonly changes: Stream.Stream<readonly ModelServingConfiguration[]>
  readonly resolve: (
    id: ModelServingConfigurationId,
  ) => Effect.Effect<Option.Option<ModelServingConfiguration>>
  readonly materialize: (
    configuration: ModelServingConfiguration,
  ) => Effect.Effect<ModelServingConfiguration, StateDocumentError>
  readonly remove: (
    id: ModelServingConfigurationId,
  ) => Effect.Effect<Option.Option<ModelServingConfiguration>, StateDocumentError>
  readonly completeRecovery: (
    defaults: readonly ModelServingConfiguration[],
  ) => Effect.Effect<readonly ModelServingConfiguration[], StateDocumentError>
}

export class RetainedModelConfigurations extends Context.Tag('RetainedModelConfigurations')<
  RetainedModelConfigurations,
  RetainedModelConfigurationsApi
>() {}

const removeReferences = (
  state: ModelState,
  id: ModelServingConfigurationId,
): ModelState => {
  const refersToConfiguration = (value: {
    readonly providerId: string
    readonly providerModelId: string
  }) => value.providerId === 'local' && value.providerModelId === id
  return {
    ...state,
    configurations: state.configurations.filter((candidate) => candidate.id !== id),
    slots: {
      primary: Option.filter(state.slots.primary, (selection) => !refersToConfiguration(selection)),
      secondary: Option.filter(state.slots.secondary, (selection) => !refersToConfiguration(selection)),
    },
    recentModels: {
      primary: state.recentModels.primary.filter((model) => !refersToConfiguration(model)),
      secondary: state.recentModels.secondary.filter((model) => !refersToConfiguration(model)),
    },
    favorites: state.favorites.filter((model) => !refersToConfiguration(model)),
  }
}

export const makeRetainedModelConfigurations = (
  state: StateHandle<ModelState, StateDocumentError>,
): RetainedModelConfigurationsApi => ({
  get: state.get.pipe(Effect.map((current) => current.configurations)),
  recoveryCompleted: state.get.pipe(Effect.map((current) => current.configurationRecoveryCompleted)),
  changes: state.changes.pipe(Stream.map((current) => current.configurations)),
  resolve: (id) => state.get.pipe(Effect.map((current) =>
    Option.fromNullable(current.configurations.find((candidate) => candidate.id === id)))),
  materialize: (configuration) => state.modify((current) => {
    const existing = current.configurations.find((candidate) => candidate.id === configuration.id)
    if (existing !== undefined) return [existing, current] as const
    const replaced = current.configurations.filter((candidate) =>
      sameServableModelBundleIdentity(candidate.bundle, configuration.bundle))
    const withoutReplacedReferences = replaced.reduce(
      (next, candidate) => removeReferences(next, candidate.id),
      current,
    )
    return [configuration, {
      ...withoutReplacedReferences,
      configurations: [...withoutReplacedReferences.configurations, configuration],
    }] as const
  }),
  remove: (id) => state.modify((current) => {
    const removed = Option.fromNullable(current.configurations.find((candidate) => candidate.id === id))
    return [removed, Option.isNone(removed) ? current : removeReferences(current, id)] as const
  }),
  completeRecovery: (defaults) => state.modify<readonly ModelServingConfiguration[]>((latest) => {
    if (latest.configurationRecoveryCompleted) {
      return [[], latest] as const
    }
    const additions: ModelServingConfiguration[] = []
    for (const candidate of defaults) {
      const accepted = [...latest.configurations, ...additions]
      if (
        accepted.some((retained) => retained.id === candidate.id)
        || accepted.some((retained) =>
          sameServableModelBundleIdentity(retained.bundle, candidate.bundle))
      ) continue
      additions.push(candidate)
    }
    return [additions, {
      ...latest,
      configurations: [...latest.configurations, ...additions],
      configurationRecoveryCompleted: true,
    }] as const
  }),
})

export const RetainedModelConfigurationsLive: Layer.Layer<
  RetainedModelConfigurations,
  never,
  MagnitudeStorage
> = Layer.effect(RetainedModelConfigurations, Effect.gen(function* () {
  const storage = yield* MagnitudeStorage
  return RetainedModelConfigurations.of(makeRetainedModelConfigurations(storage.models))
}))
