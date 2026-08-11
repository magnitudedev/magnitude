import { Context, Effect, Layer, Option, Schema, Stream } from 'effect'
import {
  ModelServingConfigurationIdSchema,
  ModelServingConfigurationSchema,
  ServingProfileSchema,
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

export class RetainedConfigurationConflict extends Schema.TaggedError<RetainedConfigurationConflict>()(
  'RetainedConfigurationConflict',
  {
    configurationId: ModelServingConfigurationIdSchema,
    reason: Schema.String,
  },
) {}

export interface RetainedModelConfigurationsApi {
  readonly get: Effect.Effect<readonly ModelServingConfiguration[]>
  readonly recoveryCompleted: Effect.Effect<boolean>
  readonly changes: Stream.Stream<readonly ModelServingConfiguration[]>
  readonly resolve: (
    id: ModelServingConfigurationId,
  ) => Effect.Effect<Option.Option<ModelServingConfiguration>>
  readonly materialize: (
    configuration: ModelServingConfiguration,
  ) => Effect.Effect<ModelServingConfiguration, StateDocumentError | RetainedConfigurationConflict>
  readonly remove: (
    id: ModelServingConfigurationId,
  ) => Effect.Effect<Option.Option<ModelServingConfiguration>, StateDocumentError>
  readonly completeRecovery: (
    defaults: readonly ModelServingConfiguration[],
  ) => Effect.Effect<readonly ModelServingConfiguration[], StateDocumentError | RetainedConfigurationConflict>
}

export class RetainedModelConfigurations extends Context.Tag('RetainedModelConfigurations')<
  RetainedModelConfigurations,
  RetainedModelConfigurationsApi
>() {}

const sameConfiguration = Schema.equivalence(ModelServingConfigurationSchema)
const sameProfile = Schema.equivalence(ServingProfileSchema)

type MaterializeResult =
  | { readonly _tag: 'Saved'; readonly configuration: ModelServingConfiguration }
  | { readonly _tag: 'Conflict'; readonly conflict: RetainedConfigurationConflict }

type RecoveryResult =
  | { readonly _tag: 'Completed'; readonly additions: readonly ModelServingConfiguration[] }
  | { readonly _tag: 'Conflict'; readonly conflict: RetainedConfigurationConflict }

const compatibilityConflict = (
  current: readonly ModelServingConfiguration[],
  incoming: ModelServingConfiguration,
): RetainedConfigurationConflict | undefined => {
  const sameId = current.find((candidate) => candidate.id === incoming.id)
  if (sameId !== undefined && !sameConfiguration(sameId, incoming)) {
    return new RetainedConfigurationConflict({
      configurationId: incoming.id,
      reason: 'The configuration ID already identifies different bundle or profile values',
    })
  }
  const sameValue = current.find((candidate) =>
    sameServableModelBundleIdentity(candidate.bundle, incoming.bundle)
    && sameProfile(candidate.profile, incoming.profile))
  if (sameValue !== undefined && sameValue.id !== incoming.id) {
    return new RetainedConfigurationConflict({
      configurationId: incoming.id,
      reason: 'The bundle and profile already carry a different configuration ID',
    })
  }
  return undefined
}

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
  materialize: (configuration) => state.modify<MaterializeResult>((current) => {
    const conflict = compatibilityConflict(current.configurations, configuration)
    if (conflict !== undefined) return [{ _tag: 'Conflict' as const, conflict }, current] as const
    const existing = current.configurations.find((candidate) => candidate.id === configuration.id)
    if (existing !== undefined) return [{ _tag: 'Saved' as const, configuration: existing }, current] as const
    const replaced = current.configurations.filter((candidate) =>
      sameServableModelBundleIdentity(candidate.bundle, configuration.bundle))
    const withoutReplacedReferences = replaced.reduce(
      (next, candidate) => removeReferences(next, candidate.id),
      current,
    )
    return [{ _tag: 'Saved' as const, configuration }, {
      ...withoutReplacedReferences,
      configurations: [...withoutReplacedReferences.configurations, configuration],
    }] as const
  }).pipe(Effect.flatMap((result) => result._tag === 'Conflict'
    ? Effect.fail(result.conflict)
    : Effect.succeed(result.configuration))),
  remove: (id) => state.modify((current) => {
    const removed = Option.fromNullable(current.configurations.find((candidate) => candidate.id === id))
    return [removed, Option.isNone(removed) ? current : removeReferences(current, id)] as const
  }),
  completeRecovery: (defaults) => state.modify<RecoveryResult>((latest) => {
    if (latest.configurationRecoveryCompleted) {
      return [{ _tag: 'Completed' as const, additions: [] }, latest] as const
    }
    const acceptedDefaults = [...latest.configurations]
    for (const candidate of defaults) {
      const conflict = compatibilityConflict(acceptedDefaults, candidate)
      if (conflict !== undefined) return [{ _tag: 'Conflict' as const, conflict }, latest] as const
      if (!acceptedDefaults.some((accepted) => accepted.id === candidate.id)) {
        acceptedDefaults.push(candidate)
      }
    }
    const additions = acceptedDefaults.slice(latest.configurations.length).filter((candidate) =>
      !latest.configurations.some((retained) =>
        sameServableModelBundleIdentity(retained.bundle, candidate.bundle)))
    return [{ _tag: 'Completed' as const, additions }, {
      ...latest,
      configurations: [...latest.configurations, ...additions],
      configurationRecoveryCompleted: true,
    }] as const
  }).pipe(Effect.flatMap((result) => result._tag === 'Conflict'
    ? Effect.fail(result.conflict)
    : Effect.succeed(result.additions))),
})

export const RetainedModelConfigurationsLive: Layer.Layer<
  RetainedModelConfigurations,
  never,
  MagnitudeStorage
> = Layer.effect(RetainedModelConfigurations, Effect.gen(function* () {
  const storage = yield* MagnitudeStorage
  return RetainedModelConfigurations.of(makeRetainedModelConfigurations(storage.models))
}))
