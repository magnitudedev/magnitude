import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelMutationFailed,
  ModelServingConfigurationSchema,
  servableModelBundlePackageIds,
  servableModelBundleTargetPackageId,
  type LocalInferenceError,
  type CatalogIdentity,
  type ModelPackage,
  type ModelServingConfiguration,
  type ModelServingConfigurationId,
  type RecommendableModel,
  type InstalledCatalogAttribution,
} from "@magnitudedev/acn-protocol"
import { IcnModels } from "@magnitudedev/icn"
import {
  catalogIdentityFromIcn,
  catalogModelDefinitionFromIcn,
  catalogModelEffectiveConfigurationFromIcn,
} from "./local-model-icn-adapter"
import {
  LocalModelAssessor,
  type CoordinatedLocalModelAssessment,
} from "./local-model-assessor"
import { LocalModelPackages } from "./local-model-packages"
import { localCatalogProviderModelId } from "./local-provider-model-id"

export interface ResolvedLocalModelConfiguration {
  readonly servingConfiguration: ModelServingConfiguration
  readonly catalogModel: Option.Option<RecommendableModel>
  readonly assessment: Option.Option<CoordinatedLocalModelAssessment["assessment"]>
}

const LocalModelTargetIdentitySchema = Schema.String.pipe(
  Schema.brand("LocalModelTargetIdentity"),
)
type LocalModelTargetIdentity = typeof LocalModelTargetIdentitySchema.Type

export interface LocalModelConfigurationResolverApi {
  readonly get: Effect.Effect<
    ReadonlyMap<LocalModelTargetIdentity, ResolvedLocalModelConfiguration>,
    LocalInferenceError
  >
  readonly changes: Stream.Stream<void>
  /**
   * True once the resolved configuration set is a complete account of the
   * local models that exist: the native catalog has been observed and every
   * currently desired assessment has completed. Absence of a configuration is
   * only meaningful when this is true.
   */
  readonly settled: Effect.Effect<boolean>
  readonly resolve: (
    configurationId: ModelServingConfigurationId,
  ) => Effect.Effect<Option.Option<ResolvedLocalModelConfiguration>, LocalInferenceError>
}

export class LocalModelConfigurationResolver extends Context.Tag(
  "LocalModelConfigurationResolver",
)<LocalModelConfigurationResolver, LocalModelConfigurationResolverApi>() {}

export const localModelTargetIdentity = (
  bundle: ModelServingConfiguration["bundle"],
): LocalModelTargetIdentity => LocalModelTargetIdentitySchema.make(
  servableModelBundleTargetPackageId(bundle),
)

export const configuredModelPackageIds = (
  configurations: Iterable<ModelServingConfiguration>,
): ReadonlySet<string> => new Set([...configurations].flatMap(({ bundle }) =>
  servableModelBundlePackageIds(bundle)))

export const isStandalonePackageCandidate = (
  modelPackage: ModelPackage,
  configuredPackageIds: ReadonlySet<string>,
): boolean => !configuredPackageIds.has(modelPackage.id)
  && modelPackage.files.some(({ role }) => role === "weights")

const sameConfiguration = Schema.equivalence(ModelServingConfigurationSchema)

export const resolveLocalModelConfigurations = (input: {
  readonly catalog: readonly RecommendableModel[]
  readonly effectiveCatalogConfigurations: readonly {
    readonly identity: CatalogIdentity
    readonly configuration: ModelServingConfiguration
  }[]
  readonly assessed: ReadonlyMap<ModelServingConfigurationId, CoordinatedLocalModelAssessment>
  readonly installedPackageIds: ReadonlySet<string>
  readonly catalogAttributionByPackageId?: ReadonlyMap<string, InstalledCatalogAttribution>
}): ReadonlyMap<LocalModelTargetIdentity, ResolvedLocalModelConfiguration> => {
  const configurations = new Map<LocalModelTargetIdentity, ModelServingConfiguration>()
  for (const { configuration, origin } of input.assessed.values()) {
    if (origin === "Standard" && servableModelBundlePackageIds(configuration.bundle).every(
      (packageId) => input.installedPackageIds.has(packageId),
    )) {
      const attribution = input.catalogAttributionByPackageId?.get(
        servableModelBundleTargetPackageId(configuration.bundle),
      )
      if (attribution?._tag !== "Attributed") {
        configurations.set(localModelTargetIdentity(configuration.bundle), configuration)
      }
    }
  }
  const catalogByIdentity = new Map(input.catalog.map((model) => [
    LocalModelTargetIdentitySchema.make(localCatalogProviderModelId(model)),
    model,
  ]))
  const effectiveCatalogByIdentity = new Map(input.effectiveCatalogConfigurations.map((entry) => [
    LocalModelTargetIdentitySchema.make(localCatalogProviderModelId(entry.identity)),
    entry.configuration,
  ]))
  for (const [identity, model] of catalogByIdentity) {
    configurations.set(
      identity,
      effectiveCatalogByIdentity.get(identity) ?? model.configuration,
    )
  }
  return new Map([...configurations].map(([identity, servingConfiguration]) => {
    const assessed = input.assessed.get(servingConfiguration.id)
    const assessment = assessed !== undefined
      && sameConfiguration(assessed.configuration, servingConfiguration)
      ? Option.some(assessed.assessment)
      : Option.none()
    const catalogModel = catalogByIdentity.get(identity)
    return [identity, {
      servingConfiguration,
      catalogModel: Option.fromNullable(catalogModel),
      assessment,
    }] as const
  }))
}

const failure = (error: unknown) => new LocalModelMutationFailed({
  code: "resolve_local_model_configurations_failed",
  message: error instanceof Error ? error.message : String(error),
  retryable: true,
})

export const LocalModelConfigurationResolverLive: Layer.Layer<
  LocalModelConfigurationResolver,
  never,
  IcnModels | LocalModelAssessor | LocalModelPackages
> = Layer.effect(LocalModelConfigurationResolver, Effect.gen(function* () {
  const models = yield* IcnModels
  const assessor = yield* LocalModelAssessor
  const packages = yield* LocalModelPackages

  const get = Effect.gen(function* () {
    const nativeCatalogModels = (yield* models.initialized)
      ? (yield* models.get).state.catalogModels
      : []
    const catalogModels = yield* Effect.forEach(
      nativeCatalogModels,
      catalogModelDefinitionFromIcn,
    )
    const effectiveCatalogConfigurationOptions = yield* Effect.forEach(
      nativeCatalogModels,
      (model) =>
        Effect.all([
          catalogIdentityFromIcn(model),
          catalogModelEffectiveConfigurationFromIcn(model),
        ]).pipe(Effect.map(([identity, configuration]) => Option.map(
          configuration,
          (effective) => [
            identity,
            effective,
          ] as const,
        ))),
    )
    const effectiveCatalogConfigurations = effectiveCatalogConfigurationOptions.flatMap((entry) =>
      Option.isSome(entry)
        ? [{ identity: entry.value[0], configuration: entry.value[1] }]
        : [])
    const packageState = (yield* packages.snapshot).state
    return resolveLocalModelConfigurations({
      catalog: catalogModels,
      effectiveCatalogConfigurations,
      assessed: yield* assessor.state,
      installedPackageIds: yield* packages.installedPackageIds,
      catalogAttributionByPackageId: new Map(packageState.entries.map((entry) => [
        entry.package.id,
        entry.catalogAttribution,
      ])),
    })
  }).pipe(Effect.mapError(failure))

  const changes = Stream.mergeAll([
    models.changes.pipe(Stream.map(() => undefined)),
    assessor.changes.pipe(Stream.map(() => undefined)),
    packages.changes.pipe(Stream.map(() => undefined)),
  ], { concurrency: "unbounded" })

  return LocalModelConfigurationResolver.of({
    get,
    changes,
    settled: Effect.gen(function* () {
      return (yield* models.initialized) && (yield* assessor.settled)
    }),
    resolve: (configurationId) => get.pipe(Effect.map((resolved) =>
      Option.fromNullable([...resolved.values()].find(({ servingConfiguration }) =>
        servingConfiguration.id === configurationId)))),
  })
}))
