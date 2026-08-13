import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelMutationFailed,
  ModelServingConfigurationSchema,
  servableModelBundlePackageIds,
  type LocalInferenceError,
  type ModelPackage,
  type ModelServingConfiguration,
  type ModelServingConfigurationId,
} from "@magnitudedev/acn-protocol"
import { IcnCatalog } from "@magnitudedev/icn"
import { recommendableModelFromIcn } from "./local-model-icn-adapter"
import {
  LocalModelAssessor,
  type CoordinatedLocalModelAssessment,
} from "./local-model-assessor"
import { LocalModelPackages } from "./local-model-packages"
import { RetainedModelConfigurations } from "./retained-model-configurations"

export interface ResolvedLocalModelConfiguration {
  readonly configuration: ModelServingConfiguration
  readonly assessment: Option.Option<CoordinatedLocalModelAssessment["assessment"]>
}

const LocalModelBundleIdentitySchema = Schema.String.pipe(
  Schema.brand("LocalModelBundleIdentity"),
)
type LocalModelBundleIdentity = typeof LocalModelBundleIdentitySchema.Type

export interface LocalModelConfigurationResolverApi {
  readonly get: Effect.Effect<
    ReadonlyMap<LocalModelBundleIdentity, ResolvedLocalModelConfiguration>,
    LocalInferenceError
  >
  readonly changes: Stream.Stream<void>
  readonly catalogReady: Effect.Effect<boolean>
  readonly resolve: (
    configurationId: ModelServingConfigurationId,
  ) => Effect.Effect<Option.Option<ResolvedLocalModelConfiguration>, LocalInferenceError>
}

export class LocalModelConfigurationResolver extends Context.Tag(
  "LocalModelConfigurationResolver",
)<LocalModelConfigurationResolver, LocalModelConfigurationResolverApi>() {}

export const localModelBundleIdentity = (
  bundle: ModelServingConfiguration["bundle"],
): LocalModelBundleIdentity => LocalModelBundleIdentitySchema.make(
  bundle._tag === "Standalone"
    ? `Standalone\0${bundle.package.id}`
    : bundle.draftSource._tag === "Embedded"
      ? `SpeculativeDecoding\0${bundle.target.id}\0Embedded\0${JSON.stringify(bundle.method)}`
      : `SpeculativeDecoding\0${bundle.target.id}\0Separate\0${bundle.draftSource.draft.id}\0${JSON.stringify(bundle.method)}`,
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
  readonly retained: readonly ModelServingConfiguration[]
  readonly catalog: readonly ModelServingConfiguration[]
  readonly assessed: ReadonlyMap<ModelServingConfigurationId, CoordinatedLocalModelAssessment>
  readonly installedPackageIds: ReadonlySet<string>
}): ReadonlyMap<LocalModelBundleIdentity, ResolvedLocalModelConfiguration> => {
  const configurations = new Map<LocalModelBundleIdentity, ModelServingConfiguration>()
  for (const { configuration, origin } of input.assessed.values()) {
    if (origin === "Standard" && servableModelBundlePackageIds(configuration.bundle).every(
      (packageId) => input.installedPackageIds.has(packageId),
    )) {
      configurations.set(localModelBundleIdentity(configuration.bundle), configuration)
    }
  }
  for (const configuration of input.catalog) {
    configurations.set(localModelBundleIdentity(configuration.bundle), configuration)
  }
  for (const configuration of input.retained) {
    configurations.set(localModelBundleIdentity(configuration.bundle), configuration)
  }
  return new Map([...configurations].map(([identity, configuration]) => {
    const assessed = input.assessed.get(configuration.id)
    const assessment = assessed !== undefined
      && sameConfiguration(assessed.configuration, configuration)
      ? Option.some(assessed.assessment)
      : Option.none()
    return [identity, { configuration, assessment }] as const
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
  RetainedModelConfigurations | IcnCatalog | LocalModelAssessor | LocalModelPackages
> = Layer.effect(LocalModelConfigurationResolver, Effect.gen(function* () {
  const retained = yield* RetainedModelConfigurations
  const catalog = yield* IcnCatalog
  const assessor = yield* LocalModelAssessor
  const packages = yield* LocalModelPackages

  const get = Effect.gen(function* () {
    const retainedConfigurations = yield* retained.get
    const catalogConfigurations = (yield* catalog.ready)
      ? yield* Effect.forEach(
          (yield* catalog.get).state.models,
          recommendableModelFromIcn,
        ).pipe(Effect.map((models) => models.map(({ configuration }) => configuration)))
      : []
    return resolveLocalModelConfigurations({
      retained: retainedConfigurations,
      catalog: catalogConfigurations,
      assessed: yield* assessor.state,
      installedPackageIds: yield* packages.installedPackageIds,
    })
  }).pipe(Effect.mapError(failure))

  const changes = Stream.mergeAll([
    retained.changes.pipe(Stream.map(() => undefined)),
    catalog.changes.pipe(Stream.map(() => undefined)),
    assessor.changes.pipe(Stream.map(() => undefined)),
    packages.changes.pipe(Stream.map(() => undefined)),
  ], { concurrency: "unbounded" })

  return LocalModelConfigurationResolver.of({
    get,
    changes,
    catalogReady: catalog.ready,
    resolve: (configurationId) => get.pipe(Effect.map((resolved) =>
      Option.fromNullable([...resolved.values()].find(({ configuration }) =>
        configuration.id === configurationId)))),
  })
}))
