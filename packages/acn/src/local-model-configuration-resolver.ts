import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelConfigurationAssessmentSchema,
  LocalModelMutationFailed,
  ModelPackageInspectionSchema,
  ModelServingConfigurationSchema,
  RecommendableModelSchema,
  servableModelBundleTargetPackageId,
  type LocalInferenceError,
  type CatalogIdentity,
  type ModelPackageEntry,
  type ModelPackageId,
  type ModelPackageInspection,
  type ModelServingConfiguration,
  type RecommendableModel,
} from "@magnitudedev/acn-protocol"
import { LocalModelCatalogAdapter } from "./local-model-catalog-adapter"
import {
  LocalModelAssessor,
  type CoordinatedLocalModelAssessment,
} from "./local-model-assessor"
import { LocalModelPackages } from "./local-model-packages"
import { localCatalogProviderModelId } from "./local-provider-model-id"
import { materializeProjection } from "./materialized-projection"

export interface ResolvedLocalModelConfiguration {
  readonly servingConfiguration: ModelServingConfiguration
  readonly catalogModel: Option.Option<RecommendableModel>
  readonly assessment: CoordinatedLocalModelAssessment["assessment"]
  readonly targetInspection: ModelPackageInspection
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
}

export class LocalModelConfigurationResolver extends Context.Tag(
  "LocalModelConfigurationResolver",
)<LocalModelConfigurationResolver, LocalModelConfigurationResolverApi>() {}

const sameConfiguration = Schema.equivalence(ModelServingConfigurationSchema)
const sameCompletedAssessment = Schema.equivalence(LocalModelConfigurationAssessmentSchema)
const sameInspection = Schema.equivalence(ModelPackageInspectionSchema)
const sameCatalogModel = Schema.equivalence(RecommendableModelSchema)
const encodeConfiguration = Schema.encodeSync(ModelServingConfigurationSchema)

const configurationKey = (configuration: ModelServingConfiguration): string =>
  JSON.stringify(encodeConfiguration(configuration))

export const resolveLocalModelConfigurations = (input: {
  readonly catalog: readonly RecommendableModel[]
  readonly effectiveCatalogConfigurations: readonly {
    readonly identity: CatalogIdentity
    readonly configuration: ModelServingConfiguration
  }[]
  readonly assessed: readonly CoordinatedLocalModelAssessment[]
  readonly packageEntries: ReadonlyMap<ModelPackageId, ModelPackageEntry>
}): ReadonlyMap<LocalModelTargetIdentity, ResolvedLocalModelConfiguration> => {
  const configurations = new Map<LocalModelTargetIdentity, ModelServingConfiguration>()
  const catalogByIdentity = new Map(input.catalog.map((model) => [
    LocalModelTargetIdentitySchema.make(localCatalogProviderModelId(model)),
    model,
  ]))
  const effectiveCatalogByIdentity = new Map(input.effectiveCatalogConfigurations.map((entry) => [
    LocalModelTargetIdentitySchema.make(localCatalogProviderModelId(entry.identity)),
    entry.configuration,
  ]))
  const assessedByConfiguration = new Map(input.assessed.map((entry) => [
    configurationKey(entry.configuration),
    entry,
  ]))
  for (const [identity, model] of catalogByIdentity) {
    configurations.set(
      identity,
      effectiveCatalogByIdentity.get(identity) ?? model.configuration,
    )
  }
  return new Map([...configurations].map(([identity, servingConfiguration]) => {
    const assessed = assessedByConfiguration.get(configurationKey(servingConfiguration))
    const assessment: CoordinatedLocalModelAssessment["assessment"] = assessed !== undefined
      ? assessed.assessment
      : { _tag: "Assessing" }
    const catalogModel = catalogByIdentity.get(identity)
    const targetEntry = input.packageEntries.get(
      servableModelBundleTargetPackageId(servingConfiguration.bundle),
    )
    const targetInspection: ModelPackageInspection = targetEntry?.localState._tag === "Installed"
      ? targetEntry.inspection
      : catalogModel !== undefined
        && sameConfiguration(catalogModel.configuration, servingConfiguration)
        ? { _tag: "Inspected", capabilities: catalogModel.capabilities }
        : { _tag: "Pending" }
    return [identity, {
      servingConfiguration,
      catalogModel: Option.fromNullable(catalogModel),
      assessment,
      targetInspection,
    }] as const
  }))
}

const failure = (error: unknown) => new LocalModelMutationFailed({
  code: "resolve_local_model_configurations_failed",
  message: error instanceof Error ? error.message : String(error),
  retryable: true,
})

const sameOptionalCatalogModel = (
  left: Option.Option<RecommendableModel>,
  right: Option.Option<RecommendableModel>,
): boolean => Option.isNone(left)
  ? Option.isNone(right)
  : Option.isSome(right) && sameCatalogModel(left.value, right.value)

const sameAssessment = (
  left: CoordinatedLocalModelAssessment["assessment"],
  right: CoordinatedLocalModelAssessment["assessment"],
): boolean => left._tag === "Assessing"
  ? right._tag === "Assessing"
  : right._tag !== "Assessing" && sameCompletedAssessment(left, right)

const sameResolvedConfigurations = (
  left: ReadonlyMap<LocalModelTargetIdentity, ResolvedLocalModelConfiguration>,
  right: ReadonlyMap<LocalModelTargetIdentity, ResolvedLocalModelConfiguration>,
): boolean => left.size === right.size && [...left].every(([identity, value]) => {
  const other = right.get(identity)
  return other !== undefined
    && sameConfiguration(value.servingConfiguration, other.servingConfiguration)
    && sameOptionalCatalogModel(value.catalogModel, other.catalogModel)
    && sameAssessment(value.assessment, other.assessment)
    && sameInspection(value.targetInspection, other.targetInspection)
})

type PackageResolutionEvidence = readonly {
  readonly packageId: ModelPackageId
  readonly installed: boolean
  readonly inspection: ModelPackageInspection
}[]

const packageResolutionEvidence = (
  entries: readonly ModelPackageEntry[],
): PackageResolutionEvidence => entries.map((entry) => ({
  packageId: entry.package.id,
  installed: entry.localState._tag === "Installed",
  inspection: entry.inspection,
}))

const samePackageResolutionEvidence = (
  left: PackageResolutionEvidence,
  right: PackageResolutionEvidence,
): boolean => left.length === right.length && left.every((entry, index) => {
  const other = right[index]
  return other !== undefined
    && entry.packageId === other.packageId
    && entry.installed === other.installed
    && sameInspection(entry.inspection, other.inspection)
})

export const LocalModelConfigurationResolverLive: Layer.Layer<
  LocalModelConfigurationResolver,
  never,
  LocalModelCatalogAdapter | LocalModelAssessor | LocalModelPackages
> = Layer.scoped(LocalModelConfigurationResolver, Effect.gen(function* () {
  const catalog = yield* LocalModelCatalogAdapter
  const assessor = yield* LocalModelAssessor
  const packages = yield* LocalModelPackages

  const project = Effect.gen(function* () {
    const catalogState = yield* catalog.state
    const packageState = yield* packages.state
    const packageEntries = new Map(packageState.entries.map((entry) => [entry.package.id, entry]))
    return resolveLocalModelConfigurations({
      catalog: catalogState.entries.map((entry) => entry.model),
      effectiveCatalogConfigurations: catalogState.entries.flatMap((entry) =>
        Option.isSome(entry.effectiveConfiguration)
          ? [{ identity: entry.identity, configuration: entry.effectiveConfiguration.value }]
          : []),
      assessed: yield* assessor.state,
      packageEntries,
    })
  }).pipe(Effect.mapError(failure))

  const sourceChanges = Stream.mergeAll([
    catalog.changes.pipe(Stream.map(() => undefined)),
    assessor.changes.pipe(Stream.map(() => undefined)),
    packages.changes.pipe(
      Stream.map((state) => packageResolutionEvidence(state.entries)),
      Stream.changesWith(samePackageResolutionEvidence),
      Stream.map(() => undefined),
    ),
  ], { concurrency: "unbounded" })
  const projection = yield* materializeProjection({
    project: project.pipe(Effect.orDie),
    invalidations: sourceChanges,
    equivalent: sameResolvedConfigurations,
  })

  return LocalModelConfigurationResolver.of({
    get: projection.get,
    changes: projection.changes.pipe(Stream.map(() => undefined)),
    settled: Effect.gen(function* () {
      return (yield* catalog.state).reconciliationComplete && (yield* assessor.settled)
    }),
  })
}))
