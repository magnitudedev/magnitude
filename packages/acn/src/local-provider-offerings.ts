import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelMutationFailed,
  type LocalInferenceError,
  ModelCapabilitiesSchema,
  type LocalProviderOffering,
  type ModelCapabilities,
  type ServableModelBundle,
  type ModelServingConfiguration,
  type ModelPackageEntry,
  type RecommendableModel,
  servableModelBundlePackageIds,
  ServableModelBundleSchema,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  ProviderModelCatalogEntrySchema,
  type ProviderModelCatalogEntry,
  type ModelServingConfigurationId,
} from "@magnitudedev/acn-protocol"
import {
  ProviderModelIdSchema,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { IcnCatalog, IcnInstalledModels } from "@magnitudedev/icn"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import {
  modelPackageFromIcn,
  packageInspectionFromIcn,
  recommendableModelFromIcn,
} from "./local-model-icn-adapter"
import { LocalModelPackages } from "./local-model-packages"
import { LocalModelConfigurationResolver } from "./local-model-configuration-resolver"
import { makeObservedState } from "./mirrored-state"
import { resolveBundlePresentation } from "./local-model-presentation"

export const localProviderModelId = (
  configurationId: ModelServingConfigurationId,
): ProviderModelId => ProviderModelIdSchema.make(configurationId)

export type ProviderOfferingPackageEvidence = readonly {
  readonly providerModelId: LocalProviderOffering["providerModelId"]
  readonly configurationId: LocalProviderOffering["configuration"]["id"]
  readonly packages: readonly {
    readonly packageId: ModelPackageEntry["package"]["id"]
    readonly installed: boolean
    readonly inspection: ModelPackageEntry["inspection"]["_tag"]
  }[]
}[]

export const providerOfferingPackageEvidence = (
  offerings: readonly Pick<LocalProviderOffering, "providerModelId" | "configuration">[],
  entries: ReadonlyMap<ModelPackageEntry["package"]["id"], ModelPackageEntry>,
): ProviderOfferingPackageEvidence => [...offerings]
  .sort((left, right) => left.providerModelId.localeCompare(right.providerModelId))
  .map((offering) => ({
    providerModelId: offering.providerModelId,
    configurationId: offering.configuration.id,
    packages: servableModelBundlePackageIds(offering.configuration.bundle).map((packageId) => {
      const entry = entries.get(packageId)
      return {
        packageId,
        installed: entry?.localState._tag === "Installed",
        inspection: entry?.inspection._tag ?? "Pending",
      }
    }).sort((left, right) => left.packageId.localeCompare(right.packageId)),
  }))

export const sameProviderOfferingPackageEvidence = (
  left: ProviderOfferingPackageEvidence,
  right: ProviderOfferingPackageEvidence,
): boolean => left.length === right.length && left.every((offering, index) => {
  const other = right[index]
  return offering.providerModelId === other?.providerModelId
    && offering.configurationId === other.configurationId
    && offering.packages.length === other.packages.length
    && offering.packages.every((modelPackage, packageIndex) => {
      const otherPackage = other.packages[packageIndex]
      return modelPackage.packageId === otherPackage?.packageId
        && modelPackage.installed === otherPackage.installed
        && modelPackage.inspection === otherPackage.inspection
    })
})

export interface LocalProviderOfferingsState {
  readonly packageEvidence: Option.Option<ProviderOfferingPackageEvidence>
  readonly entries: readonly ProviderModelCatalogEntry[]
  readonly failure: Option.Option<LocalInferenceError>
}

const failure = (operation: string, error: unknown) =>
  new LocalModelMutationFailed({
    code: operation,
    message: error instanceof Error ? error.message : String(error),
    retryable: true,
  })

const capabilitySet = (
  bundle: ServableModelBundle,
  catalog: readonly RecommendableModel[],
  installed: readonly Pick<ModelPackageEntry, "package" | "inspection">[],
): ModelCapabilities => {
  const sameBundle = Schema.equivalence(ServableModelBundleSchema)
  const recommendation = catalog.find((model) => sameBundle(model.configuration.bundle, bundle))
  if (recommendation) return recommendation.capabilities
  const primaryPackageId = bundle._tag === "Standalone" ? bundle.package.id : bundle.target.id
  const inspection = installed.find(({ package: modelPackage }) =>
    modelPackage.id === primaryPackageId)?.inspection
  return inspection?._tag === "Inspected"
    ? inspection.capabilities
    : ModelCapabilitiesSchema.make({
        vision: false,
        tools: false,
        structuredOutput: false,
        reasoning: {
          supported: false,
          efforts: [],
          defaultEffort: Option.none(),
        },
      })
}

export interface LocalProviderOfferingsApi {
  readonly list: Effect.Effect<readonly LocalProviderOffering[], LocalInferenceError>
  readonly changes: Stream.Stream<void>
  readonly catalog: Effect.Effect<readonly ProviderModelCatalogEntry[], LocalInferenceError>
  readonly state: Effect.Effect<LocalProviderOfferingsState>
  readonly catalogChanges: Stream.Stream<void>
  readonly resolve: (
    providerModelId: ProviderModelId,
  ) => Effect.Effect<LocalProviderOffering, LocalInferenceError>
}

export class LocalProviderOfferings extends Context.Tag("LocalProviderOfferings")<
  LocalProviderOfferings,
  LocalProviderOfferingsApi
>() {}

export const LocalProviderOfferingsLive: Layer.Layer<
  LocalProviderOfferings,
  never,
  IcnCatalog | IcnInstalledModels | LocalModelConfigurationResolver | LocalModelPackages
> = Layer.scoped(LocalProviderOfferings, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const installed = yield* IcnInstalledModels
  const resolver = yield* LocalModelConfigurationResolver
  const packages = yield* LocalModelPackages

  const capabilitySources = Effect.all({
    catalog: catalog.get.pipe(
      Effect.flatMap((snapshot) => Effect.forEach(
        snapshot.state.models,
        recommendableModelFromIcn,
      ).pipe(Effect.map((models) => ({ revision: snapshot.revision, models })))),
      Effect.mapError((error) => failure("read_recommendable_model_catalog_failed", error)),
    ),
    installed: installed.get.pipe(
      Effect.flatMap((snapshot) => Effect.forEach(
        snapshot.state.packages,
        (entry) => Effect.all({
          package: modelPackageFromIcn(entry.package),
          inspection: packageInspectionFromIcn(entry.inspection),
        }),
      ).pipe(Effect.map((entries) => ({ revision: snapshot.revision, entries })))),
      Effect.mapError((error) => failure("read_installed_model_capabilities_failed", error)),
    ),
  })

  const offeringsFrom = (
    configurations: readonly ModelServingConfiguration[],
    sources: Effect.Effect.Success<typeof capabilitySources>,
  ): readonly LocalProviderOffering[] => configurations.map((configuration) => ({
    providerModelId: localProviderModelId(configuration.id),
    configuration,
    capabilities: capabilitySet(
      configuration.bundle,
      sources.catalog.models,
      sources.installed.entries,
    ),
  }))

  const list: LocalProviderOfferingsApi["list"] = Effect.gen(function* () {
    const configurations = [...(yield* resolver.get).values()].map(({ configuration }) =>
      configuration)
    const sources = yield* capabilitySources
    return offeringsFrom(configurations, sources)
  }).pipe(
    Effect.mapError((error) => error instanceof LocalModelMutationFailed
      ? error
      : failure("read_local_provider_offerings_failed", error)),
  )

  const changes = Stream.mergeAll([
    resolver.changes,
    catalog.changes.pipe(Stream.map(() => undefined)),
    installed.changes.pipe(Stream.map(() => undefined)),
  ], { concurrency: "unbounded" })

  const observed = yield* makeObservedState<LocalProviderOfferingsState>({
    packageEvidence: Option.none(),
    entries: [],
    failure: Option.none(),
  })
  const entriesEquivalent = Schema.equivalence(Schema.Array(ProviderModelCatalogEntrySchema))
  const stateEquivalent = (
    left: LocalProviderOfferingsState,
    right: LocalProviderOfferingsState,
  ): boolean => Option.match(left.packageEvidence, {
    onNone: () => Option.isNone(right.packageEvidence),
    onSome: (evidence) => Option.exists(right.packageEvidence, (other) =>
      sameProviderOfferingPackageEvidence(evidence, other)),
  }) && entriesEquivalent(left.entries, right.entries)
    && Option.getOrUndefined(left.failure)?.message === Option.getOrUndefined(right.failure)?.message

  const compute = Effect.gen(function* () {
    const resolved = [...(yield* resolver.get).values()]
    const configurations = resolved.map(({ configuration }) => configuration)
    const sources = yield* capabilitySources
    const catalogModels = sources.catalog.models
    const packageSnapshot = yield* packages.snapshot
    const packageEntries = new Map(
      packageSnapshot.state.entries.map((entry) => [entry.package.id, entry]),
    )
    const configured = offeringsFrom(configurations, sources)
    const packageEvidence = providerOfferingPackageEvidence(configured, packageEntries)
    const bundleEntries = configured.map(({ configuration }) =>
      servableModelBundlePackageIds(configuration.bundle).map((id) => packageEntries.get(id)))
    const installedBundles = bundleEntries.map((entries) =>
      entries.every((entry) => entry?.localState._tag === "Installed"))
    const inspectable = bundleEntries.map((entries, index) => installedBundles[index]
      && entries.every((entry) => entry?.inspection._tag === "Inspected"))
    const sameBundle = Schema.equivalence(ServableModelBundleSchema)
    const entries = configured.map((offering, index): ProviderModelCatalogEntry => {
      const { bundle, profile } = offering.configuration
      const installed = installedBundles[index] ?? false
      const assessment = inspectable[index]
        ? Option.getOrUndefined(resolved[index]?.assessment ?? Option.none())
        : undefined
      const curated = catalogModels.find((model) =>
        sameBundle(model.configuration.bundle, bundle))
      const presentation = resolveBundlePresentation(bundle, curated && {
        displayName: curated.displayName,
        variantLabel: curated.variantLabel,
        description: curated.description,
        license: curated.license,
      })
      return {
        providerId: LOCAL_PROVIDER_ID,
        providerModelId: offering.providerModelId,
        modelFamilyId: Option.none(),
        displayName: presentation.displayName,
        variantLabel: Option.some(presentation.variantLabel),
        supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
        contextWindow: profile.contextLength,
        maxOutputTokens: profile.contextLength,
        memory: assessment?._tag === "Fits"
          ? Option.some(assessment.assessment.memory)
          : assessment?._tag === "DoesNotFit" ? Option.some(assessment.memory) : Option.none(),
        capabilities: offering.capabilities,
        availability: !installed
          ? { _tag: "Disabled", reason: "installation_unavailable" }
          : assessment?._tag === "Fits"
            ? { _tag: "Available" }
            : assessment?._tag === "DoesNotFit"
              ? { _tag: "Disabled", reason: "insufficient_resources" }
              : assessment?._tag === "Incompatible"
                ? { _tag: "Disabled", reason: "incompatible_runtime" }
                : { _tag: "Disabled", reason: "provider_unavailable" },
        pricing: Option.none(),
      }
    })
    return {
      entries,
      packageEvidence,
    }
  })

  const publishCurrent: Effect.Effect<void, LocalInferenceError> = compute.pipe(
    Effect.flatMap(({ entries, packageEvidence }) => observed.setIfChanged({
      packageEvidence: Option.some(packageEvidence),
      entries,
      failure: Option.none(),
    }, stateEquivalent)),
  )

  const project = publishCurrent.pipe(
    Effect.catchAll((error) => observed.get.pipe(Effect.flatMap(({ state }) =>
      observed.setIfChanged(
        { ...state, failure: Option.some(error) },
        stateEquivalent,
      ).pipe(Effect.asVoid)))),
    Effect.catchAllCause((cause) => Effect.logWarning(
      "Unable to project local provider offerings",
    ).pipe(Effect.annotateLogs({ cause: String(cause) }))),
  )
  yield* Stream.make(undefined).pipe(
    Stream.concat(Stream.mergeAll([
      changes,
      packages.changes.pipe(Stream.map(() => undefined)),
    ], { concurrency: "unbounded" }).pipe(Stream.debounce("25 millis"))),
    Stream.runForEach(() => project),
    Effect.forkScoped,
  )

  return LocalProviderOfferings.of({
    list,
    changes,
    catalog: observed.get.pipe(Effect.flatMap(({ state }) => Option.match(state.failure, {
      onNone: () => Effect.succeed(state.entries),
      onSome: Effect.fail,
    }))),
    state: observed.get.pipe(Effect.map(({ state }) => state)),
    catalogChanges: observed.changes.pipe(Stream.map(() => undefined)),
    resolve: (providerModelId) => list.pipe(Effect.flatMap((offerings) => {
      const offering = offerings.find((candidate) => candidate.providerModelId === providerModelId)
      return offering
        ? Effect.succeed(offering)
        : Effect.fail(new LocalModelMutationFailed({
            code: "local_provider_offering_not_found",
            message: `Local provider offering ${providerModelId} was not found`,
            retryable: false,
          }))
    })),
  })
}))
