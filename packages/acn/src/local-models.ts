import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelsMirror,
  ModelServingConfigurationSchema,
  ServableModelBundleSchema,
  type LocalModel,
  type LocalModelCatalogDownloadState,
  type LocalModelInstallation,
  type LocalModelCatalogCandidateAvailability,
  type LocalModelsState,
  type ModelFailure,
  type ModelServingConfiguration,
  type ServableModelBundle,
  type ModelPackageEntry,
  type ProviderModelCatalogEntry,
  servableModelBundlePackageIds,
} from "@magnitudedev/acn-protocol"
import type { ModelServingConfigurationId, ProviderModelId } from "@magnitudedev/sdk"
import { IcnCatalog } from "@magnitudedev/icn"
import { makeMirroredState, MirroredStateChanges } from "./mirrored-state"
import { LocalModelPackages } from "./local-model-packages"
import { LocalModelRecommendations } from "./local-model-recommendations"
import { LocalProviderOfferings } from "./local-provider-offerings"
import {
  providerOfferingPackageEvidence,
  sameProviderOfferingPackageEvidence,
} from "./local-provider-offerings"
import { LocalModelAssessor } from "./local-model-assessor"
import {
  localModelAssessmentProfiles,
  MINIMUM_LOCAL_MODEL_CONTEXT_LENGTH,
} from "./local-model-assessments"
import { recommendableModelFromIcn } from "./local-model-icn-adapter"
import { RetainedModelConfigurations } from "./retained-model-configurations"

interface ModelPresentation {
  readonly displayName: string
  readonly description: string
}

const bundlePackages = (bundle: ServableModelBundle) =>
  bundle._tag === "Standalone" ? [bundle.package] : [bundle.target, bundle.draft]
const sameConfiguration = Schema.equivalence(ModelServingConfigurationSchema)
const bundleIdentity = (bundle: ServableModelBundle): string =>
  bundle._tag === "Standalone"
    ? `Standalone\0${bundle.package.id}`
    : `SpeculativeDecodingPair\0${bundle.target.id}\0${bundle.draft.id}`

export const decideLocalModelConfigurations = (input: {
  readonly retained: readonly ModelServingConfiguration[]
  readonly catalog: readonly ModelServingConfiguration[]
  readonly assessed: readonly ModelServingConfiguration[]
}): ReadonlyMap<string, ModelServingConfiguration> => {
  const decided = new Map<string, ModelServingConfiguration>()
  const authoredIds = new Set([
    ...input.retained,
    ...input.catalog,
  ].map(({ id }) => id))
  for (const configuration of input.assessed) {
    if (!authoredIds.has(configuration.id)) {
      decided.set(bundleIdentity(configuration.bundle), configuration)
    }
  }
  for (const configuration of input.catalog) {
    decided.set(bundleIdentity(configuration.bundle), configuration)
  }
  for (const configuration of input.retained) {
    decided.set(bundleIdentity(configuration.bundle), configuration)
  }
  return decided
}

const sourceName = (bundle: ServableModelBundle): string => {
  const primary = bundle._tag === "Standalone" ? bundle.package : bundle.target
  return primary.source._tag === "HuggingFace"
    ? primary.source.repository.split("/").at(-1) ?? primary.source.repository
    : primary.files[0]?.path.split("/").at(-1) ?? primary.id
}

export const resolveBundlePresentation = (
  bundle: ServableModelBundle,
  curated: ModelPresentation | undefined,
): ModelPresentation => curated ?? {
  displayName: sourceName(bundle),
  description: "",
}

const installedBundle = (
  bundle: ServableModelBundle,
  entries: ReadonlyMap<string, ModelPackageEntry>,
): LocalModelInstallation | undefined => {
  const packages = bundlePackages(bundle)
  const packageEntries = packages.map((modelPackage) => entries.get(modelPackage.id))
  if (!packageEntries.every((entry) => entry?.localState._tag === "Installed")) {
    return undefined
  }
  const installedBytes = packages.reduce((total, modelPackage, index) =>
    total + (packageEntries[index]?.localState._tag === "Installed"
      ? modelPackage.files.reduce((sum, file) => sum + file.sizeBytes, 0)
      : 0), 0)
  const [origin, ...remainingOrigins] = [...new Set(packageEntries.flatMap((entry) =>
    entry?.localState._tag === "Installed" ? [entry.localState.origin] : []))]
  if (origin === undefined) {
    throw new Error("Installed servable model bundle has no installation origin")
  }
  return { installedBytes, origins: [origin, ...remainingOrigins] }
}

const aggregateDownload = (
  bundle: ServableModelBundle,
  entries: ReadonlyMap<string, ModelPackageEntry>,
): LocalModelCatalogDownloadState => {
  const packages = bundlePackages(bundle)
  const packageEntries = packages.map((modelPackage) => entries.get(modelPackage.id))
  const totalBytes = packages.reduce((total, modelPackage) =>
    total + modelPackage.files.reduce((sum, file) => sum + file.sizeBytes, 0), 0)
  const installedBytes = packages.reduce((total, modelPackage, index) =>
    total + (packageEntries[index]?.localState._tag === "Installed"
      ? modelPackage.files.reduce((sum, file) => sum + file.sizeBytes, 0)
      : 0), 0)
  const installation = installedBundle(bundle, entries)
  if (installation !== undefined) return { _tag: "Downloaded", ...installation }

  const downloading = packageEntries.flatMap((entry) =>
    entry?.localState._tag === "Downloading" ? [entry.localState] : [])
  const completedBytes = installedBytes + downloading.reduce(
    (total, state) => total + state.completedBytes,
    0,
  )
  if (downloading.length > 0) {
    const stages = downloading.map(({ stage }) => stage)
    const stage = stages.every((value) => value === stages[0])
      ? stages[0] ?? "queued"
      : stages.every((value) => value === "verifying" || value === "publishing")
        ? "verifying" as const
        : stages.some((value) => value === "downloading")
          ? "downloading" as const
          : stages[0] ?? "queued"
    const rates = downloading.flatMap(({ bytesPerSecond }) => Option.toArray(bytesPerSecond))
    return {
      _tag: "Downloading",
      attemptIds: downloading.map(({ attemptId }) => attemptId) as [
        typeof downloading[number]["attemptId"],
        ...Array<typeof downloading[number]["attemptId"]>,
      ],
      stage,
      completedBytes,
      totalBytes,
      bytesPerSecond: rates.length === 0
        ? Option.none()
        : Option.some(rates.reduce((total, rate) => total + rate, 0)),
    }
  }

  const failed = packageEntries.flatMap((entry) =>
    entry?.localState._tag === "DownloadFailed" ? [entry.localState] : [])[0]
  if (failed !== undefined) {
    const attemptIds = packageEntries.flatMap((entry) =>
      entry?.localState._tag === "DownloadFailed" ? [entry.localState.attemptId] : [])
    return {
      _tag: "Failed",
      attemptIds: attemptIds as [typeof attemptIds[number], ...Array<typeof attemptIds[number]>],
      completedBytes: installedBytes + packageEntries.reduce((total, entry) =>
        total + (entry?.localState._tag === "DownloadFailed" ? entry.localState.completedBytes : 0), 0),
      totalBytes,
      failure: failed.failure,
    }
  }

  const cancelledAttemptIds = packageEntries.flatMap((entry) =>
    entry?.localState._tag === "DownloadCancelled" ? [entry.localState.attemptId] : [])
  return cancelledAttemptIds.length > 0
    ? {
        _tag: "Cancelled",
        attemptIds: cancelledAttemptIds as [
          typeof cancelledAttemptIds[number],
          ...Array<typeof cancelledAttemptIds[number]>,
        ],
        completedBytes,
        totalBytes,
      }
    : { _tag: "NotDownloaded", completedBytes, totalBytes }
}

type ProviderAvailabilityProjection = Pick<ProviderModelCatalogEntry, "availability">

export const availabilityFromProviderProjection = (
  providerModelId: ProviderModelId | undefined,
  providerEntries: ReadonlyMap<ProviderModelId, ProviderAvailabilityProjection>,
  projectionCurrent: boolean,
  providerProjectionFailure: Option.Option<ModelFailure>,
): LocalModelCatalogCandidateAvailability | undefined => {
  if (providerModelId === undefined) return { _tag: "Available" }
  if (!projectionCurrent) return undefined
  const providerEntry = providerEntries.get(providerModelId)
  if (providerEntry?.availability._tag === "Available") {
    return { _tag: "Available" }
  }
  if (providerEntry?.availability._tag === "Disabled") {
    return {
      _tag: "Unavailable",
      failure: {
        code: providerEntry.availability.reason,
        message: providerEntry.availability.reason === "insufficient_resources"
          ? "This model configuration is no longer compatible with the available hardware capacity"
          : "This model configuration is not available to the local runtime",
        retryable: true,
      },
    }
  }
  if (Option.isSome(providerProjectionFailure)) {
    return { _tag: "Unavailable", failure: providerProjectionFailure.value }
  }
  return undefined
}

const aggregateAvailability = (
  bundle: ServableModelBundle,
  entries: ReadonlyMap<string, ModelPackageEntry>,
  providerModelId: ProviderModelId | undefined,
  providerEntries: ReadonlyMap<ProviderModelId, ProviderModelCatalogEntry>,
  providerProjectionCurrent: boolean,
  providerProjectionFailure: Option.Option<ModelFailure>,
): LocalModelCatalogCandidateAvailability | undefined => {
  const bundleEntries = servableModelBundlePackageIds(bundle).map((packageId) => entries.get(packageId))
  if (!bundleEntries.every((entry) => entry?.localState._tag === "Installed")) {
    return { _tag: "NotDownloaded" }
  }
  const failure = bundleEntries.flatMap((entry): readonly ModelFailure[] => {
    if (entry?.inspection._tag === "Invalid" || entry?.inspection._tag === "Incompatible") {
      return [entry.inspection.failure]
    }
    return []
  })[0]
  if (failure) return { _tag: "Unavailable", failure }
  if (bundleEntries.some((entry) => entry?.inspection._tag === "Pending")) {
    return undefined
  }
  return availabilityFromProviderProjection(
    providerModelId,
    providerEntries,
    providerProjectionCurrent,
    providerProjectionFailure,
  )
}

export interface LocalModelsApi {
  readonly snapshot: Effect.Effect<{ readonly revision: number; readonly state: LocalModelsState }>
  readonly changes: Stream.Stream<{ readonly revision: number; readonly state: LocalModelsState }>
}

export class LocalModels extends Context.Tag("LocalModels")<LocalModels, LocalModelsApi>() {}

export const LocalModelsLive: Layer.Layer<
  LocalModels,
  never,
  IcnCatalog | LocalModelPackages | LocalModelRecommendations
    | LocalModelAssessor | LocalProviderOfferings | MirroredStateChanges
    | RetainedModelConfigurations
> = Layer.scoped(LocalModels, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const packages = yield* LocalModelPackages
  const recommendations = yield* LocalModelRecommendations
  const assessments = yield* LocalModelAssessor
  const offerings = yield* LocalProviderOfferings
  const retained = yield* RetainedModelConfigurations
  const mirror = yield* makeMirroredState(LocalModelsMirror, {
    inventory: { _tag: "Initializing" },
    models: [],
    downloads: [],
    recommendations: {
      _tag: "Loading",
      progress: [],
    },
  })
  const equivalent = Schema.equivalence(LocalModelsMirror.stateSchema)
  const lock = yield* Effect.makeSemaphore(1)

  const project = lock.withPermits(1)(Effect.gen(function* () {
    const packageState = (yield* packages.snapshot).state
    const catalogModels = yield* Effect.forEach(
      (yield* catalog.get).state.models,
      recommendableModelFromIcn,
    )
    const recommendationState = (yield* recommendations.snapshot).state
    const assessmentState = yield* assessments.state
    const retainedConfigurations = yield* retained.get
    const configured = yield* offerings.list
    const projectedOfferings = yield* offerings.state
    const packageEntries = new Map(
      packageState.entries.map((entry) => [entry.package.id, entry]),
    )
    const sameBundle = Schema.equivalence(ServableModelBundleSchema)
    const providerIdByConfiguration = new Map<ModelServingConfigurationId, ProviderModelId>()
    for (const offering of configured) {
      providerIdByConfiguration.set(offering.configuration.id, offering.providerModelId)
    }
    const providerEntries = new Map(
      projectedOfferings.entries.map((entry) => [entry.providerModelId, entry]),
    )
    const currentProviderPackageEvidence = providerOfferingPackageEvidence(
      configured,
      packageEntries,
    )
    const providerProjectionCurrent = Option.exists(
      projectedOfferings.packageEvidence,
      (evidence) => sameProviderOfferingPackageEvidence(
        evidence,
        currentProviderPackageEvidence,
      ),
    )
    const providerProjectionFailure = Option.map(projectedOfferings.failure, (error): ModelFailure => ({
      code: "local_model_assessment_unavailable",
      message: error.message,
      retryable: "retryable" in error ? error.retryable : true,
    }))
    const knownConfigurations = new Map<
      ModelServingConfigurationId,
      (typeof configured)[number]["configuration"]
    >()
    const addKnownConfiguration = (
      configuration: (typeof configured)[number]["configuration"],
    ) => {
      const existing = knownConfigurations.get(configuration.id)
      if (existing !== undefined && !sameConfiguration(existing, configuration)) {
        throw new Error(`Configuration ${configuration.id} has conflicting values`)
      }
      knownConfigurations.set(configuration.id, configuration)
    }
    retainedConfigurations.forEach(addKnownConfiguration)
    catalogModels.forEach(({ configuration }) => addKnownConfiguration(configuration))
    assessmentState.forEach(({ configuration }) => addKnownConfiguration(configuration))

    const groups = new Map<string, ServableModelBundle>()
    const addBundle = (bundle: ServableModelBundle) => {
      const identity = bundleIdentity(bundle)
      const existing = groups.get(identity)
      if (existing === undefined) {
        groups.set(identity, bundle)
      } else if (!sameBundle(existing, bundle)) {
        throw new Error(`Servable model bundle ${identity} has conflicting package definitions`)
      }
    }
    knownConfigurations.forEach((configuration) => {
      if (installedBundle(configuration.bundle, packageEntries) !== undefined) {
        addBundle(configuration.bundle)
      }
    })
    for (const entry of packageState.entries) {
      if (entry.localState._tag !== "Installed") continue
      const independentlyServable = entry.package.files.some(({ role }) =>
        role === "weights" || role === "draft")
      if (independentlyServable) addBundle({ _tag: "Standalone", package: entry.package })
    }
    const configurationByBundle = decideLocalModelConfigurations({
      retained: retainedConfigurations,
      catalog: catalogModels.map(({ configuration }) => configuration),
      assessed: [...assessmentState.values()].map(({ configuration }) => configuration),
    })
    const configuredById = new Map(configured.map((offering) => [
      offering.configuration.id,
      offering,
    ]))
    const downloads = [...knownConfigurations.values()].flatMap((configuration) => {
      const download = aggregateDownload(configuration.bundle, packageEntries)
      if (download._tag === "NotDownloaded") return []
      const catalogModel = catalogModels.find((model) => model.configuration.id === configuration.id)
      const configuredOffering = configuredById.get(configuration.id)
      const primaryPackage = configuration.bundle._tag === "Standalone"
        ? configuration.bundle.package
        : configuration.bundle.target
      const inspection = packageEntries.get(primaryPackage.id)?.inspection
      const capabilities = Option.firstSomeOf([
        Option.fromNullable(catalogModel).pipe(Option.map(({ capabilities }) => capabilities)),
        Option.fromNullable(configuredOffering).pipe(
          Option.map(({ capabilities }) => capabilities),
        ),
        inspection?._tag === "Inspected"
          ? Option.some(inspection.capabilities)
          : Option.none(),
      ])
      return [{
        configuration,
        presentation: resolveBundlePresentation(configuration.bundle, catalogModel && {
          displayName: catalogModel.displayName,
          description: catalogModel.description,
        }),
        capabilities,
        state: download,
      }]
    })
    const models: LocalModel[] = [...groups.values()].map((bundle): LocalModel => {
      const curated = catalogModels.find((model) =>
        bundleIdentity(model.configuration.bundle) === bundleIdentity(bundle))
      const presentation = resolveBundlePresentation(bundle, curated && {
        displayName: curated.displayName,
        description: curated.description,
      })
      const primaryPackage = bundle._tag === "Standalone"
        ? bundle.package
        : bundle.target
      const bundleEntries = bundlePackages(bundle).map((modelPackage) =>
        packageEntries.get(modelPackage.id))
      const installation = installedBundle(bundle, packageEntries)
      if (installation === undefined) {
        throw new Error(
          `Installed servable model bundle ${bundleIdentity(bundle)} disappeared during projection`,
        )
      }
      const inspectionFailure = bundleEntries.flatMap((entry) =>
        entry?.inspection._tag === "Invalid" || entry?.inspection._tag === "Incompatible"
          ? [entry.inspection.failure]
          : [])[0]
      const inspectedCapabilities = packageEntries.get(primaryPackage.id)?.inspection
      const inspectionComplete = bundleEntries.every((entry) =>
        entry?.inspection._tag === "Inspected")
      const configuration = configurationByBundle.get(bundleIdentity(bundle))
      const assessment = configuration === undefined
        ? undefined
        : assessmentState.get(configuration.id)?.assessment
      const configuredOffering = configuration === undefined
        ? undefined
        : configuredById.get(configuration.id)
      const projectedOffering = configuredOffering === undefined
        ? undefined
        : providerEntries.get(configuredOffering.providerModelId)
      return {
        bundle,
        presentation,
        installation,
        readiness: inspectionFailure !== undefined
          ? { _tag: "Failed", failure: inspectionFailure }
          : !inspectionComplete || inspectedCapabilities?._tag !== "Inspected"
            ? { _tag: "Assessing" }
            : configuration !== undefined
              && assessment !== undefined
              && assessment._tag !== "Assessing"
              ? {
                  _tag: "Assessed",
                  capabilities: curated?.capabilities ?? inspectedCapabilities.capabilities,
                  configuration,
                  offering: Option.fromNullable(projectedOffering),
                  assessment,
                }
              : localModelAssessmentProfiles(bundle).length === 0
                ? {
                    _tag: "Failed",
                    failure: {
                      code: "unsupported_model_context_length",
                      message: `Model context length is below the ${MINIMUM_LOCAL_MODEL_CONTEXT_LENGTH.toLocaleString("en-US")}-token minimum`,
                      retryable: false,
                    },
                  }
                : { _tag: "Assessing" },
      }
    }).sort((left, right) =>
      left.presentation.displayName.localeCompare(right.presentation.displayName))
    const catalogCandidates = recommendationState._tag === "Ready"
      ? recommendationState.catalog.flatMap(({ candidate, configuration }) => {
          const availability = aggregateAvailability(
            configuration.bundle,
            packageEntries,
            providerIdByConfiguration.get(candidate.configurationId),
            providerEntries,
            providerProjectionCurrent,
            providerProjectionFailure,
          )
          return availability === undefined
            ? []
            : [{
                ...candidate,
                download: aggregateDownload(configuration.bundle, packageEntries),
                availability,
              }]
        })
      : []
    const catalogCandidatesByConfigurationId = new Map(
      catalogCandidates.map((candidate) => [candidate.configurationId, candidate]),
    )
    const recommendationLifecycle = recommendationState._tag === "Loading"
      ? {
          _tag: "Loading" as const,
          progress: recommendationState.progress,
        }
      : recommendationState._tag === "Failed"
        ? {
            _tag: "Failed" as const,
            failure: recommendationState.failure,
            progress: recommendationState.progress,
          }
        : {
            _tag: "Ready" as const,
            entries: recommendationState.recommendations.flatMap((recommendation) => {
              const entry = recommendationState.catalog.find(({ configuration }) =>
                configuration.id === recommendation.configuration.id)
              const candidate = entry
                ? catalogCandidatesByConfigurationId.get(entry.configuration.id)
                : undefined
              return candidate
                ? [{
                    id: recommendation.id,
                    intent: recommendation.intent,
                    explanation: recommendation.explanation,
                    candidate,
                  }]
                : []
            }),
            catalog: catalogCandidates,
            progress: recommendationState.progress,
          }
    yield* mirror.setIfChanged({
      inventory: packageState.inventory,
      models,
      downloads,
      recommendations: recommendationLifecycle,
    }, equivalent)
  })).pipe(Effect.catchAllCause((cause) =>
    Effect.logWarning("Unable to project local models").pipe(
      Effect.annotateLogs({ cause: String(cause) }),
    )))

  yield* project
  yield* Stream.mergeAll([
    packages.changes,
    catalog.changes,
    recommendations.changes,
    assessments.changes,
    retained.changes,
    offerings.changes,
    offerings.catalogChanges,
  ], { concurrency: "unbounded" }).pipe(
    Stream.debounce("25 millis"),
    Stream.runForEach(() => project),
    Effect.forkScoped,
  )

  return LocalModels.of({
    snapshot: mirror.get,
    changes: mirror.changes,
  })
}))
