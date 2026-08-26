import { Context, Effect, Layer, Option, Schema, Stream, SubscriptionRef } from "effect"
import type { NonEmptyReadonlyArray } from "effect/Array"
import {
  LocalModelsStateSchema,
  ModelServingConfigurationSchema,
  ServableModelBundleSchema,
  sameServableModelBundleIdentity,
  servableModelBundlePackageIds,
  servableModelBundlePackages,
  type LocalInferenceHardware,
  type LocalModel,
  type LocalModelAcquisitionState,
  type LocalModelAssessment,
  type LocalModelAvailabilityState,
  type LocalModelInstalledPackage,
  type LocalModelMemory,
  type LocalModelsState,
  type MemoryAssessment,
  type ModelDownloadFailure,
  type ModelFailure,
  type ModelBundleDownload,
  type ModelPackageEntry,
  type ModelResidency,
  type ModelServingConfiguration,
  type ModelTransferProgress,
  type ProviderModelCatalogEntry,
  type ServableModelBundle,
} from "@magnitudedev/acn-protocol"
import type { ProviderModelId } from "@magnitudedev/ai"
import { projectInferenceResidency } from "@magnitudedev/sdk"
import { IcnInstances, IcnModels } from "@magnitudedev/icn"
import type * as Generated from "@magnitudedev/icn-protocol/schemas"
import { LocalInferenceHardware as LocalInferenceHardwareService } from "./local-inference-hardware"
import { LocalModelPackages } from "./local-model-packages"
import { LocalModelRanker } from "./local-model-ranker"
import { LocalProviderOfferings, localCatalogProviderModelId } from "./local-provider-offerings"
import {
  providerOfferingPackageEvidence,
  sameProviderOfferingPackageEvidence,
} from "./local-provider-offerings"
import { catalogIdentityFromIcn, catalogModelDefinitionFromIcn } from "./local-model-icn-adapter"
import {
  LocalModelConfigurationResolver,
} from "./local-model-configuration-resolver"
import { bundlePackages, resolveBundlePresentation } from "./local-model-presentation"
export { resolveBundlePresentation } from "./local-model-presentation"

const GIB = 1024 ** 3

const bundleDownloadBytes = (bundle: ServableModelBundle): number =>
  servableModelBundlePackages(bundle).reduce((total, modelPackage) => total
    + modelPackage.files.reduce((sum, file) => sum + file.sizeBytes, 0), 0)

export interface InstalledModelFields {
  readonly installedBytes: number
  readonly packages: NonEmptyReadonlyArray<LocalModelInstalledPackage>
}

export const installedBundleFields = (
  bundle: ServableModelBundle,
  entries: ReadonlyMap<string, ModelPackageEntry>,
): InstalledModelFields | undefined => {
  const packages = servableModelBundlePackages(bundle)
  const packageEntries = packages.map((modelPackage) => entries.get(modelPackage.id))
  if (!packageEntries.every((entry) => entry?.localState._tag === "Installed")) {
    return undefined
  }
  const installedPackages = packages.map((modelPackage, index) => {
    const entry = packageEntries[index]
    if (entry?.localState._tag !== "Installed") {
      throw new Error(`Installed bundle package ${modelPackage.id} has no installed location`)
    }
    return {
      packageId: modelPackage.id,
      path: entry.localState.path,
      origin: entry.localState.origin,
    }
  })
  const [firstPackage, ...remainingPackages] = installedPackages
  if (firstPackage === undefined) {
    throw new Error("Installed servable model bundle has no installed package location")
  }
  return {
    installedBytes: bundleDownloadBytes(bundle),
    packages: [firstPackage, ...remainingPackages],
  }
}

const priorInstalledFields = (
  entries: readonly ModelPackageEntry[],
): InstalledModelFields | undefined => {
  const installedPackages = entries.flatMap((entry) => entry.localState._tag === "Installed"
    ? [{
        packageId: entry.package.id,
        path: entry.localState.path,
        origin: entry.localState.origin,
      }]
    : [])
  const [firstPackage, ...remainingPackages] = installedPackages
  if (firstPackage === undefined) return undefined
  return {
    installedBytes: entries.reduce((total, entry) => entry.localState._tag === "Installed"
      ? total + entry.package.files.reduce((sum, file) => sum + file.sizeBytes, 0)
      : total, 0),
    packages: [firstPackage, ...remainingPackages],
  }
}

type ModelTransfer =
  | { readonly _tag: "Active"; readonly progress: ModelTransferProgress }
  | { readonly _tag: "Failed"; readonly failure: ModelDownloadFailure }

/** The latest download occurrence targeting any of the model's bundles. */
export const latestBundleDownload = (
  downloads: readonly ModelBundleDownload[],
  bundles: readonly ServableModelBundle[],
): ModelBundleDownload | undefined => {
  for (let index = downloads.length - 1; index >= 0; index--) {
    const candidate = downloads[index]
    if (candidate !== undefined && bundles.some((bundle) =>
      sameServableModelBundleIdentity(candidate.bundle, bundle))) return candidate
  }
  return undefined
}

const transferOf = (download: ModelBundleDownload | undefined): ModelTransfer | undefined => {
  if (download === undefined) return undefined
  switch (download.state._tag) {
    case "Pending": return {
      _tag: "Active",
      progress: {
        stage: "queued",
        completedBytes: download.state.completedBytes,
        totalBytes: download.state.totalBytes,
        bytesPerSecond: Option.none(),
      },
    }
    case "Downloading": return {
      _tag: "Active",
      progress: {
        stage: download.state.stage,
        completedBytes: download.state.completedBytes,
        totalBytes: download.state.totalBytes,
        bytesPerSecond: download.state.bytesPerSecond,
      },
    }
    case "Failed": return download.state.acknowledged
      ? undefined
      : { _tag: "Failed", failure: download.state.failure }
    case "Completed":
    case "Cancelled":
      return undefined
  }
}

export interface ModelAcquisitionInputs {
  /** The bundle the current serving resolution targets. */
  readonly currentBundle: ServableModelBundle
  /** The bundle the curated catalog currently publishes for the model. */
  readonly desiredBundle: ServableModelBundle
  /** The current bundle's complete installation, when every package is on disk. */
  readonly currentInstalled: InstalledModelFields | undefined
  readonly downloads: readonly ModelBundleDownload[]
  /** ICN reports a newer catalog target than the installed current bundle. */
  readonly updateAvailable: boolean
  /** Installed packages attributed to this model that predate the current bundle. */
  readonly priorEntries: readonly ModelPackageEntry[]
  readonly residencyState: ModelResidency
}

/**
 * The complete per-model materialization state: disk truth, the single
 * transfer that may exist for the model, and runtime residency once any
 * version is installed.
 */
export const deriveModelAcquisitionState = (
  inputs: ModelAcquisitionInputs,
): LocalModelAcquisitionState => {
  const transfer = transferOf(latestBundleDownload(
    inputs.downloads,
    [inputs.desiredBundle, inputs.currentBundle],
  ))
  const current = inputs.currentInstalled
  if (current !== undefined) {
    const installed = { ...current, residencyState: inputs.residencyState }
    if (!inputs.updateAvailable) return { _tag: "Installed", ...installed }
    if (transfer?._tag === "Active") return { _tag: "Updating", ...installed, progress: transfer.progress }
    if (transfer?._tag === "Failed") return { _tag: "UpdateFailed", ...installed, failure: transfer.failure }
    return { _tag: "UpdateAvailable", ...installed }
  }
  const prior = priorInstalledFields(inputs.priorEntries)
  if (prior !== undefined) {
    const installed = { ...prior, residencyState: inputs.residencyState }
    if (transfer?._tag === "Active") return { _tag: "Updating", ...installed, progress: transfer.progress }
    if (transfer?._tag === "Failed") return { _tag: "UpdateFailed", ...installed, failure: transfer.failure }
    return { _tag: "UpdateAvailable", ...installed }
  }
  if (transfer?._tag === "Active") return { _tag: "Installing", progress: transfer.progress }
  if (transfer?._tag === "Failed") return { _tag: "InstallFailed", failure: transfer.failure }
  return { _tag: "NotInstalled" }
}

type ProviderAvailabilityProjection = Pick<ProviderModelCatalogEntry, "availability">

export const availabilityFromProviderProjection = (
  providerModelId: Option.Option<ProviderModelId>,
  providerEntries: ReadonlyMap<ProviderModelId, ProviderAvailabilityProjection>,
  projectionCurrent: boolean,
  providerProjectionFailure: Option.Option<ModelFailure>,
): LocalModelAvailabilityState => {
  if (Option.isNone(providerModelId)) return { _tag: "Installable" }
  const resolvedProviderModelId = providerModelId.value
  if (!projectionCurrent) return { _tag: "Preparing", providerModelId: resolvedProviderModelId }
  const providerEntry = providerEntries.get(resolvedProviderModelId)
  if (providerEntry?.availability._tag === "Available") {
    return { _tag: "Selectable", providerModelId: resolvedProviderModelId }
  }
  if (providerEntry?.availability._tag === "Disabled") {
    return {
      _tag: "Unavailable",
      providerModelId,
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
    return {
      _tag: "Unavailable",
      providerModelId,
      failure: providerProjectionFailure.value,
    }
  }
  return { _tag: "Preparing", providerModelId: resolvedProviderModelId }
}

const unavailableAssessment = (
  assessment: Exclude<LocalModelAssessment, { readonly _tag: "Fits" }>,
  providerModelId: Option.Option<ProviderModelId>,
): LocalModelAvailabilityState => ({
  _tag: "Unavailable",
  providerModelId,
  failure: assessment._tag === "Incompatible"
    ? assessment.failure
    : {
        code: "insufficient_resources",
        message: `This model exceeds available ${assessment.limitingResource} capacity by ${assessment.deficitBytes} bytes`,
        retryable: false,
      },
})

const recommendedSystemHeadroomBytes = (totalBytes: number): number =>
  Math.max(Math.floor(totalBytes / 5), 4 * GIB)

const allocationBytes = (allocation: Generated.ModelInstanceAllocation["memoryDomains"][number]): number =>
  allocation.modelBytes
  + allocation.contextBytes
  + allocation.computeBytes
  + allocation.auxiliaryBytes

const residentSystemAllocationBytes = (
  instances: Generated.ModelInstancesSnapshot,
  systemDomains: ReadonlySet<string>,
): Option.Option<number> => {
  const allocations = new Map<string, Generated.ModelInstanceAllocation>()
  for (const instance of instances.instances) {
    const lifecycle = instance.lifecycle
    if (lifecycle._tag === "Loading") return Option.none()
    if (lifecycle._tag === "Stopping") {
      if (lifecycle.allocation._tag === "Planned") return Option.none()
      allocations.set(instance.id, lifecycle.allocation.allocation)
    } else if (lifecycle._tag === "Ready") {
      allocations.set(instance.id, lifecycle.allocation)
    }
  }
  if (allocations.size > 1) {
    throw new Error("local-model projection observed multiple resident model instances")
  }
  const allocation = allocations.values().next().value as Generated.ModelInstanceAllocation | undefined
  return Option.some(allocation?.memoryDomains.reduce((total, domain) =>
    systemDomains.has(domain.memoryDomainId)
      ? total + allocationBytes(domain)
      : total, 0) ?? 0)
}

export const projectLocalModelMemory = (
  domains: readonly MemoryAssessment[],
  hardware: LocalInferenceHardware,
  instances: Generated.ModelInstancesSnapshot,
): LocalModelMemory => {
  const systemDomains = new Set(hardware.memoryDomains
    .filter(({ sharesSystemMemory }) => sharesSystemMemory)
    .map(({ memoryDomainId }) => memoryDomainId))
  const systemEvidence = domains.filter(({ memoryDomainId }) =>
    systemDomains.has(memoryDomainId))
  const requiredSystemMemoryBytes = systemEvidence.reduce(
    (total, { requiredBytes }) => total + requiredBytes,
    0,
  )
  const totalRequiredBytes = domains.reduce((total, { requiredBytes }) =>
    total + requiredBytes, 0)
  const recommendedHeadroomBytes = recommendedSystemHeadroomBytes(
    hardware.totalSystemMemoryBytes,
  )
  const predictedHeadroomBytes = Math.max(
    0,
    hardware.totalSystemMemoryBytes - requiredSystemMemoryBytes,
  )
  const systemUseState = predictedHeadroomBytes < recommendedHeadroomBytes
    ? { _tag: "High" as const, recommendedHeadroomBytes, predictedHeadroomBytes }
    : { _tag: "WithinRecommendedHeadroom" as const, recommendedHeadroomBytes, predictedHeadroomBytes }
  if (systemEvidence.length === 0) {
    return {
      domains,
      totalRequiredBytes,
      requiredSystemMemoryBytes,
      systemUseState,
      currentHeadroomState: { _tag: "NotObserved" },
    }
  }
  const residentBytes = residentSystemAllocationBytes(instances, systemDomains)
  if (Option.isNone(residentBytes)) {
    return {
      domains,
      totalRequiredBytes,
      requiredSystemMemoryBytes,
      systemUseState,
      currentHeadroomState: { _tag: "NotObserved" },
    }
  }
  const allocationHeadroomBytes = Math.min(
    hardware.systemAllocationCapacityBytes,
    hardware.systemAllocationHeadroomBytes + residentBytes.value,
  )
  const loadBoundaryBytes = requiredSystemMemoryBytes + hardware.abortReserveBytes
  const observation = {
    requiredSystemMemoryBytes,
    allocationHeadroomBytes,
    abortReserveBytes: hardware.abortReserveBytes,
    loadBoundaryBytes,
  }
  return {
    domains,
    totalRequiredBytes,
    requiredSystemMemoryBytes,
    systemUseState,
    currentHeadroomState: allocationHeadroomBytes > loadBoundaryBytes
      ? { _tag: "Sufficient", observation }
      : {
          _tag: "Insufficient",
          observation,
          minimumAdditionalAvailableBytes: loadBoundaryBytes - allocationHeadroomBytes + 1,
        },
  }
}

export interface LocalModelsApi {
  readonly state: Effect.Effect<LocalModelsState>
  readonly changes: Stream.Stream<LocalModelsState>
  /** Publishes a snapshot from the current lower-domain facts before returning. */
  readonly refresh: Effect.Effect<void>
  /**
   * The native download occurrence currently backing a model's transfer or
   * unacknowledged failure. Private command plumbing: the occurrence identity
   * never reaches the client contract.
   */
  readonly currentDownload: (
    modelId: ProviderModelId,
  ) => Effect.Effect<Option.Option<ModelBundleDownload>>
}

export class LocalModels extends Context.Tag("LocalModels")<LocalModels, LocalModelsApi>() {}

export const LocalModelsLive: Layer.Layer<
  LocalModels,
  never,
  IcnModels | LocalModelPackages | LocalModelRanker
    | LocalModelConfigurationResolver | LocalProviderOfferings | LocalInferenceHardwareService
    | IcnInstances
> = Layer.scoped(LocalModels, Effect.gen(function* () {
  const icnModels = yield* IcnModels
  const packages = yield* LocalModelPackages
  const ranker = yield* LocalModelRanker
  const resolver = yield* LocalModelConfigurationResolver
  const offerings = yield* LocalProviderOfferings
  const hardware = yield* LocalInferenceHardwareService
  const instances = yield* IcnInstances
  const current = yield* SubscriptionRef.make<LocalModelsState>({
    inventoryState: { _tag: "Initializing" },
    models: [],
    discoveryState: { _tag: "Loading", progress: [] },
  })
  const currentDownloads = yield* SubscriptionRef.make<
    ReadonlyMap<ProviderModelId, ModelBundleDownload>
  >(new Map())
  const equivalent = Schema.equivalence(LocalModelsStateSchema)
  const lock = yield* Effect.makeSemaphore(1)

  const project = lock.withPermits(1)(Effect.gen(function* () {
    const packageState = yield* packages.state
    const nativeCatalogModels = (yield* icnModels.get).state.models
    const catalogModels = yield* Effect.forEach(
      nativeCatalogModels,
      catalogModelDefinitionFromIcn,
    )
    const rankingState = yield* ranker.state
    const resolvedConfigurations = new Map<string, import("./local-model-configuration-resolver").ResolvedLocalModelConfiguration>(
      yield* resolver.get,
    )
    const configured = yield* offerings.list
    const projectedOfferings = yield* offerings.state
    const hardwareState = yield* hardware.state
    const instanceState = yield* instances.get
    const packageEntries = new Map(
      packageState.entries.map((entry) => [entry.package.id, entry]),
    )
    const sameBundle = Schema.equivalence(ServableModelBundleSchema)
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
    const providerProjectionFailure = Option.map(
      projectedOfferings.failure,
      (error): ModelFailure => ({
        code: "local_model_assessment_unavailable",
        message: error.message,
        retryable: "retryable" in error ? error.retryable : true,
      }),
    )

    const groups = new Map([...resolvedConfigurations].map(([identity, resolution]) => [
      identity,
      resolution.servingConfiguration.bundle,
    ]))

    const catalogByTarget = new Map<string, (typeof catalogModels)[number]>(catalogModels.map((model) => [
      localCatalogProviderModelId(model),
      model,
    ]))
    const nativeCatalogByTarget = new Map<string, (typeof nativeCatalogModels)[number]>(
      yield* Effect.forEach(nativeCatalogModels, (model) =>
        catalogIdentityFromIcn(model).pipe(Effect.map((identity) => [
          localCatalogProviderModelId(identity),
          model,
        ] as const))),
    )
    const rankingEntries = rankingState._tag === "Ready" ? rankingState.entries : []
    const sameConfiguration = Schema.equivalence(ModelServingConfigurationSchema)

    const downloadIndex = new Map<ProviderModelId, ModelBundleDownload>()
    const models: LocalModel[] = [...groups.entries()].map(([identity, bundle]): LocalModel => {
      const curated = catalogByTarget.get(identity)
      if (curated === undefined) {
        throw new Error(`Catalog model ${identity} has no definition`)
      }
      const presentation = resolveBundlePresentation(bundle, curated && {
        displayName: curated.displayName,
        variantLabel: curated.variantLabel,
        description: curated.description,
        license: curated.license,
      })
      const currentInstalled = installedBundleFields(bundle, packageEntries)
      const priorEntries = packageState.entries.filter((entry) =>
        entry.localState._tag === "Installed"
          && entry.catalogAttribution._tag === "Attributed"
          && localCatalogProviderModelId(entry.catalogAttribution) === identity
          && !servableModelBundlePackageIds(bundle).includes(entry.package.id))
      const catalogLocalState = nativeCatalogByTarget.get(identity)?.localState
      const updateAvailable = catalogLocalState?._tag === "Installed"
        && catalogLocalState.updateState._tag === "Available"
      const modelDownload = latestBundleDownload(
        packageState.downloads,
        [curated.configuration.bundle, bundle],
      )
      if (modelDownload !== undefined) {
        downloadIndex.set(localCatalogProviderModelId(curated), modelDownload)
      }
      const modelInstance = instanceState.instances.findLast((candidate) =>
        candidate.modelId === localCatalogProviderModelId(curated))
      const acquisitionState = deriveModelAcquisitionState({
        currentBundle: bundle,
        desiredBundle: curated.configuration.bundle,
        currentInstalled,
        downloads: packageState.downloads,
        updateAvailable,
        priorEntries,
        residencyState: modelInstance === undefined
          ? { _tag: "Unloaded" }
          : projectInferenceResidency(modelInstance),
      })
      const bundleEntries = servableModelBundlePackages(bundle).map((modelPackage) =>
        packageEntries.get(modelPackage.id))
      const primaryPackage = bundle._tag === "Standalone" ? bundle.package : bundle.target
      const dependencyEntries = bundleEntries.filter((entry) =>
        entry?.package.id !== primaryPackage.id)
      const dependencyInspectionFailure = Option.fromNullable(dependencyEntries.flatMap((entry) =>
        entry?.inspection._tag === "Invalid" || entry?.inspection._tag === "Incompatible"
          ? [entry.inspection.failure]
          : [])[0])
      const dependencyInspectionComplete = currentInstalled === undefined
        || dependencyEntries.every((entry) => entry?.inspection._tag === "Inspected")
      const resolution = Option.fromNullable(resolvedConfigurations.get(identity))
      const configuration = Option.map(
        resolution,
        ({ servingConfiguration }) => servingConfiguration,
      )
      const attribution = packageEntries.get(primaryPackage.id)?.catalogAttribution
      const catalogMembershipState: LocalModel["catalogMembershipState"] = {
        _tag: "InCatalog",
        catalogData: {
          modelId: curated.modelId,
          variantId: curated.variantId,
          releaseDate: curated.releaseDate,
          parameterization: curated.parameterization,
          intelligence: curated.intelligence,
          fidelityRank: curated.fidelityRank,
          quantizationAware: curated.quantizationAware,
        },
      }

      let servingState: LocalModel["servingState"]
      if (Option.isSome(dependencyInspectionFailure)) {
        servingState = {
          _tag: "Failed",
          configuration,
          failure: dependencyInspectionFailure.value,
        }
      } else if (Option.isNone(resolution)) {
        servingState = { _tag: "Resolving" }
      } else {
        const resolvedConfiguration = resolution.value
        const servingConfiguration = resolvedConfiguration.servingConfiguration
        const targetInspection = resolvedConfiguration.targetInspection
        const coordinatedAssessment = resolvedConfiguration.assessment
        if (targetInspection._tag === "Invalid" || targetInspection._tag === "Incompatible") {
          servingState = {
            _tag: "Failed",
            configuration: Option.some(servingConfiguration),
            failure: targetInspection.failure,
          }
        } else if (targetInspection._tag === "Pending"
          || !dependencyInspectionComplete
          || coordinatedAssessment._tag === "Assessing") {
          servingState = { _tag: "Assessing", configuration: servingConfiguration }
        } else if (coordinatedAssessment._tag === "Failed") {
          servingState = {
            _tag: "Failed",
            configuration,
            failure: coordinatedAssessment.failure,
          }
        } else {
          const assessment: LocalModelAssessment = coordinatedAssessment._tag === "Fits"
            ? {
                _tag: "Fits",
                assessmentId: coordinatedAssessment.assessment.assessmentId,
                environmentId: coordinatedAssessment.assessment.environmentId,
                profile: coordinatedAssessment.assessment.profile,
                memory: projectLocalModelMemory(
                  coordinatedAssessment.assessment.memory,
                  hardwareState,
                  instanceState,
                ),
                performance: coordinatedAssessment.assessment.performance,
              }
            : coordinatedAssessment._tag === "DoesNotFit"
              ? {
                  _tag: "DoesNotFit",
                  assessmentId: coordinatedAssessment.assessmentId,
                  environmentId: coordinatedAssessment.environmentId,
                  memoryDomains: coordinatedAssessment.memory,
                  totalRequiredBytes: coordinatedAssessment.totalRequiredBytes,
                  deficitBytes: coordinatedAssessment.deficitBytes,
                  limitingResource: coordinatedAssessment.limitingResource,
                }
              : {
                  _tag: "Incompatible",
                  environmentId: coordinatedAssessment.environmentId,
                  failure: coordinatedAssessment.failure,
                }
          const providerModelId = currentInstalled !== undefined
            ? Option.some(localCatalogProviderModelId(curated))
            : Option.none<ProviderModelId>()
          const availabilityState = assessment._tag !== "Fits"
            ? unavailableAssessment(assessment, providerModelId)
            : currentInstalled === undefined || Option.isNone(providerModelId)
              ? { _tag: "Installable" as const }
              : availabilityFromProviderProjection(
                  providerModelId,
                  providerEntries,
                  providerProjectionCurrent,
                  providerProjectionFailure,
                )
          servingState = {
            _tag: "Assessed",
            configuration: servingConfiguration,
            capabilities: targetInspection.capabilities,
            assessment,
            availabilityState,
            rankingScores: Option.fromNullable(rankingEntries.find((entry) =>
              entry.modelId === identity
              && sameConfiguration(entry.configuration, servingConfiguration))?.scores),
          }
        }
      }
      return {
        modelId: localCatalogProviderModelId(curated),
        bundle,
        presentation,
        downloadBytes: bundleDownloadBytes(bundle),
        catalogMembershipState,
        acquisitionState,
        servingState,
      }
    }).sort((left, right) => {
      const productIdentity = (model: LocalModel) => model.modelId
      return left.presentation.displayName.localeCompare(right.presentation.displayName)
        || left.presentation.variantLabel.localeCompare(right.presentation.variantLabel)
        || productIdentity(left).localeCompare(productIdentity(right))
    })

    const discoveryState = rankingState._tag === "Loading"
      ? { _tag: "Loading" as const, progress: rankingState.progress }
      : rankingState._tag === "Failed"
        ? {
            _tag: "Failed" as const,
            failure: rankingState.failure,
            progress: rankingState.progress,
          }
        : { _tag: "Ready" as const, progress: rankingState.progress }
    const next = {
      inventoryState: packageState.inventory,
      models,
      discoveryState,
    }
    yield* SubscriptionRef.set(currentDownloads, downloadIndex)
    const previous = yield* SubscriptionRef.get(current)
    if (!equivalent(previous, next)) yield* SubscriptionRef.set(current, next)
  })).pipe(Effect.catchAllCause((cause) =>
    Effect.logWarning("Unable to project local models").pipe(
      Effect.annotateLogs({ cause: String(cause) }),
    )))

  yield* project
  yield* Stream.mergeAll([
    packages.changes,
    icnModels.changes,
    ranker.changes,
    resolver.changes,
    offerings.changes,
    offerings.catalogChanges,
    hardware.changes,
    instances.changes,
  ], { concurrency: "unbounded" }).pipe(
    Stream.debounce("25 millis"),
    Stream.runForEach(() => project),
    Effect.forkScoped,
  )

  return LocalModels.of({
    state: SubscriptionRef.get(current),
    changes: current.changes,
    refresh: project,
    currentDownload: (modelId) => SubscriptionRef.get(currentDownloads).pipe(
      Effect.map((index) => Option.fromNullable(index.get(modelId))),
    ),
  })
}))
