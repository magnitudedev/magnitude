import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import type { NonEmptyReadonlyArray } from "effect/Array"
import {
  LocalModelsStateSchema,
  LocalModelMutationFailed,
  ModelPackageIdSchema,
  ModelServingConfigurationSchema,
  installedAcquisition,
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
  type ModelDownloadId,
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
import { IcnInstances } from "@magnitudedev/icn"
import * as Generated from "@magnitudedev/icn-protocol/schemas"
import { LocalInferenceHardware as LocalInferenceHardwareService } from "./local-inference-hardware"
import { LocalModelPackages } from "./local-model-packages"
import { LocalModelRanker } from "./local-model-ranker"
import {
  LocalModelAcquisitionCoordinator,
  type LocalModelSyncState,
} from "./local-model-acquisition-coordinator"
import {
  localCatalogProviderModelId,
  projectLocalProviderOfferings,
} from "./local-provider-offerings"
import { LocalModelCatalogAdapter } from "./local-model-catalog-adapter"
import {
  LocalModelConfigurationResolver,
  type ResolvedLocalModelConfiguration,
} from "./local-model-configuration-resolver"
import { bundlePackages, resolveBundlePresentation } from "./local-model-presentation"
import { materializeProjection } from "./materialized-projection"
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

type ModelTransfer =
  | { readonly _tag: "Active"; readonly progress: ModelTransferProgress }
  | { readonly _tag: "Failed"; readonly failure: ModelDownloadFailure }

/** Resolves only the exact ICN occurrence correlated by the ACN. */
export const correlatedModelSync = (
  downloads: readonly ModelBundleDownload[],
  downloadId: ModelDownloadId | undefined,
): ModelBundleDownload | undefined => downloadId === undefined
  ? undefined
  : downloads.find(({ id }) => id === downloadId)

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
  /** The current bundle's complete installation, when every package is on disk. */
  readonly currentInstalled: InstalledModelFields | undefined
  readonly download: ModelBundleDownload | undefined
  readonly syncState?: LocalModelSyncState | undefined
  readonly downloadBytes?: number
  /** ICN reports a newer catalog target than the installed current bundle. */
  readonly updateAvailable: boolean
  /** The effective installed configuration retained while the desired bundle is incomplete. */
  readonly priorInstalled: InstalledModelFields | undefined
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
  const downloadBytes = inputs.downloadBytes ?? 0
  const transfer = inputs.syncState?._tag === "Admitting"
    ? {
        _tag: "Active" as const,
        progress: {
          stage: "queued" as const,
          completedBytes: 0,
          totalBytes: downloadBytes,
          bytesPerSecond: Option.none<number>(),
        },
      }
    : inputs.syncState?._tag === "AdmissionFailed"
      ? { _tag: "Failed" as const, failure: inputs.syncState.failure }
      : inputs.syncState?._tag === "Correlated" && inputs.download?.state._tag === "Completed"
        ? {
            _tag: "Active" as const,
            progress: {
              stage: "publishing" as const,
              completedBytes: downloadBytes,
              totalBytes: downloadBytes,
              bytesPerSecond: Option.none<number>(),
            },
          }
        : inputs.syncState?._tag === "Correlated" && inputs.download === undefined
          ? {
              _tag: "Active" as const,
              progress: {
                stage: "queued" as const,
                completedBytes: 0,
                totalBytes: downloadBytes,
                bytesPerSecond: Option.none<number>(),
              },
            }
          : transferOf(inputs.download)
  const current = inputs.currentInstalled
  if (current !== undefined) {
    const installed = { ...current, residencyState: inputs.residencyState }
    if (!inputs.updateAvailable) return { _tag: "Installed", ...installed }
    if (transfer?._tag === "Active") return { _tag: "Updating", ...installed, progress: transfer.progress }
    if (transfer?._tag === "Failed") return { _tag: "UpdateFailed", ...installed, failure: transfer.failure }
    return { _tag: "UpdateAvailable", ...installed }
  }
  const prior = inputs.priorInstalled
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
): LocalModelAvailabilityState => {
  if (Option.isNone(providerModelId)) return { _tag: "Installable" }
  const resolvedProviderModelId = providerModelId.value
  const providerEntry = providerEntries.get(resolvedProviderModelId)
  if (providerEntry?.availability._tag === "Available") {
    return { _tag: "Selectable", providerModelId: resolvedProviderModelId }
  }
  if (providerEntry?.availability._tag === "Disabled") {
    if (providerEntry.availability.reason === "installation_unavailable"
      || providerEntry.availability.reason === "provider_unavailable") {
      return { _tag: "Preparing", providerModelId: resolvedProviderModelId }
    }
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
  return { _tag: "Preparing", providerModelId: resolvedProviderModelId }
}

/** A prior effective configuration remains selectable while a desired update is incomplete. */
export const selectableProviderModelId = (
  modelId: ProviderModelId,
  effectiveConfigurationInstalled: boolean,
): Option.Option<ProviderModelId> => effectiveConfigurationInstalled
  ? Option.some(modelId)
  : Option.none()

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
  readonly changes: Stream.Stream<void>
  readonly refresh: Effect.Effect<void, LocalModelMutationFailed>
}

export class LocalModels extends Context.Tag("LocalModels")<LocalModels, LocalModelsApi>() {}

export const LocalModelsLive: Layer.Layer<
  LocalModels,
  never,
  LocalModelCatalogAdapter | LocalModelPackages | LocalModelRanker
    | LocalModelConfigurationResolver | LocalInferenceHardwareService
    | IcnInstances | LocalModelAcquisitionCoordinator
> = Layer.scoped(LocalModels, Effect.gen(function* () {
  const catalog = yield* LocalModelCatalogAdapter
  const packages = yield* LocalModelPackages
  const ranker = yield* LocalModelRanker
  const resolver = yield* LocalModelConfigurationResolver
  const hardware = yield* LocalInferenceHardwareService
  const instances = yield* IcnInstances
  const acquisition = yield* LocalModelAcquisitionCoordinator

  const projectCurrent = Effect.gen(function* () {
    const packageState = yield* packages.state
    const catalogState = yield* catalog.state
    const catalogEntries = new Map<string, (typeof catalogState.entries)[number]>(
      catalogState.entries.map((entry) => [localCatalogProviderModelId(entry.identity), entry]),
    )
    const rankingState = yield* ranker.state
    const resolvedConfigurations = new Map<string, ResolvedLocalModelConfiguration>(
      yield* resolver.get,
    )
    const hardwareState = yield* hardware.state
    const instanceState = yield* instances.get
    const coordination = yield* acquisition.state
    const groups = new Map([...resolvedConfigurations].map(([identity, resolution]) => [
      identity,
      resolution.servingConfiguration.bundle,
    ]))
    const downloadsById = new Map(packageState.downloads.map((download) => [download.id, download]))
    const packageEntries = new Map(
      packageState.entries.map((entry) => [entry.package.id, entry]),
    )
    const providerEntries = new Map(projectLocalProviderOfferings(
      [...resolvedConfigurations.values()],
      packageEntries,
    ).entries.map((entry) => [entry.providerModelId, entry]))

    const rankingEntries = rankingState._tag === "Ready" ? rankingState.entries : []
    const sameConfiguration = Schema.equivalence(ModelServingConfigurationSchema)
    const sameNativeConfiguration = Schema.equivalence(Generated.ModelServingConfiguration)

    const models: LocalModel[] = [...groups.entries()].map(([identity, bundle]): LocalModel => {
      const catalogEntry = catalogEntries.get(identity)
      if (catalogEntry === undefined) {
        throw new Error(`Catalog model ${identity} has no definition`)
      }
      const curated = catalogEntry.model
      const presentation = resolveBundlePresentation(bundle, {
        displayName: curated.displayName,
        variantLabel: curated.variantLabel,
        description: curated.description,
        license: curated.license,
      })
      const catalogLocalState = catalogEntry.source.localState
      const effectivePackageIds = servableModelBundlePackageIds(bundle)
      const nativeInstalled = catalogLocalState?._tag === "Installed"
        ? new Map(catalogLocalState.installation.packages.map((entry) => [entry.package.id, entry]))
        : new Map<string, never>()
      const effectiveInstalledPackages = catalogLocalState?._tag === "Installed"
        ? catalogLocalState.installation.packages.map((entry) => ({
            packageId: ModelPackageIdSchema.make(entry.package.id),
            path: entry.path,
            origin: entry.origin,
          }))
        : []
      const [firstEffectivePackage, ...remainingEffectivePackages] = effectiveInstalledPackages
      const effectiveInstalled: InstalledModelFields | undefined = firstEffectivePackage === undefined
        ? undefined
        : {
            installedBytes: catalogLocalState?._tag === "Installed"
              ? catalogLocalState.installation.packages.reduce((total, entry) => total
                + entry.package.files.reduce((sum, file) => sum + file.sizeBytes, 0), 0)
              : 0,
            packages: [firstEffectivePackage, ...remainingEffectivePackages],
          }
      const installedPackages = effectivePackageIds.flatMap((packageId) => {
        const entry = nativeInstalled.get(packageId)
        return entry === undefined ? [] : [{
          packageId: ModelPackageIdSchema.make(entry.package.id),
          path: entry.path,
          origin: entry.origin,
        }]
      })
      const [firstInstalledPackage, ...remainingInstalledPackages] = installedPackages
      const currentInstalled: InstalledModelFields | undefined = installedPackages.length === effectivePackageIds.length
        && firstInstalledPackage !== undefined
        ? {
            installedBytes: bundleDownloadBytes(bundle),
            packages: [firstInstalledPackage, ...remainingInstalledPackages],
          }
        : undefined
      const updateAvailable = catalogLocalState?._tag === "Installed"
        && catalogLocalState.updateState._tag === "Available"
      const modelId = localCatalogProviderModelId(curated)
      const syncState = coordination.syncs.get(modelId)
      const correlatedDownload = syncState?._tag === "Correlated"
        ? downloadsById.get(syncState.downloadId)
        : undefined
      const effectiveConfiguration = catalogLocalState?._tag === "Installed"
        && catalogLocalState.installation.effectiveConfiguration._tag === "Runnable"
        ? catalogLocalState.installation.effectiveConfiguration.configuration
        : undefined
      const modelInstance = instanceState.instances.findLast((candidate) =>
        candidate.modelId === modelId
        && effectiveConfiguration !== undefined
        && sameNativeConfiguration(candidate.configuration, effectiveConfiguration))
      const derivedAcquisitionState = deriveModelAcquisitionState({
        currentInstalled,
        download: correlatedDownload,
        syncState,
        downloadBytes: bundleDownloadBytes(bundle),
        updateAvailable,
        priorInstalled: effectiveInstalled,
        residencyState: modelInstance === undefined
          ? { _tag: "Unloaded" }
          : projectInferenceResidency(modelInstance),
      })
      const removal = coordination.removals.get(modelId)
      const installed = installedAcquisition(derivedAcquisitionState)
      const acquisitionState: LocalModelAcquisitionState = removal !== undefined && installed !== undefined
        ? removal._tag === "Removing"
          ? {
              _tag: "Removing",
              installedBytes: installed.installedBytes,
              packages: installed.packages,
              residencyState: installed.residencyState,
            }
          : {
              _tag: "RemoveFailed",
              installedBytes: installed.installedBytes,
              packages: installed.packages,
              residencyState: installed.residencyState,
              failure: removal.failure,
            }
        : derivedAcquisitionState
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
          // ICN resolves a canonical model ID to its effective installed
          // configuration while a desired update remains incomplete.
          const providerModelId = selectableProviderModelId(
            localCatalogProviderModelId(curated),
            effectiveInstalled !== undefined,
          )
          const availabilityState = assessment._tag !== "Fits"
            ? unavailableAssessment(assessment, providerModelId)
            : effectiveInstalled === undefined || Option.isNone(providerModelId)
              ? { _tag: "Installable" as const }
              : availabilityFromProviderProjection(
                  providerModelId,
                  providerEntries,
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
    return {
      inventoryState: packageState.inventory,
      models,
      discoveryState,
    }
  })
  const sourceChanges = Stream.mergeAll([
    packages.changes,
    catalog.changes,
    ranker.changes,
    resolver.changes,
    hardware.changes,
    instances.changes,
    acquisition.changes,
  ], { concurrency: "unbounded" }).pipe(Stream.map(() => undefined))

  const projection = yield* materializeProjection({
    project: projectCurrent.pipe(Effect.orDie),
    invalidations: sourceChanges,
    equivalent: Schema.equivalence(LocalModelsStateSchema),
  })

  return LocalModels.of({
    state: projection.get,
    changes: projection.changes.pipe(Stream.map(() => undefined)),
    refresh: packages.refresh,
  })
}))
