import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelsMirror,
  ServableModelBundleSchema,
  sameServableModelBundleIdentity,
  servableModelBundlePackageIds,
  servableModelBundlePackages,
  type LocalInferenceHardware,
  type LocalModel,
  type LocalModelAcquisitionState,
  type LocalModelAssessment,
  type LocalModelAvailabilityState,
  type LocalModelMemory,
  type LocalModelRecommendation,
  type LocalModelsState,
  type MemoryAssessment,
  type ModelFailure,
  type ModelBundleDownload,
  type ModelPackageEntry,
  type ModelServingConfiguration,
  type ProviderModelCatalogEntry,
  type ServableModelBundle,
} from "@magnitudedev/acn-protocol"
import type { ProviderModelId } from "@magnitudedev/ai"
import { IcnCatalog, IcnInstances } from "@magnitudedev/icn"
import type * as Generated from "@magnitudedev/icn-protocol/schemas"
import { makeMirroredState, MirroredStateChanges } from "./mirrored-state"
import { LocalInferenceHardware as LocalInferenceHardwareService } from "./local-inference-hardware"
import { LocalModelPackages } from "./local-model-packages"
import { LocalModelRecommendations } from "./local-model-recommendations"
import { LocalProviderOfferings, localProviderModelId } from "./local-provider-offerings"
import {
  providerOfferingPackageEvidence,
  sameProviderOfferingPackageEvidence,
} from "./local-provider-offerings"
import {
  localModelAssessmentProfiles,
  MINIMUM_LOCAL_MODEL_CONTEXT_LENGTH,
} from "./local-model-assessments"
import { recommendableModelFromIcn } from "./local-model-icn-adapter"
import {
  configuredModelPackageIds,
  isStandalonePackageCandidate,
  LocalModelConfigurationResolver,
  localModelTargetIdentity as targetIdentity,
} from "./local-model-configuration-resolver"
import { bundlePackages, resolveBundlePresentation } from "./local-model-presentation"
export { resolveBundlePresentation } from "./local-model-presentation"

const GIB = 1024 ** 3

const bundleDownloadBytes = (bundle: ServableModelBundle): number =>
  servableModelBundlePackages(bundle).reduce((total, modelPackage) => total
    + modelPackage.files.reduce((sum, file) => sum + file.sizeBytes, 0), 0)

const installedBundle = (
  bundle: ServableModelBundle,
  entries: ReadonlyMap<string, ModelPackageEntry>,
): Extract<LocalModelAcquisitionState, { readonly _tag: "Installed" }> | undefined => {
  const packages = servableModelBundlePackages(bundle)
  const packageEntries = packages.map((modelPackage) => entries.get(modelPackage.id))
  if (!packageEntries.every((entry) => entry?.localState._tag === "Installed")) {
    return undefined
  }
  const installedBytes = bundleDownloadBytes(bundle)
  const [origin, ...remainingOrigins] = [...new Set(packageEntries.flatMap((entry) =>
    entry?.localState._tag === "Installed" ? [entry.localState.origin] : []))]
  if (origin === undefined) {
    throw new Error("Installed servable model bundle has no installation origin")
  }
  return { _tag: "Installed", installedBytes, origins: [origin, ...remainingOrigins] }
}

export const aggregateAcquisitionState = (
  bundle: ServableModelBundle,
  entries: ReadonlyMap<string, ModelPackageEntry>,
  downloads: readonly ModelBundleDownload[],
): LocalModelAcquisitionState => {
  const packages = servableModelBundlePackages(bundle)
  const packageEntries = packages.map((modelPackage) => entries.get(modelPackage.id))
  const totalBytes = bundleDownloadBytes(bundle)
  const installedBytes = packages.reduce((total, modelPackage, index) =>
    total + (packageEntries[index]?.localState._tag === "Installed"
      ? modelPackage.files.reduce((sum, file) => sum + file.sizeBytes, 0)
      : 0), 0)
  const installation = installedBundle(bundle, entries)
  if (installation !== undefined) return installation
  let current: ModelBundleDownload | undefined
  for (let index = downloads.length - 1; index >= 0; index--) {
    const candidate = downloads[index]
    if (candidate !== undefined && sameServableModelBundleIdentity(candidate.bundle, bundle)) {
      current = candidate
      break
    }
  }
  if (current === undefined) return { _tag: "NotInstalled", completedBytes: installedBytes, totalBytes }
  switch (current.state._tag) {
    case "Pending":
      return {
        _tag: "Downloading",
        downloadId: current.id,
        stage: "queued",
        completedBytes: current.state.completedBytes,
        totalBytes: current.state.totalBytes,
        bytesPerSecond: Option.none(),
      }
    case "Downloading":
      return {
        _tag: "Downloading",
        downloadId: current.id,
        stage: current.state.stage,
        completedBytes: current.state.completedBytes,
        totalBytes: current.state.totalBytes,
        bytesPerSecond: current.state.bytesPerSecond,
      }
    case "Completed":
      return { _tag: "NotInstalled", completedBytes: installedBytes, totalBytes }
    case "Failed":
      return current.state.acknowledged
        ? { _tag: "NotInstalled", completedBytes: installedBytes, totalBytes }
        : {
            _tag: "Failed",
            downloadId: current.id,
            completedBytes: current.state.completedBytes,
            totalBytes: current.state.totalBytes,
            failure: current.state.failure,
          }
    case "Cancelled":
      return {
        _tag: "Cancelled",
        downloadId: current.id,
        completedBytes: current.state.completedBytes,
        totalBytes: current.state.totalBytes,
      }
  }
}

type ProviderAvailabilityProjection = Pick<ProviderModelCatalogEntry, "availability">

export const availabilityFromProviderProjection = (
  providerModelId: ProviderModelId | undefined,
  providerEntries: ReadonlyMap<ProviderModelId, ProviderAvailabilityProjection>,
  projectionCurrent: boolean,
  providerProjectionFailure: Option.Option<ModelFailure>,
): LocalModelAvailabilityState => {
  if (providerModelId === undefined) return { _tag: "Installable" }
  if (!projectionCurrent) return { _tag: "Preparing", providerModelId }
  const providerEntry = providerEntries.get(providerModelId)
  if (providerEntry?.availability._tag === "Available") {
    return { _tag: "Selectable", providerModelId }
  }
  if (providerEntry?.availability._tag === "Disabled") {
    return {
      _tag: "Unavailable",
      providerModelId: Option.some(providerModelId),
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
      providerModelId: Option.some(providerModelId),
      failure: providerProjectionFailure.value,
    }
  }
  return { _tag: "Preparing", providerModelId }
}

const unavailableAssessment = (
  assessment: Exclude<LocalModelAssessment, { readonly _tag: "Fits" }>,
  providerModelId: ProviderModelId | undefined,
): LocalModelAvailabilityState => ({
  _tag: "Unavailable",
  providerModelId: Option.fromNullable(providerModelId),
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
  readonly snapshot: Effect.Effect<{ readonly revision: number; readonly state: LocalModelsState }>
  readonly changes: Stream.Stream<{ readonly revision: number; readonly state: LocalModelsState }>
  /** Publishes a snapshot from the current lower-domain facts before returning. */
  readonly refresh: Effect.Effect<void>
}

export class LocalModels extends Context.Tag("LocalModels")<LocalModels, LocalModelsApi>() {}

export const LocalModelsLive: Layer.Layer<
  LocalModels,
  never,
  IcnCatalog | LocalModelPackages | LocalModelRecommendations
    | LocalModelConfigurationResolver | LocalProviderOfferings | LocalInferenceHardwareService
    | IcnInstances | MirroredStateChanges
> = Layer.scoped(LocalModels, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const packages = yield* LocalModelPackages
  const recommendations = yield* LocalModelRecommendations
  const resolver = yield* LocalModelConfigurationResolver
  const offerings = yield* LocalProviderOfferings
  const hardware = yield* LocalInferenceHardwareService
  const instances = yield* IcnInstances
  const mirror = yield* makeMirroredState(LocalModelsMirror, {
    inventoryState: { _tag: "Initializing" },
    models: [],
    discoveryState: { _tag: "Loading", progress: [] },
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
    const resolvedConfigurations = yield* resolver.get
    const configured = yield* offerings.list
    const projectedOfferings = yield* offerings.state
    const hardwareState = (yield* hardware.snapshot).state
    const instanceState = yield* instances.get
    const packageEntries = new Map(
      packageState.entries.map((entry) => [entry.package.id, entry]),
    )
    const sameBundle = Schema.equivalence(ServableModelBundleSchema)
    const providerIdByConfiguration = new Map(configured.map((offering) => [
      offering.configuration.id,
      offering.providerModelId,
    ]))
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

    const groups = new Map<string, ServableModelBundle>()
    const addBundle = (bundle: ServableModelBundle) => {
      const identity = targetIdentity(bundle)
      const existing = groups.get(identity)
      if (existing === undefined) {
        groups.set(identity, bundle)
      } else if (!sameBundle(existing, bundle)) {
        throw new Error(`Servable model bundle ${identity} has conflicting package definitions`)
      }
    }
    for (const { configuration } of resolvedConfigurations.values()) {
      addBundle(configuration.bundle)
    }
    const configuredPackages = configuredModelPackageIds(
      [...resolvedConfigurations.values()].map(({ configuration }) => configuration),
    )
    for (const entry of packageState.entries) {
      if (entry.localState._tag !== "Installed") continue
      if (isStandalonePackageCandidate(entry.package, configuredPackages)) {
        addBundle({ _tag: "Standalone", package: entry.package })
      }
    }

    const catalogByTarget = new Map(catalogModels.map((model) => [
      targetIdentity(model.configuration.bundle),
      model,
    ]))
    const recommendationCandidates = recommendationState._tag === "Ready"
      ? recommendationState.catalog
      : []
    const recommendationTargetByConfiguration = new Map(recommendationCandidates.map((candidate) => [
      candidate.model.configuration.id,
      targetIdentity(candidate.model.configuration.bundle),
    ]))
    const recommendationsByTarget = new Map<string, LocalModelRecommendation[]>()
    if (recommendationState._tag === "Ready") {
      for (const recommendation of recommendationState.recommendations) {
        const target = recommendationTargetByConfiguration.get(recommendation.configurationId)
        if (target === undefined) continue
        const entries = recommendationsByTarget.get(target) ?? []
        entries.push({
          id: recommendation.id,
          intent: recommendation.intent,
          explanation: recommendation.explanation,
        })
        recommendationsByTarget.set(target, entries)
      }
    }
    const recommendationOrderByTarget = new Map(recommendationCandidates.map((candidate, index) => [
      targetIdentity(candidate.model.configuration.bundle),
      index,
    ]))

    const models: LocalModel[] = [...groups.values()].map((bundle): LocalModel => {
      const identity = targetIdentity(bundle)
      const curated = catalogByTarget.get(identity)
      const presentation = resolveBundlePresentation(bundle, curated && {
        displayName: curated.displayName,
        variantLabel: curated.variantLabel,
        description: curated.description,
        license: curated.license,
      })
      const acquisitionState = aggregateAcquisitionState(bundle, packageEntries, packageState.downloads)
      const bundleEntries = servableModelBundlePackages(bundle).map((modelPackage) =>
        packageEntries.get(modelPackage.id))
      const inspectionFailure = bundleEntries.flatMap((entry) =>
        entry?.inspection._tag === "Invalid" || entry?.inspection._tag === "Incompatible"
          ? [entry.inspection.failure]
          : [])[0]
      const primaryPackage = bundle._tag === "Standalone" ? bundle.package : bundle.target
      const primaryInspection = packageEntries.get(primaryPackage.id)?.inspection
      const inspectionComplete = acquisitionState._tag !== "Installed"
        || bundleEntries.every((entry) => entry?.inspection._tag === "Inspected")
      const resolved = resolvedConfigurations.get(identity)
      const configuration = resolved?.configuration
      const coordinatedAssessment = resolved === undefined
        ? undefined
        : Option.getOrUndefined(resolved.assessment)
      const capabilities = curated?.capabilities
        ?? (primaryInspection?._tag === "Inspected" ? primaryInspection.capabilities : undefined)
      const catalogMembershipState: LocalModel["catalogMembershipState"] = curated === undefined
        ? { _tag: "NotInCatalog" }
        : {
            _tag: "InCatalog",
            catalogData: {
              intelligenceScore: curated.qualityScore,
              intelligenceScoreSource: curated.qualityScoreProvenance,
              fidelityRank: curated.fidelityRank,
              quantizationAware: curated.quantizationAware,
              qualityNotes: curated.qualityEvidence,
            },
          }

      let servingState: LocalModel["servingState"]
      if (inspectionFailure !== undefined) {
        servingState = {
          _tag: "Failed",
          configuration: Option.fromNullable(configuration),
          failure: inspectionFailure,
        }
      } else if (configuration === undefined) {
        servingState = localModelAssessmentProfiles(bundle).length === 0
          ? {
              _tag: "Failed",
              configuration: Option.none(),
              failure: {
                code: "unsupported_model_context_length",
                message: `Model context length is below the ${MINIMUM_LOCAL_MODEL_CONTEXT_LENGTH.toLocaleString("en-US")}-token minimum`,
                retryable: false,
              },
            }
          : { _tag: "Resolving" }
      } else if (!inspectionComplete || coordinatedAssessment === undefined
        || coordinatedAssessment._tag === "Assessing") {
        servingState = { _tag: "Assessing", configuration }
      } else if (coordinatedAssessment._tag === "Failed") {
        servingState = {
          _tag: "Failed",
          configuration: Option.some(configuration),
          failure: coordinatedAssessment.failure,
        }
      } else {
        if (capabilities === undefined) {
          throw new Error(`Assessed local model ${configuration.id} has no capabilities`)
        }
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
        const providerModelId = providerIdByConfiguration.get(configuration.id)
          ?? (acquisitionState._tag === "Installed"
            ? localProviderModelId(configuration.id)
            : undefined)
        const availabilityState = assessment._tag !== "Fits"
          ? unavailableAssessment(assessment, providerModelId)
          : acquisitionState._tag !== "Installed" || providerModelId === undefined
            ? { _tag: "Installable" as const }
            : availabilityFromProviderProjection(
                providerModelId,
                providerEntries,
                providerProjectionCurrent,
                providerProjectionFailure,
              )
        servingState = {
          _tag: "Assessed",
          configuration,
          capabilities,
          assessment,
          availabilityState,
          recommendations: recommendationsByTarget.get(identity) ?? [],
        }
      }
      return {
        bundle,
        presentation,
        downloadBytes: bundleDownloadBytes(bundle),
        catalogMembershipState,
        acquisitionState,
        servingState,
      }
    }).sort((left, right) => {
      const leftOrder = recommendationOrderByTarget.get(targetIdentity(left.bundle))
        ?? Number.MAX_SAFE_INTEGER
      const rightOrder = recommendationOrderByTarget.get(targetIdentity(right.bundle))
        ?? Number.MAX_SAFE_INTEGER
      return leftOrder - rightOrder
        || left.presentation.displayName.localeCompare(right.presentation.displayName)
        || left.presentation.variantLabel.localeCompare(right.presentation.variantLabel)
        || targetIdentity(left.bundle).localeCompare(targetIdentity(right.bundle))
    })

    const discoveryState = recommendationState._tag === "Loading"
      ? { _tag: "Loading" as const, progress: recommendationState.progress }
      : recommendationState._tag === "Failed"
        ? {
            _tag: "Failed" as const,
            failure: recommendationState.failure,
            progress: recommendationState.progress,
          }
        : { _tag: "Ready" as const, progress: recommendationState.progress }
    yield* mirror.setIfChanged({
      inventoryState: packageState.inventory,
      models,
      discoveryState,
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
    snapshot: mirror.get,
    changes: mirror.changes,
    refresh: project,
  })
}))
