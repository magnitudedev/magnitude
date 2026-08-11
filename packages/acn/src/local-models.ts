import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelsMirror,
  ServableModelBundleSchema,
  servableModelBundlePackageIds,
  type LocalInferenceHardware,
  type LocalModel,
  type LocalModelAcquisitionState,
  type LocalModelAssessment,
  type LocalModelAvailabilityState,
  type LocalModelMemory,
  type LocalModelPresentation,
  type LocalModelRecommendation,
  type LocalModelsState,
  type MemoryAssessment,
  type ModelFailure,
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
import { LocalProviderOfferings } from "./local-provider-offerings"
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
  LocalModelConfigurationResolver,
  localModelBundleIdentity as bundleIdentity,
} from "./local-model-configuration-resolver"

const GIB = 1024 ** 3

interface ModelPresentationInput {
  readonly displayName: string
  readonly description: string
  readonly license?: string
}

const bundlePackages = (bundle: ServableModelBundle) =>
  bundle._tag === "Standalone" ? [bundle.package] : [bundle.target, bundle.draft]

const bundleDownloadBytes = (bundle: ServableModelBundle): number =>
  bundlePackages(bundle).reduce((total, modelPackage) => total
    + modelPackage.files.reduce((sum, file) => sum + file.sizeBytes, 0), 0)

const sourceName = (bundle: ServableModelBundle): string => {
  const primary = bundle._tag === "Standalone" ? bundle.package : bundle.target
  return primary.source._tag === "HuggingFace"
    ? primary.source.repository.split("/").at(-1) ?? primary.source.repository
    : primary.files[0]?.path.split("/").at(-1) ?? primary.id
}

export const resolveBundlePresentation = (
  bundle: ServableModelBundle,
  curated: ModelPresentationInput | undefined,
): LocalModelPresentation => ({
  displayName: curated?.displayName ?? sourceName(bundle),
  description: curated?.description ?? "",
  license: Option.fromNullable(curated?.license),
  quantization: bundlePackages(bundle)
    .map(({ properties }) => properties.quantization)
    .join(" + "),
  quantizationName: bundlePackages(bundle)
    .map(({ properties }) => properties.quantizationName)
    .join(" + "),
})

const installedBundle = (
  bundle: ServableModelBundle,
  entries: ReadonlyMap<string, ModelPackageEntry>,
): Extract<LocalModelAcquisitionState, { readonly _tag: "Installed" }> | undefined => {
  const packages = bundlePackages(bundle)
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

const aggregateAcquisitionState = (
  bundle: ServableModelBundle,
  entries: ReadonlyMap<string, ModelPackageEntry>,
): LocalModelAcquisitionState => {
  const packages = bundlePackages(bundle)
  const packageEntries = packages.map((modelPackage) => entries.get(modelPackage.id))
  const totalBytes = bundleDownloadBytes(bundle)
  const installedBytes = packages.reduce((total, modelPackage, index) =>
    total + (packageEntries[index]?.localState._tag === "Installed"
      ? modelPackage.files.reduce((sum, file) => sum + file.sizeBytes, 0)
      : 0), 0)
  const installation = installedBundle(bundle, entries)
  if (installation !== undefined) return installation

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
    : { _tag: "NotInstalled", completedBytes, totalBytes }
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
): number => {
  const allocations = new Map<string, Generated.ModelInstanceAllocation>()
  for (const instance of instances.instances) {
    if (instance.lifecycle._tag === "Ready") {
      allocations.set(instance.id, instance.lifecycle.allocation)
    } else if (instance.lifecycle._tag === "Stopping"
      && instance.lifecycle.allocation._tag === "Resident") {
      allocations.set(instance.id, instance.lifecycle.allocation.allocation)
    }
  }
  if (allocations.size > 1) {
    throw new Error("local-model projection observed multiple resident model instances")
  }
  const allocation = allocations.values().next().value as Generated.ModelInstanceAllocation | undefined
  return allocation?.memoryDomains.reduce((total, domain) =>
    systemDomains.has(domain.memoryDomainId)
      ? total + allocationBytes(domain)
      : total, 0) ?? 0
}

const projectMemory = (
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
  if (systemEvidence.length === 0) {
    return {
      domains,
      totalRequiredBytes,
      requiredSystemMemoryBytes,
      systemUseState: predictedHeadroomBytes < recommendedHeadroomBytes
        ? { _tag: "High", recommendedHeadroomBytes, predictedHeadroomBytes }
        : { _tag: "WithinRecommendedHeadroom", recommendedHeadroomBytes, predictedHeadroomBytes },
      currentHeadroomState: { _tag: "NotObserved" },
    }
  }
  const residentBytes = residentSystemAllocationBytes(instances, systemDomains)
  const allocationHeadroomBytes = Math.min(
    hardware.systemAllocationCapacityBytes,
    hardware.systemAllocationHeadroomBytes + residentBytes,
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
    systemUseState: predictedHeadroomBytes < recommendedHeadroomBytes
      ? { _tag: "High", recommendedHeadroomBytes, predictedHeadroomBytes }
      : { _tag: "WithinRecommendedHeadroom", recommendedHeadroomBytes, predictedHeadroomBytes },
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
      const identity = bundleIdentity(bundle)
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
    for (const entry of packageState.entries) {
      if (entry.localState._tag !== "Installed") continue
      const independentlyServable = entry.package.files.some(({ role }) =>
        role === "weights" || role === "draft")
      if (independentlyServable) addBundle({ _tag: "Standalone", package: entry.package })
    }

    const catalogByBundle = new Map(catalogModels.map((model) => [
      bundleIdentity(model.configuration.bundle),
      model,
    ]))
    const recommendationCandidates = recommendationState._tag === "Ready"
      ? recommendationState.catalog
      : []
    const recommendationsByConfiguration = new Map<string, LocalModelRecommendation[]>()
    if (recommendationState._tag === "Ready") {
      for (const recommendation of recommendationState.recommendations) {
        const entries = recommendationsByConfiguration.get(recommendation.configurationId) ?? []
        entries.push({
          id: recommendation.id,
          intent: recommendation.intent,
          explanation: recommendation.explanation,
        })
        recommendationsByConfiguration.set(recommendation.configurationId, entries)
      }
    }
    const recommendationOrder = new Map(recommendationCandidates.map((candidate, index) => [
      candidate.model.configuration.id,
      index,
    ]))

    const models: LocalModel[] = [...groups.values()].map((bundle): LocalModel => {
      const identity = bundleIdentity(bundle)
      const curated = catalogByBundle.get(identity)
      const presentation = resolveBundlePresentation(bundle, curated && {
        displayName: curated.displayName,
        description: curated.description,
        license: curated.license,
      })
      const acquisitionState = aggregateAcquisitionState(bundle, packageEntries)
      const bundleEntries = bundlePackages(bundle).map((modelPackage) =>
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
              memory: projectMemory(
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
          recommendations: recommendationsByConfiguration.get(configuration.id) ?? [],
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
      const leftConfigurationId = left.servingState._tag === "Resolving"
        ? undefined
        : Option.getOrUndefined(left.servingState._tag === "Failed"
          ? left.servingState.configuration
          : Option.some(left.servingState.configuration))?.id
      const rightConfigurationId = right.servingState._tag === "Resolving"
        ? undefined
        : Option.getOrUndefined(right.servingState._tag === "Failed"
          ? right.servingState.configuration
          : Option.some(right.servingState.configuration))?.id
      const leftOrder = leftConfigurationId === undefined
        ? Number.MAX_SAFE_INTEGER
        : recommendationOrder.get(leftConfigurationId) ?? Number.MAX_SAFE_INTEGER
      const rightOrder = rightConfigurationId === undefined
        ? Number.MAX_SAFE_INTEGER
        : recommendationOrder.get(rightConfigurationId) ?? Number.MAX_SAFE_INTEGER
      return leftOrder - rightOrder
        || left.presentation.displayName.localeCompare(right.presentation.displayName)
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
  })
}))
