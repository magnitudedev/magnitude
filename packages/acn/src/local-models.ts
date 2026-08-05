import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelsMirror,
  type LocalModel,
  type LocalInferenceError,
  type LocalModelDownload,
  type LocalModelCatalogCandidateAvailability,
  type LocalModelsState,
  type ModelFailure,
  type ModelOfferingTarget,
  type ModelOfferingTargetId,
  type ModelPackageEntry,
  type ProviderModelCatalogEntry,
  modelOfferingTargetPackageIds,
} from "@magnitudedev/acn-protocol"
import type { ModelServingConfigurationId, ProviderModelId } from "@magnitudedev/sdk"
import { IcnCatalog } from "@magnitudedev/icn"
import { makeMirroredState, MirroredStateChanges } from "./mirrored-state"
import { LocalModelPackages } from "./local-model-packages"
import { LocalModelRecommendations } from "./local-model-recommendations"
import { LocalProviderOfferings } from "./local-provider-offerings"
import {
  LocalProviderOfferingProjection,
  providerOfferingPackageEvidence,
  sameProviderOfferingPackageEvidence,
} from "./local-provider-offering-projection"
import { LocalModelAssessments } from "./local-model-assessments"
import { recommendableModelFromIcn } from "./local-model-icn-adapter"

interface TargetProjection {
  readonly id: ModelOfferingTargetId
  readonly target: ModelOfferingTarget
  readonly displayName: string
  readonly description: string
}

const targetPackages = (target: ModelOfferingTarget) =>
  target._tag === "Package" ? [target.package] : [target.target, target.draft]

const sourceName = (target: ModelOfferingTarget): string => {
  const primary = target._tag === "Package" ? target.package : target.target
  return primary.source._tag === "HuggingFace"
    ? primary.source.repository.split("/").at(-1) ?? primary.source.repository
    : primary.files[0]?.path.split("/").at(-1) ?? primary.id
}

const aggregateDownload = (
  target: ModelOfferingTarget,
  entries: ReadonlyMap<string, ModelPackageEntry>,
): LocalModelDownload => {
  const packages = targetPackages(target)
  const totalBytes = packages.reduce(
    (total, modelPackage) =>
      total + modelPackage.files.reduce((sum, file) => sum + file.sizeBytes, 0),
    0,
  )
  const packageEntries = packages.map((modelPackage) => entries.get(modelPackage.id))
  const installedBytes = packages.reduce((total, modelPackage, index) =>
    total + (packageEntries[index]?.localState._tag === "Installed"
      ? modelPackage.files.reduce((sum, file) => sum + file.sizeBytes, 0)
      : 0), 0)
  if (packageEntries.every((entry) => entry?.localState._tag === "Installed")) {
    return { _tag: "Downloaded", installedBytes }
  }
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
    const rates = downloading.flatMap(({ bytesPerSecond }) =>
      Option.toArray(bytesPerSecond))
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
  const failedAttemptIds = packageEntries.flatMap((entry) =>
    entry?.localState._tag === "DownloadFailed" ? [entry.localState.attemptId] : [])
  const failedBytes = installedBytes + packages.reduce((total, modelPackage, index) => {
    const entry = packageEntries[index]
    if (!entry
      || entry.localState._tag !== "DownloadFailed") return total
    return total + entry.localState.completedBytes
  }, 0)
  return failed
    ? {
        _tag: "Failed",
        attemptIds: failedAttemptIds as [
          typeof failedAttemptIds[number],
          ...Array<typeof failedAttemptIds[number]>,
        ],
        completedBytes: failedBytes,
        totalBytes,
        failure: failed.failure,
      }
    : (() => {
        const cancelledAttemptIds = packageEntries.flatMap((entry) =>
          entry?.localState._tag === "DownloadCancelled"
            ? [entry.localState.attemptId]
            : [])
        return cancelledAttemptIds.length > 0
          ? {
              _tag: "Cancelled" as const,
              attemptIds: cancelledAttemptIds as [
                typeof cancelledAttemptIds[number],
                ...Array<typeof cancelledAttemptIds[number]>,
              ],
              completedBytes,
              totalBytes,
            }
          : { _tag: "NotDownloaded" as const, completedBytes, totalBytes }
      })()
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
  target: ModelOfferingTarget,
  entries: ReadonlyMap<string, ModelPackageEntry>,
  providerModelId: ProviderModelId | undefined,
  providerEntries: ReadonlyMap<ProviderModelId, ProviderModelCatalogEntry>,
  providerProjectionCurrent: boolean,
  providerProjectionFailure: Option.Option<ModelFailure>,
): LocalModelCatalogCandidateAvailability | undefined => {
  const targetEntries = modelOfferingTargetPackageIds(target).map((packageId) => entries.get(packageId))
  if (!targetEntries.every((entry) => entry?.localState._tag === "Installed")) {
    return { _tag: "NotDownloaded" }
  }
  const failure = targetEntries.flatMap((entry): readonly ModelFailure[] => {
    if (entry?.inspection._tag === "Invalid" || entry?.inspection._tag === "Incompatible") {
      return [entry.inspection.failure]
    }
    return []
  })[0]
  if (failure) return { _tag: "Unavailable", failure }
  if (targetEntries.some((entry) => entry?.inspection._tag === "Pending")) {
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
  readonly resolveTarget: (
    targetId: ModelOfferingTargetId,
  ) => Effect.Effect<ModelOfferingTarget | undefined, LocalInferenceError>
}

export class LocalModels extends Context.Tag("LocalModels")<LocalModels, LocalModelsApi>() {}

export const LocalModelsLive: Layer.Layer<
  LocalModels,
  never,
  IcnCatalog | LocalModelPackages | LocalModelRecommendations
    | LocalModelAssessments | LocalProviderOfferingProjection | LocalProviderOfferings | MirroredStateChanges
> = Layer.scoped(LocalModels, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const packages = yield* LocalModelPackages
  const recommendations = yield* LocalModelRecommendations
  const assessments = yield* LocalModelAssessments
  const offerings = yield* LocalProviderOfferings
  const offeringProjection = yield* LocalProviderOfferingProjection
  const mirror = yield* makeMirroredState(LocalModelsMirror, {
    models: [],
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
    const recommendationEntries = recommendationState._tag === "Ready"
      ? recommendationState.recommendations
      : []
    const configured = yield* offerings.list
    const projectedOfferings = yield* offeringProjection.state
    const packageEntries = new Map(
      packageState.entries.map((entry) => [entry.package.id, entry]),
    )
    const explicitStandalonePackageIds = new Set([
      ...catalogModels.flatMap(({ target }) =>
        target._tag === "Package" ? [target.package.id] : []),
      ...recommendationEntries.flatMap(({ configuration }) =>
        configuration.target._tag === "Package"
          ? [configuration.target.package.id]
          : []),
      ...configured.flatMap(({ configuration }) =>
        configuration.target._tag === "Package"
          ? [configuration.target.package.id]
          : []),
    ])
    const speculativePackageIds = new Set([
      ...catalogModels.flatMap(({ target }) =>
        target._tag === "SpeculativeDecodingPair"
          ? [target.target.id, target.draft.id]
          : []),
      ...recommendationEntries.flatMap(({ configuration }) =>
        configuration.target._tag === "SpeculativeDecodingPair"
          ? [configuration.target.target.id, configuration.target.draft.id]
          : []),
      ...configured.flatMap(({ configuration }) =>
        configuration.target._tag === "SpeculativeDecodingPair"
          ? [configuration.target.target.id, configuration.target.draft.id]
          : []),
    ])
    const targets = new Map<ModelOfferingTargetId, TargetProjection>()
    for (const model of catalogModels) {
      targets.set(model.targetId, {
        id: model.targetId,
        target: model.target,
        displayName: model.displayName,
        description: model.description,
      })
    }
    for (const entry of packageState.entries) {
      if (Option.isNone(entry.targetId)) continue
      if (speculativePackageIds.has(entry.package.id)
        && !explicitStandalonePackageIds.has(entry.package.id)) continue
      targets.set(entry.targetId.value, {
        id: entry.targetId.value,
        target: { _tag: "Package", package: entry.package },
        displayName: sourceName({ _tag: "Package", package: entry.package }),
        description: "",
      })
    }
    for (const recommendation of recommendationEntries) {
      targets.set(recommendation.targetId, {
        id: recommendation.targetId,
        target: recommendation.configuration.target,
        displayName: recommendation.displayName,
        description: recommendation.description,
      })
    }
    for (const offering of configured) {
      const current = targets.get(offering.targetId)
      targets.set(offering.targetId, {
        id: offering.targetId,
        target: offering.configuration.target,
        displayName: current?.displayName ?? sourceName(offering.configuration.target),
        description: current?.description ?? "",
      })
    }
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
    const models: LocalModel[] = [...targets.values()].map((projection): LocalModel => {
      const modelPackages = targetPackages(projection.target)
      return {
        targetId: projection.id,
        offerings: configured
          .filter(({ targetId }) => targetId === projection.id)
          .map(({ configuration, providerModelId }) => ({
            configurationId: configuration.id,
            providerModelId,
          })),
        displayName: projection.displayName,
        description: projection.description,
        kind: projection.target._tag === "Package" ? "Standalone" : "SpeculativePair",
        quantization: modelPackages.map(({ properties }) => properties.quantization).join(" + "),
        maximumContextLength: Math.min(
          ...modelPackages.map(({ properties }) => properties.maximumContextLength),
        ),
        downloadBytes: modelPackages.reduce((total, modelPackage) =>
          total + modelPackage.files.reduce((sum, file) => sum + file.sizeBytes, 0), 0),
        download: aggregateDownload(projection.target, packageEntries),
        assessment: assessmentState.get(projection.id) ?? { _tag: "Unassessed" },
      }
    }).sort((left, right) => left.displayName.localeCompare(right.displayName))
    const catalogCandidates = recommendationState._tag === "Ready"
      ? recommendationState.catalog.flatMap(({ candidate }) => {
          const model = models.find(({ targetId }) => targetId === candidate.targetId)
          const projection = targets.get(candidate.targetId)
          const availability = projection === undefined ? undefined : aggregateAvailability(
            projection.target,
            packageEntries,
            providerIdByConfiguration.get(candidate.configurationId),
            providerEntries,
            providerProjectionCurrent,
            providerProjectionFailure,
          )
          return model === undefined || availability === undefined
            ? []
            : [{
                ...candidate,
                download: model.download,
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
      models,
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
    offerings.changes,
    offeringProjection.changes,
  ], { concurrency: "unbounded" }).pipe(
    Stream.debounce("25 millis"),
    Stream.runForEach(() => project),
    Effect.forkScoped,
  )

  return LocalModels.of({
    snapshot: mirror.get,
    changes: mirror.changes,
    resolveTarget: (targetId) => Effect.gen(function* () {
      const recommendationState = (yield* recommendations.snapshot).state
      const catalogEntry = recommendationState._tag === "Ready"
        ? recommendationState.catalog.find(({ candidate }) => candidate.targetId === targetId)
        : undefined
      if (catalogEntry) return catalogEntry.configuration.target
      const offering = (yield* offerings.list).find((candidate) => candidate.targetId === targetId)
      if (offering) return offering.configuration.target
      const entry = (yield* packages.snapshot).state.entries.find((candidate) =>
        Option.exists(candidate.targetId, (candidateTargetId) => candidateTargetId === targetId))
      return entry ? { _tag: "Package" as const, package: entry.package } : undefined
    }),
  })
}))
