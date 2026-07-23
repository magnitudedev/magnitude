import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelsMirror,
  type LocalModel,
  type LocalInferenceError,
  type LocalModelRecommendation,
  type LocalModelDownload,
  type LocalModelPreparation,
  type LocalModelsState,
  type ModelFailure,
  type ModelOfferingTarget,
  type ModelOfferingTargetId,
  type ModelPackageEntry,
  type ProviderModelCatalogEntry,
  type Recommendation,
  type RecommendableModel,
  modelOfferingTargetPackageIds,
} from "@magnitudedev/protocol"
import type { ProviderModelId } from "@magnitudedev/sdk"
import { IcnCatalog } from "@magnitudedev/icn"
import { makeMirroredState, MirroredStateChanges } from "./mirrored-state"
import { LocalModelPackages } from "./local-model-packages"
import { LocalModelRecommendations } from "./local-model-recommendations"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { LocalProviderOfferingProjection } from "./local-provider-offering-projection"
import {
  LocalModelAutoSetup,
  type LocalModelAutoSetupStatus,
} from "./local-model-auto-setup"
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
  const downloading = packageEntries.filter((entry) => entry?.localState._tag === "Downloading")
  const completedBytes = installedBytes + downloading.reduce(
    (total, entry) => total + (entry?.localState._tag === "Downloading"
      ? entry.localState.completedBytes
      : 0),
    0,
  )
  if (downloading.length > 0) return { _tag: "Downloading", completedBytes, totalBytes }
  const failed = packageEntries.flatMap((entry) =>
    entry
      && entry.localState._tag !== "Installed"
      && Option.isSome(entry.lastDownloadFailure)
      ? [entry.lastDownloadFailure.value]
      : [])[0]
  const failedBytes = installedBytes + packages.reduce((total, modelPackage, index) => {
    const entry = packageEntries[index]
    if (!entry
      || entry.localState._tag === "Installed"
      || Option.isNone(entry.lastDownloadFailure)) return total
    return total + entry.lastDownloadFailure.value.completedBytes
  }, 0)
  return failed
    ? { _tag: "Failed", completedBytes: failedBytes, totalBytes, failure: failed.failure }
    : { _tag: "NotDownloaded", completedBytes, totalBytes }
}

const aggregatePreparation = (
  modelId: ModelOfferingTargetId,
  target: ModelOfferingTarget,
  entries: ReadonlyMap<string, ModelPackageEntry>,
  configuredProviderModelIds: readonly ProviderModelId[],
  providerEntries: ReadonlyMap<ProviderModelId, ProviderModelCatalogEntry>,
  providerProjectionFailure: Option.Option<ModelFailure>,
  autoSetup: ReadonlyMap<ModelOfferingTargetId, LocalModelAutoSetupStatus>,
): LocalModelPreparation => {
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
  if (failure) return { _tag: "Unavailable", providerModelIds: configuredProviderModelIds, failure }
  const setup = autoSetup.get(modelId)
  if (setup) return setup._tag === "Preparing"
    ? setup
    : {
        _tag: "Unavailable",
        providerModelIds: configuredProviderModelIds,
        failure: setup.failure,
      }
  const availableProviderModelIds = configuredProviderModelIds.filter((providerModelId) =>
    providerEntries.get(providerModelId)?.availability._tag === "Available")
  if (availableProviderModelIds.length > 0) {
    return { _tag: "Available", providerModelIds: availableProviderModelIds }
  }
  const disabled = configuredProviderModelIds
    .map((providerModelId) => providerEntries.get(providerModelId))
    .find((entry) => entry?.availability._tag === "Disabled")
  if (disabled?.availability._tag === "Disabled") {
    return {
      _tag: "Unavailable",
      providerModelIds: configuredProviderModelIds,
      failure: {
        code: disabled.availability.reason,
        message: disabled.availability.reason === "insufficient_resources"
          ? "This model configuration no longer fits the available hardware capacity"
          : "This model configuration is not available to the local runtime",
        retryable: true,
      },
    }
  }
  if (configuredProviderModelIds.length > 0 && Option.isSome(providerProjectionFailure)) {
    return {
      _tag: "Unavailable",
      providerModelIds: configuredProviderModelIds,
      failure: providerProjectionFailure.value,
    }
  }
  return { _tag: "Preparing" }
}

const recommendationProjection = (
  recommendation: Recommendation,
  recommendable: RecommendableModel,
): LocalModelRecommendation => {
  const requiredBytes = recommendation.assessment.memory
    .reduce((total, memory) => total + memory.requiredBytes, 0)
  const availableBytes = recommendation.assessment.memory
    .reduce((total, memory) => total + memory.capacityBytes, 0)
  return {
    id: recommendation.id,
    modelId: recommendation.modelId,
    displayName: recommendation.displayName,
    intent: recommendation.intent,
    explanation: recommendation.explanation,
    sources: targetPackages(recommendation.configuration.target).map((modelPackage) => ({
      source: modelPackage.source,
      files: modelPackage.files.map(({ path, sha256 }) => ({ path, sha256 })),
    })),
    qualityScoreProvenance: recommendable.qualityScoreProvenance,
    fidelityRank: recommendable.fidelityRank,
    qualityEvidence: recommendable.qualityEvidence,
    profile: recommendation.configuration.profile,
    fit: {
      requiredBytes,
      availableBytes,
      estimatedTokensPerSecond: Option.map(
        recommendation.assessment.performance,
        ({ estimatedTokensPerSecond }) => estimatedTokensPerSecond,
      ),
    },
  }
}

export interface LocalModelsApi {
  readonly snapshot: Effect.Effect<{ readonly revision: number; readonly state: LocalModelsState }>
  readonly changes: Stream.Stream<{ readonly revision: number; readonly state: LocalModelsState }>
  readonly refresh: Effect.Effect<void>
  readonly target: (
    modelId: ModelOfferingTargetId,
  ) => Effect.Effect<ModelOfferingTarget | undefined, LocalInferenceError>
}

export class LocalModels extends Context.Tag("LocalModels")<LocalModels, LocalModelsApi>() {}

export const LocalModelsLive: Layer.Layer<
  LocalModels,
  never,
  IcnCatalog | LocalModelAutoSetup | LocalModelPackages | LocalModelRecommendations
    | LocalProviderOfferingProjection | LocalProviderOfferings | MirroredStateChanges
> = Layer.scoped(LocalModels, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const autoSetup = yield* LocalModelAutoSetup
  const packages = yield* LocalModelPackages
  const recommendations = yield* LocalModelRecommendations
  const offerings = yield* LocalProviderOfferings
  const offeringProjection = yield* LocalProviderOfferingProjection
  const mirror = yield* makeMirroredState(LocalModelsMirror, {
    models: [],
    recommendations: { _tag: "Loading" },
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
    const recommendationEntries = recommendationState._tag === "Ready"
      ? recommendationState.recommendations
      : []
    const configured = yield* offerings.list
    const projectedOfferings = yield* offeringProjection.state
    const setupStatuses = yield* autoSetup.statuses
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
      targets.set(recommendation.modelId, {
        id: recommendation.modelId,
        target: recommendation.configuration.target,
        displayName: recommendation.displayName,
        description: recommendation.description,
      })
    }
    for (const offering of configured) {
      const current = targets.get(offering.modelId)
      targets.set(offering.modelId, {
        id: offering.modelId,
        target: offering.configuration.target,
        displayName: current?.displayName ?? sourceName(offering.configuration.target),
        description: current?.description ?? "",
      })
    }
    const providerIdsByTarget = new Map<ModelOfferingTargetId, ProviderModelId[]>()
    for (const offering of configured) {
      const ids = providerIdsByTarget.get(offering.modelId) ?? []
      ids.push(offering.providerModelId)
      providerIdsByTarget.set(offering.modelId, ids)
    }
    const providerEntries = new Map(
      projectedOfferings.entries.map((entry) => [entry.providerModelId, entry]),
    )
    const providerProjectionFailure = Option.map(projectedOfferings.failure, (error): ModelFailure => ({
      code: "local_offering_assessment_unavailable",
      message: error.message,
      retryable: "retryable" in error ? error.retryable : true,
    }))
    const models: LocalModel[] = [...targets.values()].map((projection): LocalModel => {
      const modelPackages = targetPackages(projection.target)
      return {
        id: projection.id,
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
        preparation: aggregatePreparation(
          projection.id,
          projection.target,
          packageEntries,
          providerIdsByTarget.get(projection.id) ?? [],
          providerEntries,
          providerProjectionFailure,
          setupStatuses,
        ),
      }
    }).sort((left, right) => left.displayName.localeCompare(right.displayName))
    const recommendationLifecycle = recommendationState._tag === "Loading"
      ? { _tag: "Loading" as const }
      : recommendationState._tag === "Failed"
        ? { _tag: "Failed" as const, failure: recommendationState.failure }
        : {
            _tag: "Ready" as const,
            entries: recommendationState.recommendations.flatMap((recommendation) => {
              const recommendable = catalogModels.find((model) =>
                model.id === recommendation.recommendableModelId)
              return recommendable
                ? [recommendationProjection(recommendation, recommendable)]
                : []
            }),
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
    offerings.changes,
    offeringProjection.changes,
    autoSetup.changes,
  ], { concurrency: "unbounded" }).pipe(
    Stream.debounce("25 millis"),
    Stream.runForEach(() => project),
    Effect.forkScoped,
  )

  return LocalModels.of({
    snapshot: mirror.get,
    changes: mirror.changes,
    refresh: project,
    target: (modelId) => Effect.gen(function* () {
      const recommendationState = (yield* recommendations.snapshot).state
      const recommendation = recommendationState._tag === "Ready"
        ? recommendationState.recommendations.find((candidate) => candidate.modelId === modelId)
        : undefined
      if (recommendation) return recommendation.configuration.target
      const offering = (yield* offerings.list).find((candidate) => candidate.modelId === modelId)
      if (offering) return offering.configuration.target
      const entry = (yield* packages.snapshot).state.entries.find((candidate) =>
        Option.exists(candidate.targetId, (targetId) => targetId === modelId))
      return entry ? { _tag: "Package" as const, package: entry.package } : undefined
    }),
  })
}))
