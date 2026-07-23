import { Cause, Context, Effect, Layer, Option, Ref, Schema, Stream } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import { createHash } from "node:crypto"
import * as NodePath from "node:path"
import {
  ModelFailureSchema,
  LocalModelRecommendationProgressStepSchema,
  RecommendationSchema,
  type LocalModelRecommendationProgressStep,
  type LocalModelRecommendationProgressStepId,
  type ModelFailure,
  type Recommendation,
  type RecommendationId,
  type RecommendableModel,
} from "@magnitudedev/protocol"
import { IcnCatalog, IcnHardware, IcnInstalledModels } from "@magnitudedev/icn"
import {
  readStructuredFile,
  writeStructuredFileAtomic,
} from "@magnitudedev/storage"
import { makeObservedState } from "./mirrored-state"
import { LocalModelEvaluations } from "./local-model-evaluations"
import { recommendableModelFromIcn } from "./local-model-icn-adapter"
import {
  selectRecommendationPortfolio,
  type RecommendationCandidate,
} from "./local-model-recommendation-policy"

type RecommendationState =
  | {
      readonly _tag: "Loading"
      readonly progress: readonly LocalModelRecommendationProgressStep[]
    }
  | {
      readonly _tag: "Ready"
      readonly recommendations: readonly Recommendation[]
      readonly progress: readonly LocalModelRecommendationProgressStep[]
    }
  | {
      readonly _tag: "Failed"
      readonly failure: ModelFailure
      readonly progress: readonly LocalModelRecommendationProgressStep[]
    }


export interface LocalModelRecommendationsApi {
  readonly snapshot: Effect.Effect<{ readonly revision: number; readonly state: RecommendationState }>
  readonly changes: Stream.Stream<{ readonly revision: number; readonly state: RecommendationState }>
  readonly refresh: Effect.Effect<void>
  readonly get: (id: RecommendationId) => Effect.Effect<Recommendation | undefined>
}

export class LocalModelRecommendations extends Context.Tag("LocalModelRecommendations")<
  LocalModelRecommendations,
  LocalModelRecommendationsApi
>() {}

const publishedWeightBytes = (model: RecommendableModel): number => {
  const packages = model.target._tag === "Package"
    ? [model.target.package]
    : [model.target.target, model.target.draft]
  return packages.flatMap(({ files }) => files)
    .filter(({ role }) => role === "weights")
    .reduce((total, { sizeBytes }) => total + sizeBytes, 0)
}

const pendingProgress = (
  id: LocalModelRecommendationProgressStepId,
): LocalModelRecommendationProgressStep => ({
  id,
  status: { _tag: "Pending" },
  completedItems: Option.none(),
  totalItems: Option.none(),
})

const initialProgress = (): readonly LocalModelRecommendationProgressStep[] => [
  pendingProgress("hardware"),
  pendingProgress("inventory"),
  pendingProgress("catalog"),
  pendingProgress("metadata"),
  pendingProgress("assessment"),
  pendingProgress("selection"),
]

const updateProgress = (
  progress: readonly LocalModelRecommendationProgressStep[],
  id: LocalModelRecommendationProgressStepId,
  update: Partial<LocalModelRecommendationProgressStep>,
): readonly LocalModelRecommendationProgressStep[] =>
  progress.map((step) => step.id === id ? { ...step, ...update } : step)

const NonNegativeSafeInteger = Schema.Number.pipe(
  Schema.int(),
  Schema.nonNegative(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER),
)

const CachedRecommendationPortfolioSchema = Schema.Struct({
  capturedAtMs: NonNegativeSafeInteger,
  inputDigest: Schema.NonEmptyString,
  recommendations: Schema.Array(RecommendationSchema),
})

type CachedRecommendationPortfolio = typeof CachedRecommendationPortfolioSchema.Type
const RECOMMENDATION_PORTFOLIO_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1_000

export const makeLocalModelRecommendationsLive = (
  dataDir: string,
): Layer.Layer<
  LocalModelRecommendations,
  never,
  IcnCatalog | IcnHardware | IcnInstalledModels | LocalModelEvaluations | FileSystem.FileSystem
> => Layer.scoped(LocalModelRecommendations, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const hardware = yield* IcnHardware
  const installed = yield* IcnInstalledModels
  const evaluations = yield* LocalModelEvaluations
  const fs = yield* FileSystem.FileSystem
  const portfolioPath = NodePath.join(
    dataDir,
    "cache",
    "local-model-recommendations.json",
  )
  const cachedPortfolio = yield* readStructuredFile(
    portfolioPath,
    CachedRecommendationPortfolioSchema,
  ).pipe(
    Effect.provideService(FileSystem.FileSystem, fs),
    Effect.map((result): Option.Option<CachedRecommendationPortfolio> =>
      result._tag === "Present" ? Option.some(result.value) : Option.none()),
    Effect.catchAll(() => Effect.succeed(Option.none())),
  )
  const cachedPortfolioRef = yield* Ref.make(cachedPortfolio)
  const startupStartedAtMs = Date.now()
  const startupProgress = updateProgress(initialProgress(), "hardware", {
    status: { _tag: "Running", startedAtMs: startupStartedAtMs },
  })
  const mirror = yield* makeObservedState<RecommendationState>({
    _tag: "Loading",
    progress: startupProgress,
  })
  const progressRef = yield* Ref.make(startupProgress)
  const lastInputDigest = yield* Ref.make<Option.Option<string>>(Option.none())
  const recommendationsEquivalent = Schema.equivalence(Schema.Array(RecommendationSchema))
  const failuresEquivalent = Schema.equivalence(ModelFailureSchema)
  const progressEquivalent = Schema.equivalence(
    Schema.Array(LocalModelRecommendationProgressStepSchema),
  )
  const equivalent = (left: RecommendationState, right: RecommendationState): boolean =>
    left._tag === right._tag
    && progressEquivalent(left.progress, right.progress)
    && (left._tag === "Loading"
      || (left._tag === "Ready" && right._tag === "Ready"
        && recommendationsEquivalent(left.recommendations, right.recommendations))
      || (left._tag === "Failed" && right._tag === "Failed"
        && failuresEquivalent(left.failure, right.failure)))
  const lock = yield* Effect.makeSemaphore(1)

  const publishProgress = (
    progress: readonly LocalModelRecommendationProgressStep[],
  ): Effect.Effect<void> => Effect.gen(function* () {
    yield* Ref.set(progressRef, progress)
    const current = (yield* mirror.get).state
    yield* mirror.setIfChanged(
      current._tag === "Ready"
        ? { ...current, progress }
        : current._tag === "Failed"
          ? { _tag: "Loading", progress }
          : { ...current, progress },
      equivalent,
    )
  })

  const startStep = (
    progress: readonly LocalModelRecommendationProgressStep[],
    id: LocalModelRecommendationProgressStepId,
    counts?: { readonly completed: number; readonly total: number },
  ) => {
    const next = updateProgress(progress, id, {
      status: { _tag: "Running", startedAtMs: Date.now() },
      completedItems: counts ? Option.some(counts.completed) : Option.none(),
      totalItems: counts ? Option.some(counts.total) : Option.none(),
    })
    return publishProgress(next).pipe(Effect.as(next))
  }

  const completeStep = (
    progress: readonly LocalModelRecommendationProgressStep[],
    id: LocalModelRecommendationProgressStepId,
    startedAtMs: number,
    cached: boolean,
    counts?: { readonly completed: number; readonly total: number },
  ) => {
    const next = updateProgress(progress, id, {
      status: {
        _tag: "Completed",
        startedAtMs,
        durationMs: Math.max(0, Date.now() - startedAtMs),
        cached,
      },
      completedItems: counts ? Option.some(counts.completed) : Option.none(),
      totalItems: counts ? Option.some(counts.total) : Option.none(),
    })
    return publishProgress(next).pipe(Effect.as(next))
  }

  const refresh = lock.withPermits(1)(Effect.gen(function* () {
    const currentStateBeforeRefresh = (yield* mirror.get).state
    let progress = currentStateBeforeRefresh._tag === "Loading"
      ? yield* Ref.get(progressRef)
      : initialProgress()
    const hardwareStep = progress.find(({ id }) => id === "hardware")
    const hardwareStartedAt = hardwareStep?.status._tag === "Running"
      ? hardwareStep.status.startedAtMs
      : Date.now()
    if (hardwareStep?.status._tag !== "Running") {
      progress = yield* startStep(progress, "hardware")
    }
    const hardwareSnapshot = (yield* hardware.get).state
    progress = yield* completeStep(
      progress,
      "hardware",
      hardwareStartedAt,
      false,
      {
        completed: hardwareSnapshot.memory_domains.length,
        total: hardwareSnapshot.memory_domains.length,
      },
    )

    const inventoryStep = progress.find(({ id }) => id === "inventory")
    const inventoryStartedAt = inventoryStep?.status._tag === "Running"
      ? inventoryStep.status.startedAtMs
      : Date.now()
    if (inventoryStep?.status._tag !== "Running") {
      progress = yield* startStep(progress, "inventory")
    }
    if (!(yield* installed.initialized)) return
    const installedState = (yield* installed.get).state
    progress = yield* completeStep(
      progress,
      "inventory",
      inventoryStartedAt,
      false,
      {
        completed: installedState.packages.length,
        total: installedState.packages.length,
      },
    )

    const catalogStep = progress.find(({ id }) => id === "catalog")
    const catalogStartedAt = catalogStep?.status._tag === "Running"
      ? catalogStep.status.startedAtMs
      : Date.now()
    if (catalogStep?.status._tag !== "Running") {
      progress = yield* startStep(progress, "catalog")
    }
    if (!(yield* catalog.ready)) return
    const catalogState = (yield* catalog.get).state
    progress = yield* completeStep(
      progress,
      "catalog",
      catalogStartedAt,
      false,
      { completed: catalogState.models.length, total: catalogState.models.length },
    )

    const metadataStartedAt = Date.now()
    progress = yield* startStep(progress, "metadata", {
      completed: 0,
      total: catalogState.models.length,
    })
    const catalogModels = yield* Effect.forEach(
      catalogState.models,
      recommendableModelFromIcn,
    )
    progress = yield* completeStep(
      progress,
      "metadata",
      metadataStartedAt,
      false,
      { completed: catalogModels.length, total: catalogState.models.length },
    )
    const stableCapacityBytes = hardwareSnapshot.memory_domains
      .reduce((total, domain) => total + domain.stable_capacity_bytes, 0)
    const models = catalogModels.filter((model) =>
      publishedWeightBytes(model) <= stableCapacityBytes)
    const inputEvidence = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({
      catalog: models.map((model) => ({
        id: model.id,
        targetId: model.targetId,
        checkpointId: model.checkpointId,
        eligibleServingProfiles: model.eligibleServingProfiles,
        displayName: model.displayName,
        description: model.description,
        license: model.license,
        capabilities: model.capabilities,
        qualityScore: model.qualityScore,
        qualityScoreProvenance: model.qualityScoreProvenance,
        fidelityRank: model.fidelityRank,
        quantizationAware: model.quantizationAware,
        qualityEvidence: model.qualityEvidence,
        publishedWeightBytes: publishedWeightBytes(model),
      })),
      hardware: hardwareSnapshot.topology_fingerprint,
      nativeBuild: hardwareSnapshot.native_build,
      backends: hardwareSnapshot.enabled_backends,
      platform: hardwareSnapshot.platform,
      architecture: hardwareSnapshot.architecture,
      memoryDomains: hardwareSnapshot.memory_domains.map((domain) => ({
        id: domain.id,
        stableCapacityBytes: domain.stable_capacity_bytes,
        totalCapacityBytes: domain.total_capacity_bytes,
      })),
      policy: "local-model-recommendations-v4",
    })
    const inputDigest = createHash("sha256").update(inputEvidence).digest("hex")
    const previousDigest = yield* Ref.get(lastInputDigest)
    const currentState = (yield* mirror.get).state
    const persisted = yield* Ref.get(cachedPortfolioRef)
    if (Option.exists(persisted, ({ capturedAtMs, inputDigest: digest }) =>
      digest === inputDigest
      && Date.now() - capturedAtMs <= RECOMMENDATION_PORTFOLIO_MAX_AGE_MS)) {
      const recommendations = Option.getOrThrow(persisted).recommendations
      const reusedAt = Date.now()
      progress = updateProgress(progress, "assessment", {
        status: {
          _tag: "Completed",
          startedAtMs: reusedAt,
          durationMs: 0,
          cached: true,
        },
        completedItems: Option.some(models.length),
        totalItems: Option.some(models.length),
      })
      progress = updateProgress(progress, "selection", {
        status: {
          _tag: "Completed",
          startedAtMs: reusedAt,
          durationMs: 0,
          cached: true,
        },
        completedItems: Option.some(recommendations.length),
        totalItems: Option.some(4),
      })
      yield* Ref.set(progressRef, progress)
      yield* Ref.set(lastInputDigest, Option.some(inputDigest))
      yield* mirror.setIfChanged({
        _tag: "Ready",
        recommendations,
        progress,
      }, equivalent)
      return
    }
    if (Option.exists(previousDigest, (digest) => digest === inputDigest)
      && currentState._tag === "Ready") {
      const reusedAt = Date.now()
      progress = updateProgress(progress, "assessment", {
        status: {
          _tag: "Completed",
          startedAtMs: reusedAt,
          durationMs: 0,
          cached: true,
        },
        completedItems: Option.some(models.length),
        totalItems: Option.some(models.length),
      })
      progress = updateProgress(progress, "selection", {
        status: {
          _tag: "Completed",
          startedAtMs: reusedAt,
          durationMs: 0,
          cached: true,
        },
        completedItems: Option.some(currentState.recommendations.length),
        totalItems: Option.some(4),
      })
      yield* Ref.set(progressRef, progress)
      yield* mirror.setIfChanged({ ...currentState, progress }, equivalent)
      return
    }

    const assessmentStartedAt = Date.now()
    progress = yield* startStep(progress, "assessment", {
      completed: 0,
      total: models.length,
    })
    const requests = models.map((model) => ({
      target: model.target,
      profiles: model.eligibleServingProfiles,
    }))
    const results = yield* evaluations.assessManyWithProgress(
      requests,
      (completed, total) => Effect.gen(function* () {
        progress = updateProgress(progress, "assessment", {
          completedItems: Option.some(completed),
          totalItems: Option.some(total),
        })
        yield* publishProgress(progress)
      }),
    )
    progress = yield* completeStep(
      progress,
      "assessment",
      assessmentStartedAt,
      false,
      { completed: models.length, total: models.length },
    )
    const evaluated = results.flatMap((result, modelIndex): readonly RecommendationCandidate[] => {
      if (result._tag !== "Assessed") return []
      const model = models[modelIndex]
      if (!model) return []
      return result.assessments.flatMap(
        (assessment, profileIndex): readonly RecommendationCandidate[] => {
        if (assessment._tag !== "Fits") return []
        const profile = model.eligibleServingProfiles[profileIndex]
        if (!profile) return []
        return [{
          model,
          profile,
          assessment: assessment.assessment,
          artifactId: model.id,
          checkpointId: model.checkpointId,
          capability: {
            score: model.qualityScore,
            provenance: model.qualityScoreProvenance,
          },
          fidelityRank: model.fidelityRank,
          quantizationAware: model.quantizationAware,
          estimatedRuntimeBytes: assessment.assessment.memory
            .reduce((total, domain) => total + domain.requiredBytes, 0),
          stableCapacityBudgetBytes: assessment.assessment.memory
            .reduce(
              (total, domain) =>
                total + Math.max(0, domain.capacityBytes - domain.requiredReserveBytes),
              0,
            ),
          totalDownloadBytes: model.target._tag === "Package"
            ? model.target.package.files.reduce((total, file) => total + file.sizeBytes, 0)
            : [...model.target.target.files, ...model.target.draft.files]
                .reduce((total, file) => total + file.sizeBytes, 0),
        }]
      },
      )
    })
    const selectionStartedAt = Date.now()
    progress = yield* startStep(progress, "selection")
    const selected = selectRecommendationPortfolio(evaluated)
    progress = updateProgress(progress, "selection", {
      status: {
        _tag: "Completed",
        startedAtMs: selectionStartedAt,
        durationMs: Math.max(0, Date.now() - selectionStartedAt),
        cached: false,
      },
      completedItems: Option.some(selected.length),
      totalItems: Option.some(4),
    })
    yield* Ref.set(progressRef, progress)
    yield* Ref.set(lastInputDigest, Option.some(inputDigest))
    const portfolio = {
      capturedAtMs: Date.now(),
      inputDigest,
      recommendations: selected,
    } satisfies CachedRecommendationPortfolio
    yield* Ref.set(cachedPortfolioRef, Option.some(portfolio))
    yield* writeStructuredFileAtomic(
      portfolioPath,
      CachedRecommendationPortfolioSchema,
      portfolio,
    ).pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.catchAll((error) =>
        Effect.logWarning("Unable to persist local model recommendations").pipe(
          Effect.annotateLogs({ error: String(error) }),
        )),
    )
    yield* mirror.setIfChanged({
      _tag: "Ready",
      recommendations: selected,
      progress,
    }, equivalent)
  })).pipe(
    Effect.withSpan("acn.local-model-recommendations.refresh"),
    Effect.catchAllCause((cause) => Effect.gen(function* () {
    const failure = Cause.failureOption(cause)
    const message = Option.match(failure, {
      onNone: () => "",
      onSome: (error) => error.message.trim() || String(error).trim(),
    })
    const failedAtMs = Date.now()
    const failedProgress = (yield* Ref.get(progressRef)).map((step) =>
      step.status._tag === "Running"
        ? {
            ...step,
            status: {
              _tag: "Failed" as const,
              startedAtMs: step.status.startedAtMs,
              durationMs: Math.max(0, failedAtMs - step.status.startedAtMs),
              failure: {
                code: "recommendations_unavailable",
                message: message || "This step could not be completed",
                retryable: true,
              },
            },
          }
        : step)
    yield* Ref.set(progressRef, failedProgress)
    yield* mirror.setIfChanged({
      _tag: "Failed",
      failure: {
        code: "recommendations_unavailable",
        message: message || "Local model recommendations are temporarily unavailable",
        retryable: Option.match(failure, {
          onNone: () => true,
          onSome: (error) => "retryable" in error ? error.retryable : true,
        }),
      },
      progress: failedProgress,
    }, equivalent)
    yield* Effect.logWarning("Unable to refresh local model recommendations").pipe(
      Effect.annotateLogs({ cause: String(cause) }),
    )
    })),
  )

  yield* refresh.pipe(Effect.forkScoped)
  yield* Stream.merge(
    Stream.merge(catalog.changes, hardware.changes),
    installed.changes,
  ).pipe(Stream.runForEach(() => refresh), Effect.forkScoped)

  return LocalModelRecommendations.of({
    snapshot: mirror.get,
    changes: mirror.changes,
    refresh,
    get: (id) => mirror.get.pipe(Effect.map(({ state }) => state._tag === "Ready"
      ? state.recommendations.find((recommendation) => recommendation.id === id)
      : undefined)),
  })
}))
