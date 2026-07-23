import { Cause, Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  ModelFailureSchema,
  RecommendationSchema,
  type ModelFailure,
  type Recommendation,
  type RecommendationId,
  type RecommendableModel,
} from "@magnitudedev/protocol"
import { IcnCatalog, IcnHardware } from "@magnitudedev/icn"
import { makeObservedState } from "./mirrored-state"
import { LocalModelEvaluations } from "./local-model-evaluations"
import { recommendableModelFromIcn } from "./local-model-icn-adapter"
import {
  selectRecommendationPortfolio,
  type RecommendationCandidate,
} from "./local-model-recommendation-policy"

type RecommendationState =
  | { readonly _tag: "Loading" }
  | { readonly _tag: "Ready"; readonly recommendations: readonly Recommendation[] }
  | { readonly _tag: "Failed"; readonly failure: ModelFailure }


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

export const LocalModelRecommendationsLive: Layer.Layer<
  LocalModelRecommendations,
  never,
  IcnCatalog | IcnHardware | LocalModelEvaluations
> = Layer.scoped(LocalModelRecommendations, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const hardware = yield* IcnHardware
  const evaluations = yield* LocalModelEvaluations
  const mirror = yield* makeObservedState<RecommendationState>({ _tag: "Loading" })
  const recommendationsEquivalent = Schema.equivalence(Schema.Array(RecommendationSchema))
  const failuresEquivalent = Schema.equivalence(ModelFailureSchema)
  const equivalent = (left: RecommendationState, right: RecommendationState): boolean =>
    left._tag === right._tag
    && (left._tag === "Loading"
      || (left._tag === "Ready" && right._tag === "Ready"
        && recommendationsEquivalent(left.recommendations, right.recommendations))
      || (left._tag === "Failed" && right._tag === "Failed"
        && failuresEquivalent(left.failure, right.failure)))
  const lock = yield* Effect.makeSemaphore(1)

  const refresh = lock.withPermits(1)(Effect.gen(function* () {
    if (!(yield* catalog.ready)) return
    const catalogModels = yield* Effect.forEach(
      (yield* catalog.get).state.models,
      recommendableModelFromIcn,
    )
    const stableCapacityBytes = (yield* hardware.get).state.memory_domains
      .reduce((total, domain) => total + domain.stable_capacity_bytes, 0)
    const models = catalogModels.filter((model) =>
      publishedWeightBytes(model) <= stableCapacityBytes)
    const results = yield* evaluations.assessMany(models.map((model) => ({
      target: model.target,
      profiles: model.eligibleServingProfiles,
    })))
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
    yield* mirror.setIfChanged({
      _tag: "Ready",
      recommendations: selectRecommendationPortfolio(evaluated),
    }, equivalent)
  })).pipe(Effect.catchAllCause((cause) => Effect.gen(function* () {
    const failure = Cause.failureOption(cause)
    const message = Option.match(failure, {
      onNone: () => "",
      onSome: (error) => error.message.trim() || String(error).trim(),
    })
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
    }, equivalent)
    yield* Effect.logWarning("Unable to refresh local model recommendations").pipe(
      Effect.annotateLogs({ cause: String(cause) }),
    )
  })))

  yield* refresh.pipe(Effect.forkScoped)
  yield* Stream.merge(
    catalog.changes,
    hardware.changes,
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
