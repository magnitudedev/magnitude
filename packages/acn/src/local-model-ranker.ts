import {
  Cause,
  Context,
  Effect,
  Layer,
  Option,
  Schema,
  Stream,
} from "effect"
import {
  LocalModelDiscoveryProgressStepSchema,
  LocalModelMutationFailed,
  LocalModelRankingScoresSchema,
  ModelFailureSchema,
  ModelServingConfigurationSchema,
  type LocalModelDiscoveryProgressStep,
  type LocalModelDiscoveryProgressStepId,
  type LocalModelRankingScores,
  type ModelFailure,
  type ModelServingConfiguration,
  type RecommendableModel,
  type ServableModelBundle,
  servableModelBundlePackages,
} from "@magnitudedev/acn-protocol"
import { ProviderModelIdSchema, type ProviderModelId } from "@magnitudedev/ai"
import { IcnHardware } from "@magnitudedev/icn"
import { LocalModelAssessor } from "./local-model-assessor"
import {
  LocalModelCatalogAdapter,
  type AdaptedLocalModelCatalogEntry,
} from "./local-model-catalog-adapter"
import { localCatalogProviderModelId } from "./local-provider-model-id"
import { modelRankingScores } from "./local-model-ranking-policy"
import { materializeProjection } from "./materialized-projection"

const LocalModelRankingEntrySchema = Schema.Struct({
  modelId: ProviderModelIdSchema,
  configuration: ModelServingConfigurationSchema,
  scores: LocalModelRankingScoresSchema,
})
export type LocalModelRankingEntry = typeof LocalModelRankingEntrySchema.Type

const RankingStateSchema = Schema.Union(
  Schema.TaggedStruct("Loading", {
    progress: Schema.Array(LocalModelDiscoveryProgressStepSchema),
  }),
  Schema.TaggedStruct("Ready", {
    entries: Schema.Array(LocalModelRankingEntrySchema),
    progress: Schema.Array(LocalModelDiscoveryProgressStepSchema),
  }),
  Schema.TaggedStruct("Failed", {
    failure: ModelFailureSchema,
    progress: Schema.Array(LocalModelDiscoveryProgressStepSchema),
  }),
)
export type RankingState = typeof RankingStateSchema.Type

export const localModelRankingFailure = (
  error: { readonly message: string; readonly retryable?: boolean } | undefined,
): ModelFailure => error instanceof LocalModelMutationFailed
  ? { code: error.code, message: error.message, retryable: error.retryable }
  : {
      code: "model_ranking_unavailable",
      message: error?.message.trim() || "Local model ranking is temporarily unavailable",
      retryable: error?.retryable ?? true,
    }

export interface LocalModelRankerApi {
  readonly state: Effect.Effect<RankingState>
  readonly changes: Stream.Stream<void>
}

export class LocalModelRanker extends Context.Tag("LocalModelRanker")<
  LocalModelRanker,
  LocalModelRankerApi
>() {}

export const exactBundleTensorStorageBytes = (
  model: RecommendableModel,
): Option.Option<number> => exactTensorStorageBytes(model.configuration.bundle)

const exactTensorStorageBytes = (bundle: ServableModelBundle): Option.Option<number> => {
  const files = new Map(
    servableModelBundlePackages(bundle)
      .flatMap(({ files }) => files)
      .filter((file) => file.role === "weights")
      .map((file) => [file.sha256, file]),
  )
  if (files.size === 0) return Option.none()
  let total = 0
  for (const file of files.values()) {
    if (Option.isNone(file.tensorStorageBytes)) return Option.none()
    total += file.tensorStorageBytes.value
    if (!Number.isSafeInteger(total)) return Option.none()
  }
  return Option.some(total)
}

export const localModelRankingCandidates = (
  entries: readonly AdaptedLocalModelCatalogEntry[],
): readonly RecommendableModel[] => {
  const sameConfiguration = Schema.equivalence(ModelServingConfigurationSchema)
  return entries.flatMap(({ model, effectiveConfiguration }) => Option.isSome(effectiveConfiguration)
    && !sameConfiguration(model.configuration, effectiveConfiguration.value)
    ? [model, { ...model, configuration: effectiveConfiguration.value }]
    : [model])
}

const progressStep = (
  id: LocalModelDiscoveryProgressStepId,
  complete: boolean,
  startedAtMs: number,
  counts?: { readonly completed: number; readonly total: number },
): LocalModelDiscoveryProgressStep => complete
  ? {
      id,
      status: { _tag: "Completed", startedAtMs, durationMs: 0, cached: true },
      completedItems: counts ? Option.some(counts.completed) : Option.none(),
      totalItems: counts ? Option.some(counts.total) : Option.none(),
      estimatedRemainingMs: Option.none(),
    }
  : {
      id,
      status: { _tag: "Running", startedAtMs },
      completedItems: counts ? Option.some(counts.completed) : Option.none(),
      totalItems: counts ? Option.some(counts.total) : Option.none(),
      estimatedRemainingMs: Option.none(),
    }

export const makeLocalModelRankerLive = (): Layer.Layer<
  LocalModelRanker,
  never,
  LocalModelCatalogAdapter | IcnHardware | LocalModelAssessor
> => Layer.scoped(LocalModelRanker, Effect.gen(function* () {
  const catalogAdapter = yield* LocalModelCatalogAdapter
  const hardware = yield* IcnHardware
  const assessments = yield* LocalModelAssessor
  const startedAtMs = Date.now()
  const sameConfiguration = Schema.equivalence(ModelServingConfigurationSchema)

  const derive = Effect.gen(function* () {
    const catalogState = yield* catalogAdapter.state
    const catalog = catalogState.entries.map((entry) => entry.model)
    const candidates = localModelRankingCandidates(catalogState.entries)
    const hardwareState = (yield* hardware.get).state
    const coordinated = yield* assessments.state
    const installed = catalogState.entries.filter(({ source }) =>
      source.localState._tag === "Installed").length
    const aggregateCapacity = hardwareState.memory_domains.reduce(
      (total, domain) => total + domain.stable_capacity_bytes,
      0,
    )
    const assessable = catalog.filter((model) => {
      const bytes = exactBundleTensorStorageBytes(model)
      return Option.isNone(bytes) || bytes.value <= aggregateCapacity
    })
    const assessmentFor = (configuration: ModelServingConfiguration) => coordinated.find((entry) =>
      sameConfiguration(entry.configuration, configuration))?.assessment
    const completed = assessable.filter(({ configuration }) => {
      const result = assessmentFor(configuration)
      return result !== undefined && result._tag !== "Assessing"
    })
    const rejected = catalog.length - assessable.length
    const assessmentComplete = completed.length === assessable.length
    const baseProgress = [
      progressStep("hardware", true, startedAtMs, {
        completed: hardwareState.memory_domains.length,
        total: hardwareState.memory_domains.length,
      }),
      progressStep("inventory", catalogState.reconciliationComplete, startedAtMs, {
        completed: installed,
        total: installed,
      }),
      progressStep("assessment", assessmentComplete, startedAtMs, {
        completed: rejected + completed.length,
        total: catalog.length,
      }),
    ]
    const failed = assessable.flatMap(({ configuration }) => {
      const result = assessmentFor(configuration)
      return result?._tag === "Failed" ? [result.failure] : []
    })[0]
    if (failed !== undefined) {
      return {
        _tag: "Failed" as const,
        failure: failed,
        progress: [...baseProgress, progressStep("ranking", false, startedAtMs)],
      }
    }
    if (!catalogState.reconciliationComplete || !assessmentComplete) {
      return {
        _tag: "Loading" as const,
        progress: [...baseProgress, progressStep("ranking", false, startedAtMs)],
      }
    }
    const entries = (yield* Effect.forEach(candidates, (model) => {
      const result = assessmentFor(model.configuration)
      if (result?._tag !== "Fits") return Effect.succeed(Option.none<LocalModelRankingEntry>())
      return modelRankingScores({
        model,
        profile: result.assessment.profile,
        assessment: result.assessment,
      }).pipe(Effect.map((scores) => Option.some<LocalModelRankingEntry>({
        modelId: localCatalogProviderModelId(model),
        configuration: model.configuration,
        scores,
      })))
    })).flatMap((entry) => Option.isSome(entry) ? [entry.value] : [])
      .sort((left, right) => left.modelId.localeCompare(right.modelId))
    return {
      _tag: "Ready" as const,
      entries,
      progress: [...baseProgress, progressStep("ranking", true, startedAtMs, {
        completed: entries.length,
        total: candidates.length,
      })],
    }
  }).pipe(
    Effect.catchAllCause((cause) => Effect.succeed({
      _tag: "Failed" as const,
      failure: localModelRankingFailure(Option.getOrUndefined(Cause.failureOption(cause))),
      progress: [
        progressStep("hardware", false, startedAtMs),
        progressStep("inventory", false, startedAtMs),
        progressStep("assessment", false, startedAtMs),
        progressStep("ranking", false, startedAtMs),
      ],
    })),
  )

  const sourceChanges = Stream.mergeAll([
    catalogAdapter.changes.pipe(Stream.map(() => undefined)),
    hardware.assessmentChanges.pipe(Stream.map(() => undefined)),
    assessments.changes.pipe(Stream.map(() => undefined)),
  ], { concurrency: "unbounded" })
  const projection = yield* materializeProjection({
    project: derive,
    invalidations: sourceChanges,
    equivalent: Schema.equivalence(RankingStateSchema),
  })

  return LocalModelRanker.of({
    state: projection.get,
    changes: projection.changes.pipe(Stream.map(() => undefined)),
  })
}))
