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
  status: LocalModelDiscoveryProgressStep["status"],
  counts?: { readonly completed: number; readonly total: number },
): LocalModelDiscoveryProgressStep => ({
  id,
  status,
  completedItems: counts ? Option.some(counts.completed) : Option.none(),
  totalItems: counts ? Option.some(counts.total) : Option.none(),
  estimatedRemainingMs: Option.none(),
})

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
    const assessmentSnapshot = yield* assessments.snapshot
    const coordinated = assessmentSnapshot.assessments
    const installed = catalogState.entries.filter(({ source }) =>
      source.localState._tag === "Installed").length
    const assessmentFor = (configuration: ModelServingConfiguration) => coordinated.find((entry) =>
      sameConfiguration(entry.configuration, configuration))?.assessment
    const assessmentLifecycle = assessmentSnapshot.lifecycle
    const assessmentCounts = assessmentLifecycle._tag === "Discovering"
      ? undefined
      : {
          completed: assessmentLifecycle.cycle.completedTargets,
          total: assessmentLifecycle.cycle.totalTargets,
        }
    const assessmentStatus: LocalModelDiscoveryProgressStep["status"] =
      assessmentLifecycle._tag === "Assessing"
        ? { _tag: "Running", startedAtMs: assessmentLifecycle.cycle.startedAtMs }
        : assessmentLifecycle._tag === "Ready"
          ? {
              _tag: "Completed",
              startedAtMs: assessmentLifecycle.cycle.startedAtMs,
              durationMs: assessmentLifecycle.cycle.durationMs,
              cached: false,
            }
          : assessmentLifecycle._tag === "Failed"
            ? {
                _tag: "Failed",
                startedAtMs: assessmentLifecycle.cycle.startedAtMs,
                durationMs: assessmentLifecycle.cycle.durationMs,
                failure: assessmentLifecycle.cycle.failure,
              }
            : { _tag: "Pending" }
    const baseProgress = [
      progressStep("hardware", {
        _tag: "Completed",
        startedAtMs,
        durationMs: 0,
        cached: true,
      }, {
        completed: hardwareState.memory_domains.length,
        total: hardwareState.memory_domains.length,
      }),
      progressStep("inventory", catalogState.reconciliationComplete
        ? { _tag: "Completed", startedAtMs, durationMs: 0, cached: true }
        : { _tag: "Running", startedAtMs }, {
        completed: installed,
        total: installed,
      }),
      progressStep("assessment", assessmentStatus, assessmentCounts),
    ]
    const failed = candidates.flatMap(({ configuration }) => {
      const result = assessmentFor(configuration)
      return result?._tag === "Failed" ? [result.failure] : []
    })[0] ?? (assessmentLifecycle._tag === "Failed" ? assessmentLifecycle.cycle.failure : undefined)
    if (failed !== undefined) {
      return {
        _tag: "Failed" as const,
        failure: failed,
        progress: [
          ...baseProgress.slice(0, 2),
          progressStep("assessment", {
            _tag: "Failed",
            startedAtMs: assessmentLifecycle._tag === "Discovering"
              ? startedAtMs
              : assessmentLifecycle.cycle.startedAtMs,
            durationMs: assessmentLifecycle._tag === "Ready"
              || assessmentLifecycle._tag === "Failed"
              ? assessmentLifecycle.cycle.durationMs
              : 0,
            failure: failed,
          }, assessmentCounts),
          progressStep("ranking", { _tag: "Pending" }),
        ],
      }
    }
    if (!catalogState.reconciliationComplete
      || assessmentLifecycle._tag === "Discovering"
      || assessmentLifecycle._tag === "Assessing") {
      return {
        _tag: "Loading" as const,
        progress: [...baseProgress, progressStep("ranking", { _tag: "Pending" })],
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
      progress: [...baseProgress, progressStep("ranking", {
        _tag: "Completed",
        startedAtMs,
        durationMs: 0,
        cached: true,
      }, {
        completed: entries.length,
        total: candidates.length,
      })],
    }
  }).pipe(
    Effect.catchAllCause((cause) => Effect.succeed({
      _tag: "Failed" as const,
      failure: localModelRankingFailure(Option.getOrUndefined(Cause.failureOption(cause))),
      progress: [
        progressStep("hardware", { _tag: "Pending" }),
        progressStep("inventory", { _tag: "Pending" }),
        progressStep("assessment", { _tag: "Pending" }),
        progressStep("ranking", { _tag: "Pending" }),
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
