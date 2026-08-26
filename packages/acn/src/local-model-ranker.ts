import {
  Cause,
  Context,
  Effect,
  Layer,
  Option,
  Ref,
  Schema,
  Stream,
  SubscriptionRef,
} from "effect"
import { createHash } from "node:crypto"
import {
  LocalModelMutationFailed,
  LocalModelDiscoveryProgressStepSchema,
  ModelFailureSchema,
  ModelServingConfigurationSchema,
  servableModelBundlePackages,
  type LocalModelDiscoveryProgressStep,
  type LocalModelDiscoveryProgressStepId,
  type LocalModelRankingScores,
  type ServableModelBundle,
  type ModelFailure,
  type ModelServingConfiguration,
  type RecommendableModel,
} from "@magnitudedev/acn-protocol"
import type { ProviderModelId } from "@magnitudedev/ai"
import { IcnHardware, IcnModels } from "@magnitudedev/icn"
import { LocalModelAssessor } from "./local-model-assessor"
import { LocalModelPackages } from "./local-model-packages"
import { catalogModelDefinitionFromIcn } from "./local-model-icn-adapter"
import { localCatalogProviderModelId } from "./local-provider-model-id"
import { modelRankingScores } from "./local-model-ranking-policy"

export interface LocalModelRankingEntry {
  readonly modelId: ProviderModelId
  readonly configuration: ModelServingConfiguration
  readonly scores: LocalModelRankingScores
}

type RankingState =
  | {
      readonly _tag: "Loading"
      readonly progress: readonly LocalModelDiscoveryProgressStep[]
    }
  | {
      readonly _tag: "Ready"
      readonly entries: readonly LocalModelRankingEntry[]
      readonly progress: readonly LocalModelDiscoveryProgressStep[]
    }
  | {
      readonly _tag: "Failed"
      readonly failure: ModelFailure
      readonly progress: readonly LocalModelDiscoveryProgressStep[]
    }

export const localModelRankingFailure = (
  error: {
    readonly message: string
    readonly retryable?: boolean
  } | undefined,
): ModelFailure =>
  error instanceof LocalModelMutationFailed
    ? {
        code: error.code,
        message: error.message,
        retryable: error.retryable,
      }
    : {
        code: "model_ranking_unavailable",
        message:
          error?.message.trim() ||
          "Local model ranking is temporarily unavailable",
        retryable: error?.retryable ?? true,
      }

export interface LocalModelRankerApi {
  readonly state: Effect.Effect<RankingState>
  readonly changes: Stream.Stream<RankingState>
}

export class LocalModelRanker extends Context.Tag(
  "LocalModelRanker"
)<LocalModelRanker, LocalModelRankerApi>() {}

export const exactBundleTensorStorageBytes = (
  model: RecommendableModel
): Option.Option<number> => exactTensorStorageBytes(model.configuration.bundle)

const exactTensorStorageBytes = (
  bundle: ServableModelBundle,
): Option.Option<number> => {
  const packages = servableModelBundlePackages(bundle)
  const files = new Map(
    packages
      .flatMap(({ files }) => files)
      // Primary/sharded weight tensors are required for every execution of the bundle. Other
      // package roles can be optional, so counting them could create a false rejection. Native
      // assessment accounts for every selected component precisely.
      .filter((file) => file.role === "weights")
      .map((file) => [file.sha256, file])
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

const pendingProgress = (
  id: LocalModelDiscoveryProgressStepId
): LocalModelDiscoveryProgressStep => ({
  id,
  status: { _tag: "Pending" },
  completedItems: Option.none(),
  totalItems: Option.none(),
  estimatedRemainingMs: Option.none(),
})

const initialProgress = (): readonly LocalModelDiscoveryProgressStep[] => [
  pendingProgress("hardware"),
  pendingProgress("inventory"),
  pendingProgress("assessment"),
  pendingProgress("ranking"),
]

const updateProgress = (
  progress: readonly LocalModelDiscoveryProgressStep[],
  id: LocalModelDiscoveryProgressStepId,
  update: Partial<LocalModelDiscoveryProgressStep>
): readonly LocalModelDiscoveryProgressStep[] =>
  progress.map((step) => (step.id === id ? { ...step, ...update } : step))

export const makeLocalModelRankerLive = (): Layer.Layer<
  LocalModelRanker,
  never,
  IcnModels | IcnHardware | LocalModelAssessor | LocalModelPackages
> =>
  Layer.scoped(
    LocalModelRanker,
    Effect.gen(function* () {
      const models = yield* IcnModels
      const hardware = yield* IcnHardware
      const assessments = yield* LocalModelAssessor
      const packages = yield* LocalModelPackages
      const startupStartedAtMs = Date.now()
      const startupProgress = updateProgress(initialProgress(), "hardware", {
        status: { _tag: "Running", startedAtMs: startupStartedAtMs },
      })
      const current = yield* SubscriptionRef.make<RankingState>({
        _tag: "Loading",
        progress: startupProgress,
      })
      const progressRef = yield* Ref.make(startupProgress)
      const lastInputDigest = yield* Ref.make<Option.Option<string>>(Option.none())
      const entriesEquivalent = (
        left: readonly LocalModelRankingEntry[],
        right: readonly LocalModelRankingEntry[],
      ): boolean => JSON.stringify(left) === JSON.stringify(right)
      const failuresEquivalent = Schema.equivalence(ModelFailureSchema)
      const progressEquivalent = Schema.equivalence(
        Schema.Array(LocalModelDiscoveryProgressStepSchema)
      )
      const equivalent = (
        left: RankingState,
        right: RankingState
      ): boolean =>
        left._tag === right._tag &&
        progressEquivalent(left.progress, right.progress) &&
        (left._tag === "Loading" ||
          (left._tag === "Ready" &&
            right._tag === "Ready" &&
            entriesEquivalent(left.entries, right.entries)) ||
          (left._tag === "Failed" &&
            right._tag === "Failed" &&
            failuresEquivalent(left.failure, right.failure)))
      const lock = yield* Effect.makeSemaphore(1)
      const publish = (next: RankingState) => Effect.gen(function* () {
        const previous = yield* SubscriptionRef.get(current)
        if (!equivalent(previous, next)) yield* SubscriptionRef.set(current, next)
      })

      const publishProgress = (
        progress: readonly LocalModelDiscoveryProgressStep[]
      ): Effect.Effect<void> =>
        Effect.gen(function* () {
          yield* Ref.set(progressRef, progress)
          const state = yield* SubscriptionRef.get(current)
          yield* publish(
            state._tag === "Ready"
              ? { ...state, progress }
              : state._tag === "Failed"
              ? { _tag: "Loading", progress }
              : { ...state, progress },
          )
        })

      const startStep = (
        progress: readonly LocalModelDiscoveryProgressStep[],
        id: LocalModelDiscoveryProgressStepId,
        counts?: { readonly completed: number; readonly total: number }
      ) => {
        const next = updateProgress(progress, id, {
          status: { _tag: "Running", startedAtMs: Date.now() },
          completedItems: counts
            ? Option.some(counts.completed)
            : Option.none(),
          totalItems: counts ? Option.some(counts.total) : Option.none(),
          estimatedRemainingMs: Option.none(),
        })
        return publishProgress(next).pipe(Effect.as(next))
      }

      const completeStep = (
        progress: readonly LocalModelDiscoveryProgressStep[],
        id: LocalModelDiscoveryProgressStepId,
        startedAtMs: number,
        cached: boolean,
        counts?: { readonly completed: number; readonly total: number }
      ) => {
        const next = updateProgress(progress, id, {
          status: {
            _tag: "Completed",
            startedAtMs,
            durationMs: Math.max(0, Date.now() - startedAtMs),
            cached,
          },
          completedItems: counts
            ? Option.some(counts.completed)
            : Option.none(),
          totalItems: counts ? Option.some(counts.total) : Option.none(),
          estimatedRemainingMs: Option.none(),
        })
        return publishProgress(next).pipe(Effect.as(next))
      }

      const generate = lock
        .withPermits(1)(
          Effect.gen(function* () {
            const currentStateBeforeRefresh = yield* SubscriptionRef.get(current)
            const hardwareSnapshot = (yield* hardware.get).state
            if (!(yield* packages.initialized) || !(yield* models.initialized)) return
            const packageState = yield* packages.state
            const modelsState = (yield* models.get).state
            const catalogModels = yield* Effect.forEach(
              modelsState.models,
              catalogModelDefinitionFromIcn,
            )
            const coordinated = yield* assessments.state
            const encodedInput = yield* Schema.encode(
              Schema.parseJson(Schema.Unknown),
            )({
              catalog: catalogModels.map((model) => ({
                modelId: model.modelId,
                variantId: model.variantId,
                configuration: model.configuration,
                intelligenceIndexScore: model.intelligence.score,
                fidelityRank: model.fidelityRank,
                tensorStorageBytes: Option.getOrNull(exactBundleTensorStorageBytes(model)),
              })),
              assessments: coordinated,
              hardware: {
                topology: hardwareSnapshot.topology_fingerprint,
                nativeBuild: hardwareSnapshot.native_build,
                backends: hardwareSnapshot.enabled_backends,
                memoryDomains: hardwareSnapshot.memory_domains.map((domain) => ({
                  id: domain.id,
                  stableCapacityBytes: domain.stable_capacity_bytes,
                })),
              },
            })
            const inputDigest = createHash("sha256").update(encodedInput).digest("hex")
            const previousDigest = yield* Ref.get(lastInputDigest)
            if (currentStateBeforeRefresh._tag === "Ready"
              && Option.contains(previousDigest, inputDigest)) return

            let progress =
              currentStateBeforeRefresh._tag === "Loading"
                ? yield* Ref.get(progressRef)
                : initialProgress()
            const hardwareStep = progress.find(({ id }) => id === "hardware")
            const hardwareStartedAt =
              hardwareStep?.status._tag === "Running"
                ? hardwareStep.status.startedAtMs
                : Date.now()
            if (hardwareStep?.status._tag !== "Running") {
              progress = yield* startStep(progress, "hardware")
            }
            progress = yield* completeStep(
              progress,
              "hardware",
              hardwareStartedAt,
              false,
              {
                completed: hardwareSnapshot.memory_domains.length,
                total: hardwareSnapshot.memory_domains.length,
              }
            )

            const inventoryStep = progress.find(({ id }) => id === "inventory")
            const inventoryStartedAt =
              inventoryStep?.status._tag === "Running"
                ? inventoryStep.status.startedAtMs
                : Date.now()
            if (inventoryStep?.status._tag !== "Running") {
              progress = yield* startStep(progress, "inventory")
            }
            const installedCount = packageState.entries.filter(({ localState }) =>
              localState._tag === "Installed").length
            progress = yield* completeStep(
              progress,
              "inventory",
              inventoryStartedAt,
              false,
              {
                completed: installedCount,
                total: installedCount,
              }
            )

            const assessmentConfigurations = catalogModels
            const aggregateStableCapacityBytes = hardwareSnapshot.memory_domains.reduce(
              (total, domain) => total + domain.stable_capacity_bytes,
              0
            )
            const assessableConfigurations = assessmentConfigurations.filter(({ configuration }) => {
              const tensorStorageBytes = exactTensorStorageBytes(configuration.bundle)
              return Option.isSome(tensorStorageBytes)
                && tensorStorageBytes.value > aggregateStableCapacityBytes
                ? false
                : true
            })
            const rejectedCount = assessmentConfigurations.length - assessableConfigurations.length
            const assessmentStartedAt = Date.now()
            progress = yield* startStep(progress, "assessment", {
              completed: rejectedCount,
              total: assessmentConfigurations.length,
            })
            yield* publishProgress(progress)
            const sameConfiguration = Schema.equivalence(ModelServingConfigurationSchema)
            const assessmentFor = (configuration: ModelServingConfiguration) =>
              coordinated.find((entry) =>
                sameConfiguration(entry.configuration, configuration))?.assessment
            const failedAssessment = assessableConfigurations.flatMap(({ configuration }) => {
              const assessment = assessmentFor(configuration)
              return assessment?._tag === "Failed" ? [assessment.failure] : []
            })[0]
            if (failedAssessment !== undefined) {
              return yield* new LocalModelMutationFailed(failedAssessment)
            }
            const completed = assessableConfigurations.filter(({ configuration }) => {
              const assessment = assessmentFor(configuration)
              return assessment !== undefined
                && assessment._tag !== "Assessing"
            })
            progress = updateProgress(progress, "assessment", {
              completedItems: Option.some(rejectedCount + completed.length),
              totalItems: Option.some(assessmentConfigurations.length),
              estimatedRemainingMs: Option.none(),
            })
            yield* publishProgress(progress)
            if (completed.length !== assessableConfigurations.length) return
            progress = yield* completeStep(
              progress,
              "assessment",
              assessmentStartedAt,
              false,
              { completed: assessmentConfigurations.length, total: assessmentConfigurations.length }
            )
            const rankingStartedAt = Date.now()
            progress = yield* startStep(progress, "ranking")
            const entries = yield* Effect.forEach(catalogModels, (model) => {
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
            }).pipe(Effect.map((results) => results.flatMap((entry) =>
              Option.isSome(entry) ? [entry.value] : [])))
            entries.sort((left, right) => left.modelId.localeCompare(right.modelId))
            progress = updateProgress(progress, "ranking", {
              status: {
                _tag: "Completed",
                startedAtMs: rankingStartedAt,
                durationMs: Math.max(0, Date.now() - rankingStartedAt),
                cached: false,
              },
              completedItems: Option.some(entries.length),
              totalItems: Option.some(assessmentConfigurations.length),
              estimatedRemainingMs: Option.none(),
            })
            yield* Ref.set(progressRef, progress)
            yield* Ref.set(lastInputDigest, Option.some(inputDigest))
            yield* publish(
              {
                _tag: "Ready",
                entries,
                progress,
              },
            )
          })
        )
        .pipe(
          Effect.withSpan("acn.local-model-ranker.generate"),
          Effect.catchAllCause((cause) =>
            Effect.gen(function* () {
              const failure = Cause.failureOption(cause)
              const reportedFailure = localModelRankingFailure(
                Option.getOrUndefined(failure)
              )
              const failedAtMs = Date.now()
              const failedProgress = (yield* Ref.get(progressRef)).map((step) =>
                step.status._tag === "Running"
                  ? {
                      ...step,
                      estimatedRemainingMs: Option.none(),
                      status: {
                        _tag: "Failed" as const,
                        startedAtMs: step.status.startedAtMs,
                        durationMs: Math.max(
                          0,
                          failedAtMs - step.status.startedAtMs
                        ),
                        failure: {
                          ...reportedFailure,
                          message:
                            reportedFailure.message ||
                            "This step could not be completed",
                        },
                      },
                    }
                  : step
              )
              yield* Ref.set(progressRef, failedProgress)
              yield* publish(
                {
                  _tag: "Failed",
                  failure: reportedFailure,
                  progress: failedProgress,
                },
              )
              yield* Effect.logWarning(
                "Unable to prepare local model ranking scores"
              ).pipe(Effect.annotateLogs({ cause: String(cause) }))
            })
          )
        )

      yield* generate.pipe(Effect.forkScoped)
      yield* Stream.mergeAll([
        models.changes.pipe(Stream.map(() => undefined)),
        hardware.assessmentChanges.pipe(Stream.map(() => undefined)),
        packages.changes.pipe(Stream.map(() => undefined)),
        assessments.changes.pipe(Stream.map(() => undefined)),
      ], { concurrency: "unbounded" }).pipe(
        Stream.runForEach(() => generate),
        Effect.forkScoped
      )

      return LocalModelRanker.of({
        state: SubscriptionRef.get(current),
        changes: current.changes,
      })
    })
  )
