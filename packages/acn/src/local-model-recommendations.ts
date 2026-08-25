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
  ModelFailureSchema,
  ModelServingConfigurationSchema,
  LocalModelRecommendationProgressStepSchema,
  servableModelBundlePackages,
  servableModelBundleTargetPackageId,
  type LocalModelRecommendationProgressStep,
  type LocalModelRecommendationProgressStepId,
  type ServableModelBundle,
  type ModelFailure,
  type ModelServingConfiguration,
  type RecommendableModel,
} from "@magnitudedev/acn-protocol"
import { IcnHardware, IcnModels } from "@magnitudedev/icn"
import { LocalModelAssessor } from "./local-model-assessor"
import { LocalModelPackages } from "./local-model-packages"
import { catalogModelDefinitionFromIcn } from "./local-model-icn-adapter"
import {
  assembleRecommendationCatalogCandidates,
  selectRecommendationPortfolio,
  type RecommendationCandidate,
  type RecommendationSelection,
} from "./local-model-recommendation-policy"

type RecommendationState =
  | {
      readonly _tag: "Loading"
      readonly progress: readonly LocalModelRecommendationProgressStep[]
    }
  | {
      readonly _tag: "Ready"
      readonly recommendations: readonly RecommendationSelection[]
      readonly catalog: readonly RecommendationCandidate[]
      readonly progress: readonly LocalModelRecommendationProgressStep[]
    }
  | {
      readonly _tag: "Failed"
      readonly failure: ModelFailure
      readonly progress: readonly LocalModelRecommendationProgressStep[]
    }

export const localModelRecommendationFailure = (
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
        code: "recommendations_unavailable",
        message:
          error?.message.trim() ||
          "Local model recommendations are temporarily unavailable",
        retryable: error?.retryable ?? true,
      }

export interface LocalModelRecommendationsApi {
  readonly state: Effect.Effect<RecommendationState>
  readonly changes: Stream.Stream<RecommendationState>
}

export class LocalModelRecommendations extends Context.Tag(
  "LocalModelRecommendations"
)<LocalModelRecommendations, LocalModelRecommendationsApi>() {}

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
  id: LocalModelRecommendationProgressStepId
): LocalModelRecommendationProgressStep => ({
  id,
  status: { _tag: "Pending" },
  completedItems: Option.none(),
  totalItems: Option.none(),
  estimatedRemainingMs: Option.none(),
})

const initialProgress = (): readonly LocalModelRecommendationProgressStep[] => [
  pendingProgress("hardware"),
  pendingProgress("inventory"),
  pendingProgress("assessment"),
  pendingProgress("recommendations"),
]

const updateProgress = (
  progress: readonly LocalModelRecommendationProgressStep[],
  id: LocalModelRecommendationProgressStepId,
  update: Partial<LocalModelRecommendationProgressStep>
): readonly LocalModelRecommendationProgressStep[] =>
  progress.map((step) => (step.id === id ? { ...step, ...update } : step))

export const makeLocalModelRecommendationsLive = (): Layer.Layer<
  LocalModelRecommendations,
  never,
  IcnModels | IcnHardware | LocalModelAssessor | LocalModelPackages
> =>
  Layer.scoped(
    LocalModelRecommendations,
    Effect.gen(function* () {
      const models = yield* IcnModels
      const hardware = yield* IcnHardware
      const assessments = yield* LocalModelAssessor
      const packages = yield* LocalModelPackages
      const startupStartedAtMs = Date.now()
      const startupProgress = updateProgress(initialProgress(), "hardware", {
        status: { _tag: "Running", startedAtMs: startupStartedAtMs },
      })
      const current = yield* SubscriptionRef.make<RecommendationState>({
        _tag: "Loading",
        progress: startupProgress,
      })
      const progressRef = yield* Ref.make(startupProgress)
      const lastInputDigest = yield* Ref.make<Option.Option<string>>(
        Option.none()
      )
      const recommendationsEquivalent = (
        left: readonly RecommendationSelection[],
        right: readonly RecommendationSelection[],
      ): boolean => left.length === right.length && left.every((entry, index) => {
        const other = right[index]
          return other !== undefined
          && entry.id === other.id
          && entry.modelId === other.modelId
          && entry.displayName === other.displayName
          && entry.intent === other.intent
          && entry.explanation === other.explanation
      })
      const catalogEquivalent = (
        left: readonly RecommendationCandidate[],
        right: readonly RecommendationCandidate[],
      ): boolean => JSON.stringify(left) === JSON.stringify(right)
      const failuresEquivalent = Schema.equivalence(ModelFailureSchema)
      const progressEquivalent = Schema.equivalence(
        Schema.Array(LocalModelRecommendationProgressStepSchema)
      )
      const equivalent = (
        left: RecommendationState,
        right: RecommendationState
      ): boolean =>
        left._tag === right._tag &&
        progressEquivalent(left.progress, right.progress) &&
        (left._tag === "Loading" ||
          (left._tag === "Ready" &&
            right._tag === "Ready" &&
            recommendationsEquivalent(
              left.recommendations,
              right.recommendations
            ) &&
            catalogEquivalent(left.catalog, right.catalog)) ||
          (left._tag === "Failed" &&
            right._tag === "Failed" &&
            failuresEquivalent(left.failure, right.failure)))
      const lock = yield* Effect.makeSemaphore(1)
      const publish = (next: RecommendationState) => Effect.gen(function* () {
        const previous = yield* SubscriptionRef.get(current)
        if (!equivalent(previous, next)) yield* SubscriptionRef.set(current, next)
      })

      const publishProgress = (
        progress: readonly LocalModelRecommendationProgressStep[]
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
        progress: readonly LocalModelRecommendationProgressStep[],
        id: LocalModelRecommendationProgressStepId,
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
        progress: readonly LocalModelRecommendationProgressStep[],
        id: LocalModelRecommendationProgressStepId,
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
            const hardwareSnapshot = (yield* hardware.get).state
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
            if (!(yield* packages.initialized)) return
            const packageState = yield* packages.state
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

            if (!(yield* models.initialized)) return
            const modelsState = (yield* models.get).state
            const catalogModels = yield* Effect.forEach(
              modelsState.models,
              catalogModelDefinitionFromIcn
            )
            const assessmentConfigurations = catalogModels
            const inputState = yield* Schema.encode(
              Schema.parseJson(Schema.Unknown)
            )({
              catalog: catalogModels.map((model) => ({
                modelId: model.modelId,
                variantId: model.variantId,
                catalogModelId: model.modelId,
                configuration: model.configuration,
                displayName: model.displayName,
                variantLabel: model.variantLabel,
                description: model.description,
                license: model.license,
                capabilities: model.capabilities,
                qualityScore: model.qualityScore,
                qualityScoreProvenance: model.qualityScoreProvenance,
                fidelityRank: model.fidelityRank,
                quantizationAware: model.quantizationAware,
                qualityEvidence: model.qualityEvidence,
                tensorStorageBytes: Option.getOrNull(
                  exactBundleTensorStorageBytes(model)
                ),
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
            })
            const inputDigest = createHash("sha256")
              .update(inputState)
              .digest("hex")
            const previousDigest = yield* Ref.get(lastInputDigest)
            const currentState = yield* SubscriptionRef.get(current)
            if (
              Option.exists(
                previousDigest,
                (digest) => digest === inputDigest
              ) &&
              currentState._tag === "Ready"
            ) {
              const reusedAt = Date.now()
              progress = updateProgress(progress, "assessment", {
                status: {
                  _tag: "Completed",
                  startedAtMs: reusedAt,
                  durationMs: 0,
                  cached: true,
                },
                completedItems: Option.some(assessmentConfigurations.length),
                totalItems: Option.some(assessmentConfigurations.length),
                estimatedRemainingMs: Option.none(),
              })
              progress = updateProgress(progress, "recommendations", {
                status: {
                  _tag: "Completed",
                  startedAtMs: reusedAt,
                  durationMs: 0,
                  cached: true,
                },
                completedItems: Option.some(
                  currentState.recommendations.length
                ),
                totalItems: Option.some(4),
                estimatedRemainingMs: Option.none(),
              })
              yield* Ref.set(progressRef, progress)
              yield* publish({ ...currentState, progress })
              return
            }

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
            const coordinated = yield* assessments.state
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
            const evaluated = catalogModels.flatMap(
              (model): readonly RecommendationCandidate[] => {
                const result = assessmentFor(model.configuration)
                if (result?._tag !== "Fits") return []
                const profile = result.assessment.profile
                return [{
                        model,
                        profile,
                        assessment: result.assessment,
                        artifactId: servableModelBundleTargetPackageId(model.configuration.bundle),
                        catalogModelId: model.modelId,
                        capabilityScore: model.qualityScore,
                        fidelityRank: model.fidelityRank,
                        quantizationAware: model.quantizationAware,
                        estimatedLoadedBytes:
                          result.assessment.memory.reduce(
                            (total, domain) => total + domain.requiredBytes,
                            0
                          ),
                        stableCapacityBudgetBytes:
                          result.assessment.memory.reduce(
                            (total, domain) =>
                              total +
                              Math.max(
                                0,
                                domain.capacityBytes -
                                  domain.compatibilityReserveBytes
                              ),
                            0
                          ),
                      }]
              }
            )
            const selectionStartedAt = Date.now()
            progress = yield* startStep(progress, "recommendations")
            const selected = selectRecommendationPortfolio(evaluated)
            const catalogCandidates = assembleRecommendationCatalogCandidates(
              evaluated,
              selected
            )
            progress = updateProgress(progress, "recommendations", {
              status: {
                _tag: "Completed",
                startedAtMs: selectionStartedAt,
                durationMs: Math.max(0, Date.now() - selectionStartedAt),
                cached: false,
              },
              completedItems: Option.some(selected.length),
              totalItems: Option.some(4),
              estimatedRemainingMs: Option.none(),
            })
            yield* Ref.set(progressRef, progress)
            yield* Ref.set(
              lastInputDigest,
              Option.some(inputDigest)
            )
            yield* publish(
              {
                _tag: "Ready",
                recommendations: selected,
                catalog: catalogCandidates,
                progress,
              },
            )
          })
        )
        .pipe(
          Effect.withSpan("acn.local-model-recommendations.generate"),
          Effect.catchAllCause((cause) =>
            Effect.gen(function* () {
              const failure = Cause.failureOption(cause)
              const reportedFailure = localModelRecommendationFailure(
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
              const state = yield* SubscriptionRef.get(current)
              yield* publish(
                state._tag === "Ready"
                  ? { ...state, progress: failedProgress }
                  : {
                      _tag: "Failed",
                      failure: reportedFailure,
                      progress: failedProgress,
                    },
              )
              yield* Effect.logWarning(
                "Unable to generate local model recommendations"
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

      return LocalModelRecommendations.of({
        state: SubscriptionRef.get(current),
        changes: current.changes,
      })
    })
  )
