import {
  Cause,
  Context,
  Effect,
  Layer,
  Option,
  Ref,
  Schema,
  Stream,
} from "effect"
import { createHash } from "node:crypto"
import {
  LocalModelMutationFailed,
  LocalModelCatalogCandidateMetadataSchema,
  ModelFailureSchema,
  ModelServingConfigurationSchema,
  LocalModelRecommendationProgressStepSchema,
  RecommendationSchema,
  type FitsModelAssessment,
  type LocalModelRecommendationProgressStep,
  type LocalModelRecommendationProgressStepId,
  type ModelOfferingTarget,
  type ModelOfferingTargetId,
  type ModelFailure,
  type ModelServingConfiguration,
  type Recommendation,
  type RecommendableModel,
} from "@magnitudedev/acn-protocol"
import { IcnCatalog, IcnHardware } from "@magnitudedev/icn"
import { makeObservedState } from "./mirrored-state"
import {
  localModelAssessmentProfiles,
  LocalModelAssessments,
} from "./local-model-assessments"
import { LocalModelPackages } from "./local-model-packages"
import { recommendableModelFromIcn } from "./local-model-icn-adapter"
import {
  assembleRecommendationCatalogCandidates,
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
      readonly catalog: readonly CatalogEntry[]
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
  readonly snapshot: Effect.Effect<{
    readonly revision: number
    readonly state: RecommendationState
  }>
  readonly changes: Stream.Stream<{
    readonly revision: number
    readonly state: RecommendationState
  }>
  readonly getCatalogByConfigurationId: (
    configurationId: ModelServingConfiguration["id"]
  ) => Effect.Effect<Option.Option<CatalogEntry>>
}

export class LocalModelRecommendations extends Context.Tag(
  "LocalModelRecommendations"
)<LocalModelRecommendations, LocalModelRecommendationsApi>() {}

export const exactTargetTensorStorageBytes = (
  model: RecommendableModel
): Option.Option<number> => exactTensorStorageBytes(model.target)

const exactTensorStorageBytes = (
  target: ModelOfferingTarget,
): Option.Option<number> => {
  const packages =
    target._tag === "Package"
      ? [target.package]
      : [target.target, target.draft]
  const files = new Map(
    packages
      .flatMap(({ files }) => files)
      // Primary/sharded weight tensors are required for every execution of the target. Other
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

const targetPackages = (model: RecommendableModel) =>
  model.target._tag === "Package"
    ? [model.target.package]
    : [model.target.target, model.target.draft]

const CatalogEntrySchema = Schema.Struct({
  candidate: LocalModelCatalogCandidateMetadataSchema,
  configuration: ModelServingConfigurationSchema,
})
type CatalogEntry = typeof CatalogEntrySchema.Type

const catalogProjection = (
  candidate: RecommendationCandidate,
): CatalogEntry => ({
  candidate: {
    configurationId: candidate.assessment.configurationId,
    assessmentId: candidate.assessment.assessmentId,
    environmentId: candidate.assessment.environmentId,
    targetId: candidate.model.targetId,
    displayName: candidate.model.displayName,
    description: candidate.model.description,
    license: candidate.model.license,
    profile: candidate.profile,
    downloadBytes: candidate.totalDownloadBytes,
    quantization: targetPackages(candidate.model)
      .map(({ properties }) => properties.quantization)
      .join(" + "),
    quantizationName: targetPackages(candidate.model)
      .map(({ properties }) => properties.quantizationName)
      .join(" + "),
    memory: candidate.assessment.memory,
    recommendationEvidence: Option.some({
      intelligence: Option.fromNullable(candidate.capability),
      fidelityRank: candidate.fidelityRank,
      qualityEvidence: candidate.model.qualityEvidence,
    }),
    performance: candidate.assessment.performance,
    capabilities: candidate.model.capabilities,
    sources: targetPackages(candidate.model).map((modelPackage) => ({
      source: modelPackage.source,
      files: modelPackage.files.map(({ path, sha256 }) => ({ path, sha256 })),
    })),
  },
  configuration: {
    id: candidate.assessment.configurationId,
    target: candidate.model.target,
    profile: candidate.profile,
  },
})

interface InstalledAssessmentTarget {
  readonly targetId: ModelOfferingTargetId
  readonly target: ModelOfferingTarget
  readonly displayName: string
  readonly capabilities: RecommendationCandidate["model"]["capabilities"]
}

const installedCatalogProjection = (
  model: InstalledAssessmentTarget,
  assessment: FitsModelAssessment,
): CatalogEntry => {
  const packages = model.target._tag === "Package"
    ? [model.target.package]
    : [model.target.target, model.target.draft]
  return {
    candidate: {
      configurationId: assessment.configurationId,
      assessmentId: assessment.assessmentId,
      environmentId: assessment.environmentId,
      targetId: model.targetId,
      displayName: model.displayName,
      description: "",
      license: "Unknown",
      profile: assessment.profile,
      downloadBytes: packages.flatMap(({ files }) => files)
        .reduce((total, file) => total + file.sizeBytes, 0),
      quantization: packages.map(({ properties }) => properties.quantization).join(" + "),
      quantizationName: packages.map(({ properties }) => properties.quantizationName).join(" + "),
      memory: assessment.memory,
      recommendationEvidence: Option.none(),
      performance: assessment.performance,
      capabilities: model.capabilities,
      sources: packages.map((modelPackage) => ({
        source: modelPackage.source,
        files: modelPackage.files.map(({ path, sha256 }) => ({ path, sha256 })),
      })),
    },
    configuration: {
      id: assessment.configurationId,
      target: model.target,
      profile: assessment.profile,
    },
  }
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
  IcnCatalog | IcnHardware | LocalModelAssessments | LocalModelPackages
> =>
  Layer.scoped(
    LocalModelRecommendations,
    Effect.gen(function* () {
      const catalog = yield* IcnCatalog
      const hardware = yield* IcnHardware
      const assessments = yield* LocalModelAssessments
      const packages = yield* LocalModelPackages
      const startupStartedAtMs = Date.now()
      const startupProgress = updateProgress(initialProgress(), "hardware", {
        status: { _tag: "Running", startedAtMs: startupStartedAtMs },
      })
      const mirror = yield* makeObservedState<RecommendationState>({
        _tag: "Loading",
        progress: startupProgress,
      })
      const progressRef = yield* Ref.make(startupProgress)
      const lastInputDigest = yield* Ref.make<Option.Option<string>>(
        Option.none()
      )
      const recommendationsEquivalent = Schema.equivalence(
        Schema.Array(RecommendationSchema)
      )
      const catalogEquivalent = Schema.equivalence(
        Schema.Array(CatalogEntrySchema)
      )
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

      const publishProgress = (
        progress: readonly LocalModelRecommendationProgressStep[]
      ): Effect.Effect<void> =>
        Effect.gen(function* () {
          yield* Ref.set(progressRef, progress)
          const current = (yield* mirror.get).state
          yield* mirror.setIfChanged(
            current._tag === "Ready"
              ? { ...current, progress }
              : current._tag === "Failed"
              ? { _tag: "Loading", progress }
              : { ...current, progress },
            equivalent
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
            const currentStateBeforeRefresh = (yield* mirror.get).state
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
            const packageState = (yield* packages.snapshot).state
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

            if (!(yield* catalog.ready)) return
            const catalogState = (yield* catalog.get).state
            const catalogModels = yield* Effect.forEach(
              catalogState.models,
              recommendableModelFromIcn
            )
            const catalogTargetIds = new Set(catalogModels.map(({ targetId }) => targetId))
            const explicitStandalonePackageIds = new Set(catalogModels.flatMap(({ target }) =>
              target._tag === "Package" ? [target.package.id] : []))
            const pairedPackageIds = new Set(catalogModels.flatMap(({ target }) =>
              target._tag === "SpeculativeDecodingPair"
                ? [target.target.id, target.draft.id]
                : []))
            const installedTargets = packageState.entries.flatMap(
              (entry): readonly InstalledAssessmentTarget[] => {
                if (
                  entry.localState._tag !== "Installed"
                  || entry.inspection._tag !== "Inspected"
                  || Option.isNone(entry.targetId)
                  || catalogTargetIds.has(entry.targetId.value)
                  || (pairedPackageIds.has(entry.package.id)
                    && !explicitStandalonePackageIds.has(entry.package.id))
                ) return []
                const target = { _tag: "Package" as const, package: entry.package }
                const sourceName = entry.package.source._tag === "HuggingFace"
                  ? entry.package.source.repository.split("/").at(-1)
                    ?? entry.package.source.repository
                  : entry.package.files[0]?.path.split("/").at(-1) ?? entry.package.id
                return [{
                  targetId: entry.targetId.value,
                  target,
                  displayName: sourceName,
                  capabilities: entry.inspection.capabilities,
                }]
              },
            )
            const catalogAssessmentTargets = catalogModels.map((model) => ({
              ...model,
              profiles: localModelAssessmentProfiles(
                model.target,
                model.profile.contextLength,
              ),
            }))
            const installedAssessmentTargets = installedTargets.map((model) => ({
              ...model,
              profiles: localModelAssessmentProfiles(model.target),
            }))
            const assessmentTargets = [
              ...catalogAssessmentTargets,
              ...installedAssessmentTargets,
            ]
            const inputState = yield* Schema.encode(
              Schema.parseJson(Schema.Unknown)
            )({
              catalog: catalogAssessmentTargets.map((model) => ({
                id: model.id,
                targetId: model.targetId,
                checkpointId: model.checkpointId,
                assessmentProfiles: model.profiles,
                displayName: model.displayName,
                description: model.description,
                license: model.license,
                capabilities: model.capabilities,
                qualityScore: model.qualityScore,
                qualityScoreProvenance: model.qualityScoreProvenance,
                fidelityRank: model.fidelityRank,
                quantizationAware: model.quantizationAware,
                qualityEvidence: model.qualityEvidence,
                tensorStorageBytes: Option.getOrNull(
                  exactTargetTensorStorageBytes(model)
                ),
              })),
              installed: installedAssessmentTargets.map((model) => ({
                targetId: model.targetId,
                assessmentProfiles: model.profiles,
                target: model.target,
                capabilities: model.capabilities,
                tensorStorageBytes: Option.getOrNull(
                  exactTensorStorageBytes(model.target)
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
            const currentState = (yield* mirror.get).state
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
                completedItems: Option.some(assessmentTargets.length),
                totalItems: Option.some(assessmentTargets.length),
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
              yield* mirror.setIfChanged(
                { ...currentState, progress },
                equivalent
              )
              return
            }

            const aggregateStableCapacityBytes = hardwareSnapshot.memory_domains.reduce(
              (total, domain) => total + domain.stable_capacity_bytes,
              0
            )
            const assessableTargets = assessmentTargets.filter(({ target }) => {
              const tensorStorageBytes = exactTensorStorageBytes(target)
              return Option.isSome(tensorStorageBytes)
                && tensorStorageBytes.value > aggregateStableCapacityBytes
                ? false
                : true
            })
            const rejectedCount = assessmentTargets.length - assessableTargets.length
            const assessmentStartedAt = Date.now()
            progress = yield* startStep(progress, "assessment", {
              completed: rejectedCount,
              total: assessmentTargets.length,
            })
            yield* publishProgress(progress)
            const requests = assessableTargets.map(({ targetId, target, profiles }) => ({
              targetId,
              target,
              profiles,
            }))
            const assessedResults = yield* assessments.assess(
              requests,
              (completed, total) =>
                Effect.gen(function* () {
                  const elapsedMs = Math.max(
                    0,
                    Date.now() - assessmentStartedAt
                  )
                  const estimatedRemainingMs =
                    completed > 0
                      ? Math.max(
                          0,
                          Math.round(
                            (elapsedMs / completed) * (total - completed)
                          )
                        )
                      : 0
                  progress = updateProgress(progress, "assessment", {
                    completedItems: Option.some(rejectedCount + completed),
                    totalItems: Option.some(rejectedCount + total),
                    estimatedRemainingMs:
                      completed > 0
                        ? Option.some(estimatedRemainingMs)
                        : Option.none(),
                  })
                  yield* publishProgress(progress)
                })
            )
            const results = new Map(
              assessableTargets.map(({ targetId }, assessedIndex) => [
                targetId,
                assessedResults[assessedIndex],
              ])
            )
            progress = yield* completeStep(
              progress,
              "assessment",
              assessmentStartedAt,
              false,
              { completed: assessmentTargets.length, total: assessmentTargets.length }
            )
            const evaluated = catalogModels.flatMap(
              (model): readonly RecommendationCandidate[] => {
                const result = results.get(model.targetId)
                if (result?._tag !== "Assessed") return []
                return result.assessments.flatMap(
                  (assessment): readonly RecommendationCandidate[] => {
                    if (assessment._tag !== "Fits") return []
                    const profile = assessment.assessment.profile
                    return [
                      {
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
                        estimatedLoadedBytes:
                          assessment.assessment.memory.reduce(
                            (total, domain) => total + domain.requiredBytes,
                            0
                          ),
                        stableCapacityBudgetBytes:
                          assessment.assessment.memory.reduce(
                            (total, domain) =>
                              total +
                              Math.max(
                                0,
                                domain.capacityBytes -
                                  domain.compatibilityReserveBytes
                              ),
                            0
                          ),
                        totalDownloadBytes:
                          model.target._tag === "Package"
                            ? model.target.package.files.reduce(
                                (total, file) => total + file.sizeBytes,
                                0
                              )
                            : [
                                ...model.target.target.files,
                                ...model.target.draft.files,
                              ].reduce(
                                (total, file) => total + file.sizeBytes,
                                0
                              ),
                      },
                    ]
                  }
                )
              }
            )
            const selectionStartedAt = Date.now()
            progress = yield* startStep(progress, "recommendations")
            const selected = selectRecommendationPortfolio(evaluated)
            const catalogCandidates = assembleRecommendationCatalogCandidates(
              evaluated,
              selected
            ).map(catalogProjection)
            const installedCandidates = installedTargets.flatMap((model) => {
              const result = results.get(model.targetId)
              if (result?._tag !== "Assessed") return []
              const assessment = result.assessments.find((item) => item._tag === "Fits")
              return assessment === undefined
                ? []
                : [installedCatalogProjection(model, assessment.assessment)]
            })
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
            yield* mirror.setIfChanged(
              {
                _tag: "Ready",
                recommendations: selected,
                catalog: [...catalogCandidates, ...installedCandidates],
                progress,
              },
              equivalent
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
              const current = (yield* mirror.get).state
              yield* mirror.setIfChanged(
                current._tag === "Ready"
                  ? { ...current, progress: failedProgress }
                  : {
                      _tag: "Failed",
                      failure: reportedFailure,
                      progress: failedProgress,
                    },
                equivalent
              )
              yield* Effect.logWarning(
                "Unable to generate local model recommendations"
              ).pipe(Effect.annotateLogs({ cause: String(cause) }))
            })
          )
        )

      yield* generate.pipe(Effect.forkScoped)
      yield* Stream.merge(
        Stream.merge(catalog.changes, hardware.assessmentChanges),
        packages.changes
      ).pipe(
        Stream.runForEach(() => generate),
        Effect.forkScoped
      )

      return LocalModelRecommendations.of({
        snapshot: mirror.get,
        changes: mirror.changes,
        getCatalogByConfigurationId: (configurationId) =>
          mirror.get.pipe(
            Effect.map(({ state }) =>
              state._tag === "Ready"
                ? Option.fromNullable(state.catalog.find(
                    (entry) =>
                      entry.configuration.id === configurationId
                  ))
                : Option.none()
            )
          ),
      })
    })
  )
