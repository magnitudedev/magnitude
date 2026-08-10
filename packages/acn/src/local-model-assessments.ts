import {
  Context,
  Effect,
  Inspectable,
  Layer,
  Option,
  ParseResult,
  Stream,
  SubscriptionRef,
} from "effect"
import {
  FitsModelAssessmentSchema,
  AssessmentEnvironmentIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  LocalModelMutationFailed,
  MemoryAssessmentSchema,
  ModelOfferingTargetIdSchema,
  ModelServingConfigurationIdSchema,
  ModelAssessmentIdSchema,
  type LocalModelAssessmentLifecycle,
  type FitsModelAssessment,
  type AssessmentEnvironmentId,
  type LocalInferenceError,
  type ModelFailure,
  type ModelOfferingTarget,
  type ModelOfferingTargetId,
  type ModelServingConfiguration,
  type ServingProfile,
} from "@magnitudedev/acn-protocol"
import type {
  AssessModelResult,
  ModelAssessment,
} from "@magnitudedev/icn-protocol/schemas"
import { IcnClient } from "@magnitudedev/icn"
import { LocalModelPackages } from "./local-model-packages"
import {
  servingProfileFromIcn,
  servingProfileToIcn,
  targetToIcn,
} from "./local-model-icn-adapter"

const REQUIRED_RESERVE_BYTES = 1536 * 1024 * 1024
const ASSESSMENT_OPERATION_TIMEOUT_MS = 5 * 60 * 1_000
const MINIMUM_CONTEXT_LENGTH = 4_096
const DEFAULT_LOCAL_MODEL_CONTEXT_LENGTH = 100_000
const PERFORMANCE_SAMPLE_CONTEXT_LENGTHS = [25_000, 50_000, 75_000] as const
type AssessmentProfiles = readonly [] | readonly [ServingProfile]

const targetMaximumContextLength = (
  target: ModelOfferingTarget,
): number => target._tag === "Package"
    ? target.package.properties.maximumContextLength
    : Math.min(
        target.target.properties.maximumContextLength,
        target.draft.properties.maximumContextLength,
      )

const assessmentProfile = (contextLength: number): AssessmentProfiles =>
  contextLength >= MINIMUM_CONTEXT_LENGTH ? [{ contextLength }] : []

export const localModelAssessmentProfiles = (
  target: ModelOfferingTarget,
  contextLength: number = DEFAULT_LOCAL_MODEL_CONTEXT_LENGTH,
): readonly ServingProfile[] => assessmentProfile(
  Math.min(contextLength, targetMaximumContextLength(target)),
)

export const performanceSampleContextTokens = (
  profile: ServingProfile,
): readonly number[] => [...new Set([
  ...PERFORMANCE_SAMPLE_CONTEXT_LENGTHS.filter((contextLength) =>
    contextLength <= profile.contextLength),
  profile.contextLength,
])].sort((left, right) => left - right)

export type LocalModelAssessment =
  | { readonly _tag: "Fits"; readonly assessment: FitsModelAssessment }
  | {
      readonly _tag: "DoesNotFit"
      readonly profile: ServingProfile
      readonly configurationId: ModelServingConfiguration["id"]
      readonly assessmentId: FitsModelAssessment["assessmentId"]
      readonly memory: FitsModelAssessment["memory"]
      readonly deficitBytes: number
      readonly limitingResource: string
    }
  | {
      readonly _tag: "Incompatible"
      readonly profile: ServingProfile
      readonly configurationId: ModelServingConfiguration["id"]
      readonly failure: ModelFailure
    }

export interface LocalModelAssessmentRequest {
  readonly targetId: ModelOfferingTargetId
  readonly target: ModelOfferingTarget
  readonly profiles: readonly ServingProfile[]
}

export type LocalModelAssessmentResult =
  | {
      readonly _tag: "Assessed"
      readonly targetId: ModelOfferingTargetId
      readonly environmentId: AssessmentEnvironmentId
      readonly assessments: readonly LocalModelAssessment[]
    }
  | { readonly _tag: "InvalidTarget"; readonly message: string }

export const clearAssessmentLifecycle = (
  current: ReadonlyMap<ModelOfferingTargetId, LocalModelAssessmentLifecycle>,
  targetIds: readonly ModelOfferingTargetId[],
): ReadonlyMap<ModelOfferingTargetId, LocalModelAssessmentLifecycle> => {
  const next = new Map(current)
  for (const targetId of targetIds) {
    const state = next.get(targetId)
    if (state?._tag === "Assessing") {
      next.set(targetId, { _tag: "Unassessed" })
    }
  }
  return next
}

export const completeAssessmentLifecycle = (
  current: ReadonlyMap<ModelOfferingTargetId, LocalModelAssessmentLifecycle>,
  targetIds: readonly ModelOfferingTargetId[],
  completed: readonly LocalModelAssessmentResult[],
): ReadonlyMap<ModelOfferingTargetId, LocalModelAssessmentLifecycle> => {
  const next = new Map(current)
  completed.forEach((result, index) => {
    const targetId = targetIds[index]
    const state = targetId === undefined ? undefined : next.get(targetId)
    if (
      targetId === undefined
      || result._tag !== "Assessed"
      || state?._tag !== "Assessing"
    ) return
    next.set(targetId, {
      _tag: "Assessed",
      environmentId: result.environmentId,
      configurationIds: result.assessments.map((assessment) =>
        assessment._tag === "Fits"
          ? assessment.assessment.configurationId
          : assessment.configurationId),
    })
  })
  return next
}

export const formatLocalModelAssessmentFailure = (error: unknown): string => {
  try {
    const serialized = JSON.stringify(
      error,
      (_key, value: unknown) =>
        value instanceof Error
          ? {
              ...value,
              name: value.name,
              message: value.message,
              stack: value.stack,
              cause: value.cause,
            }
          : value,
      2,
    )
    if (serialized !== undefined && serialized !== "{}") return serialized
  } catch {
    // Fall through to Effect's cycle-safe unknown-value formatter.
  }
  const structured = Inspectable.toStringUnknown(error)
  if (structured !== "{}") return structured
  return error instanceof Error
    ? error.stack ?? error.message
    : structured
}

const failure = (operation: string, message: string) =>
  new LocalModelMutationFailed({
    code: operation,
    message,
    retryable: true,
  })

const logAssessmentFailure = (operation: string, error: unknown) =>
  Effect.logWarning(`Local model ${operation} failed`).pipe(
    Effect.annotateLogs({ detail: formatLocalModelAssessmentFailure(error) })
  )

const memoryAssessmentFromIcn = (
  memory: Extract<ModelAssessment, { readonly _tag: "Fits" }>["memory"][number],
) => MemoryAssessmentSchema.make({
  memoryDomainId: LocalInferenceMemoryDomainIdSchema.make(memory.memoryDomainId),
  capacityBytes: memory.capacityBytes,
  requiredBytes: memory.requiredBytes,
  compatibilityReserveBytes: memory.compatibilityReserveBytes,
  warningReserveBytes: memory.warningReserveBytes,
  remainingBytes: memory.remainingBytes,
})

const modelAssessment = (
  assessment: Extract<ModelAssessment, { readonly _tag: "Fits" }>,
  environmentId: AssessmentEnvironmentId,
) => Effect.gen(function* () {
  const profile = yield* servingProfileFromIcn(assessment.profile)
  return FitsModelAssessmentSchema.make({
    _tag: "Fits",
    profile,
    configurationId: ModelServingConfigurationIdSchema.make(assessment.configurationId),
    assessmentId: ModelAssessmentIdSchema.make(assessment.assessmentId),
    environmentId,
    memory: assessment.memory.map(memoryAssessmentFromIcn),
    performance: assessment.performance,
  })
})

const assessmentFromIcn = (
  assessment: ModelAssessment,
  environmentId: AssessmentEnvironmentId,
): Effect.Effect<LocalModelAssessment, ParseResult.ParseError> =>
  assessment._tag === "Fits"
    ? modelAssessment(assessment, environmentId).pipe(
        Effect.map((value) => ({ _tag: "Fits" as const, assessment: value })),
      )
    : assessment._tag === "DoesNotFit" ? Effect.gen(function* () {
        return {
          _tag: "DoesNotFit" as const,
          profile: yield* servingProfileFromIcn(assessment.profile),
          configurationId: ModelServingConfigurationIdSchema.make(assessment.configurationId),
          assessmentId: ModelAssessmentIdSchema.make(assessment.assessmentId),
          memory: assessment.memory.map(memoryAssessmentFromIcn),
          deficitBytes: Number(assessment.deficitBytes),
          limitingResource: String(assessment.limitingResource),
        }
      })
    : servingProfileFromIcn(assessment.profile).pipe(
        Effect.map((profile) => ({
          _tag: "Incompatible" as const,
          profile,
          configurationId: ModelServingConfigurationIdSchema.make(assessment.configurationId),
          failure: assessment.failure,
        })),
      )

export const localModelAssessmentResultFromIcn = (
  result: AssessModelResult,
  environmentId: AssessmentEnvironmentId,
): Effect.Effect<LocalModelAssessmentResult, ParseResult.ParseError> =>
  result._tag === "InvalidTarget"
    ? Effect.succeed({ _tag: "InvalidTarget", message: result.failure.message })
    : Effect.gen(function* () {
        return {
          _tag: "Assessed" as const,
          targetId: ModelOfferingTargetIdSchema.make(String(result.targetId)),
          environmentId,
          assessments: yield* Effect.all(result.profiles.map((assessment) =>
            assessmentFromIcn(assessment, environmentId))),
        }
      })

export interface LocalModelAssessmentsApi {
  readonly state: Effect.Effect<ReadonlyMap<ModelOfferingTargetId, LocalModelAssessmentLifecycle>>
  readonly changes: Stream.Stream<ReadonlyMap<ModelOfferingTargetId, LocalModelAssessmentLifecycle>>
  readonly assess: (
    requests: readonly LocalModelAssessmentRequest[],
    onProgress: (
      completed: number,
      total: number,
    ) => Effect.Effect<void>,
  ) => Effect.Effect<readonly LocalModelAssessmentResult[], LocalInferenceError>
}

export class LocalModelAssessments extends Context.Tag("LocalModelAssessments")<
  LocalModelAssessments,
  LocalModelAssessmentsApi
>() {}

export const LocalModelAssessmentsLive: Layer.Layer<
  LocalModelAssessments,
  never,
  IcnClient | LocalModelPackages
> = Layer.effect(LocalModelAssessments, Effect.gen(function* () {
  const client = yield* IcnClient
  const packages = yield* LocalModelPackages
  const lifecycle = yield* SubscriptionRef.make<
    ReadonlyMap<ModelOfferingTargetId, LocalModelAssessmentLifecycle>
  >(new Map())
  const operationLock = yield* Effect.makeSemaphore(1)

  const setLifecycle = (
    targetIds: readonly ModelOfferingTargetId[],
    value: LocalModelAssessmentLifecycle,
  ) => SubscriptionRef.update(lifecycle, (current) => {
    const next = new Map(current)
    for (const targetId of targetIds) next.set(targetId, value)
    return next
  })

  const clearLifecycle = (targetIds: readonly ModelOfferingTargetId[]) =>
    SubscriptionRef.update(lifecycle, (current) =>
      clearAssessmentLifecycle(current, targetIds))

  const assess: LocalModelAssessmentsApi["assess"] = (
    requests,
    onProgress,
  ) => {
    const deadlineAtMs = Date.now() + ASSESSMENT_OPERATION_TIMEOUT_MS
    const operation = operationLock.withPermits(1)(Effect.gen(function* () {
      if (requests.length === 0) return []
      const targetIds = requests.map(({ targetId }) => targetId)
      yield* setLifecycle(targetIds, { _tag: "Assessing" })
      const completeOwned = (completed: readonly LocalModelAssessmentResult[]) =>
        SubscriptionRef.update(lifecycle, (current) =>
          completeAssessmentLifecycle(current, targetIds, completed))
      const run = Effect.gen(function* () {
        const installedIds = yield* packages.installedPackageIds
        const nativeRequests = yield* Effect.forEach(
          requests,
          ({ target, profiles }, index) => targetToIcn(target, installedIds).pipe(
            Effect.map((nativeTarget) => ({ index, nativeTarget, profiles })),
          ),
        )
        const batchSize = 8
        const nativeResults: Array<{
          readonly environmentId: AssessmentEnvironmentId
          readonly result: AssessModelResult
        }> = []
        let expectedEnvironmentId = Option.none<AssessmentEnvironmentId>()
        for (let offset = 0; offset < nativeRequests.length; offset += batchSize) {
          const batch = nativeRequests.slice(offset, offset + batchSize)
          const response = yield* client.models.assessModels({
            payload: {
              requests: batch.map(({ index, nativeTarget, profiles }) => ({
                requestId: `assessment-${index}`,
                target: nativeTarget,
                profiles: profiles.map((profile) => ({
                  profile: servingProfileToIcn(profile),
                  performanceContextTokens: performanceSampleContextTokens(profile),
                })),
              })),
              capacityPolicy: { requiredReserveBytesPerMemoryDomain: REQUIRED_RESERVE_BYTES },
            },
          })
          const environmentId = AssessmentEnvironmentIdSchema.make(response.environmentId)
          if (
            Option.isSome(expectedEnvironmentId)
            && expectedEnvironmentId.value !== environmentId
          ) {
            return yield* new LocalModelMutationFailed({
              code: "model_assessment_environment_changed",
              message: "The hardware assessment environment changed during model assessment.",
              retryable: true,
            })
          }
          expectedEnvironmentId = Option.some(environmentId)
          nativeResults.push(...response.results.map((result) => ({ environmentId, result })))
          yield* onProgress(Math.min(offset + batch.length, requests.length), requests.length)
        }
        const byRequest = new Map(nativeResults.map(({ environmentId, result }) => [
          String(result.requestId),
          { environmentId, result },
        ]))
        return yield* Effect.forEach(
          nativeRequests,
          ({ index }) => Effect.gen(function* () {
            const found = Option.fromNullable(byRequest.get(`assessment-${index}`))
            if (Option.isNone(found)) {
              return yield* Effect.dieMessage("ICN returned no assessment result")
            }
            const decoded = yield* localModelAssessmentResultFromIcn(
              found.value.result,
              found.value.environmentId,
            ).pipe(Effect.orDie)
            if (decoded._tag === "Assessed" && decoded.targetId !== targetIds[index]) {
              return yield* Effect.dieMessage(
                "ICN returned an assessment for a different model offering target",
              )
            }
            return decoded
          }),
        )
      })
      return yield* run.pipe(
        Effect.tap(completeOwned),
        Effect.ensuring(clearLifecycle(targetIds)),
      )
    }))
    return operation.pipe(
      Effect.timeoutFail({
        duration: Math.max(0, deadlineAtMs - Date.now()),
        onTimeout: () => new LocalModelMutationFailed({
          code: "model_assessment_deadline",
          message: "Local model assessment exceeded its operation deadline.",
          retryable: true,
        }),
      }),
      Effect.tapError((error) => logAssessmentFailure("assessment", error)),
      Effect.mapError((error) => error instanceof LocalModelMutationFailed
        ? error
        : failure("assess_model_failed", "Local model assessment could not be completed.")),
    )
  }

  return LocalModelAssessments.of({
    state: SubscriptionRef.get(lifecycle),
    changes: lifecycle.changes,
    assess,
  })
}))
