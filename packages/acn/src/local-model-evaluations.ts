import { Context, Effect, Inspectable, Layer, Option, ParseResult } from "effect"
import {
  FitsOfferingAssessmentSchema,
  LocalInferenceMemoryDomainIdSchema,
  LocalModelMutationFailed,
  MemoryAssessmentSchema,
  ModelOfferingTargetIdSchema,
  ModelServingConfigurationIdSchema,
  ModelServingConfigurationSchema,
  OfferingAssessmentIdSchema,
  type FitsOfferingAssessment,
  type LocalInferenceError,
  type ModelFailure,
  type ModelOfferingTarget,
  type ModelOfferingTargetId,
  type ModelServingConfiguration,
  type ServingProfile,
} from "@magnitudedev/acn-protocol"
import type {
  AssessModelResult,
  OfferingAssessment,
} from "@magnitudedev/icn-protocol/schemas"
import { IcnClient } from "@magnitudedev/icn"
import { LocalModelPackages } from "./local-model-packages"
import {
  offeringTargetFromIcn,
  servingProfileFromIcn,
  servingProfileToIcn,
  targetToIcn,
} from "./local-model-icn-adapter"

const REQUIRED_RESERVE_BYTES = 1536 * 1024 * 1024
const MINIMUM_CONTEXT_LENGTH = 4_096
const MAXIMUM_CONTEXT_LENGTH = 200_000

export type LocalModelAssessment =
  | { readonly _tag: "Fits"; readonly assessment: FitsOfferingAssessment }
  | {
      readonly _tag: "DoesNotFit"
      readonly memory: FitsOfferingAssessment["memory"]
      readonly deficitBytes: number
      readonly limitingResource: string
    }
  | { readonly _tag: "InvalidTarget"; readonly message: string }

export interface LocalModelAssessmentRequest {
  readonly target: ModelOfferingTarget
  readonly profiles: readonly ServingProfile[]
}

export type LocalModelAssessmentResult =
  | {
      readonly _tag: "Assessed"
      readonly targetId: ModelOfferingTargetId
      readonly assessments: readonly LocalModelAssessment[]
    }
  | { readonly _tag: "AssessmentFailed"; readonly failure: ModelFailure }
  | { readonly _tag: "InvalidTarget"; readonly message: string }

export const localModelAssessmentFailure = (
  results: readonly LocalModelAssessmentResult[],
): ModelFailure | undefined => {
  const failed = results.find((result) => result._tag === "AssessmentFailed")
  return failed?.failure
}

export const formatLocalModelEvaluationFailure = (error: unknown): string => {
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

const logEvaluationFailure = (operation: string, error: unknown) =>
  Effect.logWarning(`Local model ${operation} failed`).pipe(
    Effect.annotateLogs({ detail: formatLocalModelEvaluationFailure(error) })
  )

const memoryAssessmentFromIcn = (
  memory: Extract<OfferingAssessment, { readonly _tag: "Fits" }>["memory"][number],
) => MemoryAssessmentSchema.make({
  memoryDomainId: LocalInferenceMemoryDomainIdSchema.make(memory.memoryDomainId),
  capacityBytes: memory.capacityBytes,
  requiredBytes: memory.requiredBytes,
  compatibilityReserveBytes: memory.compatibilityReserveBytes,
  warningReserveBytes: memory.warningReserveBytes,
  remainingBytes: memory.remainingBytes,
})

const fitAssessment = (
  assessment: Extract<OfferingAssessment, { readonly _tag: "Fits" }>,
) => Effect.gen(function* () {
  const profile = yield* servingProfileFromIcn(assessment.profile)
  return FitsOfferingAssessmentSchema.make({
    profile,
    configurationId: ModelServingConfigurationIdSchema.make(assessment.configurationId),
    assessmentId: OfferingAssessmentIdSchema.make(assessment.assessmentId),
    memory: assessment.memory.map(memoryAssessmentFromIcn),
    performance: assessment.performance,
  })
})

const assessmentFromIcn = (
  assessment: OfferingAssessment,
): Effect.Effect<LocalModelAssessment, ParseResult.ParseError> =>
  assessment._tag === "Fits"
    ? fitAssessment(assessment).pipe(
        Effect.map((value) => ({ _tag: "Fits" as const, assessment: value })),
      )
    : assessment._tag === "DoesNotFit" ? Effect.succeed({
        _tag: "DoesNotFit",
        memory: assessment.memory.map(memoryAssessmentFromIcn),
        deficitBytes: Number(assessment.deficitBytes),
        limitingResource: String(assessment.limitingResource),
      } as const)
    : Effect.succeed({ _tag: "InvalidTarget", message: assessment.failure.message })

export const localModelAssessmentResultFromIcn = (
  result: AssessModelResult,
): Effect.Effect<LocalModelAssessmentResult, ParseResult.ParseError> =>
  result._tag === "InvalidTarget"
    ? Effect.succeed({ _tag: "InvalidTarget", message: result.failure.message })
    : result._tag === "AssessmentFailed"
      ? Effect.succeed({ _tag: "AssessmentFailed", failure: result.failure })
      : Effect.gen(function* () {
          return {
            _tag: "Assessed" as const,
            targetId: ModelOfferingTargetIdSchema.make(String(result.targetId)),
            assessments: yield* Effect.all(result.profiles.map(assessmentFromIcn)),
          }
        })

export interface LocalModelEvaluationsApi {
  readonly assessMany: (
    requests: readonly LocalModelAssessmentRequest[],
  ) => Effect.Effect<readonly LocalModelAssessmentResult[], LocalInferenceError>
  readonly assessManyWithProgress: (
    requests: readonly LocalModelAssessmentRequest[],
    onProgress: (
      completed: number,
      total: number,
    ) => Effect.Effect<void>,
  ) => Effect.Effect<readonly LocalModelAssessmentResult[], LocalInferenceError>
  readonly assess: (
    target: ModelOfferingTarget,
    profiles: readonly ServingProfile[],
  ) => Effect.Effect<{
    readonly targetId: ModelOfferingTargetId
    readonly assessments: readonly LocalModelAssessment[]
  }, LocalInferenceError>
  readonly fit: (
    target: ModelOfferingTarget,
  ) => Effect.Effect<{
    readonly targetId: ModelOfferingTargetId
    readonly configuration: ModelServingConfiguration
    readonly assessment: FitsOfferingAssessment
  }, LocalInferenceError>
}

export class LocalModelEvaluations extends Context.Tag("LocalModelEvaluations")<
  LocalModelEvaluations,
  LocalModelEvaluationsApi
>() {}

export const LocalModelEvaluationsLive: Layer.Layer<
  LocalModelEvaluations,
  never,
  IcnClient | LocalModelPackages
> = Layer.effect(LocalModelEvaluations, Effect.gen(function* () {
  const client = yield* IcnClient
  const packages = yield* LocalModelPackages

  const targetInput = (target: ModelOfferingTarget) =>
    packages.installedPackageIds.pipe(Effect.flatMap((ids) => targetToIcn(target, ids)))

  const assessManyWithProgress: LocalModelEvaluationsApi["assessManyWithProgress"] = (
    requests,
    onProgress,
  ) =>
    Effect.gen(function* () {
      if (requests.length === 0) return []
      const installedIds = yield* packages.installedPackageIds
      const nativeTargets = yield* Effect.forEach(
        requests,
        ({ target }) => targetToIcn(target, installedIds),
      )
      const batchSize = 8
      const nativeResults: AssessModelResult[] = []
      for (let offset = 0; offset < requests.length; offset += batchSize) {
        const batch = requests.slice(offset, offset + batchSize)
        const response = yield* client.models.assessModels({
          payload: {
            requests: batch.map(({ profiles }, batchIndex) => ({
              requestId: `assessment-${offset + batchIndex}`,
              target: nativeTargets[offset + batchIndex]!,
              profiles: profiles.map(servingProfileToIcn),
            })),
            capacityPolicy: { requiredReserveBytesPerMemoryDomain: REQUIRED_RESERVE_BYTES },
          },
        })
        nativeResults.push(...response.results)
        yield* onProgress(Math.min(offset + batch.length, requests.length), requests.length)
      }
      const byRequest = new Map(nativeResults.map((result) => [String(result.requestId), result]))
      const results: LocalModelAssessmentResult[] = []
      for (let index = 0; index < requests.length; index += 1) {
        const result = byRequest.get(`assessment-${index}`)
        if (!result) {
          results.push({
            _tag: "InvalidTarget",
            message: "ICN returned no assessment result",
          })
          continue
        }
        results.push(yield* localModelAssessmentResultFromIcn(result))
      }
      return results
    }).pipe(
      Effect.tapError((error) => logEvaluationFailure("assessment", error)),
      Effect.mapError(() => failure(
        "assess_model_failed",
        "Local model assessment could not be completed.",
      )),
    )

  const assessMany: LocalModelEvaluationsApi["assessMany"] = (requests) =>
    assessManyWithProgress(requests, () => Effect.void)

  return LocalModelEvaluations.of({
    assessMany,
    assessManyWithProgress,
    assess: (target, profiles) => assessMany([{ target, profiles }]).pipe(
      Effect.flatMap((results) => {
        const result = results[0]
        if (result?._tag === "Assessed") return Effect.succeed(result)
        if (result?._tag === "AssessmentFailed") {
          return Effect.fail(new LocalModelMutationFailed(result.failure))
        }
        return Effect.fail(new LocalModelMutationFailed({
          code: "model_target_invalid",
          message: result?.message ?? "ICN returned no assessment result",
          retryable: false,
        }))
      }),
    ),
    fit: (target) => Effect.gen(function* () {
      const input = yield* targetInput(target)
      const response = yield* client.models.fitModels({
        payload: {
          targets: [{ requestId: "auto-fit", target: input }],
          capacityPolicy: { requiredReserveBytesPerMemoryDomain: REQUIRED_RESERVE_BYTES },
          minimumContextLength: MINIMUM_CONTEXT_LENGTH,
          maximumContextLength: MAXIMUM_CONTEXT_LENGTH,
        },
      })
      const result = response.results[0]
      if (!result || result._tag !== "Fitted") {
        const message = !result
          ? "ICN returned no fit result"
          : result._tag === "InvalidTarget"
            ? result.failure.message
            : `Model does not fit (${result.limitingResource}, ${result.deficitBytes} bytes short)`
        return yield* new LocalModelMutationFailed({
          code: "model_does_not_fit",
          message,
          retryable: false,
        })
      }
      if (result.assessment._tag !== "Fits") {
        return yield* new LocalModelMutationFailed({
          code: "invalid_fit_response",
          message: "ICN returned a fitted configuration without a fitting assessment",
          retryable: true,
        })
      }
      const configuration = ModelServingConfigurationSchema.make({
        id: ModelServingConfigurationIdSchema.make(result.configuration.id),
        target: yield* offeringTargetFromIcn(result.configuration.target),
        profile: yield* servingProfileFromIcn(result.configuration.profile),
      })
      return {
        targetId: ModelOfferingTargetIdSchema.make(String(result.targetId)),
        configuration,
        assessment: yield* fitAssessment(result.assessment),
      }
    }).pipe(
      Effect.tapError((error) =>
        error instanceof LocalModelMutationFailed
          ? Effect.void
          : logEvaluationFailure("fitting", error)),
      Effect.mapError((error) =>
        error instanceof LocalModelMutationFailed
          ? error
          : failure(
              "fit_model_failed",
              "The model could not be fitted to this machine.",
            )),
    ),
  })
}))
