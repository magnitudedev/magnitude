import { Context, Effect, Layer, Option, ParseResult } from "effect"
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
  type ModelOfferingTarget,
  type ModelOfferingTargetId,
  type ModelServingConfiguration,
  type ServingProfile,
} from "@magnitudedev/protocol"
import { IcnClient, type OfferingAssessment } from "@magnitudedev/icn"
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
const MAXIMUM_PARALLEL_SEQUENCES = 8

export type LocalModelAssessment =
  | { readonly _tag: "Fits"; readonly assessment: FitsOfferingAssessment }
  | {
      readonly _tag: "DoesNotFit"
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
      readonly modelId: ModelOfferingTargetId
      readonly assessments: readonly LocalModelAssessment[]
    }
  | { readonly _tag: "InvalidTarget"; readonly message: string }

const failure = (operation: string, error: unknown) =>
  new LocalModelMutationFailed({
    code: operation,
    message: error instanceof Error ? error.message : String(error),
    retryable: true,
  })

const fitAssessment = (
  assessment: Extract<OfferingAssessment, { readonly _tag: "Fits" }>,
) => Effect.gen(function* () {
  const profile = yield* servingProfileFromIcn(assessment.profile)
  return FitsOfferingAssessmentSchema.make({
    profile,
    configurationId: ModelServingConfigurationIdSchema.make(assessment.configurationId),
    assessmentId: OfferingAssessmentIdSchema.make(assessment.assessmentId),
    memory: assessment.memory.map((memory) => MemoryAssessmentSchema.make({
      memoryDomainId: LocalInferenceMemoryDomainIdSchema.make(memory.memoryDomainId),
      capacityBytes: memory.capacityBytes,
      requiredBytes: memory.requiredBytes,
      requiredReserveBytes: memory.requiredReserveBytes,
      remainingBytes: memory.remainingBytes,
    })),
    performance: Option.flatMap(assessment.performance, Option.fromNullable),
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
        deficitBytes: Number(assessment.deficitBytes),
        limitingResource: String(assessment.limitingResource),
      } as const)
    : Effect.succeed({ _tag: "InvalidTarget", message: assessment.failure.message })

export interface LocalModelEvaluationsApi {
  readonly assessMany: (
    requests: readonly LocalModelAssessmentRequest[],
  ) => Effect.Effect<readonly LocalModelAssessmentResult[], LocalInferenceError>
  readonly assess: (
    target: ModelOfferingTarget,
    profiles: readonly ServingProfile[],
  ) => Effect.Effect<{
    readonly modelId: ModelOfferingTargetId
    readonly assessments: readonly LocalModelAssessment[]
  }, LocalInferenceError>
  readonly fit: (
    target: ModelOfferingTarget,
  ) => Effect.Effect<{
    readonly modelId: ModelOfferingTargetId
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

  const assessMany: LocalModelEvaluationsApi["assessMany"] = (requests) =>
    Effect.gen(function* () {
      if (requests.length === 0) return []
      const installedIds = yield* packages.installedPackageIds
      const nativeTargets = yield* Effect.forEach(
        requests,
        ({ target }) => targetToIcn(target, installedIds),
      )
      const response = yield* client.models.assessModels({
        payload: {
          requests: requests.map(({ profiles }, index) => ({
            requestId: `assessment-${index}`,
            target: nativeTargets[index]!,
            profiles: profiles.map(servingProfileToIcn),
          })),
          capacityPolicy: { requiredReserveBytesPerMemoryDomain: REQUIRED_RESERVE_BYTES },
          includePerformance: true,
        },
      })
      const byRequest = new Map(response.results.map((result) => [String(result.requestId), result]))
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
        if (result._tag === "InvalidTarget") {
          results.push({ _tag: "InvalidTarget", message: result.failure.message })
          continue
        }
        results.push({
          _tag: "Assessed",
          modelId: ModelOfferingTargetIdSchema.make(String(result.targetId)),
          assessments: yield* Effect.all(result.profiles.map(assessmentFromIcn)),
        })
      }
      return results
    }).pipe(Effect.mapError((error) => failure("assess_model_failed", error)))

  return LocalModelEvaluations.of({
    assessMany,
    assess: (target, profiles) => assessMany([{ target, profiles }]).pipe(
      Effect.flatMap((results) => {
        const result = results[0]
        if (result?._tag === "Assessed") return Effect.succeed(result)
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
          maximumParallelSequences: MAXIMUM_PARALLEL_SEQUENCES,
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
        modelId: ModelOfferingTargetIdSchema.make(String(result.targetId)),
        configuration,
        assessment: yield* fitAssessment(result.assessment),
      }
    }).pipe(Effect.mapError((error) =>
      error instanceof LocalModelMutationFailed
        ? error
        : failure("fit_model_failed", error))),
  })
}))
