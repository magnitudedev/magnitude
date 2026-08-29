import {
  Context,
  Effect,
  Inspectable,
  Layer,
  Option,
  ParseResult,
  Schema,
  Stream,
} from "effect"
import {
  FitsModelAssessmentSchema,
  AssessmentEnvironmentIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  LocalModelMutationFailed,
  MemoryAssessmentSchema,
  ModelAssessmentIdSchema,
  servableModelBundlePackages,
  ServingProfileSchema,
  type FitsModelAssessment,
  type AssessmentEnvironmentId,
  type LocalInferenceError,
  type ModelFailure,
  type ServableModelBundle,
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
  modelServingConfigurationFromIcn,
  servingProfileToIcn,
  bundleToIcnInput,
} from "./local-model-icn-adapter"

const ASSESSMENT_OPERATION_TIMEOUT_MS = 5 * 60 * 1_000
export const MINIMUM_LOCAL_MODEL_CONTEXT_LENGTH = 4_096
const DEFAULT_LOCAL_MODEL_CONTEXT_LENGTH = 100_000
const PERFORMANCE_SAMPLE_CONTEXT_LENGTHS = [25_000, 50_000, 75_000] as const
type AssessmentProfiles = readonly [] | readonly [ServingProfile]

const bundleMaximumContextLength = (
  bundle: ServableModelBundle,
): Option.Option<number> => {
  const known = servableModelBundlePackages(bundle).flatMap(({ properties }) =>
    Option.match(properties.maximumContextLength, {
      onNone: () => [],
      onSome: (maximum) => [maximum],
    }))
  return known.length === 0 ? Option.none() : Option.some(Math.min(...known))
}

const assessmentProfile = (contextLength: number): AssessmentProfiles =>
  contextLength >= MINIMUM_LOCAL_MODEL_CONTEXT_LENGTH ? [{ contextLength }] : []

export const localModelAssessmentProfiles = (
  bundle: ServableModelBundle,
  contextLength: number = DEFAULT_LOCAL_MODEL_CONTEXT_LENGTH,
): readonly ServingProfile[] => assessmentProfile(
  Option.match(bundleMaximumContextLength(bundle), {
    onNone: () => contextLength,
    onSome: (maximum) => Math.min(contextLength, maximum),
  }),
)

export const performanceSampleContextTokens = (
  profile: ServingProfile,
): readonly number[] => [...new Set([
  ...PERFORMANCE_SAMPLE_CONTEXT_LENGTHS.filter((contextLength) =>
    contextLength <= profile.contextLength),
  profile.contextLength,
])].sort((left, right) => left - right)

export type LocalModelAssessment =
  | {
      readonly _tag: "Fits"
      readonly configuration: ModelServingConfiguration
      readonly assessment: FitsModelAssessment
    }
  | {
      readonly _tag: "DoesNotFit"
      readonly configuration: ModelServingConfiguration
      readonly assessmentId: FitsModelAssessment["assessmentId"]
      readonly memory: FitsModelAssessment["memory"]
      readonly deficitBytes: number
      readonly limitingResource: string
    }
  | {
      readonly _tag: "Incompatible"
      readonly configuration: ModelServingConfiguration
      readonly failure: ModelFailure
    }

export interface LocalModelAssessmentRequest {
  readonly bundle: ServableModelBundle
  readonly profiles: readonly ServingProfile[]
}

export type LocalModelAssessmentResult =
  | {
      readonly _tag: "Assessed"
      readonly environmentId: AssessmentEnvironmentId
      readonly assessments: readonly LocalModelAssessment[]
    }
  | { readonly _tag: "InvalidBundle"; readonly message: string }
  | { readonly _tag: "Failed"; readonly failure: ModelFailure }

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
  remainingBytes: memory.remainingBytes,
})

const modelAssessment = (
  assessment: Extract<ModelAssessment, { readonly _tag: "Fits" }>,
  environmentId: AssessmentEnvironmentId,
) => Effect.gen(function* () {
  const configuration = yield* modelServingConfigurationFromIcn(assessment.configuration)
  return {
    configuration,
    assessment: FitsModelAssessmentSchema.make({
      _tag: "Fits",
      profile: configuration.profile,
      assessmentId: ModelAssessmentIdSchema.make(assessment.assessmentId),
      environmentId,
      memory: assessment.memory.map(memoryAssessmentFromIcn),
      performance: assessment.performance,
    }),
  }
})

const assessmentFromIcn = (
  assessment: ModelAssessment,
  environmentId: AssessmentEnvironmentId,
): Effect.Effect<LocalModelAssessment, ParseResult.ParseError> =>
  assessment._tag === "Fits"
    ? modelAssessment(assessment, environmentId).pipe(
        Effect.map(({ assessment, configuration }) => ({
          _tag: "Fits" as const,
          configuration,
          assessment,
        })),
      )
    : assessment._tag === "DoesNotFit" ? Effect.gen(function* () {
        return {
          _tag: "DoesNotFit" as const,
          configuration: yield* modelServingConfigurationFromIcn(assessment.configuration),
          assessmentId: ModelAssessmentIdSchema.make(assessment.assessmentId),
          memory: assessment.memory.map(memoryAssessmentFromIcn),
          totalRequiredBytes: assessment.memory.reduce(
            (total, memory) => total + memory.requiredBytes,
            0,
          ),
          deficitBytes: Number(assessment.deficitBytes),
          limitingResource: String(assessment.limitingResource),
        }
      })
    : modelServingConfigurationFromIcn(assessment.configuration).pipe(
        Effect.map((configuration) => ({
          _tag: "Incompatible" as const,
          configuration,
          failure: assessment.failure,
        })),
      )

export const localModelAssessmentResultFromIcn = (
  result: AssessModelResult,
  environmentId: AssessmentEnvironmentId,
): Effect.Effect<LocalModelAssessmentResult, ParseResult.ParseError> =>
  result._tag === "InvalidBundle"
    ? Effect.succeed({ _tag: "InvalidBundle", message: result.failure.message })
    : result._tag === "Failed"
    ? Effect.succeed({ _tag: "Failed", failure: result.failure })
    : Effect.gen(function* () {
        return {
          _tag: "Assessed" as const,
          environmentId,
          assessments: yield* Effect.all(result.profiles.map((assessment) =>
            assessmentFromIcn(assessment, environmentId))),
        }
      })

const sameProfile = Schema.equivalence(ServingProfileSchema)

const validateRequestedProfiles = (
  requestedProfiles: readonly ServingProfile[],
): Effect.Effect<void, LocalModelMutationFailed> => Effect.gen(function* () {
  if (requestedProfiles.length === 0) {
    return yield* failure(
      "invalid_model_assessment_request",
      "A model assessment request must contain at least one profile.",
    )
  }
  if (requestedProfiles.some((profile, index) =>
    requestedProfiles.slice(0, index).some((other) => sameProfile(profile, other)))) {
    return yield* failure(
      "invalid_model_assessment_request",
      "A model assessment request contains duplicate profiles.",
    )
  }
})

export const correlateLocalModelAssessmentProfiles = (
  requestedProfiles: readonly ServingProfile[],
  assessments: readonly LocalModelAssessment[],
): Effect.Effect<readonly LocalModelAssessment[], LocalModelMutationFailed> => Effect.gen(function* () {
  yield* validateRequestedProfiles(requestedProfiles)

  const remaining = [...assessments]
  const correlated: LocalModelAssessment[] = []
  for (const profile of requestedProfiles) {
    const matchingIndexes = remaining.flatMap((assessment, index) =>
      sameProfile(assessment.configuration.profile, profile) ? [index] : [])
    if (matchingIndexes.length !== 1) {
      return yield* failure(
        "invalid_model_assessment_response",
        matchingIndexes.length === 0
          ? `Native assessment returned no result for profile ${profile.contextLength}.`
          : `Native assessment returned duplicate results for profile ${profile.contextLength}.`,
      )
    }
    correlated.push(remaining[matchingIndexes[0]!]!)
    remaining.splice(matchingIndexes[0]!, 1)
  }
  if (remaining.length !== 0) {
    return yield* failure(
      "invalid_model_assessment_response",
      "Native assessment returned results for unrequested profiles.",
    )
  }
  return correlated
})

export interface LocalModelAssessmentsApi {
  readonly assess: (
    requests: readonly LocalModelAssessmentRequest[],
    onResult: (
      requestIndex: number,
      result: LocalModelAssessmentResult,
    ) => Effect.Effect<void>,
  ) => Effect.Effect<void, LocalInferenceError>
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
  const operationLock = yield* Effect.makeSemaphore(1)

  const assess: LocalModelAssessmentsApi["assess"] = (
    requests,
    onResult,
  ) => {
    const deadlineAtMs = Date.now() + ASSESSMENT_OPERATION_TIMEOUT_MS
    const operation = operationLock.withPermits(1)(Effect.gen(function* () {
      if (requests.length === 0) return
      const run = Effect.gen(function* () {
        const installedIds = yield* packages.installedPackageIds
        const nativeRequests = yield* Effect.forEach(
          requests,
          ({ bundle, profiles }, index) => Effect.gen(function* () {
            yield* validateRequestedProfiles(profiles)
            return {
              index,
              nativeBundle: yield* bundleToIcnInput(bundle, installedIds),
              profiles,
            }
          }),
        )
        const byRequestId = new Map(nativeRequests.map((request) => [
          `assessment-${request.index}`,
          request,
        ]))
        const response = yield* client.models.assessModels({
          payload: {
            requests: nativeRequests.map(({ index, nativeBundle, profiles }) => ({
              requestId: `assessment-${index}`,
              bundle: nativeBundle,
              profiles: profiles.map((profile) => ({
                profile: servingProfileToIcn(profile),
                performanceContextTokens: performanceSampleContextTokens(profile),
              })),
            })),
          },
        })
        let environmentId = Option.none<AssessmentEnvironmentId>()
        let completed = false
        const received = new Set<string>()
        yield* response.events.pipe(Stream.runForEach((event) => Effect.gen(function* () {
          if (event._tag === "Started") {
            if (Option.isSome(environmentId) || event.totalTargets !== nativeRequests.length) {
              return yield* failure(
                "invalid_model_assessment_response",
                "Native assessment returned an invalid start event.",
              )
            }
            environmentId = Option.some(AssessmentEnvironmentIdSchema.make(event.environmentId))
            return
          }
          if (event._tag === "Completed") {
            if (
              Option.isNone(environmentId)
              || completed
              || event.environmentId !== environmentId.value
              || event.totalTargets !== nativeRequests.length
              || received.size !== nativeRequests.length
            ) {
              return yield* failure(
                "invalid_model_assessment_response",
                "Native assessment completed without exactly one result for every request.",
              )
            }
            completed = true
            return
          }
          const requestId = String(event.result.requestId)
          const request = byRequestId.get(requestId)
          if (
            Option.isNone(environmentId)
            || completed
            || request === undefined
            || received.has(requestId)
          ) {
            return yield* failure(
              "invalid_model_assessment_response",
              "Native assessment returned a result outside the declared request set.",
            )
          }
          received.add(requestId)
          const decoded = yield* localModelAssessmentResultFromIcn(
            event.result,
            environmentId.value,
          ).pipe(Effect.orDie)
          const result = decoded._tag !== "Assessed"
            ? decoded
            : {
                ...decoded,
                assessments: yield* correlateLocalModelAssessmentProfiles(
                  request.profiles,
                  decoded.assessments,
                ),
              }
          yield* onResult(request.index, result)
        })))
        if (!completed) {
          return yield* failure(
            "incomplete_model_assessment_response",
            "Native assessment ended before all requested models were assessed.",
          )
        }
      })
      return yield* run
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

  return LocalModelAssessments.of({ assess })
}))
