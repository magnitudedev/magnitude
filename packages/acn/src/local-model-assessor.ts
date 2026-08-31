import { Cause, Context, Data, Duration, Effect, Layer, Option, Ref, Schema, Stream, SubscriptionRef } from "effect"
import type {
  AssessModelsEvent,
  ModelAssessment,
  ModelAssessmentSubject,
  ReadyModel,
  ServingProfile,
} from "@magnitudedev/icn-protocol/schemas"
import { IcnClient, IcnHardware } from "@magnitudedev/icn"
import {
  LocalModelAssessmentSchema,
  localModelPerformanceContexts,
  type LocalModelAssessment,
  type ModelFailure,
  type ModelId,
} from "@magnitudedev/acn-protocol"
import { LocalModelSources, type CatalogModelSource, type LocalModelSourcesState } from "./local-model-sources"

export type CoordinatedLocalModelAssessment =
  | { readonly _tag: "Assessing" }
  | { readonly _tag: "Assessed"; readonly assessment: LocalModelAssessment }
  | { readonly _tag: "Failed"; readonly failure: ModelFailure }
export interface LocalModelAssessmentSnapshot {
  readonly assessments: ReadonlyMap<ModelId, CoordinatedLocalModelAssessment>
}
export interface LocalModelAssessorApi {
  readonly snapshot: Effect.Effect<LocalModelAssessmentSnapshot>
  readonly changes: Stream.Stream<LocalModelAssessmentSnapshot>
}
export class LocalModelAssessor extends Context.Tag("LocalModelAssessor")<LocalModelAssessor, LocalModelAssessorApi>() {}

export interface CatalogAssessmentDemand {
  readonly entry: CatalogModelSource
  readonly requestId: string
  readonly selection: "Desired" | "Effective"
  readonly ready: ReadyModel
}

export const catalogAssessmentDemands = (
  state: LocalModelSourcesState,
): readonly CatalogAssessmentDemand[] => state.catalogModels.flatMap<CatalogAssessmentDemand>((entry, index) => {
  const local = entry.source.localState
  if (local._tag === "NotInstalled") return [{
    entry, requestId: `catalog-${state.catalogRevision}-${index}`,
    selection: "Desired", ready: entry.source.desired,
  }]
  return local.effective._tag === "Ready" ? [{
    entry, requestId: `catalog-${state.catalogRevision}-${index}`,
    selection: "Effective", ready: local.effective.model,
  }] : []
})

class InvalidAssessmentResponse extends Data.TaggedError("InvalidAssessmentResponse")<{
  readonly message: string
}> {}

const projectAssessment = (environmentId: string, assessment: ModelAssessment): LocalModelAssessment => {
  if (assessment._tag === "Fits") {
    const totalRequiredBytes = assessment.memory.reduce((total, item) => total + item.requiredBytes, 0)
    const requiredSystemMemoryBytes = assessment.memory.filter((item) => item.memoryDomainId === "system")
      .reduce((total, item) => total + item.requiredBytes, 0)
    return Schema.decodeUnknownSync(LocalModelAssessmentSchema)({
      _tag: "Fits", assessmentId: assessment.assessmentId, environmentId,
      profile: assessment.profile,
      memory: {
        domains: assessment.memory, totalRequiredBytes, requiredSystemMemoryBytes,
        systemUseState: { _tag: "NotObserved" },
        currentHeadroomState: { _tag: "NotObserved" },
      },
      performance: assessment.performance,
    })
  }
  if (assessment._tag === "DoesNotFit") {
    return Schema.decodeUnknownSync(LocalModelAssessmentSchema)({
      _tag: "DoesNotFit", assessmentId: assessment.assessmentId, environmentId,
      profile: assessment.profile,
      memoryDomains: assessment.memory,
      totalRequiredBytes: assessment.memory.reduce((total, item) => total + item.requiredBytes, 0),
      deficitBytes: assessment.deficitBytes, limitingResource: assessment.limitingResource,
    })
  }
  return Schema.decodeUnknownSync(LocalModelAssessmentSchema)({
    _tag: "Incompatible", environmentId, profile: assessment.profile, failure: assessment.failure,
  })
}

export interface AssessmentExpectation {
  readonly modelId: ModelId
  readonly subject: ModelAssessmentSubject
  readonly profiles: readonly {
    readonly profile: ServingProfile
    readonly performanceContextTokens: readonly number[]
  }[]
}

const invalidResponse = (message: string) => Effect.fail(new InvalidAssessmentResponse({ message }))

const sameSubject = (left: ModelAssessmentSubject, right: ModelAssessmentSubject): boolean =>
  left._tag === right._tag
  && left.modelId === right.modelId
  && (left._tag !== "Catalog" || right._tag === "Catalog" && left.selection === right.selection)

const sameProfiles = (
  expected: AssessmentExpectation["profiles"],
  actual: readonly ModelAssessment[],
): boolean => expected.length === actual.length
  && expected.every(({ profile, performanceContextTokens }, index) => {
    const assessment = actual[index]
    if (assessment?.profile.contextLength !== profile.contextLength) return false
    return assessment._tag !== "Fits"
      || performanceContextTokens.length === assessment.performance.length
        && performanceContextTokens.every((contextTokens, sampleIndex) =>
          assessment.performance[sampleIndex]?.contextTokens === contextTokens)
  })

export const consumeAssessmentEvents = (
  response: { readonly events: Stream.Stream<AssessModelsEvent, unknown> },
  revision: number,
  byRequest: ReadonlyMap<string, AssessmentExpectation>,
  publish: (modelId: ModelId, value: CoordinatedLocalModelAssessment) => Effect.Effect<void>,
  timeout: Duration.DurationInput = "2 minutes",
) => Effect.gen(function* () {
  let environmentId: string | undefined
  let completed = false
  const pending = new Set(byRequest.keys())
  const outcome = yield* response.events.pipe(Stream.runForEach((event) => Effect.gen(function* () {
    if (event._tag === "Started") {
      if (environmentId !== undefined || completed || event.revision !== revision
        || event.totalTargets !== byRequest.size) {
        return yield* invalidResponse("ICN returned an invalid model-assessment start event")
      }
      environmentId = event.environmentId
      return
    }
    if (event._tag === "Completed") {
      if (environmentId === undefined || completed || event.environmentId !== environmentId
        || event.revision !== revision || event.totalTargets !== byRequest.size || pending.size !== 0) {
        return yield* invalidResponse("ICN completed model assessment without exactly one result for every target")
      }
      completed = true
      return
    }
    const expected = byRequest.get(event.result.requestId)
    if (environmentId === undefined || completed || expected === undefined
      || !pending.has(event.result.requestId) || !sameSubject(expected.subject, event.result.subject)) {
      return yield* invalidResponse("ICN returned an uncorrelated or duplicate model-assessment result")
    }
    if (event.result._tag === "Assessed" && !sameProfiles(expected.profiles, event.result.profiles)) {
      return yield* invalidResponse("ICN returned assessments for profiles other than those requested")
    }
    pending.delete(event.result.requestId)
    if (event.result._tag !== "Assessed") {
      yield* publish(expected.modelId, { _tag: "Failed", failure: event.result.failure })
      return
    }
    const assessment = event.result.profiles[0]
    yield* assessment === undefined
      ? publish(expected.modelId, { _tag: "Failed", failure: {
          code: "missing_assessment", message: "ICN returned no model assessment", retryable: true,
        } })
      : publish(expected.modelId, { _tag: "Assessed", assessment: projectAssessment(environmentId, assessment) })
  })), Effect.timeout(timeout), Effect.exit)
  if (outcome._tag === "Success" && !completed) {
    const failure: ModelFailure = {
      code: "incomplete_assessment_response",
      message: "ICN ended model assessment before completing the declared operation",
      retryable: true,
    }
    const incomplete = [...pending].flatMap((requestId) => {
      const expected = byRequest.get(requestId)
      return expected === undefined ? [] : [expected.modelId]
    })
    yield* Effect.forEach(incomplete, (modelId) => publish(modelId, { _tag: "Failed", failure }), {
      discard: true,
    })
    return
  }
  if (outcome._tag === "Failure") {
    const protocolFailure = Option.getOrUndefined(Cause.failureOption(outcome.cause))
    if (protocolFailure instanceof InvalidAssessmentResponse) {
      const failure: ModelFailure = {
        code: "invalid_assessment_response",
        message: protocolFailure.message,
        retryable: true,
      }
      yield* Effect.forEach(byRequest.values(), ({ modelId }) => publish(modelId, { _tag: "Failed", failure }), {
        discard: true,
      })
      return
    }
  }
  const incomplete = [...pending].flatMap((requestId) => {
    const expected = byRequest.get(requestId)
    return expected === undefined ? [] : [expected.modelId]
  })
  if (outcome._tag === "Failure") {
    const failure: ModelFailure = {
      code: "assessment_stream_failed",
      message: Cause.pretty(outcome.cause),
      retryable: true,
    }
    yield* Effect.forEach(incomplete, (modelId) => publish(modelId, { _tag: "Failed", failure }), {
      discard: true,
    })
    return
  }
  yield* Effect.forEach(incomplete, (modelId) => publish(modelId, { _tag: "Failed", failure: {
    code: "missing_assessment_result",
    message: "ICN ended model assessment without returning a result",
    retryable: true,
  } }), { discard: true })
})

export const LocalModelAssessorLive: Layer.Layer<LocalModelAssessor, never, LocalModelSources | IcnClient | IcnHardware> =
  Layer.scoped(LocalModelAssessor, Effect.gen(function* () {
    const sources = yield* LocalModelSources
    const client = yield* IcnClient
    const hardware = yield* IcnHardware
    const current = yield* SubscriptionRef.make<LocalModelAssessmentSnapshot>({ assessments: new Map() })
    const latestCycle = yield* Ref.make(0)
    const assess = (state: LocalModelSourcesState) => Effect.gen(function* () {
      const cycle = yield* Ref.updateAndGet(latestCycle, (value) => value + 1)
      const catalogTargets = catalogAssessmentDemands(state)
      const discoveryTargets = state.discoveredModels.flatMap((entry, index) => entry.source.state._tag === "Ready"
        ? [{ entry, requestId: `discovery-${state.discoveryRevision}-${index}`, ready: entry.source.state.model }]
        : [])
      const isCurrent = Effect.all([sources.state, Ref.get(latestCycle)]).pipe(Effect.map(([latest, currentCycle]) =>
        currentCycle === cycle
        && latest.catalogRevision === state.catalogRevision
        && latest.discoveryRevision === state.discoveryRevision))
      const publish = (modelId: ModelId, value: CoordinatedLocalModelAssessment) =>
        Effect.if(isCurrent, {
          onTrue: () => SubscriptionRef.update(current, ({ assessments }) => ({
            assessments: new Map(assessments).set(modelId, value),
          })),
          onFalse: () => Effect.void,
        })
      yield* Effect.if(isCurrent, {
        onTrue: () => SubscriptionRef.set(current, { assessments: new Map(
          [...catalogTargets, ...discoveryTargets].map(({ entry }) =>
            [entry.id, { _tag: "Assessing" } as const]),
        ) }),
        onFalse: () => Effect.void,
      })
      const catalogByRequest = new Map(catalogTargets.map((target) => [target.requestId, {
        modelId: target.entry.id,
        subject: { _tag: "Catalog" as const, modelId: target.entry.id, selection: target.selection },
        profiles: [{
          profile: target.ready.profile,
          performanceContextTokens: localModelPerformanceContexts(target.ready.profile.contextLength),
        }],
      }]))
      const discoveryByRequest = new Map(discoveryTargets.map((target) => [target.requestId, {
        modelId: target.entry.id,
        subject: { _tag: "Discovery" as const, modelId: target.entry.id },
        profiles: [{
          profile: target.ready.profile,
          performanceContextTokens: localModelPerformanceContexts(target.ready.profile.contextLength),
        }],
      }]))
      const failRequest = (
        targets: readonly { readonly entry: { readonly id: ModelId } }[],
        cause: Cause.Cause<unknown>,
      ) => Effect.forEach(targets, ({ entry }) => publish(entry.id, { _tag: "Failed", failure: {
        code: "assessment_request_failed",
        message: Cause.pretty(cause),
        retryable: true,
      } }), { discard: true })
      const catalogEffect = catalogTargets.length === 0 ? Effect.void : client.catalog.assessCatalogModels({ payload: {
        revision: state.catalogRevision,
        targets: catalogTargets.map(({ entry, requestId, selection, ready }) => ({ requestId,
          modelId: entry.id, selection,
          profiles: [{ profile: ready.profile, performanceContextTokens: localModelPerformanceContexts(ready.profile.contextLength) }],
        })),
      } }).pipe(
        Effect.flatMap((response) => consumeAssessmentEvents(response, state.catalogRevision, catalogByRequest, publish)),
        Effect.catchAllCause((cause) => failRequest(catalogTargets, cause)),
      )
      const discoveryEffect = discoveryTargets.length === 0 ? Effect.void : client.discovery.assessDiscoveredModels({ payload: {
        revision: state.discoveryRevision,
        targets: discoveryTargets.map(({ entry, requestId, ready }) => ({ requestId,
          modelId: entry.id,
          profiles: [{ profile: ready.profile, performanceContextTokens: localModelPerformanceContexts(ready.profile.contextLength) }],
        })),
      } }).pipe(
        Effect.flatMap((response) => consumeAssessmentEvents(response, state.discoveryRevision, discoveryByRequest, publish)),
        Effect.catchAllCause((cause) => failRequest(discoveryTargets, cause)),
      )
      yield* Effect.all([catalogEffect, discoveryEffect], { concurrency: 2, discard: true })
    }).pipe(Effect.catchAllCause((cause) => Effect.logWarning("Local model assessment failed").pipe(
      Effect.annotateLogs({ cause: String(cause) }),
    )))
    // The observed-state stream begins with its current snapshot; the explicit source cycle below
    // already assesses that environment. Only subsequent assessment-relevant hardware changes
    // should start another cycle.
    const hardwareAssessmentCycles = hardware.assessmentChanges.pipe(
      Stream.drop(1),
      Stream.mapEffect(() => sources.state),
    )
    yield* Stream.merge(
      Stream.concat(Stream.fromEffect(sources.state), sources.changes),
      hardwareAssessmentCycles,
    ).pipe(
      Stream.mapEffect(assess, { concurrency: "unbounded" }),
      Stream.runDrain,
      Effect.forkScoped,
    )
    return LocalModelAssessor.of({ snapshot: SubscriptionRef.get(current), changes: current.changes })
  }))
