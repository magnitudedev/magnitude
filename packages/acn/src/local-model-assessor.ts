import { createHash } from "node:crypto"
import { Cause, Context, Effect, Exit, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelConfigurationAssessmentSchema,
  ModelServingConfigurationSchema,
  ServableModelBundleSchema,
  ServingProfileSchema,
  servableModelBundlePackageIds,
  type ModelPackageId,
  type LocalModelConfigurationAssessment,
  type ModelServingConfiguration,
  type ModelServingConfigurationId,
  type ServableModelBundle,
  type ServingProfile,
} from "@magnitudedev/acn-protocol"
import { IcnCatalog, IcnHardware } from "@magnitudedev/icn"
import {
  LocalModelAssessments,
  localModelAssessmentProfiles,
  type LocalModelAssessmentResult,
} from "./local-model-assessments"
import { recommendableModelFromIcn } from "./local-model-icn-adapter"
import {
  configuredModelPackageIds,
  isStandalonePackageCandidate,
} from "./local-model-configuration-resolver"
import { LocalModelPackages } from "./local-model-packages"
import { makeObservedState } from "./mirrored-state"

const CoordinatedLocalModelAssessmentStateSchema = Schema.Union(
  Schema.TaggedStruct("Assessing", {}),
  LocalModelConfigurationAssessmentSchema,
)
type CoordinatedLocalModelAssessmentState =
  typeof CoordinatedLocalModelAssessmentStateSchema.Type

export interface CoordinatedLocalModelAssessment {
  readonly configuration: ModelServingConfiguration
  readonly origin: "Authored" | "Standard"
  readonly assessment: CoordinatedLocalModelAssessmentState
}

export interface LocalModelAssessorApi {
  readonly state: Effect.Effect<ReadonlyMap<ModelServingConfigurationId, CoordinatedLocalModelAssessment>>
  readonly changes: Stream.Stream<ReadonlyMap<ModelServingConfigurationId, CoordinatedLocalModelAssessment>>
}

export class LocalModelAssessor extends Context.Tag(
  "LocalModelAssessor",
)<LocalModelAssessor, LocalModelAssessorApi>() {}

type AssessmentDemandKey = string

type DesiredAssessment = {
  readonly _tag: "Authored"
  readonly configuration: ModelServingConfiguration
  readonly semanticKey: string
} | {
  readonly _tag: "Standard"
  readonly bundle: ServableModelBundle
  readonly profile: ServingProfile
  readonly semanticKey: string
}

interface AssessorState {
  readonly desired: ReadonlyMap<AssessmentDemandKey, DesiredAssessment>
  readonly published: ReadonlyMap<AssessmentDemandKey, CoordinatedLocalModelAssessment>
  readonly completedKeys: ReadonlyMap<AssessmentDemandKey, string>
}

const configurationEquivalent = Schema.equivalence(ModelServingConfigurationSchema)
const assessmentEquivalent = Schema.equivalence(CoordinatedLocalModelAssessmentStateSchema)

const publishedEquivalent = (
  left: ReadonlyMap<AssessmentDemandKey, CoordinatedLocalModelAssessment>,
  right: ReadonlyMap<AssessmentDemandKey, CoordinatedLocalModelAssessment>,
): boolean => left.size === right.size && [...left].every(([id, value]) => {
  const other = right.get(id)
  return other !== undefined
    && configurationEquivalent(value.configuration, other.configuration)
    && value.origin === other.origin
    && assessmentEquivalent(value.assessment, other.assessment)
})

const sameDesired = (
  left: ReadonlyMap<AssessmentDemandKey, DesiredAssessment>,
  right: ReadonlyMap<AssessmentDemandKey, DesiredAssessment>,
): boolean => left.size === right.size
  && [...left].every(([id, value]) => {
    const other = right.get(id)
    return other?.semanticKey === value.semanticKey && other._tag === value._tag
  })

const sameCompleted = (
  left: ReadonlyMap<AssessmentDemandKey, string>,
  right: ReadonlyMap<AssessmentDemandKey, string>,
): boolean => left.size === right.size
  && [...left].every(([id, key]) => right.get(id) === key)

const assessmentFailure = (error: unknown) => {
  const record = typeof error === "object" && error !== null
    ? error as Record<string, unknown>
    : undefined
  return {
    code: typeof record?.code === "string"
      ? record.code
      : "local_model_assessment_failed",
    message: error instanceof Error ? error.message : String(error),
    retryable: typeof record?.retryable === "boolean" ? record.retryable : true,
  }
}

const assessmentCauseFailure = (cause: Cause.Cause<unknown>) => Option.match(
  Cause.failureOption(cause),
  {
    onNone: () => ({
      code: "local_model_assessment_defect",
      message: Cause.pretty(cause),
      retryable: true,
    }),
    onSome: assessmentFailure,
  },
)

const standardDemandKey = (
  bundle: ServableModelBundle,
  profile: ServingProfile,
): AssessmentDemandKey => bundle._tag === "Standalone"
  ? `Standard\0Standalone\0${bundle.package.id}\0${profile.contextLength}`
  : `Standard\0${bundleDemandKey(bundle)}\0${profile.contextLength}`

const authoredDemandKey = (
  configuration: ModelServingConfiguration,
): AssessmentDemandKey => `Authored\0${configuration.id}`

const bundleDemandKey = (bundle: ServableModelBundle): string =>
  bundle._tag === "Standalone"
    ? `Standalone\0${bundle.package.id}`
    : bundle.draftSource._tag === "Embedded"
      ? `SpeculativeDecoding\0${bundle.target.id}\0Embedded\0${JSON.stringify(bundle.method)}`
      : `SpeculativeDecoding\0${bundle.target.id}\0Separate\0${bundle.draftSource.draft.id}\0${JSON.stringify(bundle.method)}`

const sameBundle = Schema.equivalence(ServableModelBundleSchema)
const sameProfile = Schema.equivalence(ServingProfileSchema)

const completedAssessment = (
  result: LocalModelAssessmentResult,
  request: DesiredAssessment,
): { readonly configuration: ModelServingConfiguration; readonly assessment: LocalModelConfigurationAssessment }
  | undefined => {
  if (result._tag === "InvalidBundle") {
    return request._tag === "Authored" ? {
      configuration: request.configuration,
      assessment: {
        _tag: "Failed",
        failure: {
          code: "invalid_model_bundle",
          message: result.message,
          retryable: false,
        },
      },
    } : undefined
  }
  if (result._tag === "Failed") {
    return request._tag === "Authored" ? {
      configuration: request.configuration,
      assessment: { _tag: "Failed", failure: result.failure },
    } : undefined
  }
  const resultForConfiguration = result.assessments.find(({ configuration }) =>
    request._tag === "Authored"
      ? configurationEquivalent(configuration, request.configuration)
      : sameBundle(configuration.bundle, request.bundle)
        && sameProfile(configuration.profile, request.profile))
  if (resultForConfiguration === undefined) {
    return request._tag === "Authored" ? {
      configuration: request.configuration,
      assessment: {
        _tag: "Failed",
        failure: {
          code: "model_assessment_configuration_mismatch",
          message: `Native assessment did not return configuration ${request.configuration.id}`,
          retryable: true,
        },
      },
    } : undefined
  }
  if (resultForConfiguration._tag === "Fits") {
    return {
      configuration: resultForConfiguration.configuration,
      assessment: { _tag: "Fits", assessment: resultForConfiguration.assessment },
    }
  }
  if (resultForConfiguration._tag === "DoesNotFit") {
    return {
      configuration: resultForConfiguration.configuration,
      assessment: {
        _tag: "DoesNotFit",
        assessmentId: resultForConfiguration.assessmentId,
        environmentId: result.environmentId,
        memory: resultForConfiguration.memory,
        totalRequiredBytes: resultForConfiguration.memory.reduce(
          (total, memory) => total + memory.requiredBytes,
          0,
        ),
        deficitBytes: resultForConfiguration.deficitBytes,
        limitingResource: resultForConfiguration.limitingResource,
      },
    }
  }
  return {
    configuration: resultForConfiguration.configuration,
    assessment: {
      _tag: "Incompatible",
      environmentId: result.environmentId,
      failure: resultForConfiguration.failure,
    },
  }
}

export const LocalModelAssessorLive: Layer.Layer<
  LocalModelAssessor,
  never,
  IcnCatalog | IcnHardware | LocalModelAssessments | LocalModelPackages
> = Layer.scoped(LocalModelAssessor, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const hardware = yield* IcnHardware
  const assessments = yield* LocalModelAssessments
  const packages = yield* LocalModelPackages
  const observed = yield* makeObservedState<AssessorState>({
    desired: new Map(),
    published: new Map(),
    completedKeys: new Map(),
  })
  const lock = yield* Effect.makeSemaphore(1)

  const readDesired = Effect.gen(function* () {
    const catalogConfigurations = (yield* catalog.ready)
      ? yield* Effect.forEach(
          (yield* catalog.get).state.models,
          recommendableModelFromIcn,
        ).pipe(Effect.map((models) => models.map(({ configuration }) => configuration)))
      : []
    const packageState = (yield* packages.snapshot).state
    const packageEntries = new Map(packageState.entries.map((entry) => [entry.package.id, entry]))
    const hardwareState = (yield* hardware.get).state
    const configuredPackages = configuredModelPackageIds(catalogConfigurations)

    const hardwareEvidence = {
      nativeBuild: hardwareState.native_build,
      topology: hardwareState.topology_fingerprint,
      totalSystemMemoryBytes: hardwareState.system_memory.physical_capacity_bytes,
      assessmentReserveBytes: hardwareState.system_memory.assess_reserve_bytes,
      backends: [...hardwareState.enabled_backends],
    }
    const desired = new Map<AssessmentDemandKey, DesiredAssessment>()
    for (const configuration of catalogConfigurations) {
      const packageEvidence = servableModelBundlePackageIds(configuration.bundle).map((packageId) => {
        const entry = packageEntries.get(packageId)
        return {
          packageId,
          installed: entry?.localState._tag === "Installed",
          inspection: entry?.inspection._tag ?? "Pending",
        }
      })
      const semanticInput = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({
        configuration: yield* Schema.encode(ModelServingConfigurationSchema)(configuration),
        hardware: hardwareEvidence,
        material: packageEvidence,
      })
      desired.set(authoredDemandKey(configuration), {
        _tag: "Authored",
        configuration,
        semanticKey: createHash("sha256").update(semanticInput).digest("hex"),
      })
    }
    for (const entry of packageState.entries) {
      if (entry.localState._tag !== "Installed" || entry.inspection._tag !== "Inspected") continue
      if (!isStandalonePackageCandidate(entry.package, configuredPackages)) continue
      const bundle: ServableModelBundle = { _tag: "Standalone", package: entry.package }
      for (const profile of localModelAssessmentProfiles(bundle)) {
        const packageEvidence = [{
          packageId: entry.package.id,
          installed: true,
          inspection: "Inspected",
        }]
        const semanticInput = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({
          bundle: yield* Schema.encode(ServableModelBundleSchema)(bundle),
          profile: yield* Schema.encode(ServingProfileSchema)(profile),
          hardware: hardwareEvidence,
          material: packageEvidence,
        })
        desired.set(standardDemandKey(bundle, profile), {
          _tag: "Standard",
          bundle,
          profile,
          semanticKey: createHash("sha256").update(semanticInput).digest("hex"),
        })
      }
    }
    return desired
  })

  const publish = (state: AssessorState) => observed.setIfChanged(
    state,
    (left, right) => publishedEquivalent(left.published, right.published)
      && sameDesired(left.desired, right.desired)
      && sameCompleted(left.completedKeys, right.completedKeys),
  )

  const reconcile = lock.withPermits(1)(Effect.gen(function* () {
    const desired = yield* readDesired
    const current = (yield* observed.get).state
    const pending = [...desired].filter(([demandKey, { semanticKey }]) =>
      current.completedKeys.get(demandKey) !== semanticKey)
    const published = new Map(
      [...current.published].filter(([demandKey]) => desired.has(demandKey)),
    )
    for (const [demandKey, entry] of published) {
      const next = desired.get(demandKey)
      if (next?._tag === "Authored") {
        published.set(demandKey, {
          ...entry,
          configuration: next.configuration,
          origin: "Authored",
        })
      }
    }
    for (const [demandKey, request] of pending) {
      if (request._tag === "Authored") {
        published.set(demandKey, {
          configuration: request.configuration,
          origin: "Authored",
          assessment: { _tag: "Assessing" },
        })
      } else {
        published.delete(demandKey)
      }
    }
    yield* publish({ ...current, desired, published })
    if (pending.length === 0) return

    const outcome = yield* Effect.exit(assessments.assess(
      pending.map(([, request]) => ({
        bundle: request._tag === "Authored" ? request.configuration.bundle : request.bundle,
        profiles: [request._tag === "Authored" ? request.configuration.profile : request.profile],
      })),
      () => Effect.void,
    ))
    const latestDesired = yield* readDesired
    const latest = (yield* observed.get).state
    const nextPublished = new Map(latest.published)
    const completedKeys = new Map(latest.completedKeys)
    if (Exit.isFailure(outcome)) {
      const failure = assessmentCauseFailure(outcome.cause)
      for (const [demandKey, request] of pending) {
        const latestRequest = latestDesired.get(demandKey)
        if (latestRequest?.semanticKey !== request.semanticKey) continue
        if (latestRequest._tag === "Authored") {
          nextPublished.set(demandKey, {
            configuration: latestRequest.configuration,
            origin: "Authored",
            assessment: { _tag: "Failed", failure },
          })
          completedKeys.set(demandKey, request.semanticKey)
        }
      }
    } else {
      pending.forEach(([demandKey, request], index) => {
        const latestRequest = latestDesired.get(demandKey)
        if (latestRequest?.semanticKey !== request.semanticKey) return
        const result = outcome.value[index]
        const completed = result === undefined
          ? latestRequest._tag === "Authored" ? {
              configuration: latestRequest.configuration,
              assessment: {
                _tag: "Failed",
                failure: {
                  code: "missing_model_assessment_result",
                  message: "Native assessment returned no result for this configuration",
                  retryable: true,
                },
              } as const,
            } : undefined
          : completedAssessment(result, latestRequest)
        if (completed === undefined) return
        nextPublished.set(demandKey, {
          configuration: completed.configuration,
          origin: latestRequest._tag === "Authored" ? "Authored" : "Standard",
          assessment: completed.assessment,
        })
        completedKeys.set(demandKey, request.semanticKey)
      })
    }
    for (const demandKey of nextPublished.keys()) {
      if (!latestDesired.has(demandKey)) nextPublished.delete(demandKey)
    }
    for (const demandKey of completedKeys.keys()) {
      if (!latestDesired.has(demandKey)) completedKeys.delete(demandKey)
    }
    yield* publish({ desired: latestDesired, published: nextPublished, completedKeys })
  })).pipe(
    Effect.catchAllCause((cause) => Effect.logWarning(
      "Unable to coordinate local model assessment",
    ).pipe(Effect.annotateLogs({ cause: String(cause) }))),
  )

  yield* Stream.make(undefined).pipe(
    Stream.concat(Stream.mergeAll([
      catalog.changes.pipe(Stream.map(() => undefined)),
      packages.changes.pipe(Stream.map(() => undefined)),
      hardware.assessmentChanges.pipe(Stream.map(() => undefined)),
    ], { concurrency: "unbounded" }).pipe(Stream.debounce("25 millis"))),
    Stream.runForEach(() => reconcile),
    Effect.forkScoped,
  )

  const publicState = (published: AssessorState["published"]) => new Map(
    [...published.values()].map((entry) => [entry.configuration.id, entry]),
  )

  return LocalModelAssessor.of({
    state: observed.get.pipe(Effect.map(({ state }) => publicState(state.published))),
    changes: observed.changes.pipe(Stream.map(({ state }) => publicState(state.published))),
  })
}))
