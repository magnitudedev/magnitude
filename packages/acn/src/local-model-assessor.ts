import { Cause, Context, Effect, Exit, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelConfigurationAssessmentSchema,
  ModelServingConfigurationSchema,
  servableModelBundlePackageIds,
  type ModelPackageId,
  type LocalModelConfigurationAssessment,
  type ModelServingConfiguration,
  type ServableModelBundle,
} from "@magnitudedev/acn-protocol"
import { IcnHardware, IcnModels } from "@magnitudedev/icn"
import {
  LocalModelAssessments,
  type LocalModelAssessmentResult,
} from "./local-model-assessments"
import {
  catalogModelDefinitionFromIcn,
  catalogModelEffectiveConfigurationFromIcn,
} from "./local-model-icn-adapter"
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
  readonly assessment: CoordinatedLocalModelAssessmentState
}

export interface LocalModelAssessorApi {
  readonly state: Effect.Effect<readonly CoordinatedLocalModelAssessment[]>
  readonly changes: Stream.Stream<readonly CoordinatedLocalModelAssessment[]>
  readonly settled: Effect.Effect<boolean>
}

export class LocalModelAssessor extends Context.Tag(
  "LocalModelAssessor",
)<LocalModelAssessor, LocalModelAssessorApi>() {}

type AssessmentDemandKey = string

type DesiredAssessment = {
  readonly configuration: ModelServingConfiguration
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
    && assessmentEquivalent(value.assessment, other.assessment)
})

const sameDesired = (
  left: ReadonlyMap<AssessmentDemandKey, DesiredAssessment>,
  right: ReadonlyMap<AssessmentDemandKey, DesiredAssessment>,
): boolean => left.size === right.size
  && [...left].every(([id, value]) => {
    const other = right.get(id)
    return other?.semanticKey === value.semanticKey
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

const authoredDemandKey = (
  configuration: ModelServingConfiguration,
): AssessmentDemandKey => `Authored\0${bundleDemandKey(configuration.bundle)}\0${configuration.profile.contextLength}`

const bundleDemandKey = (bundle: ServableModelBundle): string =>
  bundle._tag === "Standalone"
    ? `Standalone\0${bundle.package.id}`
    : bundle.draftSource._tag === "Embedded"
      ? `SpeculativeDecoding\0${bundle.target.id}\0Embedded\0${JSON.stringify(bundle.method)}`
      : `SpeculativeDecoding\0${bundle.target.id}\0Separate\0${bundle.draftSource.draft.id}\0${JSON.stringify(bundle.method)}`

const completedAssessment = (
  result: LocalModelAssessmentResult,
  request: DesiredAssessment,
): { readonly configuration: ModelServingConfiguration; readonly assessment: LocalModelConfigurationAssessment }
  | undefined => {
  if (result._tag === "InvalidBundle") {
    return {
      configuration: request.configuration,
      assessment: {
        _tag: "Failed",
        failure: {
          code: "invalid_model_bundle",
          message: result.message,
          retryable: false,
        },
      },
    }
  }
  if (result._tag === "Failed") {
    return {
      configuration: request.configuration,
      assessment: { _tag: "Failed", failure: result.failure },
    }
  }
  const resultForConfiguration = result.assessments[0]!
  const configuration = request.configuration
  if (resultForConfiguration._tag === "Fits") {
    return {
      configuration,
      assessment: {
        _tag: "Fits",
        assessment: {
          ...resultForConfiguration.assessment,
          profile: request.configuration.profile,
        },
      },
    }
  }
  if (resultForConfiguration._tag === "DoesNotFit") {
    return {
      configuration,
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
    configuration,
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
  IcnModels | IcnHardware | LocalModelAssessments | LocalModelPackages
> = Layer.scoped(LocalModelAssessor, Effect.gen(function* () {
  const models = yield* IcnModels
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
    const catalogModels = (yield* models.initialized)
      ? (yield* models.get).state.models
      : []
    const desiredCatalogConfigurations = yield* Effect.forEach(
      catalogModels,
      (model) => catalogModelDefinitionFromIcn(model).pipe(
        Effect.map(({ configuration }) => configuration),
      ),
    )
    const effectiveCatalogConfigurations = (
      yield* Effect.forEach(catalogModels, catalogModelEffectiveConfigurationFromIcn)
    ).flatMap((configuration) => Option.isSome(configuration) ? [configuration.value] : [])
    const catalogConfigurations = [
      ...desiredCatalogConfigurations,
      ...effectiveCatalogConfigurations,
    ]
    const packageState = (yield* packages.snapshot).state
    const packageEntries = new Map(packageState.entries.map((entry) => [entry.package.id, entry]))
    const hardwareState = (yield* hardware.get).state
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
        configuration,
        semanticKey: semanticInput,
      })
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
      if (next !== undefined) {
        published.set(demandKey, {
          ...entry,
          configuration: next.configuration,
        })
      }
    }
    for (const [demandKey, request] of pending) {
      published.set(demandKey, {
        configuration: request.configuration,
        assessment: { _tag: "Assessing" },
      })
    }
    yield* publish({ ...current, desired, published })
    if (pending.length === 0) return

    const outcome = yield* Effect.exit(assessments.assess(
      pending.map(([, request]) => ({
        bundle: request.configuration.bundle,
        profiles: [request.configuration.profile],
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
        nextPublished.set(demandKey, {
          configuration: latestRequest.configuration,
          assessment: { _tag: "Failed", failure },
        })
        completedKeys.set(demandKey, request.semanticKey)
      }
    } else {
      pending.forEach(([demandKey, request], index) => {
        const latestRequest = latestDesired.get(demandKey)
        if (latestRequest?.semanticKey !== request.semanticKey) return
        const result = outcome.value[index]
        const completed = result === undefined
          ? {
              configuration: latestRequest.configuration,
              assessment: {
                _tag: "Failed",
                failure: {
                  code: "missing_model_assessment_result",
                  message: "Native assessment returned no result for this configuration",
                  retryable: true,
                },
              } as const,
            }
          : completedAssessment(result, latestRequest)
        if (completed === undefined) return
        nextPublished.set(demandKey, {
          configuration: completed.configuration,
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
      models.changes.pipe(Stream.map(() => undefined)),
      packages.changes.pipe(Stream.map(() => undefined)),
      hardware.assessmentChanges.pipe(Stream.map(() => undefined)),
    ], { concurrency: "unbounded" }).pipe(Stream.debounce("25 millis"))),
    Stream.runForEach(() => reconcile),
    Effect.forkScoped,
  )

  const publicState = (published: AssessorState["published"]) => [...published.values()]

  return LocalModelAssessor.of({
    state: observed.get.pipe(Effect.map(({ state }) => publicState(state.published))),
    changes: observed.changes.pipe(Stream.map(({ state }) => publicState(state.published))),
    settled: Effect.gen(function* () {
      const desired = yield* readDesired
      const { completedKeys } = (yield* observed.get).state
      return [...desired].every(([demandKey, { semanticKey }]) =>
        completedKeys.get(demandKey) === semanticKey)
    }).pipe(Effect.orElseSucceed(() => false)),
  })
}))
