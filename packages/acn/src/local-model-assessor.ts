import { Cause, Context, Effect, Exit, Layer, Option, Schema, Stream, SubscriptionRef } from "effect"
import {
  LocalModelConfigurationAssessmentSchema,
  ModelServingConfigurationSchema,
  servableModelBundlePackageIds,
  type ModelPackageId,
  type LocalModelConfigurationAssessment,
  type ModelServingConfiguration,
  type ServableModelBundle,
} from "@magnitudedev/acn-protocol"
import { IcnHardware } from "@magnitudedev/icn"
import {
  LocalModelAssessments,
  type LocalModelAssessmentResult,
} from "./local-model-assessments"
import { LocalModelCatalogAdapter } from "./local-model-catalog-adapter"
import { LocalModelPackages } from "./local-model-packages"

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
  readonly runningKeys: ReadonlyMap<AssessmentDemandKey, string>
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

const sameRunning = sameCompleted

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

const assessmentExecutionKey = (
  configuration: ModelServingConfiguration,
): string => `${bundleExecutionKey(configuration.bundle)}\0${configuration.profile.contextLength}`

const bundleExecutionKey = (bundle: ServableModelBundle): string =>
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
  const configuration = request.configuration
  const resultForConfiguration = result.assessments[0]
  if (resultForConfiguration === undefined) {
    return {
      configuration,
      assessment: {
        _tag: "Failed",
        failure: {
          code: "missing_model_profile_assessment",
          message: "Native assessment returned no result for the requested serving profile",
          retryable: true,
        },
      },
    }
  }
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
  LocalModelCatalogAdapter | IcnHardware | LocalModelAssessments | LocalModelPackages
> = Layer.scoped(LocalModelAssessor, Effect.gen(function* () {
  const catalog = yield* LocalModelCatalogAdapter
  const hardware = yield* IcnHardware
  const assessments = yield* LocalModelAssessments
  const packages = yield* LocalModelPackages
  const current = yield* SubscriptionRef.make<AssessorState>({
    desired: new Map(),
    published: new Map(),
    completedKeys: new Map(),
    runningKeys: new Map(),
  })
  const lock = yield* Effect.makeSemaphore(1)

  const readDesired = Effect.gen(function* () {
    const catalogEntries = (yield* catalog.state).entries
    const desiredCatalogConfigurations = catalogEntries.map(({ model }) => model.configuration)
    const effectiveCatalogConfigurations = catalogEntries.flatMap(({ effectiveConfiguration }) =>
      Option.isSome(effectiveConfiguration) ? [effectiveConfiguration.value] : [])
    const catalogConfigurations = [
      ...desiredCatalogConfigurations,
      ...effectiveCatalogConfigurations,
    ]
    const packageState = yield* packages.state
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
      const configurationKey = yield* Schema.encode(
        Schema.parseJson(ModelServingConfigurationSchema),
      )(configuration)
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
      desired.set(configurationKey, {
        configuration,
        semanticKey: semanticInput,
      })
    }
    return desired
  })

  const publish = (state: AssessorState) => Effect.gen(function* () {
    const previous = yield* SubscriptionRef.get(current)
    const equivalent = publishedEquivalent(previous.published, state.published)
      && sameDesired(previous.desired, state.desired)
      && sameCompleted(previous.completedKeys, state.completedKeys)
      && sameRunning(previous.runningKeys, state.runningKeys)
    if (!equivalent) yield* SubscriptionRef.set(current, state)
  })

  type PendingAssessment = readonly [AssessmentDemandKey, DesiredAssessment]
  type AssessmentBatch = readonly [PendingAssessment, ...PendingAssessment[]]

  const batchesFor = (pending: readonly PendingAssessment[]): readonly AssessmentBatch[] => {
    const pendingBatches = new Map<string, AssessmentBatch>()
    for (const entry of pending) {
      const key = assessmentExecutionKey(entry[1].configuration)
      const batch = pendingBatches.get(key)
      pendingBatches.set(key, batch === undefined ? [entry] : [...batch, entry])
    }
    return [...pendingBatches.values()]
  }

  const complete = (
    pending: readonly PendingAssessment[],
    batches: readonly AssessmentBatch[],
    assessmentExit: Exit.Exit<readonly LocalModelAssessmentResult[], unknown>,
  ) => lock.withPermits(1)(Effect.gen(function* () {
    const latestDesired = yield* readDesired
    const latest = yield* SubscriptionRef.get(current)
    const nextPublished = new Map(latest.published)
    const completedKeys = new Map(latest.completedKeys)
    const runningKeys = new Map(latest.runningKeys)
    for (const [demandKey, request] of pending) {
      if (runningKeys.get(demandKey) === request.semanticKey) runningKeys.delete(demandKey)
    }
    if (Exit.isFailure(assessmentExit)) {
      const failure = assessmentCauseFailure(assessmentExit.cause)
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
      batches.forEach((batch, index) => {
        const result = assessmentExit.value[index]
        batch.forEach(([demandKey, request]) => {
          const latestRequest = latestDesired.get(demandKey)
          if (latestRequest?.semanticKey !== request.semanticKey) return
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
      })
    }
    for (const demandKey of nextPublished.keys()) {
      if (!latestDesired.has(demandKey)) nextPublished.delete(demandKey)
    }
    for (const demandKey of completedKeys.keys()) {
      if (!latestDesired.has(demandKey)) completedKeys.delete(demandKey)
    }
    for (const demandKey of runningKeys.keys()) {
      if (!latestDesired.has(demandKey)) runningKeys.delete(demandKey)
    }
    yield* publish({
      desired: latestDesired,
      published: nextPublished,
      completedKeys,
      runningKeys,
    })
  }))

  const assessPending = (
    pending: readonly PendingAssessment[],
    batches: readonly AssessmentBatch[],
  ) => Effect.exit(assessments.assess(
    batches.map((batch) => ({
      bundle: batch[0][1].configuration.bundle,
      profiles: [batch[0][1].configuration.profile],
    })),
    () => Effect.void,
  )).pipe(Effect.flatMap((outcome) => complete(pending, batches, outcome)))

  const reconcile = lock.withPermits(1)(Effect.gen(function* () {
    const desired = yield* readDesired
    const state = yield* SubscriptionRef.get(current)
    const pending = [...desired].filter(([demandKey, { semanticKey }]) =>
      state.completedKeys.get(demandKey) !== semanticKey
        && state.runningKeys.get(demandKey) !== semanticKey)
    const published = new Map(
      [...state.published].filter(([demandKey]) => desired.has(demandKey)),
    )
    const runningKeys = new Map(
      [...state.runningKeys].filter(([demandKey]) => desired.has(demandKey)),
    )
    for (const [demandKey, entry] of published) {
      const next = desired.get(demandKey)
      if (next !== undefined) published.set(demandKey, { ...entry, configuration: next.configuration })
    }
    for (const [demandKey, request] of pending) {
      published.set(demandKey, {
        configuration: request.configuration,
        assessment: { _tag: "Assessing" },
      })
      runningKeys.set(demandKey, request.semanticKey)
    }
    yield* publish({ ...state, desired, published, runningKeys })
    return pending
  })).pipe(
    Effect.flatMap((pending) => pending.length === 0
      ? Effect.void
      : assessPending(pending, batchesFor(pending)).pipe(Effect.forkScoped, Effect.asVoid)),
    Effect.catchAllCause((cause) => Effect.logWarning(
      "Unable to coordinate local model assessment",
    ).pipe(Effect.annotateLogs({ cause: String(cause) }))),
  )

  const changes = Stream.mergeAll([
    catalog.changes.pipe(Stream.map(() => undefined)),
    packages.changes.pipe(Stream.map(() => undefined)),
    hardware.assessmentChanges.pipe(Stream.map(() => undefined)),
  ], { concurrency: "unbounded" }).pipe(
    Stream.debounce("25 millis"),
  )
  yield* reconcile
  yield* changes.pipe(
    Stream.buffer({ capacity: 1, strategy: "sliding" }),
    Stream.runForEach(() => reconcile),
    Effect.forkScoped,
  )

  const publicState = (published: AssessorState["published"]) => [...published.values()]

  return LocalModelAssessor.of({
    state: SubscriptionRef.get(current).pipe(Effect.map((state) => publicState(state.published))),
    changes: current.changes.pipe(Stream.map((state) => publicState(state.published))),
    settled: Effect.gen(function* () {
      const desired = yield* readDesired
      const { completedKeys } = yield* SubscriptionRef.get(current)
      return [...desired].every(([demandKey, { semanticKey }]) =>
        completedKeys.get(demandKey) === semanticKey)
    }).pipe(Effect.orElseSucceed(() => false)),
  })
}))
