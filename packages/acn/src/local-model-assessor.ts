import { createHash } from "node:crypto"
import { Cause, Context, Effect, Exit, Layer, Option, Schema, Stream } from "effect"
import {
  LocalModelConfigurationAssessmentSchema,
  ModelServingConfigurationIdSchema,
  ModelServingConfigurationSchema,
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
import { LocalModelPackages } from "./local-model-packages"
import { makeObservedState } from "./mirrored-state"
import { RetainedModelConfigurations } from "./retained-model-configurations"

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
  readonly state: Effect.Effect<ReadonlyMap<ModelServingConfigurationId, CoordinatedLocalModelAssessment>>
  readonly changes: Stream.Stream<ReadonlyMap<ModelServingConfigurationId, CoordinatedLocalModelAssessment>>
}

export class LocalModelAssessor extends Context.Tag(
  "LocalModelAssessor",
)<LocalModelAssessor, LocalModelAssessorApi>() {}

interface DesiredAssessment {
  readonly configuration: ModelServingConfiguration
  readonly semanticKey: string
}

interface AssessorState {
  readonly desired: ReadonlyMap<ModelServingConfigurationId, DesiredAssessment>
  readonly published: ReadonlyMap<ModelServingConfigurationId, CoordinatedLocalModelAssessment>
  readonly completedKeys: ReadonlyMap<ModelServingConfigurationId, string>
}

const configurationEquivalent = Schema.equivalence(ModelServingConfigurationSchema)
const assessmentEquivalent = Schema.equivalence(CoordinatedLocalModelAssessmentStateSchema)

const publishedEquivalent = (
  left: ReadonlyMap<ModelServingConfigurationId, CoordinatedLocalModelAssessment>,
  right: ReadonlyMap<ModelServingConfigurationId, CoordinatedLocalModelAssessment>,
): boolean => left.size === right.size && [...left].every(([id, value]) => {
  const other = right.get(id)
  return other !== undefined
    && configurationEquivalent(value.configuration, other.configuration)
    && assessmentEquivalent(value.assessment, other.assessment)
})

const sameDesired = (
  left: ReadonlyMap<ModelServingConfigurationId, DesiredAssessment>,
  right: ReadonlyMap<ModelServingConfigurationId, DesiredAssessment>,
): boolean => left.size === right.size
  && [...left].every(([id, value]) => right.get(id)?.semanticKey === value.semanticKey)

const sameCompleted = (
  left: ReadonlyMap<ModelServingConfigurationId, string>,
  right: ReadonlyMap<ModelServingConfigurationId, string>,
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

const canonicalBundleKey = (bundle: ServableModelBundle): string => {
  const digest = createHash("sha256")
  digest.update("magnitude-servable-model-bundle-v1\0")
  for (const packageId of servableModelBundlePackageIds(bundle)) {
    digest.update(packageId)
    digest.update("\0")
  }
  return `bundle_${digest.digest("hex")}`
}

const derivedConfiguration = (
  bundle: ServableModelBundle,
  profile: ServingProfile,
): ModelServingConfiguration => {
  const digest = createHash("sha256")
  digest.update(canonicalBundleKey(bundle))
  const contextLength = Buffer.allocUnsafe(4)
  contextLength.writeUInt32LE(profile.contextLength)
  digest.update(contextLength)
  return ModelServingConfigurationSchema.make({
    id: ModelServingConfigurationIdSchema.make(`configuration_${digest.digest("hex")}`),
    bundle,
    profile,
  })
}

const completedAssessment = (
  result: LocalModelAssessmentResult,
  expectedConfigurationId: ModelServingConfigurationId,
): LocalModelConfigurationAssessment => {
  if (result._tag === "InvalidBundle") {
    return {
      _tag: "Failed",
      failure: {
        code: "invalid_model_bundle",
        message: result.message,
        retryable: false,
      },
    }
  }
  const resultForConfiguration = result.assessments.find((assessment) =>
    (assessment._tag === "Fits"
      ? assessment.assessment.configurationId
      : assessment.configurationId) === expectedConfigurationId)
  if (resultForConfiguration === undefined) {
    return {
      _tag: "Failed",
      failure: {
        code: "model_assessment_configuration_mismatch",
        message: `Native assessment omitted configuration ${expectedConfigurationId}`,
        retryable: true,
      },
    }
  }
  if (resultForConfiguration._tag === "Fits") {
    return { _tag: "Fits", assessment: resultForConfiguration.assessment }
  }
  if (resultForConfiguration._tag === "DoesNotFit") {
    return {
      _tag: "DoesNotFit",
      assessmentId: resultForConfiguration.assessmentId,
      environmentId: result.environmentId,
      memory: resultForConfiguration.memory,
      deficitBytes: resultForConfiguration.deficitBytes,
      limitingResource: resultForConfiguration.limitingResource,
    }
  }
  return {
    _tag: "Incompatible",
    environmentId: result.environmentId,
    failure: resultForConfiguration.failure,
  }
}

export const LocalModelAssessorLive: Layer.Layer<
  LocalModelAssessor,
  never,
  IcnCatalog | IcnHardware | LocalModelAssessments | LocalModelPackages
    | RetainedModelConfigurations
> = Layer.scoped(LocalModelAssessor, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const hardware = yield* IcnHardware
  const assessments = yield* LocalModelAssessments
  const packages = yield* LocalModelPackages
  const retained = yield* RetainedModelConfigurations
  const observed = yield* makeObservedState<AssessorState>({
    desired: new Map(),
    published: new Map(),
    completedKeys: new Map(),
  })
  const lock = yield* Effect.makeSemaphore(1)

  const readDesired = Effect.gen(function* () {
    const retainedConfigurations = yield* retained.get
    const catalogConfigurations = (yield* catalog.ready)
      ? yield* Effect.forEach(
          (yield* catalog.get).state.models,
          recommendableModelFromIcn,
        ).pipe(Effect.map((models) => models.map(({ configuration }) => configuration)))
      : []
    const packageState = (yield* packages.snapshot).state
    const packageEntries = new Map(packageState.entries.map((entry) => [entry.package.id, entry]))
    const hardwareState = (yield* hardware.get).state
    const configurations = new Map<ModelServingConfigurationId, ModelServingConfiguration>()

    const addConfiguration = (configuration: ModelServingConfiguration) => {
      const existing = configurations.get(configuration.id)
      if (existing !== undefined && !configurationEquivalent(existing, configuration)) {
        return Effect.dieMessage(
          `Configuration ${configuration.id} has conflicting definitions`,
        )
      }
      configurations.set(configuration.id, configuration)
      return Effect.void
    }
    yield* Effect.forEach(catalogConfigurations, addConfiguration, { discard: true })
    yield* Effect.forEach(retainedConfigurations, addConfiguration, { discard: true })
    const configuredBundles = new Set([
      ...catalogConfigurations,
      ...retainedConfigurations,
    ].map(({ bundle }) => canonicalBundleKey(bundle)))

    for (const entry of packageState.entries) {
      if (entry.localState._tag !== "Installed" || entry.inspection._tag !== "Inspected") continue
      const bundle: ServableModelBundle = { _tag: "Standalone", package: entry.package }
      if (configuredBundles.has(canonicalBundleKey(bundle))) continue
      for (const profile of localModelAssessmentProfiles(bundle)) {
        yield* addConfiguration(derivedConfiguration(bundle, profile))
      }
    }

    const hardwareEvidence = {
      nativeBuild: hardwareState.native_build,
      topology: hardwareState.topology_fingerprint,
      totalSystemMemoryBytes: hardwareState.system_memory.total_bytes,
      assessmentReserveBytes: hardwareState.system_memory.assess_reserve_bytes,
      backends: [...hardwareState.enabled_backends],
    }
    const catalogIds = new Set(catalogConfigurations.map(({ id }) => id))
    const desired = new Map<ModelServingConfigurationId, DesiredAssessment>()
    for (const configuration of configurations.values()) {
      const packageEvidence = servableModelBundlePackageIds(configuration.bundle).map((packageId) => {
        const entry = packageEntries.get(packageId)
        return {
          packageId,
          installed: entry?.localState._tag === "Installed",
          inspection: entry?.inspection._tag ?? "Pending",
        }
      })
      const installedAndInspected = packageEvidence.every((entry) =>
        entry.installed && entry.inspection === "Inspected")
      if (!catalogIds.has(configuration.id) && !installedAndInspected) continue
      const semanticInput = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({
        configuration: yield* Schema.encode(ModelServingConfigurationSchema)(configuration),
        hardware: hardwareEvidence,
        material: packageEvidence,
      })
      desired.set(configuration.id, {
        configuration,
        semanticKey: createHash("sha256").update(semanticInput).digest("hex"),
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
    const pending = [...desired.values()].filter(({ configuration, semanticKey }) =>
      current.completedKeys.get(configuration.id) !== semanticKey)
    const published = new Map(
      [...current.published].filter(([configurationId]) => desired.has(configurationId)),
    )
    for (const { configuration } of pending) {
      published.set(configuration.id, { configuration, assessment: { _tag: "Assessing" } })
    }
    yield* publish({ ...current, desired, published })
    if (pending.length === 0) return

    const outcome = yield* Effect.exit(assessments.assess(
      pending.map(({ configuration }) => ({
        bundle: configuration.bundle,
        profiles: [configuration.profile],
      })),
      () => Effect.void,
    ))
    const latestDesired = yield* readDesired
    const latest = (yield* observed.get).state
    const nextPublished = new Map(latest.published)
    const completedKeys = new Map(latest.completedKeys)
    if (Exit.isFailure(outcome)) {
      const failure = assessmentCauseFailure(outcome.cause)
      for (const request of pending) {
        if (latestDesired.get(request.configuration.id)?.semanticKey !== request.semanticKey) continue
        nextPublished.set(request.configuration.id, {
          configuration: request.configuration,
          assessment: {
            _tag: "Failed",
            failure,
          },
        })
        completedKeys.set(request.configuration.id, request.semanticKey)
      }
    } else {
      pending.forEach((request, index) => {
        if (latestDesired.get(request.configuration.id)?.semanticKey !== request.semanticKey) return
        const result = outcome.value[index]
        nextPublished.set(request.configuration.id, {
          configuration: request.configuration,
          assessment: result === undefined
            ? {
                _tag: "Failed",
                failure: {
                  code: "missing_model_assessment_result",
                  message: "Native assessment returned no result for this configuration",
                  retryable: true,
                },
              }
            : completedAssessment(result, request.configuration.id),
        })
        completedKeys.set(request.configuration.id, request.semanticKey)
      })
    }
    for (const configurationId of nextPublished.keys()) {
      if (!latestDesired.has(configurationId)) nextPublished.delete(configurationId)
    }
    for (const configurationId of completedKeys.keys()) {
      if (!latestDesired.has(configurationId)) completedKeys.delete(configurationId)
    }
    yield* publish({ desired: latestDesired, published: nextPublished, completedKeys })
  })).pipe(
    Effect.catchAllCause((cause) => Effect.logWarning(
      "Unable to coordinate local model assessment",
    ).pipe(Effect.annotateLogs({ cause: String(cause) }))),
  )

  yield* Stream.make(undefined).pipe(
    Stream.concat(Stream.mergeAll([
      retained.changes.pipe(Stream.map(() => undefined)),
      catalog.changes.pipe(Stream.map(() => undefined)),
      packages.changes.pipe(Stream.map(() => undefined)),
      hardware.assessmentChanges.pipe(Stream.map(() => undefined)),
    ], { concurrency: "unbounded" }).pipe(Stream.debounce("25 millis"))),
    Stream.runForEach(() => reconcile),
    Effect.forkScoped,
  )

  return LocalModelAssessor.of({
    state: observed.get.pipe(Effect.map(({ state }) => state.published)),
    changes: observed.changes.pipe(Stream.map(({ state }) => state.published)),
  })
}))
