import { Cause, Context, Data, Effect, Equal, Exit, Layer, Option, Schema, Stream, SubscriptionRef } from "effect"
import { FSM } from "@magnitudedev/utils"
import {
  LocalModelConfigurationAssessmentSchema,
  ModelFailureSchema,
  ModelServingConfigurationSchema,
  ServingProfileSchema,
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

export class LocalModelAssessmentDiscovering extends Data.TaggedClass("Discovering")<{}> {}

export class LocalModelAssessmentAssessing extends Data.TaggedClass("Assessing")<{
  readonly cycle: {
    readonly startedAtMs: number
    readonly completedTargets: number
    readonly totalTargets: number
  }
}> {}

export class LocalModelAssessmentReady extends Data.TaggedClass("Ready")<{
  readonly cycle: {
    readonly startedAtMs: number
    readonly durationMs: number
    readonly completedTargets: number
    readonly totalTargets: number
  }
}> {}

export class LocalModelAssessmentFailed extends Data.TaggedClass("Failed")<{
  readonly cycle: {
    readonly startedAtMs: number
    readonly durationMs: number
    readonly completedTargets: number
    readonly totalTargets: number
    readonly failure: typeof ModelFailureSchema.Type
  }
}> {}

export const LocalModelAssessmentLifecycle = FSM.defineFSM(
  {
    Discovering: LocalModelAssessmentDiscovering,
    Assessing: LocalModelAssessmentAssessing,
    Ready: LocalModelAssessmentReady,
    Failed: LocalModelAssessmentFailed,
  },
  {
    Discovering: ["Assessing", "Ready"],
    Assessing: ["Ready", "Failed"],
    Ready: ["Assessing"],
    Failed: ["Assessing", "Ready"],
  } as const,
)

export type LocalModelAssessmentLifecycleState =
  | LocalModelAssessmentDiscovering
  | LocalModelAssessmentAssessing
  | LocalModelAssessmentReady
  | LocalModelAssessmentFailed

export interface LocalModelAssessmentSnapshot {
  readonly assessments: readonly CoordinatedLocalModelAssessment[]
  readonly lifecycle: LocalModelAssessmentLifecycleState
}

export interface LocalModelAssessorApi {
  readonly snapshot: Effect.Effect<LocalModelAssessmentSnapshot>
  readonly changes: Stream.Stream<LocalModelAssessmentSnapshot>
}

export class LocalModelAssessor extends Context.Tag(
  "LocalModelAssessor",
)<LocalModelAssessor, LocalModelAssessorApi>() {}

type AssessmentDemandKey = string

type DesiredAssessment = {
  readonly configuration: ModelServingConfiguration
  readonly semanticKey: string
}

type AssessmentDemand = DesiredAssessment & {
  readonly assessment: CoordinatedLocalModelAssessmentState
}

interface DiscoveringCoordinatorState {
  readonly lifecycle: LocalModelAssessmentDiscovering
  readonly demands: ReadonlyMap<AssessmentDemandKey, AssessmentDemand>
}

interface AssessingCoordinatorState {
  readonly lifecycle: LocalModelAssessmentAssessing
  readonly demands: ReadonlyMap<AssessmentDemandKey, AssessmentDemand>
  readonly cycleDemandKeys: ReadonlySet<AssessmentDemandKey>
}

interface SettledCoordinatorState {
  readonly lifecycle: LocalModelAssessmentReady | LocalModelAssessmentFailed
  readonly demands: ReadonlyMap<AssessmentDemandKey, AssessmentDemand>
}

type AssessorState = DiscoveringCoordinatorState | AssessingCoordinatorState | SettledCoordinatorState

const isAssessingState = (state: AssessorState): state is AssessingCoordinatorState =>
  state.lifecycle._tag === "Assessing"

const configurationEquivalent = Schema.equivalence(ModelServingConfigurationSchema)
const assessmentEquivalent = Schema.equivalence(CoordinatedLocalModelAssessmentStateSchema)
const profileEquivalent = Schema.equivalence(ServingProfileSchema)

const demandsEquivalent = (
  left: ReadonlyMap<AssessmentDemandKey, AssessmentDemand>,
  right: ReadonlyMap<AssessmentDemandKey, AssessmentDemand>,
): boolean => left.size === right.size && [...left].every(([id, value]) => {
  const other = right.get(id)
    return other !== undefined
    && configurationEquivalent(value.configuration, other.configuration)
    && value.semanticKey === other.semanticKey
    && assessmentEquivalent(value.assessment, other.assessment)
})

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
  const resultForConfiguration = result.assessments.find((assessment) =>
    profileEquivalent(
      assessment.configuration.profile,
      request.configuration.profile,
    ))
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
    lifecycle: new LocalModelAssessmentDiscovering(),
    demands: new Map(),
  })
  const lock = yield* Effect.makeSemaphore(1)
  const coordinationLock = yield* Effect.makeSemaphore(1)

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
    const sameCycle = !isAssessingState(previous)
      || !isAssessingState(state)
      || previous.cycleDemandKeys.size === state.cycleDemandKeys.size
        && [...previous.cycleDemandKeys].every((key) => state.cycleDemandKeys.has(key))
    const equivalent = Equal.equals(previous.lifecycle, state.lifecycle)
      && demandsEquivalent(previous.demands, state.demands)
      && sameCycle
    if (!equivalent) yield* SubscriptionRef.set(current, state)
  })

  type PendingAssessment = readonly [AssessmentDemandKey, DesiredAssessment]
  interface AssessmentTarget {
    readonly demands: readonly [PendingAssessment, ...PendingAssessment[]]
    readonly request: {
      readonly bundle: ServableModelBundle
      readonly profiles: readonly [ModelServingConfiguration["profile"], ...ModelServingConfiguration["profile"][]]
    }
  }

  const targetsFor = (pending: readonly PendingAssessment[]): readonly AssessmentTarget[] => {
    const demandsByBundle = new Map<string, PendingAssessment[]>()
    for (const entry of pending) {
      const key = bundleExecutionKey(entry[1].configuration.bundle)
      const demands = demandsByBundle.get(key)
      demandsByBundle.set(key, demands === undefined ? [entry] : [...demands, entry])
    }
    return [...demandsByBundle.values()].map((demands) => {
      const profiles = demands.reduce<ModelServingConfiguration["profile"][]>((unique, demand) => {
        const profile = demand[1].configuration.profile
        return unique.some((other) => profileEquivalent(other, profile))
          ? unique
          : [...unique, profile]
      }, [])
      return {
        demands: demands as [PendingAssessment, ...PendingAssessment[]],
        request: {
          bundle: demands[0]![1].configuration.bundle,
          profiles: profiles as [
            ModelServingConfiguration["profile"],
            ...ModelServingConfiguration["profile"][],
          ],
        },
      }
    })
  }

  const completeTarget = (
    target: AssessmentTarget,
    result: LocalModelAssessmentResult,
  ) => lock.withPermits(1)(Effect.gen(function* () {
    const latest = yield* SubscriptionRef.get(current)
    if (!isAssessingState(latest)) return
    const demands = new Map(latest.demands)
    for (const [demandKey, request] of target.demands) {
      if (!latest.cycleDemandKeys.has(demandKey)) continue
      const demand = demands.get(demandKey)
      if (demand?.semanticKey !== request.semanticKey || demand.assessment._tag !== "Assessing") continue
      const completed = completedAssessment(result, demand)
      if (completed === undefined) continue
      demands.set(demandKey, {
        configuration: completed.configuration,
        semanticKey: demand.semanticKey,
        assessment: completed.assessment,
      })
    }
    const completedTargets = [...latest.cycleDemandKeys].filter((key) =>
      demands.get(key)?.assessment._tag !== "Assessing").length
    yield* publish({
      lifecycle: LocalModelAssessmentLifecycle.hold(latest.lifecycle, {
        cycle: { ...latest.lifecycle.cycle, completedTargets },
      }),
      demands,
      cycleDemandKeys: latest.cycleDemandKeys,
    })
  }))

  const finishAssessment = (
    targets: readonly AssessmentTarget[],
    received: ReadonlySet<number>,
    outcome: Exit.Exit<void, unknown>,
  ) => lock.withPermits(1)(Effect.gen(function* () {
    const latest = yield* SubscriptionRef.get(current)
    if (!isAssessingState(latest)) return
    const demands = new Map(latest.demands)
    const terminalFailure = Exit.isFailure(outcome)
      ? assessmentCauseFailure(outcome.cause)
      : received.size === targets.length
        ? undefined
        : {
            code: "incomplete_model_assessment_response",
            message: "Native assessment completed without every requested target.",
            retryable: true,
          }
    const completedTargets = latest.lifecycle.cycle.completedTargets
    if (terminalFailure !== undefined) {
      for (const [targetIndex, target] of targets.entries()) {
        if (received.has(targetIndex)) continue
        for (const [demandKey, request] of target.demands) {
          const demand = demands.get(demandKey)
          if (demand?.semanticKey !== request.semanticKey || demand.assessment._tag !== "Assessing") continue
          demands.set(demandKey, {
            configuration: demand.configuration,
            semanticKey: demand.semanticKey,
            assessment: { _tag: "Failed", failure: terminalFailure },
          })
        }
      }
    }
    const durationMs = Math.max(0, Date.now() - latest.lifecycle.cycle.startedAtMs)
    yield* publish({
      lifecycle: terminalFailure === undefined
        ? LocalModelAssessmentLifecycle.transition(latest.lifecycle, "Ready", {
            cycle: {
              ...latest.lifecycle.cycle,
              durationMs,
              completedTargets: latest.lifecycle.cycle.totalTargets,
            },
          })
        : LocalModelAssessmentLifecycle.transition(latest.lifecycle, "Failed", {
            cycle: {
              ...latest.lifecycle.cycle,
              durationMs,
              completedTargets,
              failure: terminalFailure,
            },
          }),
      demands,
    })
  }))

  const assessPending = (targets: readonly AssessmentTarget[]) => Effect.gen(function* () {
    const received = new Set<number>()
    const outcome = yield* Effect.exit(assessments.assess(
      targets.map(({ request }) => request),
      (targetIndex, result) => Effect.gen(function* () {
        yield* completeTarget(targets[targetIndex]!, result)
        received.add(targetIndex)
      }),
    ))
    yield* finishAssessment(targets, received, outcome)
  })

  const reconcile = coordinationLock.withPermits(1)(Effect.gen(function* () {
    const targets = yield* lock.withPermits(1)(Effect.gen(function* () {
      const desired = yield* readDesired
      const state = yield* SubscriptionRef.get(current)
      if (isAssessingState(state)) {
        return yield* Effect.dieMessage("Assessment reconciliation entered during an active cycle")
      }
      const demands = new Map<AssessmentDemandKey, AssessmentDemand>()
      const pending: PendingAssessment[] = []
      for (const [demandKey, request] of desired) {
        const retained = state.demands.get(demandKey)
        if (retained?.semanticKey === request.semanticKey && retained.assessment._tag !== "Assessing") {
          demands.set(demandKey, { ...retained, configuration: request.configuration })
        } else {
          const demand = { ...request, assessment: { _tag: "Assessing" as const } }
          demands.set(demandKey, demand)
          pending.push([demandKey, request])
        }
      }
      const targets = targetsFor(pending)
      if (pending.length === 0) {
        const now = Date.now()
        const lifecycle = state.lifecycle._tag === "Ready"
          ? LocalModelAssessmentLifecycle.hold(state.lifecycle, {
              cycle: {
                ...state.lifecycle.cycle,
                completedTargets: demands.size,
                totalTargets: demands.size,
              },
            })
          : LocalModelAssessmentLifecycle.transition(state.lifecycle, "Ready", {
              cycle: {
                startedAtMs: now,
                durationMs: 0,
                completedTargets: demands.size,
                totalTargets: demands.size,
              },
            })
        yield* publish({ lifecycle, demands })
        return targets
      }
      const startedAtMs = Date.now()
      const lifecycle = LocalModelAssessmentLifecycle.transition(state.lifecycle, "Assessing", {
        cycle: {
          startedAtMs,
          completedTargets: 0,
          totalTargets: pending.length,
        },
      })
      yield* publish({
        lifecycle,
        demands,
        cycleDemandKeys: new Set(pending.map(([demandKey]) => demandKey)),
      })
      return targets
    }))
    if (targets.length > 0) yield* assessPending(targets)
  })).pipe(
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
  yield* reconcile.pipe(Effect.forkScoped)
  yield* changes.pipe(
    Stream.buffer({ capacity: 1, strategy: "sliding" }),
    Stream.runForEach(() => reconcile),
    Effect.forkScoped,
  )

  const snapshot = (state: AssessorState): LocalModelAssessmentSnapshot => ({
    assessments: [...state.demands.values()].map(({ configuration, assessment }) => ({
      configuration,
      assessment,
    })),
    lifecycle: state.lifecycle,
  })

  return LocalModelAssessor.of({
    snapshot: SubscriptionRef.get(current).pipe(Effect.map(snapshot)),
    changes: current.changes.pipe(Stream.map(snapshot)),
  })
}))
