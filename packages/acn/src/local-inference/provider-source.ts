import { createHash, randomUUID } from "node:crypto"
import { Context, Effect, Layer, Option, PubSub, Ref, Schema, Scope, Stream } from "effect"
import {
  AVAILABLE_PROVIDER_MODEL,
  LlamaCppAcquisitionError,
  LlamaCppModelInfoSchema,
  LlamaCppProviderId,
  ModelCatalogError,
  type LlamaCppInferenceLease,
  type LlamaCppModelInfo,
  type LlamaCppProviderSource,
  type ProviderModelAvailability,
  ProviderModelIdSchema,
  LlamaServedModelIdSchema,
  LlamaServingRouteIdSchema,
  type ProviderModelId,
  ModelDiscoveryOperationIdSchema,
  ModelPropertyDiscoveryErrorSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  VisionProperty,
  classifyModelFamilyFromEvidence,
} from "@magnitudedev/sdk"
import { Hardware, LlamaCpp, ModelFiles } from "@magnitudedev/local-inference"
import type { LocalInferenceOperationSnapshot } from "@magnitudedev/protocol"
import { LocalInferencePlatform } from "./platform"
import {
  LocalModelConfiguration,
  type LocalModelReconciliationInput,
  type LocalSlotCandidate,
} from "./model-configuration"
import { configuredParallelSlots } from "./recommendations"
import { LOCAL_MODEL_CATALOG } from "./catalog"
import { isOpaqueLlamaRoutingName, providerModelIdForModelPath } from "./identity"

const ZERO_PRICING = { input: 0, output: 0, cached_input: null } as const
const GIBIBYTE = 1024 ** 3
const disabled = (reason: "provider_unavailable" | "model_unavailable" | "installation_unavailable" | "incompatible_runtime" | "invalid_configuration"): ProviderModelAvailability => ({ _tag: "Disabled", reason })

type ManagedRoute = {
  readonly _tag: "Managed"
  readonly request?: LlamaCpp.ManagedModelRequest
  readonly modelPath: LlamaCpp.NormalizedLlamaModelPath
  readonly record: ModelFiles.ModelFileRecord
  readonly loaded: boolean
  readonly availability: ProviderModelAvailability
  readonly fitAssessment: Option.Option<LlamaCpp.LlamaFitAssessment>
  readonly productRank: number
}

type ExternalRoute = {
  readonly _tag: "External"
  readonly request: LlamaCpp.ExternalModelRequest
  readonly modelPath: LlamaCpp.NormalizedLlamaModelPath
  readonly observation: LlamaCpp.LlamaServedModelObservation
  readonly healthy: boolean
  readonly priority: number
  readonly mode: LlamaCpp.LlamaServerMode
}

export type LlamaLogicalRoute = ManagedRoute | ExternalRoute

const LLAMA_DEFAULT_REASONING_EFFORT = LlamaCpp.llamaCppDisabledReasoningDefinition().reasoningEffort

const ManagedRouteIdSchema = Schema.String.pipe(Schema.minLength(1), Schema.brand("ManagedRouteId"))
const ExternalRouteIdSchema = Schema.String.pipe(Schema.minLength(1), Schema.brand("ExternalRouteId"))
const InternalLlamaServingRouteIdSchema = Schema.Union(ManagedRouteIdSchema, ExternalRouteIdSchema)

export const LlamaRouteRuntimeStateSchema = Schema.Union(
  Schema.TaggedStruct("Unloaded", {}),
  Schema.TaggedStruct("Loading", {}),
  Schema.TaggedStruct("Loaded", {}),
  Schema.TaggedStruct("Sleeping", {}),
  Schema.TaggedStruct("Failed", {}),
  Schema.TaggedStruct("Unknown", {}),
)
export const LlamaServingRouteSnapshotSchema = Schema.Union(
  Schema.TaggedStruct("Managed", {
    routeId: ManagedRouteIdSchema,
    providerModelId: ProviderModelIdSchema,
    modelPath: LlamaCpp.NormalizedLlamaModelPath,
    modelFileId: ModelFiles.ModelFileId,
    state: LlamaRouteRuntimeStateSchema,
    availability: Schema.Literal("available", "disabled"),
    productRank: Schema.NonNegativeInt,
  }),
  Schema.TaggedStruct("External", {
    routeId: ExternalRouteIdSchema,
    providerModelId: ProviderModelIdSchema,
    modelPath: LlamaCpp.NormalizedLlamaModelPath,
    serverId: LlamaCpp.LlamaInstanceId,
    servedModelId: LlamaCpp.LlamaServedModelId,
    state: LlamaRouteRuntimeStateSchema,
    healthy: Schema.Boolean,
    priority: Schema.NonNegativeInt,
  }),
)
export type LlamaServingRouteSnapshot = Schema.Schema.Type<typeof LlamaServingRouteSnapshotSchema>

const ModelMetadataSourceSchema = Schema.Literal("indexed_artifact", "server_reported")
const ResolvedStringSchema = Schema.Struct({ value: Schema.String, source: ModelMetadataSourceSchema })
export const ResolvedLlamaModelInformationSchema = Schema.Struct({
  providerModelId: ProviderModelIdSchema,
  displayName: Schema.String,
  displayNameSource: Schema.Literal("gguf_metadata", "server_metadata", "server_alias", "path_basename"),
  metadata: Schema.Struct({
    name: Schema.OptionFromSelf(ResolvedStringSchema),
    architecture: Schema.OptionFromSelf(ResolvedStringSchema),
    tokenizerModel: Schema.OptionFromSelf(ResolvedStringSchema),
    tokenizerPre: Schema.OptionFromSelf(ResolvedStringSchema),
    baseModelNames: Schema.Array(Schema.String),
    baseModelRepositories: Schema.Array(Schema.String),
  }),
})
export type ResolvedLlamaModelInformation = Schema.Schema.Type<typeof ResolvedLlamaModelInformationSchema>
const LogicalProjectionStateSchema = Schema.Array(Schema.Struct({
  id: ProviderModelIdSchema,
  information: ResolvedLlamaModelInformationSchema,
  model: Schema.optional(LlamaCppModelInfoSchema),
  routes: Schema.Array(LlamaServingRouteSnapshotSchema),
}))
const equivalentLogicalProjection = Schema.equivalence(LogicalProjectionStateSchema)
const equivalentProviderCatalog = Schema.equivalence(Schema.Array(LlamaCppModelInfoSchema))
const FitProjectionStateSchema = Schema.Array(Schema.Struct({
  id: ProviderModelIdSchema,
  routes: Schema.Array(Schema.Struct({
    modelFileId: ModelFiles.ModelFileId,
    assessment: Schema.OptionFromSelf(LlamaCpp.LlamaFitAssessmentSchema),
  })),
}))
const equivalentFitProjection = Schema.equivalence(FitProjectionStateSchema)
export interface LogicalLlamaModelRecord {
  readonly providerModelId: ProviderModelId
  readonly modelPath: LlamaCpp.NormalizedLlamaModelPath
  readonly routes: readonly LlamaLogicalRoute[]
  readonly routeSnapshots: readonly LlamaServingRouteSnapshot[]
  readonly information: ResolvedLlamaModelInformation
  readonly providerModel?: LlamaCppModelInfo
}

export interface LocalModelProviderSourceApi extends LlamaCppProviderSource {
  readonly warm: (providerModelId: ProviderModelId) => Effect.Effect<void, LlamaCppAcquisitionError>
  readonly operations: Effect.Effect<readonly LocalInferenceOperationSnapshot[]>
  readonly stopManaged: Effect.Effect<void, LlamaCpp.LlamaControlError | LlamaCpp.ModelInUse>
  readonly logicalModels: Effect.Effect<ReadonlyMap<ProviderModelId, LogicalLlamaModelRecord>>
  readonly getModelInformation: (providerModelId: ProviderModelId) => Effect.Effect<ResolvedLlamaModelInformation, ModelCatalogError>
  readonly selectionInput: Effect.Effect<LocalModelReconciliationInput>
  readonly selectionReady: Effect.Effect<boolean>
  readonly selectionChanges: Stream.Stream<void>
  readonly changes: Stream.Stream<void>
  readonly stateChanges: Stream.Stream<void>
}

export class LocalModelProviderSource extends Context.Tag("LocalModelProviderSource")<
  LocalModelProviderSource,
  LocalModelProviderSourceApi
>() {}

const profileFor = (contextTokens: number) => LlamaCpp.makeLlamaExecutionProfile({
  contextSize: LlamaCpp.ContextSize.Tokens({ value: contextTokens }),
  outputLimit: LlamaCpp.OutputLimit.RuntimeDefault(),
  parallelSlots: configuredParallelSlots(),
  gpuLayers: LlamaCpp.GpuLayerSelection.Fit(),
  splitMode: "layer",
  tensorSplit: Option.none(),
  kvCache: { key: "f16", value: "f16" },
  flashAttention: LlamaCpp.FlashAttentionSelection.RuntimeDefault(),
  batchSize: LlamaCpp.BatchSize.RuntimeDefault(),
  microBatchSize: LlamaCpp.MicroBatchSize.RuntimeDefault(),
  mmap: true,
  mlock: false,
})

const stableSystemCapacity = (totalBytes: number): number =>
  Math.max(0, totalBytes - Math.max(8 * GIBIBYTE, totalBytes * 0.2))

const stableDeviceCapacity = (totalBytes: number): number =>
  Math.max(0, totalBytes - Math.max(GIBIBYTE, totalBytes * 0.1))

const placementBytes = (placement: LlamaCpp.LlamaDevicePlacement): number =>
  placement.modelBytes + placement.contextBytes + placement.computeBytes

export const llamaFitEstimateFitsStableCapacity = (
  plan: LlamaCpp.LlamaFitPlan,
  host: Hardware.HostHardwareSnapshot,
  devices: readonly LlamaCpp.LlamaDevice[],
): boolean => Option.match(assessLlamaFitStableCapacity(plan, host, devices), {
  onNone: () => true,
  onSome: (assessment) => assessment.result === "likely_fits",
})

export const assessLlamaFitStableCapacity = (
  plan: LlamaCpp.LlamaFitPlan,
  host: Hardware.HostHardwareSnapshot,
  devices: readonly LlamaCpp.LlamaDevice[],
): Option.Option<LlamaCpp.LlamaFitAssessment> => {
  const systemCapacity = stableSystemCapacity(host.totalMemoryBytes)
  const appleUnifiedMemory = host.platform === "darwin" && host.nativeArchitecture === "arm64"
  if (appleUnifiedMemory) {
    const marginBytes = systemCapacity - plan.memory.estimatedTotalBytes
    return Option.some({
      estimatedTotalBytes: plan.memory.estimatedTotalBytes,
      domains: [{ memoryDomainId: "system", estimatedBytes: plan.memory.estimatedTotalBytes, stableCapacityBytes: systemCapacity, marginBytes }],
      result: marginBytes >= 0 ? "likely_fits" : "capacity_risk",
    })
  }

  const hostPlacementBytes = plan.placement
    .filter(({ device }) => String(device) === "Host")
    .reduce((total, placement) => total + placementBytes(placement), 0)
  const acceleratorPlacements = plan.placement.filter(({ device }) => String(device) !== "Host")
  const visionAdjustmentBytes = Option.match(plan.memory.vision, {
    onNone: () => 0,
    onSome: ({ estimatedProjectorBytes, uncertaintyBytes }) => estimatedProjectorBytes + uncertaintyBytes,
  })
  const devicesById = new Map(devices.map((device) => [String(device.id), device]))
  const acceleratorDomains: LlamaCpp.LlamaFitDomainAssessment[] = []
  for (const [index, placement] of acceleratorPlacements.entries()) {
    const device = devicesById.get(String(placement.device))
    if (!device || Option.isNone(device.totalMemoryBytes)) return Option.none()
    const requiredBytes = placementBytes(placement) + (index === 0 ? visionAdjustmentBytes : 0)
    const capacityBytes = stableDeviceCapacity(device.totalMemoryBytes.value)
    acceleratorDomains.push({
      memoryDomainId: String(placement.device),
      estimatedBytes: requiredBytes,
      stableCapacityBytes: capacityBytes,
      marginBytes: capacityBytes - requiredBytes,
    })
  }
  const systemEstimatedBytes = hostPlacementBytes + (acceleratorPlacements.length === 0 ? visionAdjustmentBytes : 0)
  const domains: LlamaCpp.LlamaFitDomainAssessment[] = [{
    memoryDomainId: "system",
    estimatedBytes: systemEstimatedBytes,
    stableCapacityBytes: systemCapacity,
    marginBytes: systemCapacity - systemEstimatedBytes,
  }, ...acceleratorDomains]
  return Option.some({
    estimatedTotalBytes: plan.memory.estimatedTotalBytes,
    domains,
    result: domains.every(({ marginBytes }) => marginBytes >= 0) ? "likely_fits" : "capacity_risk",
  })
}

const acquisitionError = (modelId: ProviderModelId, cause: unknown) => new LlamaCppAcquisitionError({
  modelId,
  reason: cause instanceof Error ? cause.message : String(cause),
  cause,
})

const basename = (path: string): string => {
  const value = path.split("/").filter(Boolean).at(-1) ?? path
  return value.replace(/\.gguf$/i, "") || "Local model"
}

const firstSome = <A>(items: readonly Option.Option<A>[]): Option.Option<A> => {
  for (const item of items) if (Option.isSome(item)) return item
  return Option.none()
}

const routeIsRunning = (route: LlamaLogicalRoute): boolean => route._tag === "Managed"
  ? route.loaded
  : route.healthy && (route.observation.status === "loaded" || route.observation.status === "sleeping")

const externalUsable = (route: ExternalRoute): boolean => route.healthy
  && (route.observation.status === "loaded" || route.observation.status === "sleeping")

const routeState = (status: LlamaCpp.LlamaServedModelStatus) => {
  switch (status) {
    case "loaded": return { _tag: "Loaded" as const }
    case "sleeping": return { _tag: "Sleeping" as const }
    case "loading":
    case "downloading": return { _tag: "Loading" as const }
    case "failed": return { _tag: "Failed" as const }
    case "unloaded": return { _tag: "Unloaded" as const }
    case "unknown": return { _tag: "Unknown" as const }
  }
}

const routeSnapshot = (providerModelId: ProviderModelId, route: LlamaLogicalRoute): LlamaServingRouteSnapshot => route._tag === "Managed"
  ? {
      _tag: "Managed",
      routeId: ManagedRouteIdSchema.make(`managed:${route.record.id}`),
      providerModelId,
      modelPath: route.modelPath,
      modelFileId: route.record.id,
      state: route.loaded ? { _tag: "Loaded" } : { _tag: "Unloaded" },
      availability: route.availability._tag === "Available" ? "available" : "disabled",
      productRank: route.productRank,
    }

  : {
      _tag: "External",
      routeId: ExternalRouteIdSchema.make(`external:${route.request.instanceId}:${route.request.servedModelId}`),
      providerModelId,
      modelPath: route.modelPath,
      serverId: route.request.instanceId,
      servedModelId: route.request.servedModelId,
      state: routeState(route.observation.status),
      healthy: route.healthy,
      priority: route.priority,
    }

const routeKey = (providerModelId: ProviderModelId, route: LlamaLogicalRoute): string =>
  String(routeSnapshot(providerModelId, route).routeId)

const addFingerprintField = (hash: ReturnType<typeof createHash>, name: string, value: string) => {
  hash.update(`${name.length}:${name}${value.length}:${value}`)
}

const loadedRouteProps = (
  client: LlamaCpp.LlamaServerClient,
  route: LlamaLogicalRoute,
  servedModelId: LlamaCpp.LlamaServedModelId,
) => client.props(Option.some(servedModelId)).pipe(
  Effect.catchTag("LlamaServerError", (error) => route._tag === "External"
    && route.mode === "single-model"
    && error.reason === "rejected"
    ? client.props(Option.none())
    : Effect.fail(error)),
)

const reasoningEffortsForInspections = (
  inspections: readonly LlamaCpp.LlamaCppReasoningInspection[],
) => [...new Set(inspections.flatMap((inspection) =>
  inspection.profile.effortMappings.map((mapping) => mapping.reasoningEffort)))]

const profileSupportsReasoningEffort = (
  profile: LlamaCpp.LlamaCppReasoningProfile,
  reasoningEffort: typeof ReasoningEffortSchema.Type,
): boolean => Option.isSome(LlamaCpp.resolveLlamaCppReasoningEffort(profile, reasoningEffort))

const resolveReasoningMapping = (
  profile: LlamaCpp.LlamaCppReasoningProfile,
  requestedEffort: typeof ReasoningEffortSchema.Type,
) => Option.map(
  Option.orElse(
    LlamaCpp.resolveLlamaCppReasoningEffort(profile, requestedEffort),
    () => LlamaCpp.resolveLlamaCppReasoningEffort(profile, profile.defaultReasoningEffort),
  ),
  (mapping) => ({ reasoningEffort: mapping.reasoningEffort, mapping }),
)

const resolveReasoningProperty = (
  property: LlamaCppModelInfo["properties"]["reasoning"],
  value: readonly (typeof ReasoningEffortSchema.Type)[],
  undiscovered: "preserve" | "resolve",
): LlamaCppModelInfo["properties"]["reasoning"] => ReasoningProperty.Lifecycle.match(property, {
  Deferred: (state) => undiscovered === "resolve" ? new ReasoningProperty.states.Resolved({ value: [...value] }) : state,
  Discovering: (state) => ReasoningProperty.Lifecycle.transition(state, "Resolved", { value: [...value] }),
  Cached: (state) => ReasoningProperty.Lifecycle.transition(state, "Resolved", { value: [...value] }),
  Resolved: (state) => ReasoningProperty.Lifecycle.hold(state, { value: [...value] }),
  Refreshing: (state) => ReasoningProperty.Lifecycle.transition(state, "Resolved", { value: [...value] }),
  Failed: (state) => undiscovered === "resolve" ? new ReasoningProperty.states.Resolved({ value: [...value] }) : state,
})

export const llamaCppRequestOptionsForReasoningMapping = (
  mapping: LlamaCpp.LlamaCppReasoningEffortMapping,
): {
  readonly chatTemplateKwargs: Option.Option<Readonly<Record<string, unknown>>>
  readonly thinkingBudgetTokens: Option.Option<number>
} => {
  const kwargs = {
    ...Option.match(mapping.templateOptions.enableThinking, {
      onNone: () => ({}),
      onSome: (enableThinking) => ({ enable_thinking: enableThinking }),
    }),
    ...Option.match(mapping.templateOptions.reasoningEffort, {
      onNone: () => ({}),
      onSome: (reasoningEffort) => ({ reasoning_effort: reasoningEffort }),
    }),
  }
  return {
    chatTemplateKwargs: Object.keys(kwargs).length === 0 ? Option.none() : Option.some(kwargs),
    thinkingBudgetTokens: mapping.thinkingBudget._tag === "Enabled"
      ? Option.some(mapping.thinkingBudget.tokens)
      : Option.none(),
  }
}

const bestDisabledReason = (routes: readonly LlamaLogicalRoute[]): ProviderModelAvailability => {
  if (routes.some((route) => route._tag === "Managed" && route.availability._tag === "Available")) return AVAILABLE_PROVIDER_MODEL
  if (routes.some((route) => route._tag === "External" && externalUsable(route))) return AVAILABLE_PROVIDER_MODEL
  const reasons = routes.flatMap((route) => route._tag === "Managed" && route.availability._tag === "Disabled" ? [route.availability.reason] : [])
  for (const reason of ["invalid_configuration", "incompatible_runtime", "provider_unavailable", "model_unavailable"] as const) {
    if (reasons.includes(reason)) return disabled(reason)
  }
  return disabled("model_unavailable")
}

const productRankFor = (record: ModelFiles.ModelFileRecord): number => {
  const index = LOCAL_MODEL_CATALOG.findIndex((entry) =>
    entry.files.reduce((total, file) => total + file.sizeBytes, 0) === record.sizeBytes
    && Option.contains(record.metadata.quantization, entry.quantization.format))
  return index < 0 ? Number.MAX_SAFE_INTEGER : index
}

export const resolveLlamaModelInformation = (
  providerModelId: ProviderModelId,
  path: LlamaCpp.NormalizedLlamaModelPath,
  routes: readonly LlamaLogicalRoute[],
): ResolvedLlamaModelInformation => {
  const managed = routes.filter((route): route is ManagedRoute => route._tag === "Managed")
  const external = routes.filter((route): route is ExternalRoute => route._tag === "External").sort((a, b) => a.priority - b.priority)
  const resolveString = (key: "name" | "architecture" | "tokenizerModel" | "tokenizerPre"): Option.Option<{ readonly value: string; readonly source: "indexed_artifact" | "server_reported" }> => {
    const indexed = firstSome(managed.map((route) => route.record.metadata[key]))
    if (Option.isSome(indexed)) return Option.some({ value: indexed.value, source: "indexed_artifact" })
    const server = key === "name"
      ? firstSome(external.map((route) => route.observation.serverDisplayName))
      : key === "architecture"
        ? firstSome(external.map((route) => route.observation.architecture))
        : Option.none<string>()
    return Option.map(server, (value) => ({ value, source: "server_reported" as const }))
  }
  const metadataName = resolveString("name")
  const serverName = external.map((route) => Option.getOrUndefined(route.observation.serverDisplayName)).find((name): name is string => name !== undefined)
  const alias = external.map((route) => String(route.observation.id)).find((name) => !isOpaqueLlamaRoutingName(name))
  const display = Option.isSome(metadataName) && metadataName.value.source === "indexed_artifact"
    ? { value: metadataName.value.value, source: "gguf_metadata" as const }
    : serverName
        ? { value: serverName, source: "server_metadata" as const }
        : alias
          ? { value: alias, source: "server_alias" as const }
          : { value: basename(path), source: "path_basename" as const }
  return {
    providerModelId,
    displayName: display.value,
    displayNameSource: display.source,
    metadata: {
      name: metadataName,
      architecture: resolveString("architecture"),
      tokenizerModel: resolveString("tokenizerModel"),
      tokenizerPre: resolveString("tokenizerPre"),
      baseModelNames: managed.flatMap((route) => route.record.metadata.baseModelNames),
      baseModelRepositories: managed.flatMap((route) => route.record.metadata.baseModelRepositories),
    },
  }
}

export const LocalModelProviderSourceLive: Layer.Layer<
  LocalModelProviderSource,
  never,
  LocalInferencePlatform | LocalModelConfiguration
> = Layer.scoped(LocalModelProviderSource, Effect.gen(function* () {
  const platform = yield* LocalInferencePlatform
  const configuration = yield* LocalModelConfiguration
  const logicalModels = yield* Ref.make<ReadonlyMap<ProviderModelId, LogicalLlamaModelRecord>>(new Map())
  const visionByRoute = yield* Ref.make<ReadonlyMap<string, LlamaCpp.LlamaCppVisionInspection>>(new Map())
  const reasoningByRoute = yield* Ref.make<ReadonlyMap<string, LlamaCpp.LlamaCppReasoningInspection>>(new Map())
  const activeRouteByModel = yield* Ref.make<ReadonlyMap<ProviderModelId, string>>(new Map())
  const routeLastUsed = yield* Ref.make<ReadonlyMap<string, number>>(new Map())
  const demandLoads = yield* Ref.make<ReadonlySet<ProviderModelId>>(new Set())
  const selectionReady = yield* Ref.make(false)
  const operations = yield* Ref.make<ReadonlyMap<string, LocalInferenceOperationSnapshot>>(new Map())
  const discoveryOperations = yield* Ref.make<ReadonlyMap<string, typeof ModelDiscoveryOperationIdSchema.Type>>(new Map())
  const catalogChanges = yield* PubSub.unbounded<void>()
  const selectionChanges = yield* PubSub.unbounded<void>()
  const operationChanges = yield* PubSub.unbounded<void>()
  const fitChanges = yield* PubSub.unbounded<void>()
  const fitReconcileRequests = yield* PubSub.unbounded<void>()
  const fitRevision = yield* Ref.make(0)
  const fitCommitLock = yield* Effect.makeSemaphore(1)
  const rebuildLock = yield* Effect.makeSemaphore(1)
  const discoveryLock = yield* Effect.makeSemaphore(1)
  const inspectionLock = yield* Effect.makeSemaphore(1)
  const serviceScope = yield* Scope.Scope
  const requestFitReconciliation = fitCommitLock.withPermits(1)(
    Ref.updateAndGet(fitRevision, (revision) => revision + 1).pipe(
      Effect.zipRight(PubSub.publish(fitReconcileRequests, undefined)),
      Effect.asVoid,
    ),
  )
  const updateProviderModel = (
    providerModelId: ProviderModelId,
    update: (model: LlamaCppModelInfo) => LlamaCppModelInfo,
  ) => Ref.update(logicalModels, (models) => {
    const current = models.get(providerModelId)
    if (!current?.providerModel) return models
    const next = new Map(models)
    next.set(providerModelId, { ...current, providerModel: update(current.providerModel) })
    return next
  }).pipe(
    Effect.zipRight(PubSub.publish(catalogChanges, undefined)),
    Effect.asVoid,
  )
  const setManagedSnapshotState = (providerModelId: ProviderModelId, state: LlamaServingRouteSnapshot["state"]) => Ref.update(logicalModels, (models) => {
    const current = models.get(providerModelId)
    if (!current) return models
    const next = new Map(models)
    next.set(providerModelId, {
      ...current,
      routeSnapshots: current.routeSnapshots.map((route) => route._tag === "Managed" ? { ...route, state } : route),
    })
    return next
  })
  const servingFingerprint = (
    logical: LogicalLlamaModelRecord,
    route: LlamaLogicalRoute,
    props: LlamaCpp.LlamaModelProperties,
  ) => Effect.gen(function* () {
    const resolved = route._tag === "Managed"
      ? yield* platform.files.resolve(route.record.id).pipe(Effect.option)
      : Option.none<ModelFiles.ResolvedModelFiles>()
    const hash = createHash("sha256")
    addFingerprintField(hash, "model-path", logical.modelPath)
    addFingerprintField(hash, "projector", Option.flatMap(resolved, (files) => files.projectorPath).pipe(Option.getOrElse(() => "<absent>")))
    addFingerprintField(hash, "route-config", route._tag === "Managed" ? route.request?.profile.id ?? "<invalid>" : routeKey(logical.providerModelId, route))
    addFingerprintField(hash, "build", Option.getOrElse(props.build, () => "<missing>"))
    addFingerprintField(hash, "template-default", Option.getOrElse(props.chatTemplate, () => "<missing>"))
    addFingerprintField(hash, "template-tools", Option.getOrElse(props.chatTemplateToolUse, () => "<absent>"))
    return {
      value: hash.digest("hex"),
      persistent: route._tag === "Managed"
        && Option.isSome(resolved)
        && Option.isSome(props.build)
        && Option.isSome(props.chatTemplate),
    }
  })

  const operationStage = (event: LlamaCpp.LlamaLoadEvent): LocalInferenceOperationSnapshot["stage"] => {
    switch (event._tag) {
      case "Queued": return "queued"
      case "ResolvingFiles": return "resolving_files"
      case "WritingPreset": return "writing_preset"
      case "StartingRouter": return "starting_router"
      case "UnloadingPrevious": return "unloading_previous"
      case "Loading": return "loading"
      case "Verifying": return "verifying"
      case "Loaded": return "loaded"
    }
  }

  const observeOperation = (providerModelId: ProviderModelId, operation: LlamaCpp.LlamaLoadOperation) => Effect.gen(function* () {
    const id = String(operation.id)
    if ((yield* Ref.get(operations)).has(id)) return
    const setOperation = (snapshot: LocalInferenceOperationSnapshot) => Ref.update(
      operations,
      (current) => new Map(current).set(id, snapshot),
    ).pipe(
      Effect.zipRight(PubSub.publish(operationChanges, undefined)),
      Effect.asVoid,
    )
    yield* setOperation({ operationId: id, providerModelId, status: "running", stage: "queued" })
    yield* Effect.forkIn(operation.events.pipe(Stream.runForEach((event) => setOperation({
        operationId: id,
        providerModelId,
        status: event._tag === "Loaded" ? "completed" : "running",
        stage: operationStage(event),
        ...(event._tag === "Loading" && Option.isSome(event.progress) ? { progress: event.progress.value } : {}),
      }))), serviceScope)
    yield* Effect.forkIn(operation.result.pipe(
      Effect.tap(() => setOperation({ operationId: id, providerModelId, status: "completed", stage: "loaded" })),
      Effect.tapError((cause) => setOperation({ operationId: id, providerModelId, status: "failed", stage: "loading", message: cause.reason })),
      Effect.ignore,
    ), serviceScope)
  })

  const rebuild = (
    fileRefresh: ModelFiles.ModelFileRefresh,
    assessAvailability: boolean,
    reassessManaged = true,
  ) => rebuildLock.withPermits(1)(Effect.gen(function* () {
    const records = (yield* platform.files.inspect(fileRefresh)).records
    const registry = assessAvailability
      ? Option.some(yield* platform.instances)
      : Option.none<LlamaCpp.LlamaInstanceRegistryApi>()
    const runtime = assessAvailability && Option.isSome(registry)
      ? Option.some(yield* registry.value.snapshot)
      : Option.none<LlamaCpp.LlamaInstanceSnapshot>()
    const cli = assessAvailability && reassessManaged ? yield* platform.cli.pipe(Effect.option) : Option.none<LlamaCpp.LlamaCli>()
    const previous = yield* Ref.get(logicalModels)
    const retainedManagedRoutes = new Map([...previous.values()].flatMap((logical) => logical.routes)
      .filter((route): route is ManagedRoute => route._tag === "Managed")
      .map((route) => [route.record.id, route] as const))
    const groups = new Map<ProviderModelId, { path: LlamaCpp.NormalizedLlamaModelPath; routes: LlamaLogicalRoute[] }>()
    const managedModelPaths = new Set<string>()
    const addRoute = (providerModelId: ProviderModelId, path: LlamaCpp.NormalizedLlamaModelPath, route: LlamaLogicalRoute) => {
      const group = groups.get(providerModelId) ?? { path, routes: [] }
      group.routes.push(route)
      groups.set(providerModelId, group)
    }

    yield* Effect.forEach(records, (record) => Effect.gen(function* () {
      const retainedRoute = reassessManaged
        ? undefined
        : retainedManagedRoutes.get(record.id)
      if (retainedRoute) {
        if (managedModelPaths.has(retainedRoute.modelPath)) return
        managedModelPaths.add(retainedRoute.modelPath)
        const loaded = Option.exists(runtime, ({ instances }) => instances.some((instance) =>
          instance.ownership === "managed" && instance.models.some((model) =>
            (String(model.id) === retainedRoute.request?.servedModelId || Option.contains(model.reportedModelPath, retainedRoute.modelPath))
            && (model.status === "loaded" || model.status === "sleeping")),
        ))
        addRoute(providerModelIdForModelPath(retainedRoute.modelPath), retainedRoute.modelPath, {
          ...retainedRoute,
          loaded,
        })
        return
      }
      const resolved = yield* platform.files.resolve(record.id).pipe(Effect.option)
      if (Option.isNone(resolved)) return
      const modelPath = LlamaCpp.normalizeLlamaModelPath(resolved.value.primaryPath)
      if (!modelPath || !LlamaCpp.isAbsoluteLlamaModelPath(modelPath)) return
      if (managedModelPaths.has(modelPath)) return
      managedModelPaths.add(modelPath)
      const providerModelId = providerModelIdForModelPath(modelPath)
      const context = record.metadata.trainedContextTokens
      const profile = Option.isSome(context) ? yield* profileFor(context.value) : undefined
      const request = profile ? {
        modelFileId: record.id,
        servedModelId: LlamaCpp.LlamaServedModelId.make(providerModelId),
        profile,
      } : undefined
      const loaded = Option.exists(runtime, ({ instances }) => instances.some((instance) =>
        instance.ownership === "managed" && instance.models.some((model) =>
          (String(model.id) === providerModelId || Option.contains(model.reportedModelPath, modelPath))
          && (model.status === "loaded" || model.status === "sleeping")),
      ))
      const availability = !request
        ? disabled("invalid_configuration")
        : assessAvailability
          ? loaded
            ? AVAILABLE_PROVIDER_MODEL
            : yield* Option.match(cli, {
              onNone: () => Effect.succeed(disabled("installation_unavailable")),
              onSome: () => Effect.succeed(AVAILABLE_PROVIDER_MODEL),
            })
          : AVAILABLE_PROVIDER_MODEL
      addRoute(providerModelId, modelPath, {
        _tag: "Managed",
        ...(request ? { request } : {}),
        modelPath,
        record,
        loaded,
        availability,
        fitAssessment: Option.none(),
        productRank: productRankFor(record),
      })
    }), { concurrency: 1, discard: true })

    if (Option.isSome(runtime)) {
      for (const [priority, instance] of runtime.value.instances.entries()) {
        if (instance.ownership !== "external") continue
        for (const model of instance.models) {
          if (model.status !== "loaded" && model.status !== "sleeping") continue
          if (Option.isNone(model.reportedModelPath)) continue
          const modelPath = model.reportedModelPath.value
          const providerModelId = providerModelIdForModelPath(modelPath)
          addRoute(providerModelId, modelPath, {
            _tag: "External",
            request: { instanceId: instance.id, servedModelId: model.id },
            modelPath,
            observation: model,
            healthy: instance.health === "ready",
            priority,
            mode: instance.mode,
          })
        }
      }
    }

    const selectedIds = new Set(Object.values((yield* configuration.getModels)?.slots ?? {}).flatMap((slot) =>
      slot?.providerId === "llamacpp" && slot.providerModelId && Schema.is(ProviderModelIdSchema)(slot.providerModelId)
        ? [slot.providerModelId]
        : []))
    for (const id of selectedIds) {
      if (groups.has(id)) continue
      const old = previous.get(id)
      if (old) groups.set(id, { path: old.modelPath, routes: [] })
    }

    const next = new Map<ProviderModelId, LogicalLlamaModelRecord>()
    for (const [providerModelId, group] of groups) {
      const retained = group.routes.length === 0 ? previous.get(providerModelId) : undefined
      const information = retained
        ? retained.information
        : resolveLlamaModelInformation(providerModelId, group.path, group.routes)
      const managed = group.routes.filter((route): route is ManagedRoute => route._tag === "Managed")
      const external = group.routes.filter((route): route is ExternalRoute => route._tag === "External")
      const context = firstSome([
        ...external.map((route) => route.observation.activeContextTokens),
        ...managed.map((route) => route.record.metadata.trainedContextTokens),
      ])
      const previousModel = previous.get(providerModelId)?.providerModel
      const cachedProperties = yield* platform.modelIndex.discoveredProperties(group.path)
      const cachedReasoningProfiles = Option.match(cachedProperties, {
        onNone: () => [] as readonly LlamaCpp.LlamaCppReasoningProfile[],
        onSome: (cached) => cached.reasoningInspections.map((inspection) => inspection.profile),
      })
      const cachedDefaultReasoningEffort = cachedReasoningProfiles[0]?.defaultReasoningEffort
        ?? LLAMA_DEFAULT_REASONING_EFFORT
      const indexedProperties = Option.match(cachedProperties, {
        onNone: () => ({
          vision: new VisionProperty.states.Deferred({}),
          reasoning: new ReasoningProperty.states.Deferred({}),
        }),
        onSome: (cached) => ({
          vision: cached.visionInspections.length > 0
            ? new VisionProperty.states.Cached({ value: cached.visionInspections.some((inspection) => inspection.value) })
            : new VisionProperty.states.Deferred({}),
          reasoning: cached.reasoningInspections.length > 0
            ? new ReasoningProperty.states.Cached({ value: reasoningEffortsForInspections(cached.reasoningInspections) })
            : new ReasoningProperty.states.Deferred({}),
        }),
      })
      const providerModel: LlamaCppModelInfo | undefined = retained?.providerModel
        ? { ...retained.providerModel, availability: disabled("model_unavailable") }
        : Option.isSome(context) ? {
        providerId: LlamaCppProviderId.make("llamacpp"),
        providerModelId,
        displayName: information.displayName,
        ...Option.match(classifyModelFamilyFromEvidence({
          architecture: Option.getOrUndefined(Option.map(information.metadata.architecture, ({ value }) => value)),
          tokenizerModel: Option.getOrUndefined(Option.map(information.metadata.tokenizerModel, ({ value }) => value)),
          tokenizerPre: Option.getOrUndefined(Option.map(information.metadata.tokenizerPre, ({ value }) => value)),
        }, [
          ...Option.toArray(Option.map(information.metadata.name, ({ value }) => value)),
          ...information.metadata.baseModelNames,
          ...information.metadata.baseModelRepositories,
        ]), {
          onNone: () => ({}),
          onSome: (modelFamilyId) => ({ modelFamilyId }),
        }),
        contextWindow: context.value,
        maxOutputTokens: Math.min(context.value, 8192),
        defaultReasoningEffort: previousModel?.defaultReasoningEffort ?? cachedDefaultReasoningEffort,
        properties: previousModel?.properties ?? indexedProperties,
        availability: bestDisabledReason(group.routes),
        pricing: ZERO_PRICING,
        } : undefined
      next.set(providerModelId, { providerModelId, modelPath: group.path, routes: group.routes, routeSnapshots: group.routes.map((route) => routeSnapshot(providerModelId, route)), information, ...(providerModel ? { providerModel } : {}) })
    }
    const previousProjection = [...previous.values()].map((record) => ({
      id: record.providerModelId,
      information: record.information,
      ...(record.providerModel ? { model: record.providerModel } : {}),
      routes: record.routeSnapshots,
    }))
    const nextProjection = [...next.values()].map((record) => ({
      id: record.providerModelId,
      information: record.information,
      ...(record.providerModel ? { model: record.providerModel } : {}),
      routes: record.routeSnapshots,
    }))
    const previousCatalog = [...previous.values()].flatMap((record) => record.providerModel ? [record.providerModel] : [])
    const nextCatalog = [...next.values()].flatMap((record) => record.providerModel ? [record.providerModel] : [])
    const fitProjection = (records: ReadonlyMap<ProviderModelId, LogicalLlamaModelRecord>) => [...records.values()].map((record) => ({
      id: record.providerModelId,
      routes: record.routes.flatMap((route) => route._tag === "Managed"
        ? [{ modelFileId: route.record.id, assessment: route.fitAssessment }]
        : []),
    }))
    const logicalChanged = !equivalentLogicalProjection(previousProjection, nextProjection)
    const fitChanged = !equivalentFitProjection(fitProjection(previous), fitProjection(next))
    const becameReady = assessAvailability && !(yield* Ref.get(selectionReady))
    if (becameReady) yield* Ref.set(selectionReady, true)
    if (logicalChanged || fitChanged) yield* Ref.set(logicalModels, next)
    if (logicalChanged) {
      yield* PubSub.publish(selectionChanges, undefined)
    }
    if (fitChanged) yield* PubSub.publish(fitChanges, undefined)
    if (becameReady) yield* PubSub.publish(selectionChanges, undefined)
    if (!equivalentProviderCatalog(previousCatalog, nextCatalog)) {
      yield* PubSub.publish(catalogChanges, undefined)
    }
    return yield* Schema.decodeUnknown(Schema.Array(LlamaCppModelInfoSchema))(nextCatalog)
  }).pipe(Effect.mapError((cause) => new ModelCatalogError({ message: cause instanceof Error ? cause.message : String(cause), cause }))))

  const hardwareFingerprint = (
    host: Hardware.HostHardwareSnapshot,
    devices: readonly LlamaCpp.LlamaDevice[],
  ): string => {
    const hash = createHash("sha256")
    addFingerprintField(hash, "platform", host.platform)
    addFingerprintField(hash, "architecture", host.nativeArchitecture)
    addFingerprintField(hash, "system-memory", String(host.totalMemoryBytes))
    for (const device of [...devices].sort((left, right) => String(left.id).localeCompare(String(right.id)))) {
      addFingerprintField(hash, "device-id", String(device.id))
      addFingerprintField(hash, "device-name", Option.getOrElse(device.name, () => "<absent>"))
      addFingerprintField(hash, "device-type", Option.getOrElse(device.type, () => "<absent>"))
      addFingerprintField(hash, "device-memory", Option.match(device.totalMemoryBytes, {
        onNone: () => "<absent>",
        onSome: String,
      }))
    }
    return hash.digest("hex")
  }

  const setFitAssessment = (
    providerModelId: ProviderModelId,
    modelFileId: ModelFiles.ModelFileId,
    assessment: LlamaCpp.LlamaFitAssessment,
  ) => Ref.modify(logicalModels, (models) => {
    const logical = models.get(providerModelId)
    if (!logical) return [false, models] as const
    let changed = false
    const routes = logical.routes.map((route) => {
      if (route._tag !== "Managed" || route.record.id !== modelFileId) return route
      if (Option.exists(route.fitAssessment, (current) => Schema.equivalence(LlamaCpp.LlamaFitAssessmentSchema)(current, assessment))) return route
      changed = true
      return { ...route, fitAssessment: Option.some(assessment) }
    })
    if (!changed) return [false, models] as const
    const next = new Map(models)
    next.set(providerModelId, { ...logical, routes, routeSnapshots: routes.map((route) => routeSnapshot(providerModelId, route)) })
    return [true, next] as const
  }).pipe(
    Effect.flatMap((changed) => changed ? PubSub.publish(fitChanges, undefined) : Effect.succeed(false)),
    Effect.asVoid,
  )

  const reconcileFitAssessments = Effect.gen(function* () {
    const revision = yield* Ref.get(fitRevision)
    const cli = yield* platform.cli.pipe(Effect.option)
    if (Option.isNone(cli)) return
    const [host, deviceSnapshot] = yield* Effect.all([platform.hardware.inspect, cli.value.listDevices], { concurrency: 2 })
    const topologyFingerprint = hardwareFingerprint(host, deviceSnapshot.devices)
    const logical = yield* Ref.get(logicalModels)
    const candidates = [...logical.values()].flatMap((model) => model.routes.flatMap((route) =>
      route._tag === "Managed"
        && route.request
        && !route.loaded
        && route.availability._tag === "Available"
        ? [{ providerModelId: model.providerModelId, route, request: route.request }]
        : []))
    const seenPaths = new Set<string>()
    const unique = candidates.filter(({ route }) => {
      if (seenPaths.has(route.modelPath)) return false
      seenPaths.add(route.modelPath)
      return true
    })
    yield* Effect.forEach(unique, ({ providerModelId, route, request }) => Effect.gen(function* () {
      const files = yield* platform.files.resolve(route.record.id)
      const key = LlamaCpp.makeLlamaFitAssessmentKey({
        modelPath: route.modelPath,
        fileVersion: files.version,
        projectorPath: files.projectorPath,
        profileId: request.profile.id,
        fitExecutableFingerprint: cli.value.installation.executables.fitParams.fingerprint,
        hardwareFingerprint: topologyFingerprint,
      })
      const cached = yield* platform.modelIndex.fitAssessment(route.modelPath, key)
      const assessment = yield* Option.match(cached, {
        onSome: ({ assessment }) => Effect.succeed(Option.some(assessment)),
        onNone: () => cli.value.assessFit({ files, profile: request.profile }).pipe(
          Effect.map((result) => result._tag === "Estimated"
            ? assessLlamaFitStableCapacity(result.plan, host, deviceSnapshot.devices)
            : Option.none<LlamaCpp.LlamaFitAssessment>()),
        ),
      })
      if (Option.isNone(assessment)) return
      yield* fitCommitLock.withPermits(1)(Effect.gen(function* () {
        if (revision !== (yield* Ref.get(fitRevision))) return
        yield* platform.modelIndex.putFitAssessment({ modelPath: route.modelPath, key, assessment: assessment.value })
        yield* setFitAssessment(providerModelId, route.record.id, assessment.value)
      }))
    }).pipe(
      Effect.catchAll((cause) => Effect.logWarning("Local model fit assessment failed").pipe(
        Effect.annotateLogs({ modelPath: route.modelPath, cause: String(cause) }),
      )),
    ), { concurrency: 2, discard: true })
  })

  const refresh = yield* Effect.cachedWithTTL(rebuild("changed", true).pipe(
    Effect.tap(() => requestFitReconciliation),
  ), "250 millis")
  const list = Ref.get(logicalModels).pipe(Effect.map((models) => [...models.values()].flatMap((record) => record.providerModel ? [record.providerModel] : [])))

  const inspectLoadedRoute = (
    logical: LogicalLlamaModelRecord,
    route: LlamaLogicalRoute,
    target: {
      readonly origin: URL
      readonly authorization: LlamaCppInferenceLease["authorization"]
      readonly servedModelId: LlamaCpp.LlamaServedModelId
    },
    requested: ReadonlySet<"vision" | "reasoning">,
  ) => inspectionLock.withPermits(1)(Effect.gen(function* () {
    const routeId = routeKey(logical.providerModelId, route)
    const client = yield* platform.serverClient(target.origin, target.authorization)
    const props = yield* loadedRouteProps(client, route, target.servedModelId)
    const fingerprint = yield* servingFingerprint(logical, route, props)
    const cached = yield* platform.modelIndex.discoveredProperties(logical.modelPath)

    const visionInspection: LlamaCpp.LlamaCppVisionInspection | undefined = requested.has("vision")
      ? { routeId, fingerprint: fingerprint.value, value: props.modalities.vision }
      : undefined

    let reasoningInspection: LlamaCpp.LlamaCppReasoningInspection | undefined
    if (requested.has("reasoning")) {
      const current = (yield* Ref.get(reasoningByRoute)).get(routeId)
      const indexed = Option.isSome(cached)
        ? cached.value.reasoningInspections.find((entry) => entry.routeId === routeId
          && entry.fingerprint === fingerprint.value)
        : undefined
      if (current?.fingerprint === fingerprint.value) {
        reasoningInspection = current
      } else if (indexed) {
        reasoningInspection = indexed
      } else {
        const templateInspection = yield* LlamaCpp.discoverLlamaCppReasoning({
          render: (templateRequest) => client.applyTemplate(target.servedModelId, templateRequest),
        })
        reasoningInspection = { routeId, fingerprint: fingerprint.value, ...templateInspection }
        yield* Effect.logInfo("Resolved llama.cpp reasoning profile").pipe(Effect.annotateLogs({
          providerModelId: logical.providerModelId,
          routeId,
          fingerprint: fingerprint.value,
          defaultReasoningEffort: templateInspection.profile.defaultReasoningEffort,
          reasoningEfforts: templateInspection.profile.effortMappings
            .map((mapping) => mapping.reasoningEffort)
            .join(","),
        }))
      }
    }

    if (visionInspection) {
      yield* Ref.update(visionByRoute, (current) => new Map(current).set(routeId, visionInspection))
    }
    if (reasoningInspection) {
      yield* Ref.update(reasoningByRoute, (current) => new Map(current).set(routeId, reasoningInspection!))
    }

    if (fingerprint.persistent && (visionInspection || reasoningInspection)) {
      const priorVision = Option.isSome(cached) ? cached.value.visionInspections : []
      const priorReasoning = Option.isSome(cached) ? cached.value.reasoningInspections : []
      const visionInspections = visionInspection
        ? [...priorVision.filter((entry) => entry.routeId !== routeId), visionInspection]
        : priorVision
      const reasoningInspections = reasoningInspection
        ? [...priorReasoning.filter((entry) => entry.routeId !== routeId), reasoningInspection]
        : priorReasoning
      yield* platform.modelIndex.putDiscoveredProperties({
        modelPath: logical.modelPath,
        visionInspections,
        reasoningInspections,
      })
    }

    const currentRouteIds = new Set(logical.routes.map((candidate) => routeKey(logical.providerModelId, candidate)))
    const currentVision = [...(yield* Ref.get(visionByRoute)).entries()]
      .filter(([id]) => currentRouteIds.has(id))
      .map(([, evidence]) => evidence.value)
    const currentReasoningInspections = [...(yield* Ref.get(reasoningByRoute)).entries()]
      .filter(([id]) => currentRouteIds.has(id))
      .map(([, inspection]) => inspection)

    return {
      fingerprint,
      ...(visionInspection ? { vision: currentVision.some(Boolean) } : {}),
      ...(reasoningInspection ? {
        reasoningInspection,
        reasoningDefaultReasoningEffort: reasoningInspection.profile.defaultReasoningEffort,
        reasoning: reasoningEffortsForInspections(currentReasoningInspections),
      } : {}),
    }
  }))

  const routeCanBeAcquired = (route: LlamaLogicalRoute): boolean => route._tag === "Managed"
    ? route.request !== undefined && route.availability._tag === "Available"
    : externalUsable(route)

  const acquireRoute = (
    logical: LogicalLlamaModelRecord,
    route: LlamaLogicalRoute,
  ) => Effect.gen(function* () {
    const providerModelId = logical.providerModelId
    const registry = yield* platform.instances.pipe(Effect.mapError((cause) => acquisitionError(providerModelId, cause)))
    const managed = route._tag === "Managed" ? route : undefined
    if (managed && !managed.request) return yield* acquisitionError(providerModelId, "The managed route is not configured.")
    if (managed && !managed.loaded) {
      yield* Ref.update(demandLoads, (current) => new Set(current).add(providerModelId))
      yield* setManagedSnapshotState(providerModelId, { _tag: "Loading" })
      yield* PubSub.publish(selectionChanges, undefined)
    }
    const clearDemand = Ref.update(demandLoads, (current) => {
      const next = new Set(current)
      next.delete(providerModelId)
      return next
    })
    const lease = yield* (route._tag === "External"
      ? registry.acquire(LlamaCpp.LlamaModelRequest.External({ request: route.request }))
      : Effect.gen(function* () {
        const operation = yield* registry.ensureManagedLoaded(route.request!)
        yield* observeOperation(providerModelId, operation)
        yield* operation.result.pipe(Effect.ensuring(operation.cancel))
        return yield* registry.acquireLoadedManaged(route.request!)
      })).pipe(
        Effect.mapError((cause) => acquisitionError(providerModelId, cause)),
        Effect.tapError(() => managed
          ? clearDemand.pipe(
              Effect.zipRight(setManagedSnapshotState(providerModelId, { _tag: "Failed" })),
              Effect.zipRight(PubSub.publish(selectionChanges, undefined)),
              Effect.asVoid,
            )
          : Effect.void),
      )
    if (managed) {
      yield* Ref.update(logicalModels, (models) => {
        const current = models.get(providerModelId)
        if (!current) return models
        const next = new Map(models)
        const routes = current.routes.map((candidate) => candidate === managed ? { ...candidate, loaded: true } : candidate)
        next.set(providerModelId, { ...current, routes, routeSnapshots: routes.map((candidate) => routeSnapshot(providerModelId, candidate)) })
        return next
      })
      yield* clearDemand
      yield* PubSub.publish(selectionChanges, undefined)
    }
    return lease
  })

  const inspectEligibleRoutes = (
    logical: LogicalLlamaModelRecord,
    requested: ReadonlySet<"vision" | "reasoning">,
  ) => Effect.gen(function* () {
    const routes = logical.routes.filter(routeCanBeAcquired)
    if (routes.length === 0) return yield* acquisitionError(logical.providerModelId, "No usable serving route exists for this model.")
    const outcomes = yield* Effect.forEach(routes, (route) =>
      Effect.scoped(Effect.gen(function* () {
        const lease = yield* acquireRoute(logical, route)
        return yield* inspectLoadedRoute(logical, route, {
          origin: lease.target.origin,
          authorization: lease.target.authorization,
          servedModelId: LlamaServedModelIdSchema.make(lease.target.model),
        }, requested).pipe(Effect.mapError((cause) => acquisitionError(logical.providerModelId, cause)))
      })).pipe(Effect.either), { concurrency: 1 })
    const failures = outcomes.flatMap((outcome) => outcome._tag === "Left" ? [outcome.left] : [])
    const results = outcomes.flatMap((outcome) => outcome._tag === "Right" ? [outcome.right] : [])
    yield* Effect.forEach(failures, (cause) => Effect.logWarning("Local model route inspection failed").pipe(
      Effect.annotateLogs({ providerModelId: logical.providerModelId, cause: String(cause) }),
    ), { discard: true })
    if (results.length === 0) return yield* failures[0]!
    return results.at(-1)!
  })

  const acquire = (providerModelId: ProviderModelId, requestedEffort: typeof ReasoningEffortSchema.Type | undefined): Effect.Effect<LlamaCppInferenceLease, LlamaCppAcquisitionError, Scope.Scope> => Effect.gen(function* () {
    const logical = (yield* Ref.get(logicalModels)).get(providerModelId)
    if (!logical) return yield* acquisitionError(providerModelId, "This local model is no longer available.")
    const model = logical.providerModel
    const knownEfforts = model?.properties.reasoning._tag === "Cached"
      || model?.properties.reasoning._tag === "Resolved"
      || model?.properties.reasoning._tag === "Refreshing"
      ? model.properties.reasoning.value
      : [model?.defaultReasoningEffort ?? LLAMA_DEFAULT_REASONING_EFFORT]
    let effectiveDefault = model?.defaultReasoningEffort ?? LLAMA_DEFAULT_REASONING_EFFORT
    let effectiveEffort = requestedEffort && knownEfforts.includes(requestedEffort)
      ? requestedEffort
      : model?.defaultReasoningEffort ?? LLAMA_DEFAULT_REASONING_EFFORT
    const cachedReasoning = model?.properties.reasoning._tag === "Cached" || model?.properties.reasoning._tag === "Refreshing"
    if (cachedReasoning) {
      const inspection = yield* inspectEligibleRoutes(logical, new Set(["reasoning"] as const))
      const value = inspection.reasoning!
      effectiveDefault = inspection.reasoningDefaultReasoningEffort!
      yield* updateProviderModel(providerModelId, (current) => ({
        ...current,
        defaultReasoningEffort: effectiveDefault,
        properties: { ...current.properties, reasoning: resolveReasoningProperty(current.properties.reasoning, value, "preserve") },
      }))
      if (!value.includes(effectiveEffort)) effectiveEffort = effectiveDefault
    }
    let routeInspections = yield* Ref.get(reasoningByRoute)
    if (![...routeInspections.values()].some((routeInspection) =>
      profileSupportsReasoningEffort(routeInspection.profile, effectiveEffort))) {
      const inspection = yield* inspectEligibleRoutes(logical, new Set(["reasoning"] as const))
      const value = inspection.reasoning!
      effectiveDefault = inspection.reasoningDefaultReasoningEffort!
      if (!value.includes(effectiveEffort)) effectiveEffort = effectiveDefault
      routeInspections = yield* Ref.get(reasoningByRoute)
      yield* updateProviderModel(providerModelId, (current) => ({
        ...current,
        defaultReasoningEffort: effectiveDefault,
        properties: { ...current.properties, reasoning: resolveReasoningProperty(current.properties.reasoning, value, "resolve") },
      }))
    }
    const supportsEffort = (route: LlamaLogicalRoute) => {
      const profile = routeInspections.get(routeKey(providerModelId, route))?.profile
      return profile !== undefined && profileSupportsReasoningEffort(profile, effectiveEffort)
    }
    const eligibleRoutes = logical.routes.filter(supportsEffort)
    const activeRouteId = (yield* Ref.get(activeRouteByModel)).get(providerModelId)
    const lastUsed = yield* Ref.get(routeLastUsed)
    const activeManaged = eligibleRoutes.find((candidate): candidate is ManagedRoute => candidate._tag === "Managed"
      && candidate.loaded && candidate.availability._tag === "Available" && routeKey(providerModelId, candidate) === activeRouteId)
    const loadedExternal = eligibleRoutes.filter((candidate): candidate is ExternalRoute => candidate._tag === "External"
      && candidate.healthy && candidate.observation.status === "loaded").sort((a, b) => a.priority - b.priority)[0]
    const loadedManaged = eligibleRoutes.find((candidate): candidate is ManagedRoute => candidate._tag === "Managed"
      && candidate.loaded && candidate.availability._tag === "Available")
    const sleepingExternal = eligibleRoutes.filter((candidate): candidate is ExternalRoute => candidate._tag === "External"
      && candidate.healthy && candidate.observation.status === "sleeping").sort((a, b) =>
        (lastUsed.get(routeKey(providerModelId, b)) ?? 0) - (lastUsed.get(routeKey(providerModelId, a)) ?? 0)
        || a.priority - b.priority)[0]
    const externalRoute = activeManaged ? undefined : loadedExternal ?? (loadedManaged ? undefined : sleepingExternal)
    const managedRoute = activeManaged ?? (loadedExternal ? undefined : loadedManaged)
      ?? (sleepingExternal ? undefined : eligibleRoutes.find((candidate): candidate is ManagedRoute => candidate._tag === "Managed" && candidate.availability._tag === "Available"))
    if (!externalRoute && !managedRoute) return yield* acquisitionError(providerModelId, "No usable serving route exists for this model.")
    const selectedRoute = externalRoute ?? managedRoute!
    const lease = yield* acquireRoute(logical, selectedRoute)
    const selectedRouteId = routeKey(providerModelId, selectedRoute)
    yield* Ref.update(activeRouteByModel, (current) => new Map(current).set(providerModelId, selectedRouteId))
    yield* Ref.update(routeLastUsed, (current) => new Map(current).set(selectedRouteId, Date.now()))
    const inspection = yield* inspectLoadedRoute(logical, selectedRoute, {
      origin: lease.target.origin,
      authorization: lease.target.authorization,
      servedModelId: LlamaServedModelIdSchema.make(lease.target.model),
    }, new Set(["reasoning"])).pipe(
      Effect.mapError((cause) => acquisitionError(providerModelId, cause)),
    )
    const selectedInspection = inspection.reasoningInspection!
    const value = inspection.reasoning!
    effectiveDefault = selectedInspection.profile.defaultReasoningEffort
    const selection = Option.getOrUndefined(resolveReasoningMapping(selectedInspection.profile, effectiveEffort))
    if (!selection) return yield* acquisitionError(providerModelId, "The selected route has no reasoning mapping for its default effort.")
    effectiveEffort = selection.reasoningEffort
    const reasoningRequestOptions = llamaCppRequestOptionsForReasoningMapping(selection.mapping)
    yield* updateProviderModel(providerModelId, (current) => ({
      ...current,
      defaultReasoningEffort: effectiveDefault,
      properties: { ...current.properties, reasoning: resolveReasoningProperty(current.properties.reasoning, value, "preserve") },
    }))
    return {
      providerModelId,
      routeId: LlamaServingRouteIdSchema.make(externalRoute
        ? `external:${externalRoute.request.instanceId}:${externalRoute.request.servedModelId}`
        : `managed:${managedRoute!.record.id}`),
      origin: lease.target.origin,
      authorization: lease.target.authorization,
      servedModelId: LlamaServedModelIdSchema.make(lease.target.model),
      reasoningEffort: effectiveEffort,
      chatTemplateKwargs: reasoningRequestOptions.chatTemplateKwargs,
      thinkingBudgetTokens: reasoningRequestOptions.thinkingBudgetTokens,
    }
  })

  const startModelPropertyDiscovery = (request: Parameters<LocalModelProviderSourceApi["discoverModelProperties"]>[0], discoveryKey: string) => Effect.gen(function* () {
    const operationId = ModelDiscoveryOperationIdSchema.make(randomUUID())
    const requested = new Set(request.properties)
    const begin = updateProviderModel(request.providerModelId, (model) => ({
      ...model,
      properties: {
        vision: requested.has("vision")
          ? VisionProperty.Lifecycle.match(model.properties.vision, {
              Deferred: (state) => VisionProperty.Lifecycle.transition(state, "Discovering", { operationId, phase: "loading" }),
              Discovering: (state) => VisionProperty.Lifecycle.hold(state, { operationId, phase: "loading" }),
              Cached: (state) => VisionProperty.Lifecycle.transition(state, "Refreshing", { operationId, phase: "loading" }),
              Resolved: (state) => VisionProperty.Lifecycle.transition(state, "Refreshing", { operationId, phase: "loading" }),
              Refreshing: (state) => VisionProperty.Lifecycle.hold(state, { operationId, phase: "loading" }),
              Failed: (state) => VisionProperty.Lifecycle.transition(state, "Discovering", { operationId, phase: "loading" }),
            })
          : model.properties.vision,
        reasoning: requested.has("reasoning")
          ? ReasoningProperty.Lifecycle.match(model.properties.reasoning, {
              Deferred: (state) => ReasoningProperty.Lifecycle.transition(state, "Discovering", { operationId, phase: "loading" }),
              Discovering: (state) => ReasoningProperty.Lifecycle.hold(state, { operationId, phase: "loading" }),
              Cached: (state) => ReasoningProperty.Lifecycle.transition(state, "Refreshing", { operationId, phase: "loading" }),
              Resolved: (state) => ReasoningProperty.Lifecycle.transition(state, "Refreshing", { operationId, phase: "loading" }),
              Refreshing: (state) => ReasoningProperty.Lifecycle.hold(state, { operationId, phase: "loading" }),
              Failed: (state) => ReasoningProperty.Lifecycle.transition(state, "Discovering", { operationId, phase: "loading" }),
            })
          : model.properties.reasoning,
      },
    }))
    yield* begin

    const work = Effect.gen(function* () {
      yield* updateProviderModel(request.providerModelId, (model) => ({
        ...model,
        properties: {
          vision: requested.has("vision") && (model.properties.vision._tag === "Discovering" || model.properties.vision._tag === "Refreshing")
            ? VisionProperty.Lifecycle.hold(model.properties.vision, { phase: "inspecting" })
            : model.properties.vision,
          reasoning: requested.has("reasoning") && (model.properties.reasoning._tag === "Discovering" || model.properties.reasoning._tag === "Refreshing")
            ? ReasoningProperty.Lifecycle.hold(model.properties.reasoning, { phase: "inspecting" })
            : model.properties.reasoning,
        },
      }))
      const logical = (yield* Ref.get(logicalModels)).get(request.providerModelId)
      if (!logical) return yield* new LlamaCpp.LlamaCppReasoningInspectionError({ message: "The llama.cpp model is no longer registered" })
      const inspection = yield* inspectEligibleRoutes(logical, requested)
      yield* updateProviderModel(request.providerModelId, (model) => ({
        ...model,
        ...(requested.has("reasoning") && inspection.reasoningDefaultReasoningEffort !== undefined
          ? { defaultReasoningEffort: inspection.reasoningDefaultReasoningEffort }
          : {}),
        properties: {
          vision: requested.has("vision") && inspection.vision !== undefined
            && (model.properties.vision._tag === "Discovering" || model.properties.vision._tag === "Refreshing")
            ? VisionProperty.Lifecycle.transition(model.properties.vision, "Resolved", { value: inspection.vision })
            : model.properties.vision,
          reasoning: requested.has("reasoning") && inspection.reasoning !== undefined
            && (model.properties.reasoning._tag === "Discovering" || model.properties.reasoning._tag === "Refreshing")
            ? ReasoningProperty.Lifecycle.transition(model.properties.reasoning, "Resolved", { value: inspection.reasoning })
            : model.properties.reasoning,
        },
      }))
    }).pipe(
      Effect.catchAll((cause) => Effect.logWarning("Local model property discovery failed").pipe(
        Effect.annotateLogs({ providerModelId: request.providerModelId, cause: String(cause) }),
        Effect.zipRight(updateProviderModel(request.providerModelId, (model) => {
          const error = { code: "discovery_failed", message: cause instanceof Error ? cause.message : String(cause), retryable: true }
          return {
            ...model,
            properties: {
              vision: requested.has("vision")
                ? model.properties.vision._tag === "Refreshing"
                  ? VisionProperty.Lifecycle.transition(model.properties.vision, "Cached", {})
                  : model.properties.vision._tag === "Discovering"
                    ? VisionProperty.Lifecycle.transition(model.properties.vision, "Failed", { error })
                    : model.properties.vision
                : model.properties.vision,
              reasoning: requested.has("reasoning")
                ? model.properties.reasoning._tag === "Refreshing"
                  ? ReasoningProperty.Lifecycle.transition(model.properties.reasoning, "Cached", {})
                  : model.properties.reasoning._tag === "Discovering"
                    ? ReasoningProperty.Lifecycle.transition(model.properties.reasoning, "Failed", { error })
                    : model.properties.reasoning
                : model.properties.reasoning,
            },
          }
        })),
      )),
    )
    yield* Ref.update(discoveryOperations, (current) => new Map(current).set(discoveryKey, operationId))
    yield* Effect.forkIn(work.pipe(Effect.ensuring(Ref.update(discoveryOperations, (current) => {
      const next = new Map(current)
      next.delete(discoveryKey)
      return next
    }))), serviceScope)
    return operationId
  })
  const discoverModelProperties: LocalModelProviderSourceApi["discoverModelProperties"] = (request) => discoveryLock.withPermits(1)(Effect.gen(function* () {
    const discoveryKey = `${request.providerModelId}\0${[...request.properties].sort().join(",")}`
    const existing = (yield* Ref.get(discoveryOperations)).get(discoveryKey)
    if (existing) return existing
    return yield* startModelPropertyDiscovery(request, discoveryKey)
  }))

  const selectionInput = Effect.all([Ref.get(logicalModels), Ref.get(demandLoads)]).pipe(Effect.map(([models, loading]) => ({
    authoritativeModelIds: new Set([...models.values()]
      .filter((model) => model.routes.length > 0)
      .map((model) => model.providerModelId)),
    candidates: [...models.values()].map((model): LocalSlotCandidate => {
      const managed = model.routes.filter((route): route is ManagedRoute => route._tag === "Managed")
      const external = model.routes.filter((route): route is ExternalRoute => route._tag === "External")
      return {
        providerModelId: model.providerModelId,
        availability: model.providerModel?.availability._tag === "Available" ? "available" : "disabled",
        externalLoaded: external.some((route) => route.healthy && route.observation.status === "loaded"),
        managedLoaded: managed.some((route) => route.loaded && route.availability._tag === "Available"),
        sleeping: external.some((route) => route.healthy && route.observation.status === "sleeping"),
        managedRestorable: managed.some((route) => route.availability._tag === "Available"),
        demandLoading: loading.has(model.providerModelId),
        productRank: Math.min(...managed.map((route) => route.productRank), Number.MAX_SAFE_INTEGER),
        externalPriority: Math.min(...external.map((route) => route.priority), Number.MAX_SAFE_INTEGER),
      }
    }),
  })))

  const warm = (providerModelId: ProviderModelId) => Effect.scoped(acquire(providerModelId, undefined)).pipe(Effect.asVoid)
  const status = platform.instances.pipe(
    Effect.flatMap((registry) => registry.snapshot),
    Effect.map((snapshot) => snapshot.instances.some((instance) => instance.health === "ready")
      ? { status: "ok" as const }
      : { status: "not_found" as const, message: "No local model is currently serving." }),
  )
  yield* rebuild("cached", false).pipe(Effect.catchAll((cause) => Effect.logWarning("Cached local model projection failed").pipe(
    Effect.annotateLogs({ operation: "hydrate-local-model-catalog", cause: String(cause) }),
  )))
  yield* Stream.fromPubSub(fitReconcileRequests).pipe(
    Stream.debounce("25 millis"),
    Stream.runForEach(() => reconcileFitAssessments.pipe(
      Effect.catchAll((cause) => Effect.logWarning("Local model fit reconciliation failed").pipe(
        Effect.annotateLogs({ cause: String(cause) }),
      )),
    )),
    Effect.forkIn(serviceScope),
  )
  yield* Effect.forkIn(refresh.pipe(Effect.catchAll((cause) => Effect.logWarning("Initial local model discovery failed").pipe(Effect.annotateLogs({ cause: String(cause) })))), serviceScope)
  yield* Effect.forkIn(Effect.gen(function* () {
    const registry = yield* platform.instances
    yield* platform.files.changes.pipe(
      Stream.debounce("25 millis"),
      Stream.runForEach(() => rebuild("cached", true).pipe(
        Effect.tap(() => requestFitReconciliation),
        Effect.catchAll((cause) => Effect.logWarning("Local model projection update failed").pipe(
          Effect.annotateLogs({ cause: String(cause) }),
        )),
      )),
      Effect.forkScoped,
    )
    yield* platform.installations.changes.pipe(
      Stream.debounce("25 millis"),
      Stream.runForEach(() => rebuild("cached", true, true).pipe(
          Effect.tap(() => requestFitReconciliation),
          Effect.catchAll((cause) => Effect.logWarning("Local installation-dependent model update failed").pipe(
          Effect.annotateLogs({ cause: String(cause) }),
        )),
      )),
      Effect.forkScoped,
    )
    yield* Stream.concat(
      Stream.make({ registry, reassessManaged: false }),
      platform.instanceChanges.pipe(Stream.map((current) => ({ registry: current, reassessManaged: true }))),
    ).pipe(
      Stream.flatMap(({ registry: current, reassessManaged }) => Stream.concat(
        Stream.make(reassessManaged),
        current.changes.pipe(Stream.map(() => false)),
      ), {
        concurrency: 1,
        switch: true,
      }),
      Stream.debounce("25 millis"),
      Stream.runForEach((reassessManaged) => rebuild("cached", true, reassessManaged).pipe(
        Effect.tap(() => reassessManaged ? requestFitReconciliation : Effect.void),
        Effect.catchAll((cause) => Effect.logWarning("Local serving projection update failed").pipe(
          Effect.annotateLogs({ cause: String(cause) }),
        )),
      )),
    )
  }), serviceScope)

  return LocalModelProviderSource.of({
    catalog: {
      list,
      refresh,
      get: (providerId, providerModelId) => providerId !== "llamacpp"
        ? Effect.fail(new ModelCatalogError({ message: `Model not found: ${providerId}/${providerModelId}` }))
        : Ref.get(logicalModels).pipe(Effect.flatMap((models) => {
          const model = models.get(providerModelId)?.providerModel
          return model ? Effect.succeed(model) : Effect.fail(new ModelCatalogError({ message: `Model not found: ${providerId}/${providerModelId}` }))
        })),
    },
    discoverModelProperties,
    acquire,
    warm,
    operations: Ref.get(operations).pipe(Effect.map((current) => [...current.values()])),
    status,
    stopManaged: platform.instances.pipe(Effect.flatMap((registry) => registry.stopManaged)),
    logicalModels: Ref.get(logicalModels),
    getModelInformation: (providerModelId) => Ref.get(logicalModels).pipe(Effect.flatMap((models) => {
      const information = models.get(providerModelId)?.information
      return information
        ? Effect.succeed(information)
        : Effect.fail(new ModelCatalogError({ message: `Model information not found: llamacpp/${providerModelId}` }))
    })),
    selectionInput,
    selectionReady: Ref.get(selectionReady),
    selectionChanges: Stream.fromPubSub(selectionChanges),
    changes: Stream.fromPubSub(catalogChanges),
    stateChanges: Stream.mergeAll([
      Stream.fromPubSub(catalogChanges),
      Stream.fromPubSub(selectionChanges),
      Stream.fromPubSub(operationChanges),
      Stream.fromPubSub(fitChanges),
    ], { concurrency: "unbounded" }),
  })
}))
