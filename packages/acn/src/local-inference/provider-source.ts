import { Context, Effect, Layer, Option, PubSub, Ref, Schema, Scope, Stream } from "effect"
import {
  AVAILABLE_PROVIDER_MODEL,
  LlamaCppAcquisitionError,
  LlamaCppModelInfoSchema,
  ModelCatalogError,
  type LlamaCppInferenceLease,
  type LlamaCppModelInfo,
  type LlamaCppProviderSource,
  type ProviderModelAvailability,
  classifyModelFamilyFromEvidence,
} from "@magnitudedev/sdk"
import { LlamaCpp, ModelFiles } from "@magnitudedev/local-inference"
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
const disabled = (reason: "insufficient_resources" | "provider_unavailable" | "model_unavailable" | "incompatible_runtime" | "invalid_configuration"): ProviderModelAvailability => ({ _tag: "Disabled", reason })

type ManagedRoute = {
  readonly _tag: "Managed"
  readonly request?: LlamaCpp.ManagedModelRequest
  readonly modelPath: LlamaCpp.NormalizedLlamaModelPath
  readonly record: ModelFiles.ModelFileRecord
  readonly loaded: boolean
  readonly availability: ProviderModelAvailability
  readonly productRank: number
}

type ExternalRoute = {
  readonly _tag: "External"
  readonly request: LlamaCpp.ExternalModelRequest
  readonly modelPath: LlamaCpp.NormalizedLlamaModelPath
  readonly observation: LlamaCpp.LlamaServedModelObservation
  readonly healthy: boolean
  readonly priority: number
}

export type LlamaLogicalRoute = ManagedRoute | ExternalRoute

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
    routeId: Schema.String,
    providerModelId: Schema.String,
    modelPath: Schema.String,
    modelFileId: Schema.String,
    state: LlamaRouteRuntimeStateSchema,
    availability: Schema.Literal("available", "disabled"),
    productRank: Schema.NonNegativeInt,
  }),
  Schema.TaggedStruct("External", {
    routeId: Schema.String,
    providerModelId: Schema.String,
    modelPath: Schema.String,
    serverId: Schema.String,
    servedModelId: Schema.String,
    state: LlamaRouteRuntimeStateSchema,
    healthy: Schema.Boolean,
    priority: Schema.NonNegativeInt,
  }),
)
export type LlamaServingRouteSnapshot = Schema.Schema.Type<typeof LlamaServingRouteSnapshotSchema>

const OpportunisticModelMetadataSchema = Schema.Struct({
  name: Schema.optional(Schema.String),
  architecture: Schema.optional(Schema.String),
  tokenizerModel: Schema.optional(Schema.String),
  tokenizerPre: Schema.optional(Schema.String),
  baseModelNames: Schema.Array(Schema.String),
  baseModelRepositories: Schema.Array(Schema.String),
})
export const LlamaModelMetadataEvidenceSchema = Schema.Union(
  Schema.TaggedStruct("IndexedArtifact", {
    modelFileId: Schema.String,
    artifactDisplayName: Schema.String,
    metadata: OpportunisticModelMetadataSchema,
  }),
  Schema.TaggedStruct("SourceManifest", {
    sourceId: Schema.String,
    metadata: OpportunisticModelMetadataSchema,
  }),
  Schema.TaggedStruct("ServerReported", {
    routeId: Schema.String,
    alias: Schema.String,
    metadata: OpportunisticModelMetadataSchema,
  }),
)
export type LlamaModelMetadataEvidence = Schema.Schema.Type<typeof LlamaModelMetadataEvidenceSchema>
const ModelMetadataSourceSchema = Schema.Literal("indexed_artifact", "source_manifest", "server_reported")
const ResolvedStringSchema = Schema.Struct({ value: Schema.String, source: ModelMetadataSourceSchema })
export const ResolvedLlamaModelInformationSchema = Schema.Struct({
  providerModelId: Schema.String,
  displayName: Schema.String,
  displayNameSource: Schema.Literal("gguf_metadata", "source_manifest", "server_metadata", "server_alias", "path_basename"),
  metadata: Schema.Struct({
    name: Schema.optional(ResolvedStringSchema),
    architecture: Schema.optional(ResolvedStringSchema),
    tokenizerModel: Schema.optional(ResolvedStringSchema),
    tokenizerPre: Schema.optional(ResolvedStringSchema),
    baseModelNames: Schema.Array(Schema.String),
    baseModelRepositories: Schema.Array(Schema.String),
  }),
})
export type ResolvedLlamaModelInformation = Schema.Schema.Type<typeof ResolvedLlamaModelInformationSchema>
const LogicalProjectionSignatureSchema = Schema.Array(Schema.Struct({
  id: Schema.String,
  information: ResolvedLlamaModelInformationSchema,
  model: Schema.optional(LlamaCppModelInfoSchema),
  routes: Schema.Array(LlamaServingRouteSnapshotSchema),
}))

export interface LogicalLlamaModelRecord {
  readonly providerModelId: string
  readonly modelPath: LlamaCpp.NormalizedLlamaModelPath
  readonly routes: readonly LlamaLogicalRoute[]
  readonly routeSnapshots: readonly LlamaServingRouteSnapshot[]
  readonly evidence: readonly LlamaModelMetadataEvidence[]
  readonly information: ResolvedLlamaModelInformation
  readonly providerModel?: LlamaCppModelInfo
}

export interface LocalModelProviderSourceApi extends LlamaCppProviderSource {
  readonly warm: (providerModelId: string) => Effect.Effect<void, LlamaCppAcquisitionError>
  readonly operations: Effect.Effect<readonly LocalInferenceOperationSnapshot[]>
  readonly stopManaged: Effect.Effect<void, LlamaCpp.LlamaControlError | LlamaCpp.ModelInUse | LlamaCpp.LlamaDistributionError>
  readonly logicalModels: Effect.Effect<ReadonlyMap<string, LogicalLlamaModelRecord>>
  readonly getModelInformation: (providerModelId: string) => Effect.Effect<ResolvedLlamaModelInformation, ModelCatalogError>
  readonly selectionInput: Effect.Effect<LocalModelReconciliationInput>
  readonly selectionReady: Effect.Effect<boolean>
  readonly selectionChanges: Stream.Stream<void>
  readonly changes: Stream.Stream<void>
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

const acquisitionError = (modelId: string, cause: unknown) => new LlamaCppAcquisitionError({
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

const routeSnapshot = (providerModelId: string, route: LlamaLogicalRoute): LlamaServingRouteSnapshot => route._tag === "Managed"
  ? {
      _tag: "Managed",
      routeId: `managed:${route.record.id}`,
      providerModelId,
      modelPath: String(route.modelPath),
      modelFileId: String(route.record.id),
      state: route.loaded ? { _tag: "Loaded" } : { _tag: "Unloaded" },
      availability: route.availability._tag === "Available" ? "available" : "disabled",
      productRank: route.productRank,
    }
  : {
      _tag: "External",
      routeId: `external:${route.request.instanceId}:${route.request.servedModelId}`,
      providerModelId,
      modelPath: String(route.modelPath),
      serverId: String(route.request.instanceId),
      servedModelId: String(route.request.servedModelId),
      state: routeState(route.observation.status),
      healthy: route.healthy,
      priority: route.priority,
    }

const bestDisabledReason = (routes: readonly LlamaLogicalRoute[]): ProviderModelAvailability => {
  if (routes.some((route) => route._tag === "Managed" && route.availability._tag === "Available")) return AVAILABLE_PROVIDER_MODEL
  if (routes.some((route) => route._tag === "External" && externalUsable(route))) return AVAILABLE_PROVIDER_MODEL
  const reasons = routes.flatMap((route) => route._tag === "Managed" && route.availability._tag === "Disabled" ? [route.availability.reason] : [])
  for (const reason of ["invalid_configuration", "incompatible_runtime", "insufficient_resources", "provider_unavailable", "model_unavailable"] as const) {
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
  providerModelId: string,
  path: LlamaCpp.NormalizedLlamaModelPath,
  routes: readonly LlamaLogicalRoute[],
): { readonly information: ResolvedLlamaModelInformation; readonly evidence: readonly LlamaModelMetadataEvidence[] } => {
  const managed = routes.filter((route): route is ManagedRoute => route._tag === "Managed")
  const external = routes.filter((route): route is ExternalRoute => route._tag === "External").sort((a, b) => a.priority - b.priority)
  const evidence: readonly LlamaModelMetadataEvidence[] = [
    ...managed.map((route) => ({
      _tag: "IndexedArtifact" as const,
      modelFileId: String(route.record.id),
      artifactDisplayName: route.record.displayName,
      metadata: {
        ...(Option.isSome(route.record.metadata.name) ? { name: route.record.metadata.name.value } : {}),
        ...(Option.isSome(route.record.metadata.architecture) ? { architecture: route.record.metadata.architecture.value } : {}),
        ...(Option.isSome(route.record.metadata.tokenizerModel) ? { tokenizerModel: route.record.metadata.tokenizerModel.value } : {}),
        ...(Option.isSome(route.record.metadata.tokenizerPre) ? { tokenizerPre: route.record.metadata.tokenizerPre.value } : {}),
        baseModelNames: route.record.metadata.baseModelNames,
        baseModelRepositories: route.record.metadata.baseModelRepositories,
      },
    })),
    ...external.map((route) => ({
      _tag: "ServerReported" as const,
      routeId: `${route.request.instanceId}\0${route.request.servedModelId}`,
      alias: String(route.observation.id),
      metadata: {
        ...(Option.isSome(route.observation.serverDisplayName) ? { name: route.observation.serverDisplayName.value } : {}),
        ...(Option.isSome(route.observation.architecture) ? { architecture: route.observation.architecture.value } : {}),
        baseModelNames: [],
        baseModelRepositories: [],
      },
    })),
  ]
  const resolveString = (key: "name" | "architecture" | "tokenizerModel" | "tokenizerPre") => {
    const indexed = evidence.find((item) => item._tag === "IndexedArtifact" && item.metadata[key] !== undefined)
    if (indexed?.metadata[key] !== undefined) return { value: indexed.metadata[key], source: "indexed_artifact" as const }
    const manifest = evidence.find((item) => item._tag === "SourceManifest" && item.metadata[key] !== undefined)
    if (manifest?.metadata[key] !== undefined) return { value: manifest.metadata[key], source: "source_manifest" as const }
    const server = evidence.find((item) => item._tag === "ServerReported" && item.metadata[key] !== undefined)
    return server?.metadata[key] !== undefined ? { value: server.metadata[key], source: "server_reported" as const } : undefined
  }
  const metadataName = resolveString("name")
  const serverName = external.map((route) => Option.getOrUndefined(route.observation.serverDisplayName)).find((name): name is string => name !== undefined)
  const alias = external.map((route) => String(route.observation.id)).find((name) => !isOpaqueLlamaRoutingName(name))
  const display = metadataName?.source === "indexed_artifact"
    ? { value: metadataName.value, source: "gguf_metadata" as const }
    : metadataName?.source === "source_manifest"
      ? { value: metadataName.value, source: "source_manifest" as const }
    : serverName
        ? { value: serverName, source: "server_metadata" as const }
        : alias
          ? { value: alias, source: "server_alias" as const }
          : { value: basename(path), source: "path_basename" as const }
  return { evidence, information: {
    providerModelId,
    displayName: display.value,
    displayNameSource: display.source,
    metadata: {
      ...(metadataName ? { name: metadataName } : {}),
      ...(resolveString("architecture") ? { architecture: resolveString("architecture")! } : {}),
      ...(resolveString("tokenizerModel") ? { tokenizerModel: resolveString("tokenizerModel")! } : {}),
      ...(resolveString("tokenizerPre") ? { tokenizerPre: resolveString("tokenizerPre")! } : {}),
      baseModelNames: evidence.flatMap((item) => item.metadata.baseModelNames),
      baseModelRepositories: evidence.flatMap((item) => item.metadata.baseModelRepositories),
    },
  } }
}

export const LocalModelProviderSourceLive: Layer.Layer<
  LocalModelProviderSource,
  never,
  LocalInferencePlatform | LocalModelConfiguration
> = Layer.scoped(LocalModelProviderSource, Effect.gen(function* () {
  const platform = yield* LocalInferencePlatform
  const configuration = yield* LocalModelConfiguration
  const logicalModels = yield* Ref.make<ReadonlyMap<string, LogicalLlamaModelRecord>>(new Map())
  const logicalSignature = yield* Ref.make("")
  const catalogSignature = yield* Ref.make("")
  const demandLoads = yield* Ref.make<ReadonlySet<string>>(new Set())
  const selectionReady = yield* Ref.make(false)
  const operations = yield* Ref.make<ReadonlyMap<string, LocalInferenceOperationSnapshot>>(new Map())
  const catalogChanges = yield* PubSub.unbounded<void>()
  const selectionChanges = yield* PubSub.unbounded<void>()
  const serviceScope = yield* Scope.Scope
  const setManagedSnapshotState = (providerModelId: string, state: LlamaServingRouteSnapshot["state"]) => Ref.update(logicalModels, (models) => {
    const current = models.get(providerModelId)
    if (!current) return models
    const next = new Map(models)
    next.set(providerModelId, {
      ...current,
      routeSnapshots: current.routeSnapshots.map((route) => route._tag === "Managed" ? { ...route, state } : route),
    })
    return next
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

  const observeOperation = (providerModelId: string, operation: LlamaCpp.LlamaLoadOperation) => Effect.gen(function* () {
    const id = String(operation.id)
    if ((yield* Ref.get(operations)).has(id)) return
    yield* Ref.update(operations, (current) => new Map(current).set(id, {
      operationId: id, providerModelId, status: "running", stage: "queued",
    }))
    yield* Effect.forkIn(operation.events.pipe(Stream.runForEach((event) => Ref.update(operations, (current) => new Map(current).set(id, {
      operationId: id,
      providerModelId,
      status: event._tag === "Loaded" ? "completed" : "running",
      stage: operationStage(event),
      ...(event._tag === "Loading" && Option.isSome(event.progress) ? { progress: event.progress.value } : {}),
    })))), serviceScope)
    yield* Effect.forkIn(operation.result.pipe(
      Effect.tap(() => Ref.update(operations, (current) => new Map(current).set(id, { operationId: id, providerModelId, status: "completed", stage: "loaded" }))),
      Effect.tapError((cause) => Ref.update(operations, (current) => new Map(current).set(id, { operationId: id, providerModelId, status: "failed", stage: "loading", message: cause.reason }))),
      Effect.ignore,
    ), serviceScope)
  })

  const rebuild = (fileRefresh: ModelFiles.ModelFileRefresh, assessAvailability: boolean) => Effect.gen(function* () {
    const records = (yield* platform.files.inspect(fileRefresh)).records
    const registry = assessAvailability
      ? Option.some(yield* platform.instances)
      : Option.none<LlamaCpp.LlamaInstanceRegistryApi>()
    const runtime = assessAvailability && Option.isSome(registry)
      ? Option.some(yield* registry.value.inspect)
      : Option.none<LlamaCpp.LlamaInstanceSnapshot>()
    const cli = assessAvailability ? yield* platform.cli.pipe(Effect.option) : Option.none<LlamaCpp.LlamaCli>()
    const groups = new Map<string, { path: LlamaCpp.NormalizedLlamaModelPath; routes: LlamaLogicalRoute[] }>()
    const addRoute = (providerModelId: string, path: LlamaCpp.NormalizedLlamaModelPath, route: LlamaLogicalRoute) => {
      const group = groups.get(providerModelId) ?? { path, routes: [] }
      group.routes.push(route)
      groups.set(providerModelId, group)
    }

    yield* Effect.forEach(records, (record) => Effect.gen(function* () {
      const resolved = yield* platform.files.resolve(record.id).pipe(Effect.option)
      if (Option.isNone(resolved)) return
      const modelPath = LlamaCpp.normalizeLlamaModelPath(resolved.value.primaryPath)
      if (!modelPath || !LlamaCpp.isAbsoluteLlamaModelPath(modelPath)) return
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
          ? yield* Option.match(cli, {
            onNone: () => Effect.succeed(disabled("provider_unavailable")),
            onSome: (availableCli) => availableCli.assessFit({ files: resolved.value, profile: request.profile }).pipe(
              Effect.flatMap((fit) => {
                if (fit._tag !== "Measured") return Effect.succeed(disabled("incompatible_runtime"))
                if (loaded) return Effect.succeed(AVAILABLE_PROVIDER_MODEL)
                return Effect.all([platform.hardware.inspect, availableCli.listDevices], { concurrency: 2 }).pipe(
                  Effect.map(([host, devices]) => {
                    const deviceById = new Map(devices.devices.map((device) => [String(device.id), device]))
                    const fitsLivePressure = fit.plan.placement.every((placement) => {
                      const required = placement.modelBytes + placement.contextBytes + placement.computeBytes
                      if (String(placement.device) === "Host") {
                        const reserve = Math.max(8 * 1024 ** 3, host.totalMemoryBytes * 0.2)
                        return required <= Math.max(0, host.availableMemoryBytes - reserve)
                      }
                      const device = deviceById.get(String(placement.device))
                      if (!device || Option.isNone(device.freeMemoryBytes)) return true
                      const free = device.freeMemoryBytes.value
                      const reserve = Math.max(1024 ** 3, Option.getOrElse(device.totalMemoryBytes, () => free) * 0.1)
                      return required <= Math.max(0, free - reserve)
                    })
                    return fitsLivePressure ? AVAILABLE_PROVIDER_MODEL : disabled("insufficient_resources")
                  }),
                )
              }),
              Effect.catchAll(() => Effect.succeed(disabled("incompatible_runtime"))),
            ),
          })
          : AVAILABLE_PROVIDER_MODEL
      addRoute(providerModelId, modelPath, { _tag: "Managed", ...(request ? { request } : {}), modelPath, record, loaded, availability, productRank: productRankFor(record) })
    }), { concurrency: 4, discard: true })

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
          })
        }
      }
    }

    const previous = yield* Ref.get(logicalModels)
    const selectedIds = new Set(Object.values((yield* configuration.getModels)?.slots ?? {}).flatMap((slot) => slot?.providerId === "llamacpp" && slot.providerModelId ? [slot.providerModelId] : []))
    for (const id of selectedIds) {
      if (groups.has(id)) continue
      const old = previous.get(id)
      if (old) groups.set(id, { path: old.modelPath, routes: [] })
    }

    const next = new Map<string, LogicalLlamaModelRecord>()
    for (const [providerModelId, group] of groups) {
      const retained = group.routes.length === 0 ? previous.get(providerModelId) : undefined
      const resolvedInformation = retained
        ? { information: retained.information, evidence: retained.evidence }
        : resolveLlamaModelInformation(providerModelId, group.path, group.routes)
      const information = resolvedInformation.information
      const managed = group.routes.filter((route): route is ManagedRoute => route._tag === "Managed")
      const external = group.routes.filter((route): route is ExternalRoute => route._tag === "External")
      const context = firstSome([
        ...external.filter(externalUsable).map((route) => route.observation.activeContextTokens),
        ...managed.map((route) => route.record.metadata.trainedContextTokens),
      ])
      const modalities = firstSome([
        ...managed.map((route) => route.record.metadata.inputModalities),
        ...external.map((route) => route.observation.inputModalities),
      ])
      const providerModel: LlamaCppModelInfo | undefined = retained?.providerModel
        ? { ...retained.providerModel, availability: disabled("model_unavailable") }
        : Option.isSome(context) ? {
        providerId: "llamacpp",
        providerModelId,
        displayName: information.displayName,
        ...Option.match(classifyModelFamilyFromEvidence({
          architecture: information.metadata.architecture?.value,
          tokenizerModel: information.metadata.tokenizerModel?.value,
          tokenizerPre: information.metadata.tokenizerPre?.value,
        }, [information.metadata.name?.value, ...information.metadata.baseModelNames, ...information.metadata.baseModelRepositories]), {
          onNone: () => ({}),
          onSome: (modelFamilyId) => ({ modelFamilyId }),
        }),
        contextWindow: context.value,
        maxOutputTokens: Math.min(context.value, 8192),
        capabilities: Option.isSome(modalities) ? { vision: modalities.value.includes("image") } : {},
        availability: bestDisabledReason(group.routes),
        pricing: ZERO_PRICING,
        reasoningEfforts: ["none"],
        } : undefined
      next.set(providerModelId, { providerModelId, modelPath: group.path, routes: group.routes, routeSnapshots: group.routes.map((route) => routeSnapshot(providerModelId, route)), evidence: resolvedInformation.evidence, information, ...(providerModel ? { providerModel } : {}) })
    }
    const signature = yield* Schema.encode(Schema.parseJson(LogicalProjectionSignatureSchema))([...next.values()].map((record) => ({
      id: record.providerModelId,
      information: record.information,
      ...(record.providerModel ? { model: record.providerModel } : {}),
      routes: record.routeSnapshots,
    })))
    const nextCatalogSignature = yield* Schema.encode(Schema.parseJson(Schema.Array(LlamaCppModelInfoSchema)))(
      [...next.values()].flatMap((record) => record.providerModel ? [record.providerModel] : []),
    )
    const becameReady = assessAvailability && !(yield* Ref.get(selectionReady))
    if (becameReady) yield* Ref.set(selectionReady, true)
    if ((yield* Ref.get(logicalSignature)) !== signature) {
      yield* Ref.set(logicalModels, next)
      yield* Ref.set(logicalSignature, signature)
      yield* PubSub.publish(selectionChanges, undefined)
    }
    if (becameReady) yield* PubSub.publish(selectionChanges, undefined)
    if ((yield* Ref.get(catalogSignature)) !== nextCatalogSignature) {
      yield* Ref.set(catalogSignature, nextCatalogSignature)
      yield* PubSub.publish(catalogChanges, undefined)
    }
    return yield* Schema.decodeUnknown(Schema.Array(LlamaCppModelInfoSchema))(
      [...next.values()].flatMap((record) => record.providerModel ? [record.providerModel] : []),
    )
  }).pipe(Effect.mapError((cause) => new ModelCatalogError({ message: cause instanceof Error ? cause.message : String(cause), cause })))

  const refresh = yield* Effect.cachedWithTTL(rebuild("changed", true), "250 millis")
  const list = Ref.get(logicalModels).pipe(Effect.map((models) => [...models.values()].flatMap((record) => record.providerModel ? [record.providerModel] : [])))

  const acquire = (providerModelId: string): Effect.Effect<LlamaCppInferenceLease, LlamaCppAcquisitionError, Scope.Scope> => Effect.gen(function* () {
    const logical = (yield* Ref.get(logicalModels)).get(providerModelId)
    if (!logical) return yield* acquisitionError(providerModelId, "This local model is no longer available.")
    const loadedManagedRoute = logical.routes.find((candidate): candidate is ManagedRoute => candidate._tag === "Managed" && candidate.loaded && candidate.availability._tag === "Available")
    const externalRoute = loadedManagedRoute ? undefined : logical.routes.filter((candidate): candidate is ExternalRoute => candidate._tag === "External" && externalUsable(candidate)).sort((a, b) => a.priority - b.priority)[0]
    const managedRoute = loadedManagedRoute ?? logical.routes.find((candidate): candidate is ManagedRoute => candidate._tag === "Managed" && candidate.availability._tag === "Available")
    if (!externalRoute && !managedRoute) return yield* acquisitionError(providerModelId, "No usable serving route exists for this model.")
    const registry = yield* platform.instances.pipe(Effect.mapError((cause) => acquisitionError(providerModelId, cause)))
    if (managedRoute && !externalRoute && !managedRoute.loaded) {
      yield* Ref.update(demandLoads, (current) => new Set(current).add(providerModelId))
      yield* setManagedSnapshotState(providerModelId, { _tag: "Loading" })
      yield* PubSub.publish(selectionChanges, undefined)
    }
    const clearDemand = Ref.update(demandLoads, (current) => {
      const next = new Set(current)
      next.delete(providerModelId)
      return next
    })
    const lease = yield* (externalRoute
      ? registry.acquire(LlamaCpp.LlamaModelRequest.External({ request: externalRoute.request }))
      : Effect.gen(function* () {
        const operation = yield* registry.ensureManagedLoaded(managedRoute!.request!)
        yield* observeOperation(providerModelId, operation)
        yield* operation.result.pipe(Effect.ensuring(operation.cancel))
        return yield* registry.acquireLoadedManaged(managedRoute!.request!)
      })).pipe(
        Effect.mapError((cause) => acquisitionError(providerModelId, cause)),
        Effect.tapError(() => clearDemand.pipe(
          Effect.zipRight(setManagedSnapshotState(providerModelId, { _tag: "Failed" })),
          Effect.zipRight(PubSub.publish(selectionChanges, undefined)),
          Effect.asVoid,
        )),
      )
    if (managedRoute && !externalRoute) {
      yield* Ref.update(logicalModels, (models) => {
        const current = models.get(providerModelId)
        if (!current) return models
        const next = new Map(models)
        const routes = current.routes.map((route) => route === managedRoute ? { ...route, loaded: true } : route)
        next.set(providerModelId, { ...current, routes, routeSnapshots: routes.map((route) => routeSnapshot(providerModelId, route)) })
        return next
      })
      yield* clearDemand
      yield* PubSub.publish(selectionChanges, undefined)
    }
    return {
      origin: lease.target.origin,
      authorization: lease.target.authorization,
      servedModelId: String(lease.target.model),
      requestStarted: configuration.recordUse("selected", providerModelId).pipe(Effect.ignore),
    }
  })

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

  const warm = (providerModelId: string) => Effect.scoped(acquire(providerModelId)).pipe(Effect.asVoid)
  const status = platform.instances.pipe(
    Effect.flatMap((registry) => registry.inspect),
    Effect.map((snapshot) => snapshot.instances.some((instance) => instance.health === "ready")
      ? { status: "ok" as const }
      : { status: "not_found" as const, message: "No local model is currently serving." }),
    Effect.catchAll((cause) => Effect.succeed({ status: "error" as const, message: cause instanceof Error ? cause.message : String(cause) })),
  )

  yield* rebuild("cached", false).pipe(Effect.catchAll(() => Effect.void))
  yield* Effect.forkIn(refresh.pipe(Effect.catchAll((cause) => Effect.logWarning("Initial local model discovery failed").pipe(Effect.annotateLogs({ cause: String(cause) })))), serviceScope)
  yield* Effect.forkIn(Stream.tick("3 seconds").pipe(
    Stream.runForEach(() => rebuild("cached", true).pipe(
      Effect.catchAll((cause) => Effect.logWarning("External llama.cpp observation failed").pipe(Effect.annotateLogs({ cause: String(cause) }))),
    )),
  ), serviceScope)

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
  })
}))
