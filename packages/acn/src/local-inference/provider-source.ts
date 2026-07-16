import { createHash } from "node:crypto"
import { Context, Effect, Layer, Option, PubSub, Ref, Scope, Stream } from "effect"
import {
  AVAILABLE_PROVIDER_MODEL,
  LlamaCppAcquisitionError,
  ModelCatalogError,
  type LlamaCppProviderSource,
  type LlamaCppModelInfo,
} from "@magnitudedev/sdk"
import { LlamaCpp, ModelFiles } from "@magnitudedev/local-inference"
import type { LocalInferenceOperationSnapshot } from "@magnitudedev/protocol"
import { LocalInferencePlatform } from "./platform"
import { LocalModelConfiguration } from "./model-configuration"
import { configuredParallelSlots } from "./recommendations"

const ZERO_PRICING = { input: 0, output: 0, cached_input: null } as const
const disabled = (reason: "insufficient_resources" | "provider_unavailable" | "model_unavailable" | "incompatible_runtime" | "invalid_configuration") => ({ _tag: "Disabled", reason } as const)
const managedProviderModelId = (id: ModelFiles.ModelFileId): string =>
  `local-${createHash("sha256").update(String(id)).digest("hex").slice(0, 24)}`
const externalProviderModelId = (instanceId: string, servedModelId: string): string =>
  `external-${createHash("sha256").update(`${instanceId}\0${servedModelId}`).digest("hex").slice(0, 24)}`

type Route =
  | { readonly _tag: "Managed"; readonly request: LlamaCpp.ManagedModelRequest }
  | { readonly _tag: "External"; readonly request: LlamaCpp.ExternalModelRequest }

export interface LocalModelProviderSourceApi extends LlamaCppProviderSource {
  readonly warm: (providerModelId: string) => Effect.Effect<void, LlamaCppAcquisitionError>
  readonly operations: Effect.Effect<readonly LocalInferenceOperationSnapshot[]>
  readonly stopManaged: Effect.Effect<void, LlamaCpp.LlamaControlError | LlamaCpp.ModelInUse | LlamaCpp.LlamaDistributionError>
  /** Invalidates catalog snapshots after an observed local state change. */
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

export const LocalModelProviderSourceLive: Layer.Layer<
  LocalModelProviderSource,
  never,
  LocalInferencePlatform | LocalModelConfiguration
> = Layer.scoped(LocalModelProviderSource, Effect.gen(function* () {
  const platform = yield* LocalInferencePlatform
  const configuration = yield* LocalModelConfiguration
  const routes = yield* Ref.make<ReadonlyMap<string, Route>>(new Map())
  const catalogCache = yield* Ref.make<Option.Option<{ readonly at: number; readonly models: readonly LlamaCppModelInfo[] }>>(Option.none())
  const operations = yield* Ref.make<ReadonlyMap<string, LocalInferenceOperationSnapshot>>(new Map())
  const catalogChanges = yield* PubSub.unbounded<void>()
  const serviceScope = yield* Scope.Scope

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
    if ((yield* Ref.get(operations)).has(id)) return operation
    yield* Ref.update(operations, (current) => new Map(current).set(id, {
      operationId: id,
      providerModelId,
      status: "running",
      stage: "queued",
    }))
    const trackEvents = operation.events.pipe(Stream.runForEach((event) => Ref.update(operations, (current) => {
      const next = new Map(current)
      const progress = event._tag === "Loading" ? Option.getOrUndefined(event.progress) : undefined
      next.set(id, {
        operationId: id,
        providerModelId,
        status: event._tag === "Loaded" ? "completed" : "running",
        stage: operationStage(event),
        ...(progress === undefined ? {} : { progress }),
      })
      return next
    })))
    yield* Effect.forkIn(trackEvents, serviceScope)
    yield* Effect.forkIn(operation.result.pipe(
      Effect.tap(() => Ref.update(operations, (current) => new Map(current).set(id, {
        operationId: id, providerModelId, status: "completed", stage: "loaded",
      }))),
      Effect.tapError((cause) => Ref.update(operations, (current) => new Map(current).set(id, {
        operationId: id, providerModelId, status: "failed", stage: "loading",
        message: cause.reason,
      }))),
      Effect.ignore,
    ), serviceScope)
    return operation
  })

  const rebuild = (fileRefresh: ModelFiles.ModelFileRefresh, assessAvailability: boolean) => Effect.gen(function* () {
    const records = (yield* platform.files.inspect(fileRefresh)).records
    const runtime = assessAvailability
      ? yield* platform.instances.pipe(Effect.flatMap((instances) => instances.refreshExternal), Effect.option)
      : Option.none<LlamaCpp.LlamaInstanceSnapshot>()
    const cli = assessAvailability ? yield* platform.cli.pipe(Effect.option) : Option.none<LlamaCpp.LlamaCli>()
    const nextRoutes = new Map<string, Route>()
    const managed = yield* Effect.forEach(records, (record) => Effect.gen(function* () {
      const contextWindow = Option.getOrElse(record.metadata.trainedContextTokens, () => 32_768)
      const profile = yield* profileFor(contextWindow)
      const providerModelId = managedProviderModelId(record.id)
      nextRoutes.set(providerModelId, {
        _tag: "Managed",
        request: {
          modelFileId: record.id,
          servedModelId: LlamaCpp.LlamaServedModelId.make(providerModelId),
          profile,
        },
      })
      const loaded = Option.exists(runtime, ({ instances }) => instances.some((instance) =>
        instance.ownership === "managed" && instance.models.some((model) => String(model.id) === providerModelId && (model.status === "loaded" || model.status === "sleeping")),
      ))
      const availability = assessAvailability ? yield* Option.match(cli, {
        onNone: () => Effect.succeed(disabled("provider_unavailable")),
        onSome: (availableCli) => platform.files.resolve(record.id).pipe(
          Effect.flatMap((files) => availableCli.assessFit({ files, profile })),
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
                  const free = Option.getOrElse(device.freeMemoryBytes, () => 0)
                  const reserve = Math.max(1024 ** 3, Option.getOrElse(device.totalMemoryBytes, () => free) * 0.1)
                  return required <= Math.max(0, free - reserve)
                })
                return fitsLivePressure ? AVAILABLE_PROVIDER_MODEL : disabled("insufficient_resources")
              }),
            )
          }),
          Effect.catchAll(() => Effect.succeed(disabled("incompatible_runtime"))),
        ),
      }) : AVAILABLE_PROVIDER_MODEL
      return {
        providerId: "llamacpp" as const,
        providerModelId,
        displayName: record.displayName,
        modelFamilyId: "unknown",
        contextWindow,
        maxOutputTokens: Math.min(contextWindow, 8192),
        capabilities: { vision: Option.exists(record.metadata.inputModalities, (items) => items.includes("image")) },
        availability,
        pricing: ZERO_PRICING,
        reasoningEfforts: ["none"],
        ownership: "managed" as const,
        residency: loaded ? "loaded" as const : "unloaded" as const,
        productRank: 0,
        servedModelId: providerModelId,
        managedArtifactId: String(record.id),
        metadataName: Option.getOrUndefined(record.metadata.name),
        modelArchitecture: Option.getOrUndefined(record.metadata.architecture),
        tokenizerModel: Option.getOrUndefined(record.metadata.tokenizerModel),
        tokenizerPre: Option.getOrUndefined(record.metadata.tokenizerPre),
        baseModelNames: record.metadata.baseModelNames,
        baseModelRepositories: record.metadata.baseModelRepositories,
        serverContextSize: contextWindow,
      }
    }), { concurrency: 2 })

    const observedExternal = Option.match(runtime, {
      onNone: () => [],
      onSome: ({ instances }) => instances.flatMap((instance, externalPriority) => {
        if (instance.ownership !== "external") return []
        return instance.models.flatMap((model) => {
          const contextWindow = Option.getOrElse(model.activeContextTokens, () => 32_768)
          const providerModelId = externalProviderModelId(String(instance.id), String(model.id))
          nextRoutes.set(providerModelId, {
            _tag: "External",
            request: { instanceId: instance.id, servedModelId: model.id },
          })
          const usable = instance.health === "ready" && (model.status === "loaded" || model.status === "sleeping")
          return [{
            providerId: "llamacpp" as const,
            providerModelId,
            displayName: Option.getOrElse(model.serverDisplayName, () => String(model.id)),
            modelFamilyId: "unknown",
            contextWindow,
            maxOutputTokens: Math.min(contextWindow, 8192),
            capabilities: { vision: Option.exists(model.inputModalities, (items) => items.includes("image")) },
            availability: usable ? AVAILABLE_PROVIDER_MODEL : disabled(instance.health === "ready" ? "model_unavailable" : "provider_unavailable"),
            pricing: ZERO_PRICING,
            reasoningEfforts: ["none"],
            ownership: "external" as const,
            residency: model.status === "downloading" ? "loading" as const : model.status,
            productRank: Number.MAX_SAFE_INTEGER,
            externalPriority,
            servedModelId: String(model.id),
            externalServerId: String(instance.id),
            serverContextSize: contextWindow,
            modelArchitecture: Option.getOrUndefined(model.architecture),
          }]
        })
      }),
    })
    const previousCatalog = yield* Ref.get(catalogCache)
    const previousRoutes = yield* Ref.get(routes)
    const observedExternalIds = new Set(observedExternal.map((model) => model.providerModelId))
    const unavailableExternal = Option.match(previousCatalog, {
      onNone: () => [] as readonly LlamaCppModelInfo[],
      onSome: ({ models }) => models.filter((model) => model.ownership === "external" && !observedExternalIds.has(model.providerModelId)).map((model) => {
        const route = previousRoutes.get(model.providerModelId)
        if (route?._tag === "External") nextRoutes.set(model.providerModelId, route)
        return { ...model, availability: disabled("provider_unavailable"), residency: "unknown" as const }
      }),
    })
    const models: readonly LlamaCppModelInfo[] = [...managed, ...observedExternal, ...unavailableExternal]
    yield* Ref.set(routes, nextRoutes)
    yield* Ref.set(catalogCache, Option.some({ at: Date.now(), models }))
    yield* PubSub.publish(catalogChanges, undefined)
    return models
  }).pipe(Effect.mapError((cause) => new ModelCatalogError({ message: cause instanceof Error ? cause.message : String(cause), cause })))

  // Coalesce concurrent startup/manual/acquisition refreshes into one scan.
  const refresh = yield* Effect.cachedWithTTL(rebuild("changed", true), "250 millis")

  // Listing is observational and never performs discovery. The supervised
  // startup refresh below populates this cache and emits an invalidation.
  const list = Ref.get(catalogCache).pipe(Effect.map(Option.match({
    onNone: () => [] as readonly LlamaCppModelInfo[],
    onSome: (value) => value.models,
  })))

  const acquire = (providerModelId: string) => Effect.gen(function* () {
    yield* refresh.pipe(Effect.mapError((cause) => acquisitionError(providerModelId, cause)))
    const route = (yield* Ref.get(routes)).get(providerModelId)
    if (!route) return yield* acquisitionError(providerModelId, "This local model is no longer available.")
    const registry = yield* platform.instances.pipe(Effect.mapError((cause) => acquisitionError(providerModelId, cause)))
    if (route._tag === "Managed") {
      const operation = yield* registry.ensureManagedLoaded(route.request)
      yield* observeOperation(providerModelId, operation)
      yield* operation.result.pipe(
        Effect.ensuring(operation.cancel),
        Effect.mapError((cause) => acquisitionError(providerModelId, cause)),
      )
    }
    const lease = yield* registry.acquire(route._tag === "Managed"
      ? LlamaCpp.LlamaModelRequest.Managed({ request: route.request })
      : LlamaCpp.LlamaModelRequest.External({ request: route.request })).pipe(
        Effect.mapError((cause) => acquisitionError(providerModelId, cause)),
      )
    yield* configuration.recordUse("selected", providerModelId).pipe(Effect.ignore)
    return {
      origin: lease.target.origin,
      authorization: lease.target.authorization,
      servedModelId: String(lease.target.model),
    }
  })

  const warm = (providerModelId: string) => Effect.scoped(acquire(providerModelId)).pipe(Effect.asVoid)
  const status = platform.instances.pipe(
    Effect.flatMap((registry) => registry.inspect),
    Effect.map((snapshot) => snapshot.instances.some((instance) => instance.health === "ready")
      ? { status: "ok" as const }
      : { status: "not_found" as const, message: "No local model is currently serving." }),
    Effect.catchAll((cause) => Effect.succeed({ status: "error" as const, message: cause instanceof Error ? cause.message : String(cause) })),
  )

  // Hydrate the last validated file index before exposing the provider. This
  // does no file discovery or GGUF parsing; live fit/status replaces it below.
  yield* rebuild("cached", false).pipe(Effect.catchAll(() => Effect.void))
  yield* Effect.forkIn(refresh.pipe(
    Effect.catchAll((cause) => Effect.logWarning("Initial local model discovery failed").pipe(
      Effect.annotateLogs({ cause: String(cause) }),
    )),
  ), serviceScope)

  return LocalModelProviderSource.of({
    catalog: {
      list,
      refresh,
      get: (providerId, providerModelId) => providerId !== "llamacpp"
        ? Effect.fail(new ModelCatalogError({ message: `Model not found: ${providerId}/${providerModelId}` }))
        : refresh.pipe(Effect.flatMap((models) => {
          const model = models.find((candidate) => candidate.providerModelId === providerModelId)
          return model ? Effect.succeed(model) : Effect.fail(new ModelCatalogError({ message: `Model not found: ${providerId}/${providerModelId}` }))
        })),
    },
    acquire,
    warm,
    operations: Ref.get(operations).pipe(Effect.map((current) => [...current.values()])),
    status,
    stopManaged: platform.instances.pipe(Effect.flatMap((registry) => registry.stopManaged)),
    changes: Stream.fromPubSub(catalogChanges),
  })
}))
