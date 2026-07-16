import { Context, Effect, Layer, Option, Ref, Schema, Stream } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import {
  LocalInferenceError,
  type MirroredResourceInvalidation,
  type LocalInferenceHostProfile,
  type LocalInferenceState,
  type LocalInferenceUsageSelection,
  type LocalModelChoice,
} from "@magnitudedev/protocol"
import { HuggingFace, LlamaCpp, ModelFiles } from "@magnitudedev/local-inference"
import type { DurableLocalModelBinding } from "@magnitudedev/storage"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { LOCAL_MODEL_CATALOG } from "./catalog"
import { LocalModelConfiguration, type ModelConfigurationError } from "./model-configuration"
import { LocalInferencePlatform } from "./platform"
import { LocalModelProviderSource, type LlamaLogicalRoute } from "./provider-source"
import { configuredParallelSlots } from "./recommendations"
import { recommendLocalModels } from "./recommendations"

export interface LocalInferenceApi {
  readonly state: Effect.Effect<LocalInferenceState, LocalInferenceError>
  readonly watchState: Stream.Stream<MirroredResourceInvalidation, LocalInferenceError>
  readonly configureUsage: (selection: LocalInferenceUsageSelection) => Effect.Effect<void, LocalInferenceError>
  readonly installDistribution: Effect.Effect<void, LocalInferenceError>
  readonly downloadModel: (configurationId: string) => Effect.Effect<void, LocalInferenceError>
  readonly activateModel: (selectionId: string) => Effect.Effect<void, LocalInferenceError>
  readonly deleteModel: (selectionId: string) => Effect.Effect<void, LocalInferenceError>
  readonly restart: Effect.Effect<void, LocalInferenceError>
  readonly disable: Effect.Effect<void, LocalInferenceError>
}

export class LocalInference extends Context.Tag("LocalInference")<LocalInference, LocalInferenceApi>() {}

const localError = (code: LocalInferenceError["code"], operation: string, message: string, retryable = false) =>
  new LocalInferenceError({ code, operation, message, retryable })
const fromConfiguration = (error: ModelConfigurationError) =>
  localError("configuration_failed", error.operation, error.reason, true)
const storedProviderModelId = (value: string) =>
  Schema.is(ProviderModelIdSchema)(value) ? value : undefined
const mapFailure = (operation: string, cause: unknown) => localError(
  "configuration_failed",
  operation,
  cause instanceof Error ? cause.message : String(cause),
  true,
)

const hostToWire = (
  host: { readonly totalMemoryBytes: number; readonly availableMemoryBytes: number; readonly cpuModel: Option.Option<string>; readonly logicalCores: number },
  devices: readonly LlamaCpp.LlamaDevice[],
): LocalInferenceHostProfile => ({
  systemMemoryBytes: host.totalMemoryBytes,
  cpuModel: Option.getOrNull(host.cpuModel),
  logicalCores: host.logicalCores,
  memoryDomains: [
    {
      id: "system",
      kind: "system",
      stableCapacityBytes: Math.max(0, host.totalMemoryBytes - Math.max(8 * 1024 ** 3, host.totalMemoryBytes * 0.2)),
      currentFreeBytes: host.availableMemoryBytes,
      sharesSystemMemory: false,
      deviceNames: [],
      splitGroupId: null,
    },
    ...devices.flatMap((device) => Option.match(device.totalMemoryBytes, {
      onNone: () => [],
      onSome: (total) => [{
        id: String(device.id),
        kind: "physical_device" as const,
        stableCapacityBytes: Math.max(0, total - Math.max(1024 ** 3, total * 0.1)),
        currentFreeBytes: Option.getOrNull(device.freeMemoryBytes),
        sharesSystemMemory: false,
        deviceNames: [Option.getOrElse(device.name, () => String(device.id))],
        splitGroupId: null,
      }],
    })),
  ],
})

export const LocalInferenceLive: Layer.Layer<
  LocalInference,
  never,
  LocalInferencePlatform | LocalModelProviderSource | LocalModelConfiguration | HttpClient.HttpClient
> = Layer.effect(LocalInference, Effect.gen(function* () {
  const platform = yield* LocalInferencePlatform
  const source = yield* LocalModelProviderSource
  const configuration = yield* LocalModelConfiguration
  const http = yield* HttpClient.HttpClient
  const lock = yield* Effect.makeSemaphore(1)
  const distributionCache = yield* Ref.make<Option.Option<LlamaCpp.LlamaDistributionStatus>>(Option.none())
  const distributionStatus = Ref.get(distributionCache).pipe(Effect.flatMap(Option.match({
    onNone: () => platform.distribution.status.pipe(Effect.tap((status) => Ref.set(distributionCache, Option.some(status)))),
    onSome: Effect.succeed,
  })))
  const configured = configuration.get.pipe(Effect.mapError(fromConfiguration))

  const configureUsage = (selection: LocalInferenceUsageSelection) => lock.withPermits(1)(
    configuration.updateUsage(selection).pipe(Effect.mapError(fromConfiguration)),
  )

  const installDistribution = platform.hardware.inspect.pipe(
    Effect.flatMap((host) => {
      const id = host.platform === "darwin"
        ? host.nativeArchitecture === "arm64" ? "macos-arm64-metal" : "macos-x64-cpu"
        : host.platform === "linux"
          ? host.nativeArchitecture === "arm64" ? "linux-arm64-cpu" : "linux-x64-cpu"
          : null
      return platform.distribution.install(id ? Option.some(LlamaCpp.LlamaDistributionVariantId.make(id)) : Option.none())
    }),
    Effect.asVoid,
    Effect.tap(() => platform.distribution.status.pipe(Effect.flatMap((status) => Ref.set(distributionCache, Option.some(status))))),
    Effect.mapError((cause) => mapFailure("install llama.cpp distribution", cause)),
  )

  const downloadModel = (configurationId: string) => Effect.gen(function* () {
    const entry = LOCAL_MODEL_CATALOG.find((candidate) => candidate.id === configurationId)
    if (!entry) return yield* localError("invalid_selection", "download local model", "The requested catalog model is unknown.")
    if (entry.license.acknowledgementRequired) return yield* localError("license_required", "download local model", `License acknowledgement is required at ${entry.license.url}`)
    const artifact = yield* platform.hub.resolveArtifact({
      repository: HuggingFace.HuggingFaceRepositoryId.make(entry.repo),
      revision: HuggingFace.HuggingFaceRevision.make(entry.revision),
      files: entry.files.map((file) => ({ path: file.path, role: "primary" as const, shardIndex: Option.none() })),
      relationships: [],
    }).pipe(Effect.mapError((cause) => mapFailure("resolve local model download", cause)))
    yield* platform.downloads.download(artifact).pipe(
      Stream.runDrain,
      Effect.mapError((cause) => mapFailure("download local model", cause)),
    )
    yield* platform.files.inspect("changed")
  })

  const catalogModels = source.catalog.list.pipe(
    Effect.mapError((cause) => mapFailure("inspect local models", cause)),
    Effect.provideService(HttpClient.HttpClient, http),
  )

  const matchCatalogEntry = (record: ModelFiles.ModelFileRecord | undefined) => record
    ? LOCAL_MODEL_CATALOG.find((entry) =>
        entry.files.reduce((total, file) => total + file.sizeBytes, 0) === record.sizeBytes
        && Option.contains(record.metadata.quantization, entry.quantization.format),
      )
    : undefined

  const resolveSelectedModel = (selectionId: string) => Effect.gen(function* () {
    const logical = yield* source.logicalModels
    for (const record of logical.values()) {
      const managed = record.routes.find((route): route is Extract<LlamaLogicalRoute, { readonly _tag: "Managed" }> => route._tag === "Managed")
      if (record.providerModelId === selectionId || matchCatalogEntry(managed?.record)?.id === selectionId) return record
    }
    return undefined
  })

  const activateModel = (selectionId: string) => lock.withPermits(1)(Effect.gen(function* () {
    const logical = yield* resolveSelectedModel(selectionId)
    const model = logical?.providerModel
    if (!logical || !model || model.availability._tag === "Disabled") return yield* localError("invalid_selection", "activate local model", "The requested local model is unavailable.")
    yield* source.warm(logical.providerModelId).pipe(Effect.mapError((cause) => mapFailure("activate local model", cause)))
    const managed = logical.routes.find((route): route is Extract<LlamaLogicalRoute, { readonly _tag: "Managed" }> => route._tag === "Managed")
    const external = logical.routes.find((route): route is Extract<LlamaLogicalRoute, { readonly _tag: "External" }> => route._tag === "External")
    const binding: DurableLocalModelBinding = !managed && external
      ? {
          _tag: "External",
          selectionId,
          endpointConfigId: String(external.request.instanceId),
          providerModelId: logical.providerModelId,
          contextTokens: model.contextWindow,
        }
      : {
          _tag: "Managed",
          selectionId,
          artifactId: managed ? String(managed.record.id) : "",
          providerModelId: logical.providerModelId,
          contextTokens: model.contextWindow,
          parallelSlots: configuredParallelSlots(),
        }
    yield* configuration.activateLocal(binding).pipe(Effect.mapError(fromConfiguration))
  }))

  const deleteModel = (selectionId: string) => lock.withPermits(1)(Effect.gen(function* () {
    const config = yield* configured
    if (config.binding?.providerModelId === selectionId) return yield* localError("artifact_active", "delete local model", "Disable the active local model before deleting it.")
    const model = yield* resolveSelectedModel(selectionId)
    const managed = model?.routes.find((route): route is Extract<LlamaLogicalRoute, { readonly _tag: "Managed" }> => route._tag === "Managed")
    if (!managed) return yield* localError("invalid_selection", "delete local model", "The requested stored model is unknown.")
    const id = managed.record.id
    const record = yield* platform.files.get(id).pipe(Effect.mapError((cause) => mapFailure("inspect local model", cause)))
    if (!record.operations.delete) return yield* localError("artifact_not_owned", "delete local model", "Only Magnitude-owned model artifacts can be deleted.")
    yield* platform.files.remove(id).pipe(Effect.mapError((cause) => mapFailure("delete local model", cause)))
  }))

  const restart = lock.withPermits(1)(Effect.gen(function* () {
    const config = yield* configured
    if (!config.binding || config.binding._tag !== "Managed") return yield* localError("invalid_selection", "restart local model", "There is no managed local model to restart.")
    const providerModelId = storedProviderModelId(config.binding.providerModelId)
    if (!providerModelId) return yield* localError("invalid_selection", "restart local model", "The configured local model ID is invalid.")
    yield* source.stopManaged.pipe(Effect.mapError((cause) => mapFailure("restart local model", cause)))
    yield* source.warm(providerModelId).pipe(Effect.mapError((cause) => mapFailure("restart local model", cause)))
  }))

  const disable = lock.withPermits(1)(Effect.gen(function* () {
    yield* configuration.disableLocal.pipe(Effect.mapError(fromConfiguration))
    yield* source.stopManaged.pipe(Effect.mapError((cause) => mapFailure("disable local inference", cause)))
  }))

  const state = Effect.gen(function* () {
    const [config, distribution, hostResult, models, logicalModels, files, operations] = yield* Effect.all([
      configured,
      distributionStatus,
      platform.hardware.inspect.pipe(Effect.either),
      catalogModels,
      source.logicalModels,
      platform.files.inspect("cached"),
      source.operations,
    ], { concurrency: 4 })
    const choices: LocalModelChoice[] = [...logicalModels.values()].map((logical) => {
      const model = logical.providerModel
      const managed = logical?.routes.find((route): route is Extract<LlamaLogicalRoute, { readonly _tag: "Managed" }> => route._tag === "Managed")
      const external = logical?.routes.find((route): route is Extract<LlamaLogicalRoute, { readonly _tag: "External" }> => route._tag === "External")
      const record = managed?.record
      const curated = matchCatalogEntry(record)
      const runningExternal = logical?.routes.some((route) => route._tag === "External" && route.healthy && (route.observation.status === "loaded" || route.observation.status === "sleeping")) ?? false
      const runningManaged = logical?.routes.some((route) => route._tag === "Managed" && route.loaded) ?? false
      const residency = runningExternal || runningManaged ? "loaded" as const : "unloaded" as const
      const common = {
        choiceId: curated?.id ?? logical.providerModelId,
        displayName: logical.information.displayName,
        providerModelId: logical.providerModelId,
        ...(model ? { contextTokens: model.contextWindow } : {}),
        fitClass: "unknown" as const,
        compatible: model?.availability._tag === "Available",
        explanation: !model ? "Metadata required for safe inference is unavailable." : model.availability._tag === "Available" ? "Runnable with the active llama.cpp distribution." : `Unavailable: ${model.availability.reason}`,
        residency,
        ...(record ? { sizeBytes: record.sizeBytes } : {}),
      }
      if (runningExternal && !runningManaged) return { _tag: "RunningExternal" as const, ...common }
      if (runningManaged) return { _tag: "RunningManaged" as const, ...common }
      return record?.ownership === "magnitude"
        ? { _tag: "StoredOwned" as const, ...common }
        : { _tag: "StoredExternal" as const, ...common }
    })
    const binary = Option.orElse(distribution.configured, () => Option.orElse(distribution.managed, () => distribution.path))
    const activeProviderModelId = config.binding ? storedProviderModelId(config.binding.providerModelId) : undefined
    return {
      usage: config.usage ?? null,
      activeBinding: config.binding && activeProviderModelId
        ? config.binding._tag === "Managed"
          ? { _tag: "Managed" as const, selectionId: config.binding.selectionId, providerModelId: activeProviderModelId, contextTokens: config.binding.contextTokens }
          : { _tag: "External" as const, selectionId: config.binding.selectionId, providerModelId: activeProviderModelId, contextTokens: config.binding.contextTokens }
        : null,
      distribution: Option.match(binary, {
        onNone: () => ({ _tag: "Missing" as const }),
        onSome: (value) => ({ _tag: "Ready" as const, build: Option.getOrElse(value.build.buildNumber, () => 1), source: value.source === "path" ? "configured" as const : value.source }),
      }),
      host: hostResult._tag === "Right"
        ? { _tag: "Available" as const, profile: hostToWire(hostResult.right, []) }
        : { _tag: "Unavailable" as const, message: hostResult.left.message },
      choices,
      operations,
      recommendations: config.usage && hostResult._tag === "Right"
        ? recommendLocalModels({ systemMemoryBytes: hostResult.right.totalMemoryBytes, acceleratorDomains: [] }, config.usage)
        : [],
      warnings: files.issues.map((issue) => ({ code: issue.code, message: issue.message })),
    } satisfies LocalInferenceState
  })

  const watchState = Stream.concat(Stream.make(undefined), Stream.tick("500 millis")).pipe(
    Stream.mapEffect(() => state),
    Stream.changesWith((previous, current) => JSON.stringify(previous) === JSON.stringify(current)),
    Stream.mapAccum(0, (revision): readonly [number, MirroredResourceInvalidation] => [
      revision + 1,
      { _tag: "changed", revision: revision + 1 },
    ]),
  )

  return LocalInference.of({ state, watchState, configureUsage, installDistribution, downloadModel, activateModel, deleteModel, restart, disable })
}))

