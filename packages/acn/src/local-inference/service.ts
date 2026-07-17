import { Context, Effect, Layer, Option, Ref, Schema, Stream } from "effect"
import {
  LocalInferenceError,
  type MirroredResourceInvalidation,
  type LocalInferenceHostProfile,
  type LocalInferenceSnapshot,
  LocalInferenceState,
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
import { makeMirroredResource } from "../mirrored-resource"

export interface LocalInferenceApi {
  readonly state: Effect.Effect<LocalInferenceSnapshot, LocalInferenceError>
  readonly watchState: Stream.Stream<MirroredResourceInvalidation, LocalInferenceError>
  readonly configureUsage: (selection: LocalInferenceUsageSelection) => Effect.Effect<void, LocalInferenceError>
  readonly installLlamaCpp: Effect.Effect<string, LocalInferenceError>
  readonly refreshInstallations: Effect.Effect<void, LocalInferenceError>
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
  LocalInferencePlatform | LocalModelProviderSource | LocalModelConfiguration
> = Layer.scoped(LocalInference, Effect.gen(function* () {
  const platform = yield* LocalInferencePlatform
  const source = yield* LocalModelProviderSource
  const configuration = yield* LocalModelConfiguration
  const lock = yield* Effect.makeSemaphore(1)
  const hostSnapshot = yield* platform.hardware.inspect.pipe(Effect.either)
  const initialRegistry = yield* platform.instances.pipe(Effect.option)
  const activeRegistry = yield* Ref.make(initialRegistry)
  const configured = configuration.get.pipe(Effect.mapError(fromConfiguration))

  const configureUsage = (selection: LocalInferenceUsageSelection) => lock.withPermits(1)(
    configuration.updateUsage(selection).pipe(Effect.mapError(fromConfiguration)),
  )

  const installLlamaCpp = platform.installations.installManaged.pipe(
    Effect.map(String),
    Effect.mapError((cause) => mapFailure("install managed llama.cpp", cause)),
  )
  const refreshInstallations = platform.installations.refresh.pipe(
    Effect.zipRight(platform.instances.pipe(Effect.asVoid)),
    Effect.mapError((cause) => mapFailure("refresh llama.cpp installations", cause)),
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
    yield* source.warm(logical.providerModelId).pipe(Effect.mapError((cause) => mapFailure("activate local model", cause)))
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

  const buildState = Effect.gen(function* () {
    const [config, installationState, logicalModels, files, operations, instanceSnapshot] = yield* Effect.all([
      configured,
      platform.installations.snapshot,
      source.logicalModels,
      platform.files.inspect("cached"),
      source.operations,
      Ref.get(activeRegistry).pipe(Effect.flatMap(Option.match({
        onNone: () => Effect.succeed({
          instances: [],
          failures: [],
          capturedAt: new Date(),
          activeManagedInstallationId: Option.none(),
        } satisfies LlamaCpp.LlamaInstanceSnapshot),
        onSome: (registry) => registry.snapshot,
      }))),
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
      const availability = model?.availability ?? { _tag: "Disabled" as const, reason: "model_unavailable" as const }
      const fitAssessment = managed && !runningManaged
        ? Option.match(managed.fitAssessment, {
          onNone: () => ({ _tag: "NotAssessed" as const }),
          onSome: (assessment) => ({ _tag: "Estimated" as const, ...assessment }),
        })
        : { _tag: "NotAssessed" as const }
      const common = {
        choiceId: curated?.id ?? logical.providerModelId,
        displayName: logical.information.displayName,
        providerModelId: logical.providerModelId,
        ...(model ? { contextTokens: model.contextWindow } : {}),
        fitClass: "unknown" as const,
        availability,
        fitAssessment,
        explanation: !model
          ? "Metadata required for safe inference is unavailable."
          : availability._tag === "Disabled"
            ? `Unavailable: ${availability.reason}`
            : fitAssessment._tag === "Estimated" && fitAssessment.result === "capacity_risk"
              ? "Estimated memory exceeds stable capacity; loading may fail or affect system performance."
              : "Runnable with the selected llama.cpp installation.",
        residency,
        ...(record ? { sizeBytes: record.sizeBytes } : {}),
      }
      if (runningExternal && !runningManaged) return { _tag: "RunningExternal" as const, ...common }
      if (runningManaged) return { _tag: "RunningManaged" as const, ...common }
      return record?.ownership === "magnitude"
        ? { _tag: "StoredOwned" as const, ...common }
        : { _tag: "StoredExternal" as const, ...common }
    })
    const activeProviderModelId = config.binding ? storedProviderModelId(config.binding.providerModelId) : undefined
    const installOperation = installationState.managedInstall.operation
    const clientInstallOperation = installOperation._tag === "Running"
      ? {
          ...installOperation,
          operationId: String(installOperation.operationId),
          stage: installOperation.stage === "Resolving"
            ? "preparing" as const
            : installOperation.stage === "Downloading"
              ? "downloading" as const
              : "installing" as const,
        }
      : installOperation._tag === "Failed"
        ? { ...installOperation, operationId: String(installOperation.operationId) }
        : installOperation
    return {
      usage: config.usage ?? null,
      activeBinding: config.binding && activeProviderModelId
        ? config.binding._tag === "Managed"
          ? { _tag: "Managed" as const, selectionId: config.binding.selectionId, providerModelId: activeProviderModelId, contextTokens: config.binding.contextTokens }
          : { _tag: "External" as const, selectionId: config.binding.selectionId, providerModelId: activeProviderModelId, contextTokens: config.binding.contextTokens }
        : null,
      llamaCpp: {
        minimumBuild: installationState.minimumBuild,
        recommendedBuild: installationState.recommendedBuild,
        installations: installationState.installations.map((installation) => ({
          id: String(installation.id),
          build: installation.build,
          ownership: installation.ownership,
          executables: {
            serverPath: installation.executables.server.path,
            fitParamsPath: installation.executables.fitParams.path,
          },
          discoveries: installation.discoveries,
        })),
        selectedInstallationId: Option.map(installationState.selectedInstallationId, String),
        activeManagedInstallationId: Option.map(instanceSnapshot.activeManagedInstallationId, String),
        managedInstall: {
          availability: installationState.managedInstall.availability._tag === "Available"
            ? { _tag: "Available" as const, build: installationState.managedInstall.availability.build }
            : installationState.managedInstall.availability,
          operation: clientInstallOperation,
        },
        diagnostics: installationState.diagnostics.map((diagnostic) => ({ code: diagnostic._tag, message: diagnostic.reason })),
      },
      host: hostSnapshot._tag === "Right"
        ? { _tag: "Available" as const, profile: hostToWire(hostSnapshot.right, []) }
        : { _tag: "Unavailable" as const, message: hostSnapshot.left.message },
      choices,
      operations,
      recommendations: config.usage && hostSnapshot._tag === "Right"
        ? recommendLocalModels({ systemMemoryBytes: hostSnapshot.right.totalMemoryBytes, acceleratorDomains: [] }, config.usage)
        : [],
      warnings: [
        ...files.issues.map((issue) => ({ code: issue.code, message: issue.message })),
        ...instanceSnapshot.instances.flatMap((instance) => instance.diagnostics.map((diagnostic) => ({
          code: diagnostic.code,
          message: diagnostic.message,
        }))),
        ...instanceSnapshot.failures.map((failure) => ({
          code: `runtime_${failure.reason}`,
          message: `Unable to observe llama.cpp instance ${failure.instanceId}: ${failure.reason}`,
        })),
      ],
    } satisfies LocalInferenceState
  })

  const initialState = yield* buildState.pipe(Effect.orDie)
  const stateResource = yield* makeMirroredResource(initialState)
  const publishState = buildState.pipe(
    Effect.flatMap((next) => stateResource.setIfChanged(next, Schema.equivalence(LocalInferenceState))),
    Effect.asVoid,
  )
  const registryChanges = Stream.concat(
    Option.match(initialRegistry, {
      onNone: () => Stream.empty,
      onSome: Stream.make,
    }),
    platform.instanceChanges,
  ).pipe(
    Stream.tap((registry) => Ref.set(activeRegistry, Option.some(registry))),
    Stream.flatMap((registry) => Stream.concat(Stream.make(undefined), registry.changes), {
      concurrency: 1,
      switch: true,
    }),
  )
  const triggers = Stream.mergeAll([
    configuration.changes,
    platform.installations.changes,
    source.stateChanges,
    platform.files.changes.pipe(Stream.map(() => undefined)),
    registryChanges,
  ], { concurrency: "unbounded" })
  yield* triggers.pipe(
    Stream.debounce("50 millis"),
    Stream.runForEach(() => publishState.pipe(
      Effect.catchAll((cause) => Effect.logWarning("Local inference state projection failed").pipe(
        Effect.annotateLogs({ cause: String(cause) }),
      )),
    )),
    Effect.forkScoped,
  )

  return LocalInference.of({
    state: stateResource.get,
    watchState: stateResource.changes,
    configureUsage,
    installLlamaCpp: installLlamaCpp.pipe(Effect.tap(() => publishState)),
    refreshInstallations: refreshInstallations.pipe(Effect.tap(() => publishState)),
    downloadModel,
    activateModel,
    deleteModel,
    restart,
    disable,
  })
}))
