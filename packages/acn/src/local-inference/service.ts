import { Context, Effect, Layer, Stream } from "effect"
import {
  LocalInferenceError,
  type LocalInferenceHostProfile,
  type LocalInferenceState,
  type LocalInferenceUsageSelection,
  type LocalModelChoice,
  type LocalModelRecommendation,
} from "@magnitudedev/protocol"
import {
  DistributionInstallError,
  LlamaCppDistribution,
  LlamaCppHost,
  LlamaCppModelStore,
  LlamaCppModelStoreError,
  LlamaCppRuntime,
  LlamaCppRuntimeError,
  type ArtifactDownloadPlan,
  type LlamaCppHostProfile,
  type ModelArtifactSummary,
  type ModelFitPlan,
} from "@magnitudedev/llamacpp"
import type { DurableLocalModelBinding } from "@magnitudedev/storage"
import { LOCAL_MODEL_CATALOG } from "./catalog"
import {
  artifactIdForCatalogEntry,
  providerModelIdForArtifact,
  runningModelChoiceId,
  storedArtifactChoiceId,
} from "./identity"
import { LocalModelConfiguration, type ModelConfigurationError } from "./model-configuration"
import {
  estimateRuntimeOverheadPerSlot,
  contextTargetsForUsage,
  parallelSlotsForUsage,
  recommendLocalModels,
  resolveConfiguration,
  stableCapacityFromHost,
} from "./recommendations"
import type { EvaluatedLocalConfiguration, LocalModelCatalogEntry } from "./types"

export interface LocalInferenceApi {
  readonly state: Effect.Effect<LocalInferenceState, LocalInferenceError>
  readonly configureUsage: (selection: LocalInferenceUsageSelection) => Effect.Effect<void, LocalInferenceError>
  readonly installDistribution: Effect.Effect<void, LocalInferenceError>
  readonly downloadModel: (configurationId: string) => Effect.Effect<void, LocalInferenceError>
  readonly activateModel: (selectionId: string) => Effect.Effect<void, LocalInferenceError>
  readonly deleteModel: (selectionId: string) => Effect.Effect<void, LocalInferenceError>
  readonly restart: Effect.Effect<void, LocalInferenceError>
  readonly disable: Effect.Effect<void, LocalInferenceError>
}

export class LocalInference extends Context.Tag("LocalInference")<LocalInference, LocalInferenceApi>() {}

const localError = (
  code: LocalInferenceError["code"],
  operation: string,
  message: string,
  retryable = false,
): LocalInferenceError => new LocalInferenceError({ code, operation, message, retryable })

const localInferenceErrorFromConfiguration = (
  error: ModelConfigurationError,
): LocalInferenceError => localError("configuration_failed", error.operation, error.reason, true)

export const localInferenceErrorFromDistribution = (
  error: DistributionInstallError,
): LocalInferenceError => {
  switch (error.code) {
    case "unsupported_platform":
      return localError("unsupported_platform", "install llama.cpp distribution", error.reason)
    case "integrity_failed":
      return localError("integrity_failed", "install llama.cpp distribution", error.reason)
    case "download_failed":
    case "storage_failed":
      return localError("configuration_failed", "install llama.cpp distribution", error.reason, true)
  }
}

export const localInferenceErrorFromModelStore = (
  error: LlamaCppModelStoreError,
): LocalInferenceError => {
  switch (error.code) {
    case "artifact_not_found":
      return localError("artifact_unavailable", `${error.operation} local model`, error.reason)
    case "artifact_not_owned":
      return localError("artifact_not_owned", `${error.operation} local model`, error.reason)
    case "invalid_plan":
      return localError("invalid_selection", `${error.operation} local model`, error.reason)
    case "insufficient_space":
      return localError("insufficient_disk_space", `${error.operation} local model`, error.reason)
    case "integrity_failed":
      return localError("integrity_failed", `${error.operation} local model`, error.reason)
    case "download_failed":
      return localError("artifact_unavailable", `${error.operation} local model`, error.reason, true)
    case "storage_failed":
      return localError("configuration_failed", `${error.operation} local model`, error.reason, true)
  }
}

export const localInferenceErrorFromRuntime = (
  error: LlamaCppRuntimeError,
): LocalInferenceError => {
  switch (error.code) {
    case "distribution_unavailable":
      return localError("distribution_missing", error.operation, error.reason)
    case "model_unavailable":
      return localError("artifact_unavailable", error.operation, error.reason)
    case "external_unavailable":
      return localError("external_server_unavailable", error.operation, error.reason, true)
    case "server_start_failed":
    case "server_timeout":
      return localError("server_start_failed", error.operation, error.reason, true)
    case "identity_mismatch":
      return localError("invalid_selection", error.operation, error.reason)
    case "context_mismatch":
      return localError("context_mismatch", error.operation, error.reason)
    case "endpoint_failed":
      return localError("runtime_probe_failed", error.operation, error.reason, true)
  }
}

const fromDistributionError = localInferenceErrorFromDistribution
const fromStoreError = localInferenceErrorFromModelStore
const fromRuntimeError = localInferenceErrorFromRuntime

const downloadPlanFor = (configuration: EvaluatedLocalConfiguration): ArtifactDownloadPlan => ({
  artifactId: artifactIdForCatalogEntry(configuration.entry),
  repo: configuration.entry.repo,
  revision: configuration.entry.revision,
  files: configuration.entry.files,
  safetyReserveBytes: Math.max(2 * 1024 ** 3, Math.ceil(configuration.entry.files.reduce(
    (total, file) => total + file.sizeBytes,
    0,
  ) * 0.05)),
})

const entryForArtifactId = (artifactId: string): LocalModelCatalogEntry | null =>
  LOCAL_MODEL_CATALOG.find((entry) => artifactIdForCatalogEntry(entry) === artifactId) ?? null

const hostToWire = (host: LlamaCppHostProfile): LocalInferenceHostProfile => ({
  systemMemoryBytes: host.system.totalMemoryBytes,
  cpuModel: host.system.cpuModel,
  logicalCores: host.system.logicalCores,
  memoryDomains: host.memoryDomains.map((memory) => ({
    id: memory.id,
    kind: memory.kind,
    stableCapacityBytes: memory.stableCapacityBytes,
    currentFreeBytes: memory.currentFreeBytes,
    sharesSystemMemory: memory.sharesSystemMemory,
    deviceNames: memory.devices.map((device) => device.name),
    splitGroupId: memory.splitGroupId,
  })),
})

const recommendationForArtifact = (
  artifact: ModelArtifactSummary,
  recommendations: readonly LocalModelRecommendation[],
): LocalModelRecommendation | null => {
  const entry = entryForArtifactId(artifact.modelId)
  return entry
    ? recommendations.find((recommendation) => recommendation.catalogModelId === entry.id) ?? null
    : null
}

const storedChoice = (
  artifact: ModelArtifactSummary,
  recommendations: readonly LocalModelRecommendation[],
): LocalModelChoice => {
  const recommendation = recommendationForArtifact(artifact, recommendations)
  const common = {
    choiceId: recommendation?.configurationId ?? storedArtifactChoiceId(artifact.modelId),
    displayName: artifact.metadata.displayName,
    providerModelId: providerModelIdForArtifact(artifact.modelId),
    contextTokens: recommendation?.contextTokens ?? artifact.metadata.contextLength ?? 32_768,
    fitClass: recommendation?.fitClass ?? "unknown" as const,
    compatible: true,
    explanation: recommendation?.explanation ?? "Discovered model; activate after a host fit plan is available.",
    ...(artifact.metadata.quantization
      ? {
          quantization: {
            format: artifact.metadata.quantization,
            bitsClass: "other" as const,
            quantAwareCheckpoint: false,
            fidelityLabel: "Discovered model",
            fidelityEvidence: "No curated fidelity evidence is available for this artifact.",
            fidelitySourceUrl: "https://github.com/ggml-org/llama.cpp",
          },
        }
      : {}),
    sizeBytes: artifact.sizeBytes,
    ...(recommendation ? { servingProfile: recommendation.servingProfile } : {}),
  }
  return artifact.source._tag === "MagnitudeOwned"
    ? { _tag: "StoredOwned", ...common }
    : { _tag: "StoredExternal", ...common }
}

export const LocalInferenceLive: Layer.Layer<
  LocalInference,
  never,
  | LlamaCppDistribution
  | LlamaCppHost
  | LlamaCppModelStore
  | LlamaCppRuntime
  | LocalModelConfiguration
> = Layer.effect(
  LocalInference,
  Effect.gen(function* () {
    const distribution = yield* LlamaCppDistribution
    const host = yield* LlamaCppHost
    const models = yield* LlamaCppModelStore
    const runtime = yield* LlamaCppRuntime
    const configuration = yield* LocalModelConfiguration
    const lifecycleLock = yield* Effect.makeSemaphore(1)
    const getConfiguration = configuration.get.pipe(Effect.mapError(localInferenceErrorFromConfiguration))

    const hostProfile = host.inspect.pipe(
      Effect.mapError((error) => localError("runtime_probe_failed", "inspect local inference host", error.reason, true)),
    )

    const resolveCurated = (configurationId: string) => Effect.gen(function* () {
      const config = yield* getConfiguration
      if (!config.usage) {
        return yield* localError(
          "invalid_selection",
          "resolve local model recommendation",
          "Choose local inference usage before selecting a model.",
        )
      }
      const profile = yield* hostProfile
      const selected = resolveConfiguration(configurationId, stableCapacityFromHost(profile), config.usage)
      if (!selected) {
        return yield* localError(
          "invalid_selection",
          "resolve local model recommendation",
          "The server-issued recommendation is unknown or no longer fits stable host capacity.",
        )
      }
      return selected
    })

    const fitFor = (
      artifact: ModelArtifactSummary,
      contextTokens: number,
      parallelSlots: number,
    ): Effect.Effect<ModelFitPlan, LocalInferenceError> => host.plan({
      modelBytes: artifact.sizeBytes,
      contextBytesPerSlot: estimateRuntimeOverheadPerSlot(
        artifact.sizeBytes,
        contextTokens,
        parallelSlots,
      ),
      parallelSlots,
      modelLayerCount: artifact.metadata.layerCount,
    }).pipe(
      Effect.mapError((error) => localError("runtime_probe_failed", "plan local model runtime", error.reason, true)),
    )

    interface ManagedSelection {
      readonly selectionId: string
      readonly artifactId: string
      readonly providerModelId: string
      readonly contextTokens: number
      readonly parallelSlots: number
    }

    const contextForArtifact = (
      artifact: ModelArtifactSummary,
      usage: LocalInferenceUsageSelection,
    ): number => {
      const targets = contextTargetsForUsage(usage)
      const maximum = artifact.metadata.contextLength
      if (maximum === null) return targets[0]
      return targets.find((target) => target <= maximum) ?? Math.max(1, Math.floor(maximum))
    }

    const resolveManagedSelection = (
      selectionId: string,
    ): Effect.Effect<ManagedSelection | null, LocalInferenceError> => Effect.gen(function* () {
      const config = yield* getConfiguration
      if (!config.usage) {
        return yield* localError(
          "invalid_selection",
          "resolve local model selection",
          "Choose local inference usage before selecting a model.",
        )
      }
      const [profile, snapshot] = yield* Effect.all([
        hostProfile,
        models.inspect.pipe(Effect.mapError(fromStoreError)),
      ], { concurrency: 2 })
      const capacity = stableCapacityFromHost(profile)
      const selected = resolveConfiguration(selectionId, capacity, config.usage)
      if (selected) {
        const artifactId = artifactIdForCatalogEntry(selected.entry)
        return {
          selectionId,
          artifactId,
          providerModelId: providerModelIdForArtifact(artifactId),
          contextTokens: selected.contextTokens,
          parallelSlots: selected.servingProfile.parallelSlots,
        }
      }
      const recommendations = recommendLocalModels(capacity, config.usage)
      const artifact = snapshot.artifacts.find(
        (candidate) => storedChoice(candidate, recommendations).choiceId === selectionId,
      )
      if (!artifact) return null
      return {
        selectionId,
        artifactId: artifact.modelId,
        providerModelId: providerModelIdForArtifact(artifact.modelId),
        contextTokens: contextForArtifact(artifact, config.usage),
        parallelSlots: parallelSlotsForUsage(config.usage),
      }
    })

    const ensureManagedSelection = (
      selected: ManagedSelection,
      onStage?: (
        stage: "planning" | "starting" | "loading" | "verifying",
      ) => Effect.Effect<void, LocalInferenceError>,
    ) => Effect.gen(function* () {
      const artifact = yield* models.resolve(selected.artifactId).pipe(Effect.mapError(fromStoreError))
      if (onStage) yield* onStage("planning")
      const fitPlan = yield* fitFor(artifact, selected.contextTokens, selected.parallelSlots)
      if (!fitPlan.fits) {
        return yield* localError(
          "artifact_unavailable",
          "activate local model",
          "The selected model no longer fits stable host capacity.",
        )
      }
      if (onStage) {
        yield* onStage("starting")
        yield* onStage("loading")
      }
      const target = yield* runtime.ensureServing({
        _tag: "Managed",
        modelId: selected.artifactId,
        providerModelId: selected.providerModelId,
        contextTokens: selected.contextTokens,
        fitPlan,
      }).pipe(Effect.mapError(fromRuntimeError))
      if (onStage) yield* onStage("verifying")
      const binding: DurableLocalModelBinding = {
        _tag: "Managed",
        selectionId: selected.selectionId,
        artifactId: selected.artifactId,
        providerModelId: selected.providerModelId,
        contextTokens: target.configuredContextTokens,
        parallelSlots: selected.parallelSlots,
      }
      return { binding, target }
    })

    const ensureExternalSelection = (
      selectionId: string,
      onStage?: (stage: "verifying") => Effect.Effect<void, LocalInferenceError>,
    ) => Effect.gen(function* () {
      const snapshot = yield* runtime.inspect.pipe(Effect.mapError(fromRuntimeError))
      for (const server of snapshot.external) {
        if (server.health !== "ready") continue
        for (const model of server.models) {
          if (runningModelChoiceId(server.serverId, model.providerModelId) !== selectionId) continue
          if (model.contextTokens === null) {
            return yield* localError(
              "context_mismatch",
              "activate external local model",
              "The external endpoint did not report its configured context size.",
            )
          }
          if (onStage) yield* onStage("verifying")
          const target = yield* runtime.ensureServing({
            _tag: "External",
            connectionId: server.serverId,
            providerModelId: model.providerModelId,
            contextTokens: model.contextTokens,
          }).pipe(Effect.mapError(fromRuntimeError))
          const binding: DurableLocalModelBinding = {
            _tag: "External",
            selectionId,
            endpointConfigId: server.serverId,
            providerModelId: model.providerModelId,
            contextTokens: model.contextTokens,
          }
          return { binding, target }
        }
      }
      return null
    })

    const resolveStoredArtifact = (
      selectionId: string,
    ): Effect.Effect<ModelArtifactSummary | null, LocalInferenceError> => Effect.gen(function* () {
      const [config, snapshot, inspectedHost] = yield* Effect.all([
        getConfiguration,
        models.inspect.pipe(Effect.mapError(fromStoreError)),
        hostProfile.pipe(Effect.either),
      ], { concurrency: 3 })
      const recommendations = config.usage && inspectedHost._tag === "Right"
        ? recommendLocalModels(stableCapacityFromHost(inspectedHost.right), config.usage)
        : []
      return snapshot.artifacts.find(
        (artifact) => storedChoice(artifact, recommendations).choiceId === selectionId,
      ) ?? null
    })

    const configureUsage = (selection: LocalInferenceUsageSelection) =>
      lifecycleLock.withPermits(1)(Effect.gen(function* () {
        const before = yield* getConfiguration
        yield* configuration.updateUsage(selection).pipe(
          Effect.mapError(localInferenceErrorFromConfiguration),
        )
        if (before.binding && (
          before.usage?.localModelRole !== selection.localModelRole
          || before.usage.sessionConcurrency !== selection.sessionConcurrency
        )) {
          yield* runtime.stopManaged.pipe(Effect.mapError(fromRuntimeError))
        }
      }))

    const installDistribution = distribution.install.pipe(
      Stream.runDrain,
      Effect.mapError((error) => error instanceof LocalInferenceError
        ? error
        : fromDistributionError(error)),
    )

    const downloadModel = (configurationId: string) => Effect.gen(function* () {
      const selected = yield* resolveCurated(configurationId)
      if (selected.entry.license.acknowledgementRequired) {
        return yield* localError(
          "license_required",
          "download local model",
          `License acknowledgement is required at ${selected.entry.license.url}`,
        )
      }
      yield* models.download(downloadPlanFor(selected)).pipe(
        Stream.runDrain,
        Effect.mapError((error) => error instanceof LocalInferenceError
          ? error
          : fromStoreError(error)),
      )
    })

    const activateModel = (selectionId: string) =>
      lifecycleLock.withPermits(1)(Effect.gen(function* () {
        const before = yield* getConfiguration
        if (before.binding?.selectionId === selectionId) return

        const external = yield* ensureExternalSelection(selectionId)
        const activated = external ?? (yield* Effect.gen(function* () {
          const selected = yield* resolveManagedSelection(selectionId)
          if (!selected) {
            return yield* localError(
              "invalid_selection",
              "activate local model",
              "The server-issued model selection is unknown or no longer available.",
            )
          }
          return yield* ensureManagedSelection(selected)
        }))

        yield* configuration.activateLocal(activated.binding).pipe(
          Effect.mapError(localInferenceErrorFromConfiguration),
        )
        if (before.binding?._tag === "Managed" && activated.binding._tag === "External") {
          yield* runtime.stopManaged.pipe(Effect.mapError(fromRuntimeError))
        }
      }))

    const deleteModel = (selectionId: string) =>
      lifecycleLock.withPermits(1)(Effect.gen(function* () {
        const artifact = yield* resolveStoredArtifact(selectionId)
        if (!artifact) {
          return yield* localError(
            "invalid_selection",
            "delete local model",
            "The server-issued stored model selection is unknown or no longer available.",
          )
        }
        if (artifact.source._tag !== "MagnitudeOwned") {
          return yield* localError(
            "artifact_not_owned",
            "delete local model",
            "Only Magnitude-owned model artifacts can be deleted.",
          )
        }
        const config = yield* getConfiguration
        if (config.binding?._tag === "Managed" && config.binding.artifactId === artifact.modelId) {
          return yield* localError(
            "artifact_active",
            "delete local model",
            "Disable the active local model before deleting it.",
          )
        }
        yield* models.deleteOwned(artifact.modelId).pipe(Effect.mapError(fromStoreError))
      }))

    const restart = lifecycleLock.withPermits(1)(Effect.gen(function* () {
      const config = yield* getConfiguration
      if (config.binding?._tag !== "Managed") {
        return yield* localError(
          "invalid_selection",
          "restart local model",
          "There is no managed local model binding to restart.",
        )
      }
      const selected: ManagedSelection = {
        selectionId: config.binding.selectionId,
        artifactId: config.binding.artifactId,
        providerModelId: config.binding.providerModelId,
        contextTokens: config.binding.contextTokens,
        parallelSlots: config.binding.parallelSlots,
      }
      yield* runtime.stopManaged.pipe(Effect.mapError(fromRuntimeError))
      yield* ensureManagedSelection(selected)
    }))

    const disable = lifecycleLock.withPermits(1)(Effect.gen(function* () {
      yield* configuration.disableLocal.pipe(Effect.mapError(localInferenceErrorFromConfiguration))
      yield* runtime.stopManaged.pipe(Effect.mapError(fromRuntimeError))
    }))

    const state = Effect.gen(function* () {
      const [config, distributionState, hostResult, modelSnapshot, runtimeSnapshot] = yield* Effect.all([
        getConfiguration,
        distribution.inspect.pipe(
          Effect.mapError((error) => localError("runtime_probe_failed", "inspect llama.cpp distribution", error.reason, true)),
        ),
        hostProfile.pipe(Effect.either),
        models.inspect.pipe(Effect.mapError(fromStoreError)),
        runtime.inspect.pipe(Effect.mapError(fromRuntimeError)),
      ], { concurrency: "unbounded" })

      const recommendations = config.usage && hostResult._tag === "Right"
        ? recommendLocalModels(stableCapacityFromHost(hostResult.right), config.usage)
        : []
      const choices: LocalModelChoice[] = modelSnapshot.artifacts.map((artifact) => storedChoice(artifact, recommendations))
      for (const server of [...(runtimeSnapshot.managed ? [runtimeSnapshot.managed] : []), ...runtimeSnapshot.external]) {
        for (const model of server.models) {
          if (server.ownership === "managed" && choices.some(
            (choice) => choice.providerModelId === model.providerModelId,
          )) continue
          const common = {
            choiceId: runningModelChoiceId(server.serverId, model.providerModelId),
            displayName: model.providerModelId,
            providerModelId: model.providerModelId,
            contextTokens: model.contextTokens ?? 32_768,
            fitClass: "unknown" as const,
            compatible: model.contextTokens !== null,
            explanation: server.ownership === "external" ? "External server; observed read-only." : "Magnitude-managed running model.",
          }
          choices.push(server.ownership === "external"
            ? { _tag: "RunningExternal", ...common }
            : { _tag: "RunningManaged", ...common })
        }
      }

      const activeBinding = config.binding
        ? {
            _tag: config.binding._tag,
            selectionId: config.binding.selectionId,
            providerModelId: config.binding.providerModelId,
            contextTokens: config.binding.contextTokens,
          } as const
        : null

      return {
        schemaVersion: 3,
        usage: config.usage ?? null,
        activeBinding,
        distribution: distributionState._tag === "Ready"
          ? { _tag: "Ready", build: distributionState.distribution.build, source: distributionState.distribution.source }
          : distributionState._tag === "UnsupportedPlatform"
            ? { _tag: "Unsupported", message: `${distributionState.platform}/${distributionState.architecture}` }
            : distributionState._tag === "Invalid"
              ? { _tag: "Invalid", message: distributionState.reason }
              : { _tag: "Missing" },
        host: hostResult._tag === "Right"
          ? { _tag: "Available", profile: hostToWire(hostResult.right) }
          : { _tag: "Unavailable", message: hostResult.left.message },
        choices,
        recommendations,
        warnings: [
          ...modelSnapshot.warnings,
          ...(hostResult._tag === "Right" ? hostResult.right.warnings : []),
        ],
      } satisfies LocalInferenceState
    })

    return LocalInference.of({
      state,
      configureUsage,
      installDistribution,
      downloadModel,
      activateModel,
      deleteModel,
      restart,
      disable,
    })
  }),
)
