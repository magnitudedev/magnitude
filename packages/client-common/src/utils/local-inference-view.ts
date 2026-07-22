import { Option } from "effect"
import {
  ProviderModelIdSchema,
  type IcnHardwareState,
  type IcnInventoryState,
  type ModelRecipesState,
} from "@magnitudedev/sdk"
import type {
  LocalInferenceHostProfile,
  LocalInferenceOperationSnapshot,
  LocalInferenceState,
  LocalModelChoice,
  LocalModelFitAssessment,
} from "../types/local-inference"

const displayBackendName = (backend: string): string =>
  backend.toUpperCase() === "MTL" ? "Metal" : backend

export const icnHardwareToLocalHost = (
  hardware: IcnHardwareState,
): LocalInferenceHostProfile => {
  const resident = Option.getOrNull(hardware.resident_memory)
  return {
    platform: hardware.platform,
    architecture: hardware.architecture,
    topologyFingerprint: hardware.topology_fingerprint,
    systemMemoryBytes: hardware.system_memory.total_bytes,
    cpuModel: Option.getOrNull(hardware.cpu_model),
    logicalCores: Math.max(1, hardware.logical_cores),
    memoryDomains: hardware.memory_domains.map((domain) => ({
      id: domain.id,
      kind: domain.kind,
      totalCapacityBytes: domain.total_capacity_bytes,
      stableCapacityBytes: domain.stable_capacity_bytes,
      currentFreeBytes: Option.getOrNull(domain.current_free_bytes),
      sharesSystemMemory: domain.shares_system_memory,
      backendNames: [...new Set(domain.devices
        .filter((device) => device.kind !== "cpu")
        .map((device) => displayBackendName(device.backend)))],
      deviceNames: domain.devices
        .filter((device) => device.kind !== "cpu")
        .map((device) => device.description),
      splitGroupId: null,
    })),
    residentMemory: resident === null
      ? null
      : {
          modelId: resident.model_id,
          runtimeGeneration: resident.runtime_generation,
          domains: resident.domains.map((domain) => ({
            memoryDomainId: domain.memory_domain_id,
            modelBytes: domain.model_bytes,
            contextBytes: domain.context_bytes,
            computeBytes: domain.compute_bytes,
            auxiliaryBytes: domain.auxiliary_bytes,
          })),
        },
  }
}

const fitAssessment = (
  assessment: IcnInventoryState["data"][number]["hardware"],
): LocalModelFitAssessment => assessment.type === "fits" || assessment.type === "does_not_fit"
  ? {
      _tag: "Assessed",
      requiredTotalBytes: assessment.memory.required_bytes,
      domains: assessment.memory.domains.map((domain) => ({
        memoryDomainId: domain.memory_domain,
        requiredBytes: domain.required_bytes,
        stableCapacityBytes: domain.available_bytes,
        marginBytes: domain.margin_bytes,
      })),
      result: assessment.type,
    }
  : { _tag: "NotAssessed" }

const choiceResidency = (
  residency: IcnInventoryState["data"][number]["residency"],
): LocalModelChoice["residency"] => {
  switch (residency.type) {
    case "loaded": return "loaded"
    case "loading":
    case "unloading": return "loading"
    case "load_failed": return "failed"
    case "not_resident": return "unloaded"
  }
}

const choicesFromInventory = (inventory: IcnInventoryState): readonly LocalModelChoice[] => {
  const activeId = inventory.data.find((model) => model.residency.type === "loaded")?.id
  return inventory.data
    .filter((model) => model.availability.type === "available")
    .map((model) => {
      const properties = model.properties.type === "inspected" ? model.properties : null
      const quantization = properties ? Option.getOrNull(properties.quantization) : null
      const contextTokens = Option.getOrNull(model.serving_configuration)?.profile.context_length
      const sizeBytes = model.location.type === "file"
        ? model.location.component.size_bytes
        : model.location.total_bytes
      return {
        _tag: activeId === model.id ? "Running" : "Stored",
        choiceId: model.id,
        displayName: Option.getOrNull(model.name)?.trim() || model.id,
        providerModelId: ProviderModelIdSchema.make(model.id),
        contextTokens: Option.fromNullable(contextTokens),
        fitClass: model.hardware.type === "fits"
          ? model.hardware.profile.acceleration.toLowerCase().includes("hybrid")
            ? "hybrid"
            : "full_accelerator"
          : "unknown",
        availability: { _tag: "Available" },
        fitAssessment: fitAssessment(model.hardware),
        explanation: model.hardware.type === "fits"
          ? `${model.hardware.profile.acceleration} placement`
          : "Stored local model",
        residency: choiceResidency(model.residency),
        quantization: quantization === null
          ? Option.none()
          : Option.some({
            format: quantization,
            quantAwareCheckpoint: false,
            fidelityLabel: "Inspected artifact",
            fidelityEvidence: "Artifact properties inspected by ICN.",
            fidelitySourceUrl: "",
          }),
        sizeBytes: Option.some(sizeBytes),
      } satisfies LocalModelChoice
    })
}

const timestamp = (milliseconds: number): string => new Date(milliseconds).toISOString()

const operationsFromInventory = (
  inventory: IcnInventoryState,
): readonly LocalInferenceOperationSnapshot[] => inventory.data.flatMap(
  (model): readonly LocalInferenceOperationSnapshot[] => {
  if (model.availability.type === "downloading") {
    return [{
      operationId: model.availability.operation_id,
      kind: "download",
      selectionId: model.id,
      providerModelId: ProviderModelIdSchema.make(model.id),
      status: "running",
      stage: model.availability.stage,
      progress: Option.some({
        completedBytes: model.availability.completed_bytes,
        totalBytes: model.availability.total_bytes,
      }),
      failure: Option.none(),
      startedAt: timestamp(model.availability.started_at),
      updatedAt: timestamp(model.availability.updated_at),
    } satisfies LocalInferenceOperationSnapshot]
  }
  if (model.residency.type === "loading") {
    const fraction = Option.getOrNull(model.residency.fraction)
    return [{
      operationId: model.residency.load_id,
      kind: "activate",
      selectionId: model.id,
      providerModelId: ProviderModelIdSchema.make(model.id),
      status: "running",
      stage: "loading",
      progress: Option.fromNullable(fraction).pipe(Option.map((value) => ({ fraction: value }))),
      failure: Option.none(),
      startedAt: timestamp(model.residency.started_at),
      updatedAt: timestamp(model.residency.started_at),
    } satisfies LocalInferenceOperationSnapshot]
  }
  if (model.residency.type === "load_failed") {
    const operationId = `${model.id}:${model.residency.attempted_at}`
    return [{
      operationId,
      kind: "activate",
      selectionId: model.id,
      providerModelId: ProviderModelIdSchema.make(model.id),
      status: "failed",
      stage: "loading",
      progress: Option.none(),
      failure: Option.some({
        code: model.residency.code,
        message: model.residency.message,
        retryable: model.residency.retryable,
      }),
      startedAt: timestamp(model.residency.attempted_at),
      updatedAt: timestamp(model.residency.attempted_at),
    } satisfies LocalInferenceOperationSnapshot]
  }
  return []
})

export const deriveLocalInferenceView = (
  hardware: IcnHardwareState,
  inventory: IcnInventoryState,
  recipes: ModelRecipesState,
): LocalInferenceState => {
  const active = inventory.data.find((model) => model.residency.type === "loaded")
  const recommendations = recipes._tag === "Ready"
    ? { _tag: "Ready" as const, recommendations: recipes.recommendations }
    : recipes._tag === "Loading"
      ? { _tag: "Loading" as const }
      : { _tag: "Failed" as const, message: recipes.message }

  return {
    activeBinding: active && active.residency.type === "loaded"
      ? {
          selectionId: active.id,
          providerModelId: ProviderModelIdSchema.make(active.id),
          contextTokens: Option.getOrNull(active.serving_configuration)?.profile.context_length
            ?? active.residency.context_length,
        }
      : null,
    host: icnHardwareToLocalHost(hardware),
    choices: choicesFromInventory(inventory),
    operations: operationsFromInventory(inventory),
    recommendationState: recommendations,
    warnings: recipes._tag === "Ready" && recipes.failureCount > 0
      ? [{
          code: "preview_failed",
          message: "ICN could not assess every local model recipe.",
        }]
      : [],
  }
}
