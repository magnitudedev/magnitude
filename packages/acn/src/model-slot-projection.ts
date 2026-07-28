import { Option } from "effect"
import {
  ModelInstanceIdSchema,
  ModelServingConfigurationIdSchema,
  type ModelInstanceAllocation,
  type ModelLoadPreview,
  type ModelCapabilities,
  type ModelSlotAction,
  type ModelSlotAvailability,
  type ModelSlotConfiguredLocal,
  type ModelSlotInstance,
  type ProviderModelCatalogEntry,
  type SlotId,
} from "@magnitudedev/protocol"
import type { Generated } from "@magnitudedev/icn"

export const projectModelInstanceAllocation = (
  allocation: Generated.ModelInstanceAllocation,
): ModelInstanceAllocation => ({
  contextWindowTokens: allocation.contextWindowTokens,
  parallelSequences: allocation.parallelSequences,
  physicalContextTokens: allocation.physicalContextTokens,
  memoryDomains: allocation.memoryDomains.map((domain) => ({
    memoryDomainId: domain.memoryDomainId as ModelInstanceAllocation["memoryDomains"][number]["memoryDomainId"],
    modelBytes: domain.modelBytes,
    contextBytes: domain.contextBytes,
    computeBytes: domain.computeBytes,
    auxiliaryBytes: domain.auxiliaryBytes,
  })),
})

export const projectModelLoadPreview = (
  allocation: Generated.ModelLoadAllocation,
): ModelLoadPreview => ({
  contextWindowTokens: allocation.contextWindowTokens,
  parallelSequences: allocation.parallelSequences,
  physicalContextTokens: allocation.physicalContextTokens,
  requiredSystemMemoryBytes: allocation.requiredSystemMemoryBytes,
})

export const projectModelInstance = (
  instance: Generated.ModelInstance,
): ModelSlotInstance => ({
  id: ModelInstanceIdSchema.make(instance.id),
  configurationId: ModelServingConfigurationIdSchema.make(instance.configurationId),
  lifecycle: (() => {
    switch (instance.lifecycle._tag) {
      case "Loading":
        return {
          _tag: "Loading" as const,
          stage: instance.lifecycle.stage,
          progress: Option.flatMap(instance.lifecycle.progress, Option.fromNullable),
          plannedAllocation: Option.map(
            instance.lifecycle.plannedAllocation,
            projectModelLoadPreview,
          ),
        }
      case "Ready":
        return {
          _tag: "Ready" as const,
          allocation: projectModelInstanceAllocation(instance.lifecycle.allocation),
        }
      case "Stopping":
        return {
          _tag: "Stopping" as const,
          reason: instance.lifecycle.reason,
          allocation: instance.lifecycle.allocation._tag === "Resident"
            ? {
                _tag: "Resident" as const,
                allocation: projectModelInstanceAllocation(
                  instance.lifecycle.allocation.allocation,
                ),
              }
            : {
                _tag: "Planned" as const,
                allocation: Option.map(
                  instance.lifecycle.allocation.allocation,
                  projectModelLoadPreview,
                ),
              },
        }
      case "Stopped":
        return { _tag: "Stopped" as const, reason: instance.lifecycle.reason }
      case "Failed":
        return { _tag: "Failed" as const, failure: instance.lifecycle.failure }
    }
  })(),
})

export const modelSlotActions = (
  availability: ModelSlotAvailability,
  readiness: ModelSlotConfiguredLocal["readiness"],
  instance: Option.Option<ModelSlotInstance>,
): readonly ModelSlotAction[] => {
  if (availability._tag !== "Available") return []
  return Option.match(instance, {
    onNone: () => readiness._tag === "Loadable" ? ["Load"] : [],
    onSome: (current) => {
      switch (current.lifecycle._tag) {
        case "Loading":
        case "Ready":
          return ["Stop"]
        case "Stopping":
          return []
        case "Stopped":
          return readiness._tag === "Loadable" ? ["Load"] : []
        case "Failed":
          return current.lifecycle.failure.retryable && readiness._tag === "Loadable"
            ? ["RetryLoad"]
            : []
      }
    },
  })
}

export const localModelAvailability = (
  catalogAvailability: ModelSlotAvailability,
  offeringExists: boolean,
  installed: boolean,
): ModelSlotAvailability => {
  if (!offeringExists) {
    return {
      _tag: "Unavailable",
      failure: {
        code: "local_offering_unavailable",
        message: "The selected local configuration is unavailable",
        retryable: true,
      },
    }
  }
  if (!installed) {
    return {
      _tag: "Unavailable",
      failure: {
        code: "local_model_not_installed",
        message: "The selected local model is not downloaded",
        retryable: true,
      },
    }
  }
  return catalogAvailability
}

export interface LocalOfferingSelectionEvidence {
  readonly capabilities: ModelCapabilities
  readonly packageIds: readonly string[]
}

export const selectableModelCapabilities = (
  slotId: SlotId,
  catalogModel: ProviderModelCatalogEntry | undefined,
  localOffering: LocalOfferingSelectionEvidence | undefined,
  installedPackageIds: ReadonlySet<string>,
): ModelCapabilities | undefined => {
  if (localOffering) {
    if (localOffering.packageIds.some((packageId) => !installedPackageIds.has(packageId))) {
      return undefined
    }
    if (catalogModel && !catalogModel.supportedSlots.includes(slotId)) {
      return undefined
    }
    return catalogModel?.capabilities ?? localOffering.capabilities
  }
  return catalogModel?.availability._tag === "Available"
    && catalogModel.supportedSlots.includes(slotId)
    ? catalogModel.capabilities
    : undefined
}
