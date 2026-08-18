import { Option } from "effect"
import {
  ModelInstanceIdSchema,
  ModelServingConfigurationIdSchema,
  modelSlotActions,
  type ModelResidency,
  type ModelInstanceAllocation,
  type ModelLoadPlan,
  type ModelPackagesState,
  type ModelCapabilities,
  type ModelSlotAvailability,
  type ProviderModelCatalogEntry,
  type SlotId,
} from "@magnitudedev/acn-protocol"
import type * as Generated from "@magnitudedev/icn-protocol/schemas"

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

export const projectModelLoadPlan = (
  plan: Generated.ModelLoadPlan,
): ModelLoadPlan => ({
  contextWindowTokens: plan.contextWindowTokens,
  parallelSequences: plan.parallelSequences,
  physicalContextTokens: plan.physicalContextTokens,
  requiredSystemMemoryBytes: plan.requiredSystemMemoryBytes,
})

export const projectModelResidency = (
  instance: Generated.ModelInstance,
): ModelResidency => {
  const identity = {
    instanceId: ModelInstanceIdSchema.make(instance.id),
    configurationId: ModelServingConfigurationIdSchema.make(instance.configurationId),
  }
  switch (instance.lifecycle._tag) {
    case "Loading":
      return {
        _tag: "Loading" as const,
        ...identity,
        stage: instance.lifecycle.stage,
        progress: Option.flatMap(instance.lifecycle.progress, Option.fromNullable),
        plannedAllocation: Option.map(
          instance.lifecycle.plannedAllocation,
          projectModelLoadPlan,
        ),
      }
    case "Ready":
      return {
        _tag: "Ready" as const,
        ...identity,
        allocation: projectModelInstanceAllocation(instance.lifecycle.allocation),
      }
    case "Stopping":
      return {
        _tag: "Stopping" as const,
        ...identity,
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
                projectModelLoadPlan,
              ),
            },
      }
    case "Stopped":
      return instance.lifecycle.reason === "memory_pressure"
        ? {
            _tag: "Failed" as const,
            failure: {
              code: "low_memory",
              message: "The model stopped because available memory became too low",
              retryable: true,
            },
          }
        : { _tag: "Unloaded" as const }
    case "Failed":
      return {
        _tag: "Failed" as const,
        failure: instance.lifecycle.failure._tag === "LowMemory"
          ? {
              ...instance.lifecycle.failure,
              code: "low_memory" as const,
            }
          : {
              code: instance.lifecycle.failure.code,
              message: instance.lifecycle.failure.message,
              retryable: instance.lifecycle.failure.retryable,
            },
      }
  }
}

export const localModelSlotAvailability = (
  input: {
    readonly catalogIdentityPending: boolean
    readonly offeringsReady: boolean
    readonly inventory: ModelPackagesState["inventory"]
    readonly offeringExists: boolean
    readonly installed: boolean
  },
): ModelSlotAvailability => {
  if (input.catalogIdentityPending
    || input.inventory._tag === "Initializing") {
    return { _tag: "Pending" }
  }
  if (input.inventory._tag === "Degraded") {
    return { _tag: "Unavailable", failure: input.inventory.failure }
  }
  if (!input.offeringExists) {
    if (!input.offeringsReady) return { _tag: "Pending" }
    return {
      _tag: "Unavailable",
      failure: {
        code: "local_offering_unavailable",
        message: "The selected local configuration is unavailable",
        retryable: true,
      },
    }
  }
  if (!input.installed) {
    return {
      _tag: "Unavailable",
      failure: {
        code: "local_model_not_installed",
        message: "The selected local model is not downloaded",
        retryable: true,
      },
    }
  }
  return { _tag: "Available" }
}

export interface LocalOfferingSelectionEvidence {
  readonly capabilities: ModelCapabilities
}

export const selectableModelCapabilities = (
  slotId: SlotId,
  catalogModel: ProviderModelCatalogEntry | undefined,
  localOffering: LocalOfferingSelectionEvidence | undefined,
): ModelCapabilities | undefined => {
  if (localOffering) {
    if (catalogModel && !catalogModel.supportedSlots.includes(slotId)) {
      return undefined
    }
    return localOffering.capabilities
  }
  return catalogModel?.availability._tag === "Available"
    && catalogModel.supportedSlots.includes(slotId)
    ? catalogModel.capabilities
    : undefined
}
