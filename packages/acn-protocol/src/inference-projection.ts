import { Option } from "effect"
import type * as InferenceSchema from "@magnitudedev/icn-protocol/schemas"
import type {
  ModelInstanceAllocation,
  ModelLoadPlan,
  ModelResidency,
} from "@magnitudedev/acn-protocol"

export const projectInferenceAllocation = (
  allocation: InferenceSchema.ModelInstanceAllocation,
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

export const projectInferenceLoadPlan = (
  plan: InferenceSchema.ModelLoadPlan,
): ModelLoadPlan => ({
  contextWindowTokens: plan.contextWindowTokens,
  parallelSequences: plan.parallelSequences,
  physicalContextTokens: plan.physicalContextTokens,
  requiredSystemMemoryBytes: plan.requiredSystemMemoryBytes,
})

export const projectInferenceResidency = (
  instance: InferenceSchema.ModelInstance,
): ModelResidency => {
  switch (instance.lifecycle._tag) {
    case "Loading": return {
      _tag: "Loading",
      stage: instance.lifecycle.stage,
      progress: Option.flatMap(instance.lifecycle.progress, Option.fromNullable),
      plannedAllocation: Option.map(instance.lifecycle.plannedAllocation, projectInferenceLoadPlan),
    }
    case "Ready": return {
      _tag: "Ready",
      allocation: projectInferenceAllocation(instance.lifecycle.allocation),
    }
    case "Stopping": return {
      _tag: "Stopping",
      reason: instance.lifecycle.reason,
      allocation: instance.lifecycle.allocation._tag === "Resident"
        ? { _tag: "Resident", allocation: projectInferenceAllocation(instance.lifecycle.allocation.allocation) }
        : { _tag: "Planned", allocation: Option.map(instance.lifecycle.allocation.allocation, projectInferenceLoadPlan) },
    }
    case "Stopped": return instance.lifecycle.reason === "memory_pressure"
      ? {
          _tag: "Failed",
          failure: {
            code: "low_memory",
            message: "The model stopped because available memory became too low",
            retryable: true,
          },
        }
      : { _tag: "Unloaded" }
    case "Failed": return {
      _tag: "Failed",
      failure: instance.lifecycle.failure._tag === "LowMemory"
        ? { ...instance.lifecycle.failure, code: "low_memory" }
        : {
            code: instance.lifecycle.failure.code,
            message: instance.lifecycle.failure.message,
            retryable: instance.lifecycle.failure.retryable,
          },
    }
  }
}
