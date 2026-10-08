import { Option } from "effect"
import type * as InferenceSchema from "@magnitudedev/icn-protocol/schemas"
import type {
  ModelInstanceAllocation,
  ModelLoadDevice,
  ModelLoadPlan,
  ModelResidency,
} from "@magnitudedev/acn-protocol"
import { LocalInferenceDeviceIdSchema } from "./schemas/model-state"

export const projectInferenceAllocation = (
  allocation: InferenceSchema.ModelInstanceAllocation,
): ModelInstanceAllocation => ({
  contextWindowTokens: allocation.contextWindowTokens,
  memoryDomains: allocation.memoryDomains.map((domain) => ({
    memoryDomainId: domain.memoryDomainId as ModelInstanceAllocation["memoryDomains"][number]["memoryDomainId"],
    modelBytes: domain.modelBytes,
    contextBytes: domain.contextBytes,
    computeBytes: domain.computeBytes,
    auxiliaryBytes: domain.auxiliaryBytes,
  })),
})

export const projectInferenceLoadDevice = (
  device: InferenceSchema.ModelLoadDevice,
): ModelLoadDevice => ({
  deviceId: LocalInferenceDeviceIdSchema.make(device.id),
  backend: device.backend,
})

export const projectInferenceLoadPlan = (
  plan: InferenceSchema.ModelLoadPlan,
): ModelLoadPlan => ({
  contextWindowTokens: plan.contextWindowTokens,
  requiredMemoryBytes: plan.requiredMemoryBytes,
  device: projectInferenceLoadDevice(plan.device),
})

export const projectInferenceResidency = (
  instance: InferenceSchema.ModelInstance,
): ModelResidency => {
  switch (instance.lifecycle._tag) {
    case "Loading": return {
      _tag: "Loading",
      stage: instance.lifecycle.stage,
      fraction: instance.lifecycle.fraction,
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
    case "Stopped": return { _tag: "Stopped", reason: instance.lifecycle.reason }
    case "Failed": return {
      _tag: "Failed",
      failure: instance.lifecycle.failure._tag === "MemoryShortage"
        ? { ...instance.lifecycle.failure, code: "memory_shortage" }
        : {
            code: instance.lifecycle.failure.code,
            message: instance.lifecycle.failure.message,
            retryable: instance.lifecycle.failure.retryable,
          },
    }
  }
}
