import { Option } from "effect"
import type * as InferenceSchema from "@magnitudedev/icn-protocol/schemas"
import type {
  ModelInstanceAllocation,
  ModelLoadDevice,
  ModelLoadPlan,
  ModelResidency,
} from "@magnitudedev/acn-protocol"
import {
  LocalInferenceDeviceIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  type ModelStreamingStatus,
  type StreamingResize,
} from "./schemas/model-state"

const projectInferenceResize = (resize: InferenceSchema.StreamingResize): StreamingResize =>
  resize._tag === "Shrank"
    ? { _tag: "Shrank", cause: resize.cause, to: resize.to, by: resize.by, atUnixMs: resize.atUnixMs }
    : { _tag: "Grew", from: resize.from, by: resize.by, atUnixMs: resize.atUnixMs }

const projectInferenceStreaming = (
  streaming: InferenceSchema.ModelStreamingStatus,
): ModelStreamingStatus => ({
  experts: streaming.experts,
  slots: streaming.slots,
  floorSlots: streaming.floorSlots,
  ceilingSlots: streaming.ceilingSlots,
  tiers: streaming.tiers.map((tier) => ({
    memoryDomainId: LocalInferenceMemoryDomainIdSchema.make(tier.memoryDomainId),
    bytes: tier.bytes,
  })),
  host: Option.map(streaming.host, (host) => ({
    memoryDomainId: LocalInferenceMemoryDomainIdSchema.make(host.memoryDomainId),
    slots: host.slots,
    ceilingSlots: host.ceilingSlots,
    bytes: host.bytes,
    lastResize: Option.map(host.lastResize, projectInferenceResize),
  })),
  diskBytes: streaming.diskBytes,
  tables: streaming.tables.map((table) => ({
    records: table.records,
    floorRecords: table.floorRecords,
    ceilingRecords: table.ceilingRecords,
    bytes: table.bytes,
    lastResize: Option.map(table.lastResize, projectInferenceResize),
  })),
  lastResize: Option.map(streaming.lastResize, projectInferenceResize),
})

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
  promptGroupPlaces: allocation.promptGroupPlaces,
  streaming: Option.map(allocation.streaming, projectInferenceStreaming),
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
      device: projectInferenceLoadDevice(instance.lifecycle.device),
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
