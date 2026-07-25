import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  LocalInferenceMemoryDomainIdSchema,
  type LocalInferenceHardware,
  type ProviderModelCatalogEntry,
} from "@magnitudedev/sdk"
import { providerModelMemoryConditions } from "./model-memory"

const systemMemoryDomainId = LocalInferenceMemoryDomainIdSchema.make("system")

const hardware: LocalInferenceHardware = {
  platform: "Linux",
  architecture: "X64",
  productName: Option.none(),
  processor: Option.none(),
  logicalCores: 8,
  totalSystemMemoryBytes: 16,
  availableSystemMemoryBytes: 8,
  warningReserveBytes: 4,
  assessReserveBytes: 2,
  abortReserveBytes: 2,
  accelerators: [],
  memoryDomains: [{
    memoryDomainId: systemMemoryDomainId,
    kind: "System",
    totalBytes: 16,
    stableCapacityBytes: 14,
    availableBytes: Option.some(8),
    sharesSystemMemory: true,
  }],
  residentMemory: Option.none(),
  runtimeFailure: Option.none(),
}

const model = {
  availability: { _tag: "Available" },
  memory: Option.some([{
    memoryDomainId: systemMemoryDomainId,
    capacityBytes: 16,
    requiredBytes: 6,
    compatibilityReserveBytes: 2,
    warningReserveBytes: 4,
    remainingBytes: 8,
  }]),
} as unknown as ProviderModelCatalogEntry

describe("model memory presentation", () => {
  it("uses the same strict boundary as load admission", () => {
    expect(providerModelMemoryConditions(model, hardware).lacksCurrentHeadroom).toBe(true)
    expect(providerModelMemoryConditions(model, {
      ...hardware,
      availableSystemMemoryBytes: 9,
    }).lacksCurrentHeadroom).toBe(false)
  })

  it("keeps stable incompatibility distinct from temporary headroom", () => {
    expect(providerModelMemoryConditions({
      ...model,
      memory: Option.some([{
        memoryDomainId: systemMemoryDomainId,
        capacityBytes: 10,
        requiredBytes: 9,
        compatibilityReserveBytes: 2,
        warningReserveBytes: 4,
        remainingBytes: -1,
      }]),
    }, hardware).exceedsCapacity).toBe(true)
  })

  it("derives the warning state from assessment quantities", () => {
    expect(providerModelMemoryConditions({
      ...model,
      memory: Option.some([{
        memoryDomainId: systemMemoryDomainId,
        capacityBytes: 10,
        requiredBytes: 7,
        compatibilityReserveBytes: 2,
        warningReserveBytes: 4,
        remainingBytes: 1,
      }]),
    }, undefined).belowWarningReserve).toBe(true)
  })

  it("marks current headroom evidence unavailable until hardware is known", () => {
    expect(providerModelMemoryConditions(model, undefined).evidenceUnavailable).toBe(true)
    expect(providerModelMemoryConditions(model, hardware).evidenceUnavailable).toBe(false)
    expect(providerModelMemoryConditions({
      ...model,
      memory: Option.none(),
    }, hardware).evidenceUnavailable).toBe(true)
  })

  it("counts the current singleton residency as reclaimable before replacement", () => {
    expect(providerModelMemoryConditions(model, {
      ...hardware,
      memoryDomains: [{
        memoryDomainId: systemMemoryDomainId,
        kind: "System",
        totalBytes: 16,
        stableCapacityBytes: 14,
        availableBytes: Option.some(8),
        sharesSystemMemory: true,
      }],
      residentMemory: Option.some({
        domains: [{
          memoryDomainId: systemMemoryDomainId,
          modelBytes: 1,
          contextBytes: 0,
          computeBytes: 0,
          auxiliaryBytes: 0,
        }],
      }),
    }).lacksCurrentHeadroom).toBe(false)
  })
})
