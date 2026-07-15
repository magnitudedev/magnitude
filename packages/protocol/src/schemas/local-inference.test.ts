import { describe, expect, test } from "vitest"
import { Schema } from "effect"
import {
  LocalInferenceCapabilities,
  LocalInferenceServingProfile,
  LocalInferenceUsageSelection,
  LocalModelDownloadProgress,
} from "./local-inference"

describe("local inference protocol schemas", () => {
  test("accepts stable capacity plus transient diagnostic memory", () => {
    const decoded = Schema.decodeUnknownSync(LocalInferenceCapabilities)({
      binary: { identity: "managed-llama-server" },
      system: { totalMemoryBytes: 64 * 1024 ** 3, logicalCores: 12 },
      accelerators: [{
        id: "CUDA0",
        backend: "CUDA",
        description: "consumer RTX",
        capacityBytes: 24 * 1024 ** 3,
        capacityKind: "physical-device-memory",
        memoryDomainId: "pci:1",
        sharesSystemMemory: false,
        currentFreeBytes: 1,
      }],
      warnings: [],
    })
    expect(decoded.accelerators[0]?.currentFreeBytes).toBe(1)
  })

  test("rejects negative byte counts", () => {
    expect(() => Schema.decodeUnknownSync(LocalModelDownloadProgress)({
      operationId: "operation",
      status: "downloading",
      completedBytes: -1,
      totalBytes: 100,
      resumable: true,
    })).toThrow()
  })

  test("accepts only the two supported usage choices", () => {
    expect(Schema.decodeUnknownSync(LocalInferenceUsageSelection)({
      localModelRole: "subagent",
      sessionConcurrency: "up_to_three",
    })).toEqual({
      localModelRole: "subagent",
      sessionConcurrency: "up_to_three",
    })
    expect(() => Schema.decodeUnknownSync(LocalInferenceUsageSelection)({
      localModelRole: "both",
      sessionConcurrency: "unlimited",
    })).toThrow()
  })

  test("requires positive uniform serving-profile dimensions", () => {
    const profile = {
      localModelRole: "main",
      sessionConcurrency: "up_to_three",
      parallelSlots: 3,
      contextTokensPerSlot: 100_000,
      totalContextCapacityTokens: 300_000,
      slotAllocation: "uniform",
      runtimeProfileId: "profile",
    }
    expect(Schema.decodeUnknownSync(LocalInferenceServingProfile)(profile)).toEqual(profile)
    expect(() => Schema.decodeUnknownSync(LocalInferenceServingProfile)({
      ...profile,
      parallelSlots: 0,
    })).toThrow()
  })
})
