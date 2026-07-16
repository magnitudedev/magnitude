import { describe, expect, test } from "vitest"
import { Schema } from "effect"
import {
  ActiveLocalBindingSummary,
  LocalInferenceDistributionState,
  LocalInferenceErrorCode,
  LocalInferenceHostProfile,
  LocalModelChoice,
  LocalInferenceServingProfile,
  LocalInferenceState,
  LocalInferenceUsageSelection,
} from "./local-inference"

describe("local inference protocol schemas", () => {
  test("keeps stable capacity distinct from point-in-time free memory", () => {
    const decoded = Schema.decodeUnknownSync(LocalInferenceHostProfile)({
      systemMemoryBytes: 64 * 1024 ** 3,
      cpuModel: "test cpu",
      logicalCores: 12,
      memoryDomains: [{
        id: "gpu:0",
        kind: "physical_device",
        stableCapacityBytes: 24 * 1024 ** 3,
        currentFreeBytes: 1,
        sharesSystemMemory: false,
        deviceNames: ["test gpu"],
        splitGroupId: null,
      }],
    })
    expect(decoded.memoryDomains[0]?.stableCapacityBytes).toBe(24 * 1024 ** 3)
    expect(decoded.memoryDomains[0]?.currentFreeBytes).toBe(1)
  })

  test("accepts only the supported usage choices", () => {
    expect(Schema.decodeUnknownSync(LocalInferenceUsageSelection)({
      sessionConcurrency: "up_to_three",
    })).toEqual({
      sessionConcurrency: "up_to_three",
    })
    expect(() => Schema.decodeUnknownSync(LocalInferenceUsageSelection)({
      sessionConcurrency: "unlimited",
    })).toThrow()
  })

  test("requires positive uniform serving-profile dimensions", () => {
    const profile = {
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

  test("local inference state cannot contain onboarding state", () => {
    const state = {
      usage: null,
      activeBinding: null,
      distribution: { _tag: "Missing" },
      host: { _tag: "Unavailable", message: "distribution missing" },
      choices: [],
      operations: [],
      recommendations: [],
      warnings: [],
    }
    expect(Schema.decodeUnknownSync(LocalInferenceState)(state)).toEqual(state)
  })

  test("round-trips every tagged choice, binding, and distribution variant", () => {
    const choiceFields = {
      choiceId: "opaque-choice",
      displayName: "Model",
      providerModelId: "model",
      contextTokens: 8192,
      fitClass: "unknown",
      compatible: true,
      explanation: "test",
      residency: "loaded",
    }
    for (const _tag of ["RunningExternal", "RunningManaged", "StoredOwned", "StoredExternal"] as const) {
      const choice = { _tag, ...choiceFields }
      expect(Schema.decodeUnknownSync(LocalModelChoice)(choice)).toEqual(choice)
    }

    for (const binding of [
      { _tag: "Managed", selectionId: "managed", providerModelId: "model", contextTokens: 8192 },
      { _tag: "External", selectionId: "external", providerModelId: "model", contextTokens: 8192 },
    ] as const) {
      expect(Schema.decodeUnknownSync(ActiveLocalBindingSummary)(binding)).toEqual(binding)
    }

    for (const distribution of [
      { _tag: "Missing" },
      { _tag: "Unsupported", message: "unsupported" },
      { _tag: "Invalid", message: "invalid" },
      { _tag: "Ready", build: 10011, source: "managed" },
    ] as const) {
      expect(Schema.decodeUnknownSync(LocalInferenceDistributionState)(distribution)).toEqual(distribution)
    }
  })

  test("accepts every error code in the closed wire vocabulary", () => {
    const codes = [
      "distribution_missing",
      "unsupported_platform",
      "invalid_selection",
      "artifact_unavailable",
      "license_required",
      "insufficient_disk_space",
      "integrity_failed",
      "artifact_not_owned",
      "artifact_active",
      "context_mismatch",
      "server_start_failed",
      "external_server_unavailable",
      "configuration_failed",
      "runtime_probe_failed",
      "cancelled",
    ] as const
    for (const code of codes) {
      expect(Schema.decodeUnknownSync(LocalInferenceErrorCode)(code)).toBe(code)
    }
  })
})
