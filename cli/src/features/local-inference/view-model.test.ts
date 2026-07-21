import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { ProviderModelIdSchema, type LocalInferenceState, type LocalModelRecommendation } from "@magnitudedev/sdk"
import {
  buildLocalInferenceSelections,
  describeLocalHardware,
  selectionCapacityWarning,
  selectionMetadata,
} from "./view-model"

const recommendation: LocalModelRecommendation = {
  configurationId: "configuration-1",
  catalogModelId: "catalog-1",
  badge: "recommended",
  displayName: "Recommended model",
  family: "test",
  architecture: "dense",
  quantization: {
    format: "Q4_K_M",
    quantAwareCheckpoint: false,
    fidelityLabel: "Test fidelity",
    fidelityEvidence: "Test evidence",
    fidelitySourceUrl: "https://example.invalid/model",
  },
  quantTag: "Q4_K_M",
  repo: "example/model",
  revision: "revision",
  files: [{ path: "model.gguf", role: "weights", sizeBytes: 4_000, sha256: "sha256" }],
  totalDownloadBytes: 4_000,
  sourcePageUrl: "https://example.invalid/model",
  license: { id: "test", url: "https://example.invalid/license", acknowledgementRequired: false },
  contextTokens: 32_768,
  modelMaximumContextTokens: 32_768,
  estimatedRuntimeBytes: 5_000,
  stableCapacityBudgetBytes: 10_000,
  fitMarginBytes: 5_000,
  fitClass: "cpu_or_unified",
  constrainedContext: false,
  explanation: "Fits the test host.",
}

const baseState = {
  activeBinding: null,
  host: { _tag: "Unavailable", message: "not needed" },
  operations: [],
  warnings: [],
} satisfies Omit<LocalInferenceState, "choices" | "recommendationState">

describe("local inference selection view model", () => {
  it("presents Apple Silicon as one unified-memory system", () => {
    expect(describeLocalHardware({
      platform: "macos",
      architecture: "aarch64",
      topologyFingerprint: "test",
      systemMemoryBytes: 64 * 1024 ** 3,
      cpuModel: "Apple M4 Max",
      logicalCores: 16,
      memoryDomains: [{
        id: "system",
        kind: "unified_memory",
        totalCapacityBytes: 64 * 1024 ** 3,
        stableCapacityBytes: 51.2 * 1024 ** 3,
        currentFreeBytes: null,
        sharesSystemMemory: true,
        backendNames: ["Metal"],
        deviceNames: ["Apple M4 Max"],
        splitGroupId: null,
      }],
      residentMemory: null,
    })).toEqual({
      system: {
        name: "Apple M4 Max",
        details: [
          "macOS · Apple Silicon · 16 logical CPU cores",
          "64.0 GiB unified memory · Metal GPU acceleration",
        ],
      },
      accelerators: [],
    })
  })

  it("uses ICN unified-domain semantics without platform-specific topology inference", () => {
    const gib = 1024 ** 3
    expect(describeLocalHardware({
      platform: "linux",
      architecture: "x86_64",
      topologyFingerprint: "test",
      systemMemoryBytes: 32 * gib,
      cpuModel: "Example CPU",
      logicalCores: 8,
      memoryDomains: [{
        id: "system",
        kind: "unified_memory",
        totalCapacityBytes: 32 * gib,
        stableCapacityBytes: 30.5 * gib,
        currentFreeBytes: null,
        sharesSystemMemory: true,
        backendNames: ["Vulkan"],
        deviceNames: ["Integrated GPU"],
        splitGroupId: null,
      }],
      residentMemory: null,
    })).toEqual({
      system: {
        name: "Example CPU",
        details: [
          "Linux · x86-64 · 8 logical CPU cores",
          "32.0 GiB unified memory · Vulkan GPU acceleration",
        ],
      },
      accelerators: [],
    })
  })

  it("presents every discrete GPU with its actual VRAM and backend", () => {
    const gib = 1024 ** 3
    expect(describeLocalHardware({
      platform: "linux",
      architecture: "x64",
      topologyFingerprint: "test",
      systemMemoryBytes: 128 * gib,
      cpuModel: "AMD Ryzen Threadripper",
      logicalCores: 64,
      memoryDomains: [{
        id: "system",
        kind: "system",
        totalCapacityBytes: 128 * gib,
        stableCapacityBytes: 102.4 * gib,
        currentFreeBytes: null,
        sharesSystemMemory: false,
        backendNames: [],
        deviceNames: [],
        splitGroupId: null,
      }, ...[0, 1].map((index) => ({
        id: `cuda:${index}`,
        kind: "physical_device" as const,
        totalCapacityBytes: 24 * gib,
        stableCapacityBytes: 21.6 * gib,
        currentFreeBytes: null,
        sharesSystemMemory: false,
        backendNames: ["CUDA"],
        deviceNames: [`NVIDIA GeForce RTX 4090 #${index + 1}`],
        splitGroupId: "cuda:group",
      }))],
      residentMemory: null,
    })).toEqual({
      system: {
        name: "AMD Ryzen Threadripper",
        details: [
          "Linux · x86-64 · 64 logical CPU cores",
          "128.0 GiB system memory",
        ],
      },
      accelerators: [{
        name: "NVIDIA GeForce RTX 4090 #1",
        details: "24.0 GiB VRAM · CUDA GPU acceleration",
      }, {
        name: "NVIDIA GeForce RTX 4090 #2",
        details: "24.0 GiB VRAM · CUDA GPU acceleration",
      }],
    })
  })

  it("does not duplicate a downloaded recommendation", () => {
    const state: LocalInferenceState = {
      ...baseState,
      choices: [{
        _tag: "Stored",
        choiceId: recommendation.configurationId,
        displayName: recommendation.displayName,
        providerModelId: ProviderModelIdSchema.make("local-model"),
        contextTokens: recommendation.contextTokens,
        fitClass: recommendation.fitClass,
        availability: { _tag: "Available" },
        fitAssessment: { _tag: "NotAssessed" },
        explanation: "Stored and ready.",
        residency: "unloaded",
      }],
      recommendationState: { _tag: "Ready", recommendations: [recommendation] },
    }

    expect(buildLocalInferenceSelections(state)).toEqual([expect.objectContaining({
      kind: "stored",
      id: recommendation.configurationId,
    })])
  })

  it("keeps an available external server as an actionable running selection", () => {
    const state: LocalInferenceState = {
      ...baseState,
      choices: [{
        _tag: "Running",
        choiceId: "external-choice",
        displayName: "External model",
        providerModelId: ProviderModelIdSchema.make("external-model"),
        contextTokens: 48_000,
        fitClass: "unknown",
        availability: { _tag: "Available" },
        fitAssessment: { _tag: "NotAssessed" },
        explanation: "Observed read-only endpoint.",
        residency: "loaded",
        quantization: {
          format: "UD-Q6_K_XL",
          quantAwareCheckpoint: false,
          fidelityLabel: "Very high fidelity with minimal quality loss",
          fidelityEvidence: "Catalog evidence.",
          fidelitySourceUrl: "https://example.invalid/model",
        },
        sizeBytes: 32_600_719_872,
      }],
      recommendationState: { _tag: "Ready", recommendations: [] },
    }

    const selections = buildLocalInferenceSelections(state)
    expect(selections).toEqual([expect.objectContaining({
      kind: "running",
      id: "external-choice",
    })])
    expect(selectionMetadata(selections[0]!)).toBe("UD-Q6_K_XL · 30.4 GiB · 48K context")
  })

  it("keeps a capacity-risk model actionable and exposes its warning", () => {
    const state: LocalInferenceState = {
      ...baseState,
      choices: [{
        _tag: "Stored",
        choiceId: "large-model",
        displayName: "Large model",
        providerModelId: ProviderModelIdSchema.make("large-model"),
        contextTokens: 32_768,
        fitClass: "cpu_or_unified",
        availability: { _tag: "Available" },
        fitAssessment: {
          _tag: "Assessed",
          requiredTotalBytes: 24 * 1024 ** 3,
          domains: [{ memoryDomainId: "system", requiredBytes: 24 * 1024 ** 3, stableCapacityBytes: 20 * 1024 ** 3, marginBytes: -4 * 1024 ** 3 }],
          result: "does_not_fit",
        },
        explanation: "Loading may fail.",
        residency: "unloaded",
      }],
      recommendationState: { _tag: "Ready", recommendations: [] },
    }

    const selections = buildLocalInferenceSelections(state)
    expect(selections).toHaveLength(1)
    expect(selectionCapacityWarning(selections[0]!)).toContain("estimated 24.0 GiB")
  })

  it("excludes a hard-disabled model independently of fit state", () => {
    const state: LocalInferenceState = {
      ...baseState,
      choices: [{
        _tag: "Stored",
        choiceId: "unavailable",
        displayName: "Unavailable model",
        providerModelId: ProviderModelIdSchema.make("unavailable"),
        fitClass: "unknown",
        availability: { _tag: "Disabled", reason: "installation_unavailable" },
        fitAssessment: { _tag: "NotAssessed" },
        explanation: "No installation.",
        residency: "unloaded",
      }],
      recommendationState: { _tag: "Ready", recommendations: [] },
    }

    expect(buildLocalInferenceSelections(state)).toEqual([])
  })
})
