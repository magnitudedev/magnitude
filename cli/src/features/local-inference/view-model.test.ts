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
  files: [{ path: "model.gguf", sizeBytes: 4_000, sha256: "sha256", downloadUrl: "https://example.invalid/model.gguf" }],
  totalDownloadBytes: 4_000,
  sourcePageUrl: "https://example.invalid/model",
  license: { id: "test", url: "https://example.invalid/license", acknowledgementRequired: false },
  contextTokens: 32_768,
  servingProfile: {
    sessionConcurrency: "one",
    parallelSlots: 1,
    contextTokensPerSlot: 32_768,
    totalContextCapacityTokens: 32_768,
    slotAllocation: "uniform",
    runtimeProfileId: "test",
  },
  modelMaximumContextTokens: 32_768,
  estimatedRuntimeBytes: 5_000,
  stableCapacityBudgetBytes: 10_000,
  fitMarginBytes: 5_000,
  fitClass: "cpu_or_unified",
  constrainedContext: false,
  explanation: "Fits the test host.",
}

const baseState = {
  usage: { sessionConcurrency: "one" },
  activeBinding: null,
  llamaCpp: {
    minimumBuild: 8868,
    recommendedBuild: 10011,
    installations: [{
      id: "managed",
      executables: { serverPath: "/managed/llama-server", fitParamsPath: "/managed/llama-fit-params" },
      build: 10011,
      ownership: "magnitude",
      discoveries: [{ _tag: "Managed", markerPath: "/managed/current.json", release: "test" }],
    }],
    selectedInstallationId: Option.some("managed"),
    activeManagedInstallationId: Option.none(),
    managedInstall: { availability: { _tag: "Available", build: 10011 }, operation: { _tag: "Idle" } },
    diagnostics: [],
  },
  host: { _tag: "Unavailable", message: "not needed" },
  operations: [],
  warnings: [],
} satisfies Omit<LocalInferenceState, "choices" | "recommendations">

describe("local inference selection view model", () => {
  it("presents Apple Silicon as one unified-memory system", () => {
    expect(describeLocalHardware({
      platform: "darwin",
      architecture: "arm64",
      systemMemoryBytes: 64 * 1024 ** 3,
      cpuModel: "Apple M4 Max",
      logicalCores: 16,
      memoryDomains: [{
        id: "system",
        kind: "system",
        totalCapacityBytes: 64 * 1024 ** 3,
        stableCapacityBytes: 51.2 * 1024 ** 3,
        currentFreeBytes: null,
        sharesSystemMemory: false,
        backendNames: [],
        deviceNames: [],
        splitGroupId: null,
      }, {
        id: "metal",
        kind: "unified_working_set",
        totalCapacityBytes: 48 * 1024 ** 3,
        stableCapacityBytes: 43.2 * 1024 ** 3,
        currentFreeBytes: null,
        sharesSystemMemory: true,
        backendNames: ["Metal"],
        deviceNames: ["Apple M4 Max"],
        splitGroupId: null,
      }],
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

  it("presents every discrete GPU with its actual VRAM and backend", () => {
    const gib = 1024 ** 3
    expect(describeLocalHardware({
      platform: "linux",
      architecture: "x64",
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
        splitGroupId: "llamacpp:cuda",
      }))],
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
        _tag: "StoredOwned",
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
      recommendations: [recommendation],
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
        _tag: "RunningExternal",
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
      recommendations: [],
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
        _tag: "StoredOwned",
        choiceId: "large-model",
        displayName: "Large model",
        providerModelId: ProviderModelIdSchema.make("large-model"),
        contextTokens: 32_768,
        fitClass: "cpu_or_unified",
        availability: { _tag: "Available" },
        fitAssessment: {
          _tag: "Estimated",
          estimatedTotalBytes: 24 * 1024 ** 3,
          domains: [{ memoryDomainId: "system", estimatedBytes: 24 * 1024 ** 3, stableCapacityBytes: 20 * 1024 ** 3, marginBytes: -4 * 1024 ** 3 }],
          result: "capacity_risk",
        },
        explanation: "Loading may fail.",
        residency: "unloaded",
      }],
      recommendations: [],
    }

    const selections = buildLocalInferenceSelections(state)
    expect(selections).toHaveLength(1)
    expect(selectionCapacityWarning(selections[0]!)).toContain("estimated 24.0 GiB")
  })

  it("excludes a hard-disabled model independently of fit state", () => {
    const state: LocalInferenceState = {
      ...baseState,
      choices: [{
        _tag: "StoredOwned",
        choiceId: "unavailable",
        displayName: "Unavailable model",
        providerModelId: ProviderModelIdSchema.make("unavailable"),
        fitClass: "unknown",
        availability: { _tag: "Disabled", reason: "installation_unavailable" },
        fitAssessment: { _tag: "NotAssessed" },
        explanation: "No installation.",
        residency: "unloaded",
      }],
      recommendations: [],
    }

    expect(buildLocalInferenceSelections(state)).toEqual([])
  })
})
