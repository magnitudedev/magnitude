import { describe, expect, it } from "vitest"
import type { LocalInferenceState, LocalModelRecommendation } from "@magnitudedev/sdk"
import { buildLocalInferenceSelections, selectionMetadata } from "./view-model"

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
    localModelRole: "main",
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
  usage: { localModelRole: "main", sessionConcurrency: "one" },
  activeBinding: null,
  distribution: { _tag: "Ready", build: 10011, source: "managed" },
  host: { _tag: "Unavailable", message: "not needed" },
  operations: [],
  warnings: [],
} satisfies Omit<LocalInferenceState, "choices" | "recommendations">

describe("local inference selection view model", () => {
  it("does not duplicate a downloaded recommendation", () => {
    const state: LocalInferenceState = {
      ...baseState,
      choices: [{
        _tag: "StoredOwned",
        choiceId: recommendation.configurationId,
        displayName: recommendation.displayName,
        providerModelId: "local-model",
        contextTokens: recommendation.contextTokens,
        fitClass: recommendation.fitClass,
        compatible: true,
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

  it("keeps a compatible external server as an actionable running selection", () => {
    const state: LocalInferenceState = {
      ...baseState,
      choices: [{
        _tag: "RunningExternal",
        choiceId: "external-choice",
        displayName: "External model",
        providerModelId: "external-model",
        contextTokens: 48_000,
        fitClass: "unknown",
        compatible: true,
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
})
