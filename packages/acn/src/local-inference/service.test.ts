import { describe, expect, it } from "vitest"
import { Schema } from "effect"
import { Generated } from "@magnitudedev/icn"
import type { LocalModelRecommendation } from "@magnitudedev/protocol"
import { LOCAL_MODEL_CATALOG } from "./catalog"
import {
  hostToWire,
  selectRecommendationPortfolio,
  type RankedRecommendationCandidate,
} from "./service"

const recommendationCandidate = (input: {
  modelId: string
  family?: string
  quality: number
  fidelity: number
  context: 100_000 | 200_000
  runtime: number
  quantTag?: string
}): RankedRecommendationCandidate => {
  const quantTag = input.quantTag ?? `Q${input.fidelity}`
  const value: LocalModelRecommendation = {
    configurationId: `${input.modelId}:${quantTag}:ctx${input.context}`,
    catalogModelId: `${input.modelId}:${quantTag}`,
    badge: "alternative",
    displayName: input.modelId,
    family: input.family ?? "family-a",
    architecture: "dense",
    quantization: {
      format: quantTag,
      quantAwareCheckpoint: false,
      fidelityLabel: quantTag,
      fidelityEvidence: "test",
      fidelitySourceUrl: "https://example.invalid/fidelity",
    },
    quantTag,
    repo: `example/${input.modelId}`,
    revision: "revision",
    files: [{ path: "model.gguf", role: "weights", sizeBytes: input.runtime, sha256: "sha256" }],
    totalDownloadBytes: input.runtime,
    sourcePageUrl: "https://example.invalid/model",
    license: { id: "test", url: "https://example.invalid/license", acknowledgementRequired: false },
    contextTokens: input.context,
    modelMaximumContextTokens: 200_000,
    estimatedRuntimeBytes: input.runtime,
    stableCapacityBudgetBytes: 1_000,
    fitMarginBytes: 1_000 - input.runtime,
    fitClass: "full_accelerator",
    constrainedContext: input.context === 100_000,
    explanation: "test",
  }
  return {
    value,
    modelId: input.modelId,
    modelQualityRank: input.quality,
    fidelityRank: input.fidelity,
  }
}

describe("local inference hardware projection", () => {
  it("projects Apple unified memory once and uses accelerator descriptions", () => {
    const gib = 1024 ** 3
    const hardware = Schema.decodeUnknownSync(Generated.HardwareSnapshotSchema)({
      captured_at: 1,
      platform: "macos",
      architecture: "aarch64",
      cpu_model: "Apple M4 Max",
      logical_cores: 16,
      system_memory: {
        total_bytes: 64 * gib,
        current_available_bytes: 40 * gib,
      },
      native_build: "test",
      enabled_backends: ["CPU", "MTL"],
      assessment_policy: "test",
      capacity_policy: "test",
      topology_fingerprint: "test",
      memory_domains: [{
        id: "system",
        kind: "unified_memory",
        total_capacity_bytes: 64 * gib,
        stable_capacity_bytes: 62.5 * gib,
        current_free_bytes: 40 * gib,
        shares_system_memory: true,
        devices: [{
          id: "cpu",
          backend: "CPU",
          name: "CPU",
          description: "Apple M4 Max",
          kind: "cpu",
          memory_limit: null,
        }, {
          id: "metal",
          backend: "MTL",
          name: "MTL0",
          description: "Apple M4 Max",
          kind: "gpu",
          memory_limit: {
            kind: "recommended_working_set",
            total_bytes: 48 * gib,
            stable_bytes: 46.5 * gib,
            current_free_bytes: 30 * gib,
          },
        }],
      }],
    })

    expect(hostToWire(hardware)).toEqual({
      platform: "macos",
      architecture: "aarch64",
      systemMemoryBytes: 64 * gib,
      cpuModel: "Apple M4 Max",
      logicalCores: 16,
      memoryDomains: [{
        id: "system",
        kind: "unified_memory",
        totalCapacityBytes: 64 * gib,
        stableCapacityBytes: 62.5 * gib,
        currentFreeBytes: 40 * gib,
        sharesSystemMemory: true,
        backendNames: ["Metal"],
        deviceNames: ["Apple M4 Max"],
        splitGroupId: null,
      }],
    })
  })
})

describe("local model recommendation policy", () => {
  it("assesses only 100K and 200K product contexts", () => {
    expect(new Set(LOCAL_MODEL_CATALOG.flatMap((entry) => entry.supportedContextTokens)))
      .toEqual(new Set([100_000, 200_000]))
  })

  it("keeps one configuration per base model, preferring fidelity before context", () => {
    const recommendations = selectRecommendationPortfolio([
      recommendationCandidate({ modelId: "model-a", quality: 10, fidelity: 80, context: 100_000, runtime: 80 }),
      recommendationCandidate({ modelId: "model-a", quality: 10, fidelity: 80, context: 200_000, runtime: 90 }),
      recommendationCandidate({ modelId: "model-a", quality: 10, fidelity: 60, context: 200_000, runtime: 70 }),
    ])

    expect(recommendations).toHaveLength(1)
    expect(recommendations[0]).toMatchObject({
      displayName: "model-a",
      quantTag: "Q80",
      contextTokens: 200_000,
      badge: "recommended",
    })
  })

  it("uses 100K as the fallback when the preferred 200K profile does not fit", () => {
    const recommendations = selectRecommendationPortfolio([
      recommendationCandidate({ modelId: "model-a", quality: 10, fidelity: 80, context: 100_000, runtime: 80 }),
    ])

    expect(recommendations[0]?.contextTokens).toBe(100_000)
  })

  it("builds a diverse portfolio and assigns badges from actual differences", () => {
    const recommendations = selectRecommendationPortfolio([
      recommendationCandidate({ modelId: "primary", quality: 100, fidelity: 60, context: 200_000, runtime: 100 }),
      recommendationCandidate({ modelId: "higher-fidelity", quality: 90, fidelity: 80, context: 100_000, runtime: 90 }),
      recommendationCandidate({ modelId: "lighter", quality: 80, fidelity: 60, context: 200_000, runtime: 70 }),
      recommendationCandidate({ modelId: "other-family", family: "family-b", quality: 70, fidelity: 60, context: 200_000, runtime: 95 }),
    ])

    expect(recommendations.map(({ displayName, badge }) => ({ displayName, badge }))).toEqual([
      { displayName: "primary", badge: "recommended" },
      { displayName: "lighter", badge: "lighter" },
      { displayName: "higher-fidelity", badge: "higher_fidelity" },
      { displayName: "other-family", badge: "alternative" },
    ])
    expect(new Set(recommendations.map((recommendation) => recommendation.displayName)).size)
      .toBe(recommendations.length)
  })
})
