import { describe, expect, test } from "vitest"
import type { LocalInferenceCapabilities } from "@magnitudedev/protocol"
import {
  BASELINE_CONTEXT_TOKENS,
  GIB,
  recommendLocalModels,
  stableCapacityFromCapabilities,
  systemCapacityBudget,
} from "./recommendations"
import type { StableInferenceCapacity } from "./types"

const cpu = (gib: number): StableInferenceCapacity => ({
  systemMemoryBytes: gib * GIB,
  acceleratorDomains: [],
})

describe("local inference recommendation policy", () => {
  test.each([
    [8, undefined],
    [16, "Qwen3.5 9B"],
    [24, "Gemma 4 26B-A4B"],
    [32, "Qwen3.6 35B-A3B"],
    [48, "Qwen3.6 35B-A3B"],
    [64, "Qwen3.6 35B-A3B"],
    [96, "Qwen3.6 35B-A3B"],
    [128, "NVIDIA Nemotron 3 Super 120B-A12B"],
    [640, "GLM 5.2"],
  ])("returns a deterministic balanced tier for %i GiB", (gib, expected) => {
    const recommendation = recommendLocalModels(cpu(gib))[0]
    expect(recommendation?.displayName).toBe(expected)
    if (recommendation) {
      expect(recommendation.contextTokens).toBeGreaterThanOrEqual(BASELINE_CONTEXT_TOKENS)
      expect(recommendation.estimatedRuntimeBytes).toBeLessThanOrEqual(recommendation.stableCapacityBudgetBytes)
      expect(recommendation.files.every((file) => file.downloadUrl.includes(recommendation.revision))).toBe(true)
    }
  })

  test("changes quant tiers with headroom instead of always defaulting to Q4", () => {
    expect(recommendLocalModels(cpu(32))[0]?.quantization.bitsClass).toBe("q4")
    expect(recommendLocalModels(cpu(48))[0]?.quantization.bitsClass).toBe("q6")
    expect(recommendLocalModels(cpu(64))[0]?.quantization.bitsClass).toBe("q8")
  })

  test("gives a lighter model its largest fitting context instead of minimizing context", () => {
    const recommendations = recommendLocalModels(cpu(64))
    const lighter = recommendations.find((item) => item.badge === "lighter")

    expect(lighter?.displayName).toBe("Gemma 4 26B-A4B")
    expect(lighter?.contextTokens).toBe(131_072)
  })

  test("uses the named stable OS reserve", () => {
    expect(systemCapacityBudget(16 * GIB)).toBe(8 * GIB)
    expect(systemCapacityBudget(64 * GIB)).toBeCloseTo(51.2 * GIB, -2)
  })

  test("transient free RAM and VRAM cannot change recommendations", () => {
    const capabilities = (free: number): LocalInferenceCapabilities => ({
      binary: { identity: "managed-test-binary" },
      system: { totalMemoryBytes: 64 * GIB },
      accelerators: [{
        id: "CUDA0",
        backend: "CUDA",
        description: "pre-Blackwell RTX",
        capacityBytes: 24 * GIB,
        capacityKind: "physical-device-memory",
        memoryDomainId: "pci:1",
        sharesSystemMemory: false,
        currentFreeBytes: free,
      }],
      warnings: [],
    })
    const busy = recommendLocalModels(stableCapacityFromCapabilities(capabilities(512 * 1024 ** 2)))
    const idle = recommendLocalModels(stableCapacityFromCapabilities(capabilities(23 * GIB)))
    expect(busy).toEqual(idle)
  })

  test("does not double-count Apple unified memory or duplicate backend domains", () => {
    const capabilities: LocalInferenceCapabilities = {
      binary: { identity: "managed-metal-binary" },
      system: { totalMemoryBytes: 32 * GIB },
      accelerators: [
        {
          id: "MTL0",
          backend: "Metal",
          description: "Apple GPU",
          capacityBytes: 28 * GIB,
          capacityKind: "recommended-working-set",
          memoryDomainId: "unified:0",
          sharesSystemMemory: true,
        },
        {
          id: "Vulkan0",
          backend: "Vulkan",
          description: "same Apple GPU",
          capacityBytes: 30 * GIB,
          capacityKind: "recommended-working-set",
          memoryDomainId: "unified:0",
          sharesSystemMemory: true,
        },
      ],
      warnings: [],
    }
    const stable = stableCapacityFromCapabilities(capabilities)
    expect(stable.acceleratorDomains).toHaveLength(1)
    expect(stable.acceleratorDomains[0]?.capacityBytes).toBe(28 * GIB)
    expect(recommendLocalModels(stable)[0]?.displayName).toBe("Qwen3.6 35B-A3B")
  })

  test("combines discrete accelerator capacity only for an explicit supported split group", () => {
    const base: StableInferenceCapacity = {
      systemMemoryBytes: 16 * GIB,
      acceleratorDomains: [
        { memoryDomainId: "gpu:0", capacityBytes: 16 * GIB, sharesSystemMemory: false, preferredBackend: "CUDA" },
        { memoryDomainId: "gpu:1", capacityBytes: 16 * GIB, sharesSystemMemory: false, preferredBackend: "CUDA" },
      ],
    }
    const withoutSplit = recommendLocalModels(base)[0]
    const withSplit = recommendLocalModels({
      ...base,
      acceleratorDomains: base.acceleratorDomains.map((domain) => ({ ...domain, modelSplitGroupId: "cuda:all" })),
    })[0]
    expect(withoutSplit?.displayName).toBe("Qwen3.6 27B")
    expect(withSplit?.displayName).toBe("Qwen3.6 35B-A3B")
  })

  test("keeps exact model, quant, and context bound in opaque configuration ids", () => {
    const recommendations = recommendLocalModels(cpu(32))
    for (const recommendation of recommendations) {
      expect(recommendation.configurationId).toBe(
        `${recommendation.catalogModelId}@${recommendation.revision}@ctx-${recommendation.contextTokens}`,
      )
    }
  })

  test("treats very large models as normal capacity-gated recommendations", () => {
    expect(recommendLocalModels(cpu(128))[0]?.quantization.bitsClass).toBe("mxfp4")
    expect(recommendLocalModels(cpu(640))[0]?.displayName).toBe("GLM 5.2")
  })
})
