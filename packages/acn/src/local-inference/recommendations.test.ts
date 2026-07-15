import { describe, expect, test } from "vitest"
import type {
  LocalInferenceCapabilities,
  LocalInferenceUsageSelection,
} from "@magnitudedev/protocol"
import { LOCAL_MODEL_CATALOG } from "./catalog"
import {
  GIB,
  estimateRuntimeBytes,
  parallelSlotsForUsage,
  recommendLocalModels,
  stableCapacityFromCapabilities,
  systemCapacityBudget,
} from "./recommendations"
import type { StableInferenceCapacity } from "./types"

const cpu = (gib: number): StableInferenceCapacity => ({
  systemMemoryBytes: gib * GIB,
  acceleratorDomains: [],
})

const MAIN_ONE = {
  localModelRole: "main",
  sessionConcurrency: "one",
} as const satisfies LocalInferenceUsageSelection

const MAIN_THREE = {
  localModelRole: "main",
  sessionConcurrency: "up_to_three",
} as const satisfies LocalInferenceUsageSelection

const SUBAGENT_ONE = {
  localModelRole: "subagent",
  sessionConcurrency: "one",
} as const satisfies LocalInferenceUsageSelection

const SUBAGENT_THREE = {
  localModelRole: "subagent",
  sessionConcurrency: "up_to_three",
} as const satisfies LocalInferenceUsageSelection

const ALL_USAGE = [MAIN_ONE, MAIN_THREE, SUBAGENT_ONE, SUBAGENT_THREE] as const

describe("local inference recommendation policy", () => {
  test.each([
    [MAIN_ONE, 1],
    [MAIN_THREE, 3],
    [SUBAGENT_ONE, 3],
    [SUBAGENT_THREE, 9],
  ] as const)("derives deterministic uniform parallelism from %o", (usage, expected) => {
    expect(parallelSlotsForUsage(usage)).toBe(expected)
  })

  test.each([
    [8, undefined],
    [16, "Qwen3.5 4B"],
    [24, "Gemma 4 12B"],
    [32, "Qwen3.6 27B"],
    [48, "Qwen3.6 35B-A3B"],
    [64, "Qwen3.6 35B-A3B"],
    [128, "NVIDIA Nemotron 3 Super 120B-A12B"],
    [256, "DeepSeek V4 Flash 284B-A13B"],
    [512, "NVIDIA Nemotron 3 Ultra 550B-A55B"],
    [768, "GLM 5.2 753B-A40B"],
  ])("returns a deterministic one-main-agent tier for %i GiB", (gib, expected) => {
    const recommendation = recommendLocalModels(cpu(gib), MAIN_ONE)[0]
    expect(recommendation?.displayName).toBe(expected)
    if (recommendation) {
      expect([100_000, 200_000]).toContain(recommendation.contextTokens)
      expect(recommendation.servingProfile.parallelSlots).toBe(1)
      expect(recommendation.estimatedRuntimeBytes).toBeLessThanOrEqual(
        recommendation.stableCapacityBudgetBytes,
      )
    }
  })

  test("changes recommendations when identical hardware reserves more context windows", () => {
    expect(recommendLocalModels(cpu(32), MAIN_ONE)[0]).toMatchObject({
      displayName: "Qwen3.6 27B",
      contextTokens: 200_000,
      servingProfile: { parallelSlots: 1 },
    })
    expect(recommendLocalModels(cpu(32), MAIN_THREE)[0]).toMatchObject({
      displayName: "Gemma 4 26B-A4B",
      contextTokens: 100_000,
      servingProfile: { parallelSlots: 3 },
    })
    expect(recommendLocalModels(cpu(32), SUBAGENT_ONE)[0]).toMatchObject({
      displayName: "Qwen3.6 27B",
      contextTokens: 64_000,
      servingProfile: { parallelSlots: 3 },
    })
    expect(recommendLocalModels(cpu(32), SUBAGENT_THREE)[0]).toMatchObject({
      displayName: "Qwen3.5 9B",
      contextTokens: 64_000,
      servingProfile: { parallelSlots: 9 },
    })
  })

  test("enforces the role-specific context ladders and uniform total capacity", () => {
    for (const usage of ALL_USAGE) {
      for (const gib of [16, 24, 32, 48, 64, 128, 256, 512, 768, 1024]) {
        for (const recommendation of recommendLocalModels(cpu(gib), usage)) {
          const expectedContexts = usage.localModelRole === "main"
            ? [200_000, 100_000]
            : [100_000, 64_000]
          expect(expectedContexts).toContain(recommendation.contextTokens)
          expect(recommendation.servingProfile.contextTokensPerSlot).toBe(
            recommendation.contextTokens,
          )
          expect(recommendation.servingProfile.totalContextCapacityTokens).toBe(
            recommendation.contextTokens * recommendation.servingProfile.parallelSlots,
          )
          expect(recommendation.servingProfile.slotAllocation).toBe("uniform")
        }
      }
    }
  })

  test("counts weights once and scales only runtime context with total reserved windows", () => {
    const entry = LOCAL_MODEL_CATALOG.find((item) => item.id === "qwen3.6-35b-a3b:UD-Q6_K_XL")!
    const one = estimateRuntimeBytes(entry, 100_000, 1)
    const three = estimateRuntimeBytes(entry, 100_000, 3)
    const nine = estimateRuntimeBytes(entry, 100_000, 9)
    const weights = entry.files.reduce((total, file) => total + file.sizeBytes, 0)

    expect(one).toBeGreaterThan(weights)
    expect(three).toBeGreaterThan(one)
    expect(nine).toBeGreaterThan(three)
    expect(nine).toBeLessThan(one * 9)
  })

  test("uses distinct configuration IDs for the two different three-slot routing intents", () => {
    const main = recommendLocalModels(cpu(64), MAIN_THREE)[0]!
    const subagent = recommendLocalModels(cpu(64), SUBAGENT_ONE)[0]!

    expect(main.servingProfile.parallelSlots).toBe(3)
    expect(subagent.servingProfile.parallelSlots).toBe(3)
    expect(main.configurationId).not.toBe(subagent.configurationId)
    expect(main.configurationId).toContain("@role-main@sessions-up_to_three@p-3@")
    expect(subagent.configurationId).toContain("@role-subagent@sessions-one@p-3@")
  })

  test("changes quant tiers with headroom instead of always defaulting to Q4", () => {
    expect(recommendLocalModels(cpu(48), MAIN_ONE)[0]?.quantization.bitsClass).toBe("q5")
    expect(recommendLocalModels(cpu(64), MAIN_ONE)[0]?.quantization.bitsClass).toBe("q8")
    expect(recommendLocalModels(cpu(64), MAIN_THREE)[0]?.quantization.bitsClass).toBe("q4")
  })

  test("returns three useful choices whenever three configurations fit", () => {
    for (const gib of [24, 32, 48, 64, 128, 256, 512, 768]) {
      const recommendations = recommendLocalModels(cpu(gib), MAIN_ONE)
      expect(recommendations, `${gib} GiB`).toHaveLength(3)
      expect(
        new Set(recommendations.map((item) => item.catalogModelId.split(":")[0])).size,
        `${gib} GiB`,
      ).toBeGreaterThanOrEqual(2)
    }
  })

  test("uses the higher-fidelity badge only for another quant of the primary model", () => {
    for (const usage of ALL_USAGE) {
      for (let gib = 12; gib <= 768; gib += 4) {
        const recommendations = recommendLocalModels(cpu(gib), usage)
        const primary = recommendations.find((item) => item.badge === "recommended")
        const higherFidelity = recommendations.find((item) => item.badge === "higher_fidelity")
        if (higherFidelity) {
          expect(higherFidelity.catalogModelId.split(":")[0], `${gib} GiB`).toBe(
            primary?.catalogModelId.split(":")[0],
          )
        }
      }
    }
  })

  test("uses the named stable OS reserve", () => {
    expect(systemCapacityBudget(16 * GIB)).toBe(8 * GIB)
    expect(systemCapacityBudget(64 * GIB)).toBeCloseTo(51.2 * GIB, -2)
  })

  test("transient free RAM and VRAM cannot change profile-aware recommendations", () => {
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
    const busy = recommendLocalModels(
      stableCapacityFromCapabilities(capabilities(512 * 1024 ** 2)),
      SUBAGENT_THREE,
    )
    const idle = recommendLocalModels(
      stableCapacityFromCapabilities(capabilities(23 * GIB)),
      SUBAGENT_THREE,
    )
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
    expect(recommendLocalModels(stable, MAIN_ONE)[0]?.displayName).toBe("Qwen3.6 27B")
  })

  test("combines discrete accelerator capacity only for an explicit supported split group", () => {
    const base: StableInferenceCapacity = {
      systemMemoryBytes: 16 * GIB,
      acceleratorDomains: [
        { memoryDomainId: "gpu:0", capacityBytes: 16 * GIB, sharesSystemMemory: false, preferredBackend: "CUDA" },
        { memoryDomainId: "gpu:1", capacityBytes: 16 * GIB, sharesSystemMemory: false, preferredBackend: "CUDA" },
      ],
    }
    const withoutSplit = recommendLocalModels(base, MAIN_ONE)[0]
    const withSplit = recommendLocalModels({
      ...base,
      acceleratorDomains: base.acceleratorDomains.map((domain) => ({
        ...domain,
        modelSplitGroupId: "cuda:all",
      })),
    }, MAIN_ONE)[0]
    expect(withoutSplit?.estimatedRuntimeBytes).toBeLessThanOrEqual(
      withoutSplit?.stableCapacityBudgetBytes ?? 0,
    )
    expect(withSplit?.estimatedRuntimeBytes).toBeLessThanOrEqual(
      withSplit?.stableCapacityBudgetBytes ?? 0,
    )
    expect(withSplit?.displayName).not.toBeUndefined()
  })

  test("keeps every serving input bound in opaque configuration IDs", () => {
    for (const usage of ALL_USAGE) {
      for (const recommendation of recommendLocalModels(cpu(64), usage)) {
        expect(recommendation.configurationId).toContain(recommendation.catalogModelId)
        expect(recommendation.configurationId).toContain(recommendation.revision)
        expect(recommendation.configurationId).toContain(`@role-${usage.localModelRole}`)
        expect(recommendation.configurationId).toContain(`@sessions-${usage.sessionConcurrency}`)
        expect(recommendation.configurationId).toContain(
          `@p-${recommendation.servingProfile.parallelSlots}@ctx-${recommendation.contextTokens}`,
        )
      }
    }
  })

  test("keeps very large models as normal profile-gated recommendations", () => {
    expect(recommendLocalModels(cpu(128), MAIN_ONE)[0]?.displayName).toBe(
      "NVIDIA Nemotron 3 Super 120B-A12B",
    )
    expect(recommendLocalModels(cpu(256), MAIN_ONE)[0]?.displayName).toBe(
      "DeepSeek V4 Flash 284B-A13B",
    )
    expect(recommendLocalModels(cpu(512), MAIN_ONE)[0]?.displayName).toBe(
      "NVIDIA Nemotron 3 Ultra 550B-A55B",
    )
    expect(recommendLocalModels(cpu(768), MAIN_ONE)[0]?.displayName).toBe(
      "GLM 5.2 753B-A40B",
    )
  })
})
