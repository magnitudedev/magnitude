import { Option } from "effect"
import { describe, expect, it } from "vitest"
import type { LocalInferenceHardware, LocalModel, ProviderModelId } from "@magnitudedev/sdk"
import {
  localModelRankingUtility,
  rankedLocalModelOptions,
  targetPhysicalMemoryBytes,
  type LocalModelOption,
} from "./options"

const option = (
  modelId: string,
  totalRequiredBytes: number,
  scores: { intelligence: number; speed: number; quality: number },
  kind: LocalModelOption["kind"] = "downloadable",
): LocalModelOption => ({
  id: `${kind}:${modelId}`,
  kind,
  model: {
    modelId: modelId as ProviderModelId,
    servingState: {
      _tag: "Assessed",
      assessment: {
        _tag: "Fits",
        memory: { totalRequiredBytes },
      },
      rankingScores: Option.some(scores),
    },
  } as unknown as LocalModel,
})

describe("local model ranking", () => {
  it("moves utility from speed toward intelligence while quality always contributes", () => {
    const scores = { intelligence: 0.8, speed: 0.4, quality: 0.5 }
    expect(localModelRankingUtility(scores, 0)).toBeCloseTo(0.4 ** 0.9 * 0.5 ** 0.1)
    expect(localModelRankingUtility(scores, 1)).toBeCloseTo(0.8 ** 0.9 * 0.5 ** 0.1)
  })

  it("filters by the exact memory budget before sorting and truncating", () => {
    const fast = option("fast", 8, { intelligence: 0.4, speed: 1, quality: 1 })
    const smart = option("smart", 8, { intelligence: 1, speed: 0.4, quality: 1 })
    const overBudget = option("over", 9, { intelligence: 1, speed: 1, quality: 1 })
    expect(rankedLocalModelOptions(
      [smart, overBudget, fast],
      { fastToSmart: 0, memoryBudgetBytes: 8 },
      1,
    )).toEqual([fast])
  })

  it("breaks equal utility by canonical model ID", () => {
    const scores = { intelligence: 0.8, speed: 0.8, quality: 0.8 }
    const second = option("b", 1, scores)
    const first = option("a", 1, scores)
    expect(rankedLocalModelOptions(
      [second, first],
      { fastToSmart: 0.5, memoryBudgetBytes: 1 },
    )).toEqual([first, second])
  })

  it("ranks installed and downloadable choices together", () => {
    const stored = option("stored", 1, { intelligence: 1, speed: 1, quality: 1 }, "stored")
    const downloadable = option("downloadable", 1, { intelligence: 0.5, speed: 0.5, quality: 1 })

    expect(rankedLocalModelOptions(
      [downloadable, stored],
      { fastToSmart: 0.5, memoryBudgetBytes: 1 },
    )).toEqual([stored, downloadable])
  })

  it("sums distinct normalized physical memory domains without adding system memory twice", () => {
    const hardware = {
      totalSystemMemoryBytes: 64,
      memoryDomains: [{ totalBytes: 64 }, { totalBytes: 24 }],
    } as unknown as LocalInferenceHardware
    expect(targetPhysicalMemoryBytes(hardware)).toBe(88)
  })
})
