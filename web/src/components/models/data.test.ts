import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { CatalogFormModelIdSchema, type LocalModel, type RecommendedModel } from "@magnitudedev/sdk"
import { makeSetupModel } from "@magnitudedev/client-common/fixtures/model"
import { offeredFromRecommendation, ownedModel, transferLabel } from "./data"

const GB = 1e9
const rung = (id: string, storageBytes: number, installed = false): LocalModel =>
  ({ ...makeSetupModel(installed), modelId: CatalogFormModelIdSchema.make(id), storageBytes })
const ladder = [rung("flash:gguf:q4", 115 * GB), rung("flash:gguf:q6", 173 * GB)]
const recommendation = {
  base: "flash", name: "Qwen3.8 Flash Next", developer: { lab: "alibaba", name: "Alibaba", country: "CN" },
  chosen: Option.some("flash:gguf:q4"), rungs: ladder.map(model => model.modelId),
} as unknown as RecommendedModel
const offered = (freeDiskBytes: number | null, models = ladder) =>
  offeredFromRecommendation(recommendation, new Map(models.map(model => [model.modelId, model])), "Device", freeDiskBytes)!

describe("a recommended model's sizes", () => {
  it("mark a size without room on disk, as the service counts it, so it can't be picked (6-bit 173 GB with 153 GB free)", () => {
    const quants = offered(153 * GB).quants
    expect(quants.map(quant => [quant.key, quant.missingDiskBytes])).toEqual([["flash:gguf:q4", 0], ["flash:gguf:q6", 20 * GB]])
  })

  it("mark nothing when free disk is unknown, or for a size already downloaded", () => {
    expect(offered(null).quants.every(quant => quant.missingDiskBytes === 0)).toBe(true)
    const here = [ladder[0]!, rung("flash:gguf:q6", 173 * GB, true)]
    expect(offered(153 * GB, here).quants.every(quant => quant.missingDiskBytes === 0)).toBe(true)
  })
})

describe("a model whose download failed", () => {
  it("says why on its row, not only on hover", () => {
    const failed = { ...ladder[1]!, acquisitionState: { _tag: "InstallFailed", failure: { _tag: "InsufficientDiskSpace", requiredBytes: 172_859_493_888, availableBytes: 152_652_071_760 } } } as unknown as LocalModel
    expect(ownedModel(failed, "Device").downloadFailure).toBe("Not enough disk space. Free at least 20.2 GB and try again.")
    expect(ownedModel(ladder[0]!, "Device").downloadFailure).toBeNull()
  })
})

describe("a model's quant name", () => {
  it("is its quant's full name, the name Chat gives it too, not the shorthand of its variant label", () => {
    const model = { ...ladder[0]!, presentation: { ...ladder[0]!.presentation, variantLabel: "Q4 QAT" },
      catalogData: { ...(ladder[0] as { catalogData: object }).catalogData, quant: { name: "QAD-Q4_0", repository: "LiquidAI/LFM3-1.7B-GGUF" } } } as unknown as LocalModel
    expect(ownedModel(model, "Device").quant.name).toBe("QAD-Q4_0")
  })
})

describe("a transfer's label", () => {
  const progress = (stage: "queued" | "resolving" | "downloading", completedBytes: number, totalBytes: number) =>
    ({ stage, completedBytes, totalBytes, bytesPerSecond: Option.none() })

  it("says no 0% before bytes are downloading against a known total, as while queued to resume after a restart", () => {
    expect(transferLabel(progress("queued", 0, 0))).toEqual({ text: "Queued…", fraction: null })
    expect(transferLabel(progress("resolving", 0, 3_470_000_000))).toEqual({ text: "Preparing…", fraction: null })
    expect(transferLabel(progress("downloading", 0, 0))).toEqual({ text: "Downloading…", fraction: null })
  })

  it("gives the real percentage once downloading, the bytes already here included", () => {
    expect(transferLabel(progress("downloading", 1_353_300_000, 3_470_000_000))).toEqual({ text: "Downloading 39%", fraction: 0.39 })
  })
})
