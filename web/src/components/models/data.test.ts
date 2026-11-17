import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { CatalogFormModelIdSchema, type LocalModel, type RecommendedModel } from "@magnitudedev/sdk"
import { makeSetupModel } from "@magnitudedev/client-common/fixtures/model"
import { offeredFromRecommendation, ownedModel } from "./data"

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
