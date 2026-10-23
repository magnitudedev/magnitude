import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { CatalogFormModelIdSchema, CatalogSupportSchema, CustomFormModelIdSchema, HuggingFaceRepositoryIdSchema } from "@magnitudedev/sdk"
import type { CatalogLocalModel, CatalogSupport, LocalInferenceHardware, LocalModel, ProviderModelId } from "@magnitudedev/sdk"
import {
  localModelConnectionOptions,
  localModelRankingUtility,
  featuredCatalogModels,
  rankedLocalModelOptions,
  targetPhysicalMemoryBytes,
  type LocalModelOption,
} from "./options"
import { makeFoundModel, makeSetupModel } from "../application/fixtures/model"

describe("connection model choices", () => {
  it("includes all five available catalog, Hugging Face and found models without ranking custom models", () => {
    const catalog = ["Gemma", "Liquid", "MiniCPM"].map(name => ({
      ...makeSetupModel(true),
      modelId: CatalogFormModelIdSchema.make(`${name.toLowerCase()}:gguf:q4`),
      presentation: { ...makeSetupModel(true).presentation, displayName: name },
    }))
    const found = makeFoundModel({ modelId: CustomFormModelIdSchema.make("local:community-q4:abcdef123456") })
    const added = makeFoundModel({
      modelId: CustomFormModelIdSchema.make("hf:owner/finetune/model.gguf"),
      origin: { _tag: "Added", repository: HuggingFaceRepositoryIdSchema.make("owner/finetune"), commit: "pinned" },
      acquisitionState: makeSetupModel(true).acquisitionState,
      presentation: { ...found.presentation, displayName: "Added finetune" },
    })
    const models = [found, ...catalog, added]
    const choices = localModelConnectionOptions(models)
    expect(choices).toHaveLength(5)
    expect(new Set(choices.map(choice => choice.id))).toEqual(new Set(models.map(model => model.modelId)))
    expect(choices[0]).toEqual({ id: added.modelId, label: "Added finetune (Q4_K_P)" })
    expect(models[0]).toBe(found)
    expect(rankedLocalModelOptions(models.map(model => ({ id: model.modelId, kind: "stored", model })), {
      fastToSmart: 0.5, memoryBudgetBytes: 100,
    })).toHaveLength(3)
  })

  it("leaves out models that are unavailable, still assessing, unsupported or too large", () => {
    const found = makeFoundModel()
    const serving = found.servingState
    if (serving._tag !== "Assessed") throw new Error("expected assessed fixture")
    const assessment = serving.assessment
    if (assessment._tag !== "Fits") throw new Error("expected fitting fixture")
    const unavailable = makeFoundModel({ acquisitionState: { _tag: "NotInstalled" } })
    const assessing = makeFoundModel({ servingState: { _tag: "Assessing", profile: { contextLength: 32_768 } } })
    const unsupported = makeFoundModel({ servingState: { ...serving, assessment: {
      _tag: "Unsupported", environmentId: assessment.environmentId, profile: Option.none(),
      reasons: [{ _tag: "Unreadable" }], detail: "Unsupported model",
    } } })
    const tooLarge = makeFoundModel({ servingState: { ...serving, assessment: {
      _tag: "DoesNotFit", profile: { contextLength: 32_768 }, memoryDomains: [],
      assessmentId: assessment.assessmentId, environmentId: assessment.environmentId,
      totalRequiredBytes: 0, limitingResource: "memory", deficitBytes: 1,
    } } })
    expect(localModelConnectionOptions([makeSetupModel(false), unavailable, assessing, unsupported, tooLarge])).toEqual([])
    expect(localModelConnectionOptions([found])).toEqual([{ id: found.modelId, label: "Setup Model Uncensored (Q4_K_P)" }])
  })
})

const option = (
  modelId: string,
  totalRequiredBytes: number,
  scores: { intelligence: number; speed: number | null; fidelity: number },
  kind: LocalModelOption["kind"] = "downloadable",
  support: CatalogSupport = { _tag: "Supported" },
): LocalModelOption => ({
  id: `${kind}:${modelId}`,
  kind,
  model: {
    _tag: "Catalog",
    modelId: modelId as ProviderModelId,
    catalogData: { support },
    servingState: {
      _tag: "Assessed",
      assessment: {
        _tag: "Fits",
        memory: { totalRequiredBytes },
      },
      rankingScores: { ...scores, speed: Option.fromNullable(scores.speed) },
    },
  } as unknown as LocalModel,
})

describe("local model ranking", () => {
  it("features up to two configurations per base while preserving ranking order", () => {
    const scores = { intelligence: 1, speed: 1, fidelity: 1 }
    const model = (id: string) => option(id, 1, scores).model as CatalogLocalModel
    const best = model("first:gguf:q8")
    const sameBase = model("first:gguf:q4")
    const thirdQuant = model("first:gguf:q6")
    const second = model("second:gguf:q4")
    const third = model("third:gguf:q4")
    const fourth = model("fourth:gguf:q4")
    const ranked = [best, second, sameBase, thirdQuant, third, fourth]
    expect(featuredCatalogModels(ranked)).toEqual([best, second, sameBase, third, fourth])
    expect(featuredCatalogModels(ranked, 2)).toEqual([best, second])
    expect(featuredCatalogModels(ranked, 0)).toEqual([])
    expect(featuredCatalogModels([sameBase, thirdQuant, best, second])).toEqual([sameBase, thirdQuant, second])
    expect(featuredCatalogModels([best, sameBase, thirdQuant])).toEqual([best, sameBase])
  })

  it("moves utility from speed toward intelligence while fidelity always contributes", () => {
    const scores = { intelligence: 0.8, speed: Option.some(0.4), fidelity: 0.5 }
    expect(localModelRankingUtility(scores, 0)).toBeCloseTo(0.4 ** 0.9 * 0.5 ** 0.1)
    expect(localModelRankingUtility(scores, 1)).toBeCloseTo(0.8 ** 0.9 * 0.5 ** 0.1)
    const pending = { ...scores, speed: Option.none() }
    expect(localModelRankingUtility(pending, 0.5)).toBeCloseTo(0.8 ** 0.45 * 0.5 ** 0.1)
  })

  it("filters by the exact memory budget before sorting and truncating", () => {
    const fast = option("fast", 8, { intelligence: 0.4, speed: 1, fidelity: 1 })
    const smart = option("smart", 8, { intelligence: 1, speed: 0.4, fidelity: 1 })
    const overBudget = option("over", 9, { intelligence: 1, speed: 1, fidelity: 1 })
    expect(rankedLocalModelOptions(
      [smart, overBudget, fast],
      { fastToSmart: 0, memoryBudgetBytes: 8 },
      1,
    )).toEqual([fast])
  })

  it("breaks equal utility by canonical model ID", () => {
    const scores = { intelligence: 0.8, speed: 0.8, fidelity: 0.8 }
    const second = option("b", 1, scores)
    const first = option("a", 1, scores)
    expect(rankedLocalModelOptions(
      [second, first],
      { fastToSmart: 0.5, memoryBudgetBytes: 1 },
    )).toEqual([first, second])
  })

  it("orders fitting models without a speed estimate after every estimated model", () => {
    const estimated = option("z-estimated", 1, { intelligence: 0.1, speed: 0.1, fidelity: 0.1 })
    const pending = option("a-pending", 1, { intelligence: 1, speed: null, fidelity: 1 })
    expect(rankedLocalModelOptions(
      [pending, estimated],
      { fastToSmart: 0.5, memoryBudgetBytes: 1 },
    )).toEqual([estimated, pending])
  })

  it("ranks models without a speed estimate by intelligence and fidelity, not by name", () => {
    const smarter = option("z-smarter", 1, { intelligence: 0.9, speed: null, fidelity: 0.5 })
    const faithful = option("m-faithful", 1, { intelligence: 0.5, speed: null, fidelity: 1 })
    const weaker = option("a-weaker", 1, { intelligence: 0.5, speed: null, fidelity: 0.5 })
    expect(rankedLocalModelOptions(
      [weaker, faithful, smarter],
      { fastToSmart: 0.5, memoryBudgetBytes: 1 },
    )).toEqual([smarter, faithful, weaker])
  })

  it("ranks installed and downloadable choices together", () => {
    const stored = option("stored", 1, { intelligence: 1, speed: 1, fidelity: 1 }, "stored")
    const downloadable = option("downloadable", 1, { intelligence: 0.5, speed: 0.5, fidelity: 1 })

    expect(rankedLocalModelOptions(
      [downloadable, stored],
      { fastToSmart: 0.5, memoryBudgetBytes: 1 },
    )).toEqual([stored, downloadable])
  })

  it("never ranks a disabled or deprecated model", () => {
    const scores = { intelligence: 1, speed: 1, fidelity: 1 }
    const disabled = option("disabled", 1, scores, "downloadable",
      { _tag: "Disabled", reason: "not yet qualified" })
    const deprecated = option("deprecated", 1, scores, "stored", Schema.decodeUnknownSync(CatalogSupportSchema)({
      _tag: "Deprecated",
      since: "2026-09-27",
      replacement: "supported:gguf:q4",
      reason: "unsupported architecture",
    }))
    expect(rankedLocalModelOptions(
      [deprecated, disabled],
      { fastToSmart: 0.5, memoryBudgetBytes: 1 },
    )).toEqual([])
  })

  it("sums distinct normalized physical memory domains without adding system memory twice", () => {
    const hardware = {
      totalSystemMemoryBytes: 64,
      memoryDomains: [{ totalBytes: 64 }, { totalBytes: 24 }],
    } as unknown as LocalInferenceHardware
    expect(targetPhysicalMemoryBytes(hardware)).toBe(88)
  })
})
