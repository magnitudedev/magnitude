import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { CatalogFormModelIdSchema, CustomFormModelIdSchema, HuggingFaceRepositoryIdSchema } from "@magnitudedev/sdk"
import type { LocalInferenceHardware } from "@magnitudedev/sdk"
import { localModelConnectionOptions, targetPhysicalMemoryBytes } from "./options"
import { makeFoundModel, makeSetupModel } from "../application/fixtures/model"

describe("connection model choices", () => {
  it("includes all five available catalog, Hugging Face and found models", () => {
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
      _tag: "DoesNotFit", profile: { contextLength: 32_768 }, shortfall: { contextLength: 32_768 }, memoryDomains: [],
      assessmentId: assessment.assessmentId, environmentId: assessment.environmentId,
      totalRequiredBytes: 0, limitingResource: "memory", deficitBytes: 1,
    } } })
    expect(localModelConnectionOptions([makeSetupModel(false), unavailable, assessing, unsupported, tooLarge])).toEqual([])
    expect(localModelConnectionOptions([found])).toEqual([{ id: found.modelId, label: "Setup Model Uncensored (Q4_K_P)" }])
  })
})

describe("local model memory", () => {
  it("sums distinct normalized physical memory domains without adding system memory twice", () => {
    const hardware = {
      totalSystemMemoryBytes: 64,
      memoryDomains: [{ totalBytes: 64 }, { totalBytes: 24 }],
    } as unknown as LocalInferenceHardware
    expect(targetPhysicalMemoryBytes(hardware)).toBe(88)
  })
})
