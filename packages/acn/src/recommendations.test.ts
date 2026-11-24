import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  CountryCodeSchema,
  RecommendationsSchema,
  type LocalInferenceHardware,
  type LocalModel,
  type LocalModelFamily,
} from "@magnitudedev/acn-protocol"
import type { CatalogModelSource } from "./local-model-sources"
import { projectRecommendations, recommendationViews, sameRecommendationViews } from "./recommendations"

const GB = 1_000_000_000

interface RungFixture {
  readonly id: string
  readonly storageBytes: number
  readonly tokensPerSecond?: number
  readonly fits?: boolean
  readonly streamed?: boolean
  readonly assessing?: boolean
  readonly installed?: boolean
  readonly tools?: boolean
  readonly speedUnavailable?: boolean
  readonly support?: "Supported" | "Disabled"
  readonly country?: string
  readonly intelligence?: number
  readonly quantName?: string
  readonly quantRepository?: string
}

const defaultQuant = (id: string): { readonly name: string; readonly repository: string } => {
  const variant = id.split(":")[2] ?? ""
  if (variant.includes("nvfp4")) return { name: "NVFP4", repository: "magnitudedev/model" }
  if (variant.includes("mxfp4")) return { name: "MXFP4_MOE", repository: "magnitudedev/model" }
  if (variant.includes("iq2")) return { name: "IQ2_XXS", repository: "unsloth/model" }
  if (variant.includes("q2")) return { name: "UD-Q2_K_XL", repository: "unsloth/model" }
  if (variant.includes("iq3")) return { name: "UD-IQ3_XXS", repository: "unsloth/model" }
  if (variant.includes("q3")) return { name: "UD-Q3_K_XL", repository: "unsloth/model" }
  if (variant.includes("q4")) return { name: "UD-Q4_K_XL", repository: "unsloth/model" }
  if (variant.includes("q6")) return { name: "UD-Q6_K_XL", repository: "unsloth/model" }
  if (variant.includes("q8")) return { name: "UD-Q8_K_XL", repository: "unsloth/model" }
  return { name: "UD-Q4_K_XL", repository: "unsloth/model" }
}

interface FixtureModel {
  readonly model: LocalModel
  readonly quant: { readonly name: string; readonly repository: string }
}

const rung = (fixture: RungFixture): FixtureModel => {
  const fits = fixture.fits ?? true
  const support = fixture.support ?? "Supported"
  const fallbackQuant = defaultQuant(fixture.id)
  const quant = {
    name: fixture.quantName ?? fallbackQuant.name,
    repository: fixture.quantRepository ?? fallbackQuant.repository,
  }
  const model = {
    _tag: "Catalog",
    modelId: fixture.id,
    storageBytes: fixture.storageBytes,
    presentation: { displayName: fixture.id.split(":")[0], variantLabel: "Q4", description: "", license: Option.none(), sourceUrls: [] },
    catalogData: {
      developer: { lab: "lab", name: "Lab", country: fixture.country ?? "US" },
      support: { _tag: support, reason: "withheld" },
      intelligence: support === "Disabled"
        ? Option.fromNullable(fixture.intelligence)
        : fixture.intelligence ?? 50,
      drafter: Option.some({ method: { _tag: "Mtp" }, repository: "publisher/drafter", sizeBytes: 1 * GB }),
    },
    acquisitionState: fixture.installed ? { _tag: "Installed" } : { _tag: "NotInstalled" },
    drafterWithheld: false,
    servingState: fixture.assessing
      ? { _tag: "Assessing", profile: { contextLength: 131_072 } }
      : fits
      ? {
          _tag: "Assessed",
          capabilities: { vision: false, tools: fixture.tools ?? true, structuredOutput: true, reasoning: { supported: false, efforts: [], defaultEffort: Option.none() } },
          assessment: {
            _tag: "Fits",
            served: { contextLength: 131_072 },
            residency: {
              _tag: fixture.streamed ? "Streamed" : "Resident",
              performance: fixture.speedUnavailable
                ? { _tag: "Unavailable", failure: { code: "performance_unavailable", message: "cannot be measured", retryable: false } }
                : fixture.tokensPerSecond === undefined
                ? { _tag: "Pending", measurement: { _tag: "Measuring", progress: Option.none() } }
                : {
                    _tag: "Estimated",
                    samples: [
                      { contextTokens: 25_000, estimatedTokensPerSecond: fixture.tokensPerSecond * 2 },
                      { contextTokens: 50_000, estimatedTokensPerSecond: fixture.tokensPerSecond },
                    ],
                  },
            },
          },
        }
      : { _tag: "Assessed", capabilities: { vision: false, tools: fixture.tools ?? true, structuredOutput: true, reasoning: { supported: false, efforts: [], defaultEffort: Option.none() } }, assessment: { _tag: "DoesNotFit" } },
  } as unknown as LocalModel
  return { model, quant }
}

const source = (id: string, architecture: string, quant: { readonly name: string; readonly repository: string }): CatalogModelSource =>
  ({ id, source: { desired: { metadata: { architecture } }, quant } }) as unknown as CatalogModelSource

const families: readonly LocalModelFamily[] = [
  { name: "Qwen", developer: { lab: "alibaba", name: "Alibaba", country: "CN" }, architectures: ["qwen35"], status: "Supported" },
  { name: "DeepSeek", developer: { lab: "deepseek", name: "DeepSeek", country: "CN" }, architectures: ["deepseek41"], status: "ComingSoon" },
] as unknown as readonly LocalModelFamily[]

const hardware = (freeBytes?: number): LocalInferenceHardware =>
  ({ modelStorage: Option.fromNullable(freeBytes === undefined ? undefined : { freeBytes }) }) as unknown as LocalInferenceHardware

const project = (entries: readonly FixtureModel[], options: { freeBytes?: number; country?: string } = {}) => {
  const recommendations = projectRecommendations({
    models: entries.map(({ model }) => model),
    catalog: entries.map(({ model, quant }) => source(model.modelId, model.modelId.startsWith("deepseek") ? "deepseek41" : "qwen35", quant)),
    families,
    hardware: hardware(options.freeBytes),
    developerCountry: Option.map(Option.fromNullable(options.country), CountryCodeSchema.make),
  })
  expect(() => Schema.encodeSync(RecommendationsSchema)(recommendations)).not.toThrow()
  return recommendations
}

describe("recommendations", () => {
  it("is complete only once every rung of a listed model is measured, speed included", () => {
    const measured = rung({ id: "a:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 30 })
    expect(project([measured]).complete).toBe(true)
    expect(project([measured, rung({ id: "a:gguf:q8", storageBytes: 2 * GB, assessing: true })])).toMatchObject({ complete: false, measured: 1, rungs: 2 })
    expect(project([measured, rung({ id: "a:gguf:q8", storageBytes: 2 * GB })]).complete).toBe(false)
    expect(project([measured, rung({ id: "a:gguf:q8", storageBytes: 2 * GB, fits: false })]).complete).toBe(true)
  })

  it("prefers ISTA-DASLab 2-bit over larger quants even when NVFP4 and Q6 reach 10 tok/s, listing rungs smallest first", () => {
    const { models: [model] } = project([
      rung({ id: "flash:gguf:q6", storageBytes: 170 * GB, tokensPerSecond: 20, quantName: "UD-Q6_K_XL", quantRepository: "unsloth/Flash-GGUF" }),
      rung({ id: "flash:gguf:nvfp4", storageBytes: 129 * GB, tokensPerSecond: 30, quantName: "NVFP4", quantRepository: "magnitudedev/Flash-NVFP4-GGUF" }),
      rung({ id: "flash:gguf:iq3", storageBytes: 85 * GB, tokensPerSecond: 35, quantName: "IQ3_S", quantRepository: "ISTA-DASLab/Flash-GSQ-RCO-GGUF" }),
      rung({ id: "flash:gguf:q2", storageBytes: 67 * GB, tokensPerSecond: 45, quantName: "Q2_0", quantRepository: "ISTA-DASLab/Flash-GSQ-RCO-GGUF" }),
    ])
    expect(model?.rungs).toEqual(["flash:gguf:q2", "flash:gguf:iq3", "flash:gguf:nvfp4", "flash:gguf:q6"])
    expect(model?.chosen).toEqual(Option.some("flash:gguf:q2"))
    expect(model?.meetsTarget).toBe(true)
  })

  it("prefers native 4-bit (NVFP4/MXFP4) when there is no ISTA 2-bit, falling back to 2-bit when NVFP4 does not fit or is too slow", () => {
    const glm = (nvfp4: Partial<RungFixture>) => [
      rung({ id: "glm:gguf:iq2", storageBytes: 103 * GB, tokensPerSecond: 25, quantName: "UD-IQ2_XXS", quantRepository: "unsloth/GLM-GGUF" }),
      rung({ id: "glm:gguf:iq3", storageBytes: 122 * GB, tokensPerSecond: 20, quantName: "UD-IQ3_XXS", quantRepository: "unsloth/GLM-GGUF" }),
      rung({ id: "glm:gguf:nvfp4", storageBytes: 187 * GB, tokensPerSecond: 18, quantName: "NVFP4", quantRepository: "magnitudedev/GLM-GGUF", ...nvfp4 }),
    ]
    expect(project(glm({})).models[0]?.chosen).toEqual(Option.some("glm:gguf:nvfp4"))
    expect(project(glm({ fits: false })).models[0]?.chosen).toEqual(Option.some("glm:gguf:iq2"))
    expect(project(glm({ streamed: true, tokensPerSecond: 2 })).models[0]?.chosen).toEqual(Option.some("glm:gguf:iq2"))
  })

  it("picks 2-bit on Unsloth-only Q2–Q6 ladders and smallest (Q4) on Q4–Q8 ladders", () => {
    const { models: [moe] } = project([
      rung({ id: "moe:gguf:q2", storageBytes: 14 * GB, tokensPerSecond: 40 }),
      rung({ id: "moe:gguf:q4", storageBytes: 24 * GB, tokensPerSecond: 30 }),
      rung({ id: "moe:gguf:q6", storageBytes: 34 * GB, tokensPerSecond: 20 }),
    ])
    expect(moe?.chosen).toEqual(Option.some("moe:gguf:q2"))

    const { models: [dense] } = project([
      rung({ id: "dense:gguf:q4", storageBytes: 4 * GB, tokensPerSecond: 60 }),
      rung({ id: "dense:gguf:q6", storageBytes: 5 * GB, tokensPerSecond: 45 }),
      rung({ id: "dense:gguf:q8", storageBytes: 7 * GB, tokensPerSecond: 35 }),
    ])
    expect(dense?.chosen).toEqual(Option.some("dense:gguf:q4"))
  })

  it("skips a rung without room on disk unless it is already downloaded", () => {
    const ladder = (installed: boolean) => [
      rung({ id: "big:gguf:iq2", storageBytes: 10 * GB, tokensPerSecond: 40 }),
      rung({ id: "big:gguf:nvfp4", storageBytes: 20 * GB, tokensPerSecond: 25, installed }),
    ]
    expect(project(ladder(false), { freeBytes: 15 * GB }).models[0]?.chosen).toEqual(Option.some("big:gguf:iq2"))
    expect(project(ladder(true), { freeBytes: 15 * GB }).models[0]?.chosen).toEqual(Option.some("big:gguf:nvfp4"))
  })

  it("chooses no rung without room on disk, even when none reaches 10 tok/s, and none still being measured", () => {
    const slow = [
      rung({ id: "big:gguf:q2", storageBytes: 100 * GB, tokensPerSecond: 6 }),
      rung({ id: "big:gguf:nvfp4", storageBytes: 173 * GB, tokensPerSecond: 6 }),
    ]
    expect(project(slow, { freeBytes: 16 * GB }).models[0]?.chosen).toEqual(Option.none())
    const unmeasured = [rung({ id: "big:gguf:q2", storageBytes: 100 * GB }), rung({ id: "small:gguf:q4", storageBytes: 1 * GB })]
    expect(project(unmeasured, { freeBytes: 16 * GB }).models.map(({ base, chosen }) => [base, Option.getOrNull(chosen)]))
      .toEqual([["big", null], ["small", null]])
  })

  it("offers the best measured rung while another stays measuring, and never the one still measuring", () => {
    const stuck = [
      rung({ id: "qwen:gguf:iq2", storageBytes: 5 * GB, tokensPerSecond: 30 }),
      rung({ id: "qwen:gguf:nvfp4", storageBytes: 10 * GB }),
    ]
    const recommendations = project(stuck, { freeBytes: 100 * GB })
    expect(recommendations.complete).toBe(false)
    expect(recommendations.models[0]?.chosen).toEqual(Option.some("qwen:gguf:iq2"))
    expect(project([rung({ id: "lfm:gguf:q4", storageBytes: 1 * GB })]).models[0]?.chosen).toEqual(Option.none())
  })

  it("offers the preferred resident rung where no speed can be measured, such as on a CPU, without claiming a speed", () => {
    const cpu = [
      rung({ id: "lfm:gguf:q4", storageBytes: 1 * GB, speedUnavailable: true }),
      rung({ id: "lfm:gguf:q8", storageBytes: 2 * GB, speedUnavailable: true }),
    ]
    const [model] = project(cpu).models
    expect(model?.chosen).toEqual(Option.some("lfm:gguf:q4"))
    expect(model?.meetsTarget).toBe(false)
  })

  it("names the model that would lead with more free disk, and how much more it needs", () => {
    const models = [
      rung({ id: "big:gguf:iq2", storageBytes: 100 * GB, tokensPerSecond: 30, intelligence: 90 }),
      rung({ id: "big:gguf:nvfp4", storageBytes: 173 * GB, tokensPerSecond: 25, intelligence: 90 }),
      rung({ id: "small:gguf:q4", storageBytes: 2 * GB, tokensPerSecond: 60, intelligence: 40 }),
    ]
    const short = project(models, { freeBytes: 16 * GB })
    expect(short.models[0]?.base).toBe("small")
    expect(short.roomShortfall).toEqual(Option.some({ base: "big", name: "big", rung: "big:gguf:nvfp4", missingBytes: 157 * GB }))
    expect(project(models, { freeBytes: 200 * GB }).roomShortfall).toEqual(Option.none())
    expect(project(models).roomShortfall).toEqual(Option.none())
    const larger = project([rung({ id: "big:gguf:iq2", storageBytes: 10 * GB, tokensPerSecond: 40 }), ...models.slice(1, 2)], { freeBytes: 16 * GB })
    expect(larger.roomShortfall).toEqual(Option.none())
  })

  it("prefers resident rungs over slower streamed rungs when none reaches 10 tok/s, and smallest when all are streamed", () => {
    const { models: [residentVsStreamed] } = project([
      rung({ id: "slow:gguf:iq2", storageBytes: 103 * GB, tokensPerSecond: 7 }),
      rung({ id: "slow:gguf:nvfp4", storageBytes: 187 * GB, tokensPerSecond: 2, streamed: true }),
    ])
    expect(residentVsStreamed?.chosen).toEqual(Option.some("slow:gguf:iq2"))
    expect(residentVsStreamed?.meetsTarget).toBe(false)

    const { models: [allStreamed] } = project([
      rung({ id: "stream:gguf:iq2", storageBytes: 103 * GB, tokensPerSecond: 5, streamed: true }),
      rung({ id: "stream:gguf:nvfp4", storageBytes: 187 * GB, tokensPerSecond: 2, streamed: true }),
    ])
    expect(allStreamed?.chosen).toEqual(Option.some("stream:gguf:iq2"))
    expect(allStreamed?.meetsTarget).toBe(false)
  })

  it("orders target-speed models by intelligence, then usable ones, then slow ones, then those that do not fit", () => {
    const { models } = project([
      rung({ id: "fast-dim:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 50, intelligence: 20 }),
      rung({ id: "slow-smart:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 0.5, intelligence: 90 }),
      rung({ id: "usable:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 6, intelligence: 40 }),
      rung({ id: "fast-smart:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 14, intelligence: 60 }),
      rung({ id: "huge:gguf:q4", storageBytes: 1 * GB, fits: false, intelligence: 95 }),
    ])
    expect(models.map(({ base }) => base)).toEqual(["fast-smart", "fast-dim", "usable", "slow-smart", "huge"])
    expect(models.map(({ chosen }) => Option.isSome(chosen))).toEqual([true, true, true, true, false])
  })

  it("ranks usable models ahead of sub-1 tok/s models when none reaches the target speed", () => {
    const { models } = project([
      rung({ id: "crawling-27b:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 0.3, intelligence: 85 }),
      rung({ id: "usable-slow:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 3, intelligence: 55 }),
      rung({ id: "usable-target:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 12, intelligence: 40 }),
      rung({ id: "unmeasured:gguf:q4", storageBytes: 1 * GB, speedUnavailable: true, intelligence: 70 }),
      rung({ id: "too-big:gguf:q4", storageBytes: 1 * GB, fits: false, intelligence: 95 }),
    ])
    expect(models.map(({ base }) => base)).toEqual([
      "usable-target",
      "usable-slow",
      "crawling-27b",
      "unmeasured",
      "too-big",
    ])
  })

  it("leaves out disabled models except those of a family that is coming soon, which have no chosen rung", () => {
    const { models } = project([
      rung({ id: "qwen:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 50 }),
      rung({ id: "withheld:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 50, support: "Disabled" }),
      rung({ id: "deepseek:gguf:mxfp4", storageBytes: 1 * GB, tokensPerSecond: 50, support: "Disabled" }),
    ])
    expect(models.map(({ base, chosen }) => [base, Option.getOrNull(chosen)])).toEqual([
      ["qwen", "qwen:gguf:q4"],
      ["deepseek", null],
    ])
  })

  it("recommends only models whose developer is from the requested country", () => {
    const { models, families: listed } = project([
      rung({ id: "us:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 50, country: "US" }),
      rung({ id: "cn:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 50, country: "CN", intelligence: 90 }),
    ], { country: "US" })
    expect(models.map(({ base }) => base)).toEqual(["us"])
    expect(listed).toEqual(families)
  })

  it("never recommends a model that cannot use tools", () => {
    const { models } = project([
      rung({ id: "agentic:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 50 }),
      rung({ id: "chatonly:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 50, intelligence: 90, tools: false }),
    ])
    expect(models.map(({ base }) => base)).toEqual(["agentic"])
  })
})

describe("republishing recommendations", () => {
  const ladder = [
    rung({ id: "big:gguf:iq2", storageBytes: 10 * GB, tokensPerSecond: 40 }),
    rung({ id: "big:gguf:nvfp4", storageBytes: 20 * GB, tokensPerSecond: 25, country: "US" }),
  ]
  const views = (freeBytes: number) => recommendationViews({
    models: ladder.map(({ model }) => model),
    catalog: ladder.map(({ model, quant }) => source(model.modelId, "qwen35", quant)),
    families,
    hardware: hardware(freeBytes),
  })

  it("doesn't happen when free disk moves by a few KiB without changing any offer", () => {
    expect(sameRecommendationViews(views(109_362_847_744), views(109_362_839_552))).toBe(true)
  })

  it("happens when free disk changes which size has room to download", () => {
    expect(sameRecommendationViews(views(25 * GB), views(15 * GB))).toBe(false)
  })
})
