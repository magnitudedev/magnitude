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
import { projectRecommendations } from "./recommendations"

const GB = 1_000_000_000

interface RungFixture {
  readonly id: string
  readonly storageBytes: number
  readonly tokensPerSecond?: number
  readonly fits?: boolean
  readonly assessing?: boolean
  readonly installed?: boolean
  readonly support?: "Supported" | "Disabled"
  readonly country?: string
  readonly intelligence?: number
}

const rung = (fixture: RungFixture): LocalModel => {
  const fits = fixture.fits ?? true
  const support = fixture.support ?? "Supported"
  return {
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
          assessment: {
            _tag: "Fits",
            served: { contextLength: 131_072 },
            residency: {
              _tag: "Resident",
              performance: fixture.tokensPerSecond === undefined
                ? { _tag: "Pending", measurement: { _tag: "Measuring" } }
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
      : { _tag: "Assessed", assessment: { _tag: "DoesNotFit" } },
  } as unknown as LocalModel
}

const source = (id: string, architecture: string): CatalogModelSource =>
  ({ id, source: { desired: { metadata: { architecture } } } }) as unknown as CatalogModelSource

const families: readonly LocalModelFamily[] = [
  { name: "Qwen", developer: { lab: "alibaba", name: "Alibaba", country: "CN" }, architectures: ["qwen35"], status: "Supported" },
  { name: "DeepSeek", developer: { lab: "deepseek", name: "DeepSeek", country: "CN" }, architectures: ["deepseek41"], status: "ComingSoon" },
] as unknown as readonly LocalModelFamily[]

const hardware = (freeBytes?: number): LocalInferenceHardware =>
  ({ modelStorage: Option.fromNullable(freeBytes === undefined ? undefined : { freeBytes }) }) as unknown as LocalInferenceHardware

const project = (models: readonly LocalModel[], options: { freeBytes?: number; country?: string } = {}) => {
  const recommendations = projectRecommendations({
    models,
    catalog: models.map((model) => source(model.modelId, model.modelId.startsWith("deepseek") ? "deepseek41" : "qwen35")),
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

  it("chooses the highest rung that fits, has room and reaches 20 tok/s, listing rungs smallest first", () => {
    const { models: [model] } = project([
      rung({ id: "big:gguf:q6", storageBytes: 30 * GB, tokensPerSecond: 15 }),
      rung({ id: "big:gguf:q2", storageBytes: 10 * GB, tokensPerSecond: 40 }),
      rung({ id: "big:gguf:q4", storageBytes: 20 * GB, tokensPerSecond: 25 }),
    ])
    expect(model?.rungs).toEqual(["big:gguf:q2", "big:gguf:q4", "big:gguf:q6"])
    expect(model?.chosen).toEqual(Option.some("big:gguf:q4"))
    expect([model?.meetsTarget, model?.meetsOnboardingFloor]).toEqual([true, true])
  })

  it("skips a rung without room on disk unless it is already downloaded", () => {
    const ladder = (installed: boolean) => [
      rung({ id: "big:gguf:q2", storageBytes: 10 * GB, tokensPerSecond: 40 }),
      rung({ id: "big:gguf:q4", storageBytes: 20 * GB, tokensPerSecond: 25, installed }),
    ]
    expect(project(ladder(false), { freeBytes: 15 * GB }).models[0]?.chosen).toEqual(Option.some("big:gguf:q2"))
    expect(project(ladder(true), { freeBytes: 15 * GB }).models[0]?.chosen).toEqual(Option.some("big:gguf:q4"))
  })

  it("falls back to the highest fitting rung near the fastest when none reaches 20 tok/s", () => {
    const { models: [model] } = project([
      rung({ id: "slow:gguf:q2", storageBytes: 10 * GB, tokensPerSecond: 12 }),
      rung({ id: "slow:gguf:q4", storageBytes: 20 * GB, tokensPerSecond: 8 }),
      rung({ id: "slow:gguf:q6", storageBytes: 30 * GB, fits: false }),
    ])
    expect(model?.chosen).toEqual(Option.some("slow:gguf:q2"))
    expect([model?.meetsTarget, model?.meetsOnboardingFloor]).toEqual([false, true])
    const { models: [alike] } = project([
      rung({ id: "alike:gguf:q2", storageBytes: 10 * GB, tokensPerSecond: 19 }),
      rung({ id: "alike:gguf:q4", storageBytes: 20 * GB, tokensPerSecond: 18 }),
    ])
    expect(alike?.chosen).toEqual(Option.some("alike:gguf:q4"))
  })

  it("orders fast models by intelligence, then slow ones, then those that do not fit", () => {
    const { models } = project([
      rung({ id: "fast-dim:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 50, intelligence: 20 }),
      rung({ id: "slow-smart:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 5, intelligence: 90 }),
      rung({ id: "fast-smart:gguf:q4", storageBytes: 1 * GB, tokensPerSecond: 30, intelligence: 60 }),
      rung({ id: "huge:gguf:q4", storageBytes: 1 * GB, fits: false, intelligence: 95 }),
    ])
    expect(models.map(({ base }) => base)).toEqual(["fast-smart", "fast-dim", "slow-smart", "huge"])
    expect(models.map(({ chosen }) => Option.isSome(chosen))).toEqual([true, true, true, false])
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
})
