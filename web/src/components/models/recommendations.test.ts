import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { RecommendedModel } from "@magnitudedev/sdk"
import { onboardingChoices, recommendedForYourMachine, stillMeasuring, type OfferedRecommendation } from "./recommendations"
import type { OfferedModel } from "./expanding-model"

const offer = (base: string, options: { chosen?: boolean; floor?: boolean; speed?: number | null; measuring?: boolean }): OfferedRecommendation => {
  const key = `${base}:gguf:q4`
  return {
    recommendation: { base, chosen: options.chosen === false ? Option.none() : Option.some(key), meetsOnboardingFloor: options.floor ?? false } as unknown as RecommendedModel,
    model: { key: base, recommended: options.chosen === false ? null : key, quants: [{ key, bits: 4, speed: options.speed ?? null, measuring: options.measuring ?? false }] } as unknown as OfferedModel,
  }
}

describe("onboarding's choices", () => {
  it("offers the models fast enough to start with, while others are still being measured", () => {
    const { choices, speedUnknown } = onboardingChoices([offer("lfm", { floor: true, speed: 40 }), offer("qwen", { chosen: false })])
    expect(choices.map(entry => entry.model.key)).toEqual(["lfm"])
    expect(speedUnknown).toBe(false)
  })

  it("offers models that fit with their speed unknown where no speed can be measured, such as on a CPU", () => {
    const { choices, speedUnknown } = onboardingChoices([offer("lfm3", { speed: null }), offer("lfm25", { speed: null })])
    expect(choices.map(entry => entry.model.key)).toEqual(["lfm3", "lfm25"])
    expect(speedUnknown).toBe(true)
  })

  it("never offers a model measured as too slow, or one still being measured", () => {
    expect(onboardingChoices([offer("slow", { speed: 3 }), offer("pending", { speed: null, measuring: true })]).choices).toEqual([])
  })
})

describe("Recommended for your machine", () => {
  const notes = (offered: readonly OfferedRecommendation[], owned: readonly string[] = []) =>
    recommendedForYourMachine(offered, new Set(owned)).map(entry => [entry.model.key, entry.note])

  it("offers only downloadable models fast enough to start with, at most three", () => {
    expect(notes([offer("a", { chosen: false }), offer("b", { floor: true, speed: 20 }), offer("c", { floor: true, speed: 30 }),
      offer("d", { floor: true, speed: 10 }), offer("e", { floor: true, speed: 50 })]).map(([key]) => key)).toEqual(["b", "c", "d"])
  })

  it("calls the first the smartest that runs well and names the fastest", () => {
    expect(notes([offer("smart", { floor: true, speed: 20 }), offer("quick", { floor: true, speed: 60 }), offer("other", { floor: true, speed: 30 })]))
      .toEqual([["smart", "Smartest that runs well here"], ["quick", "Fastest"], ["other", undefined]])
  })

  it("says which quant is already here, and that the speed is unknown where it can't be measured", () => {
    expect(notes([offer("lfm", { floor: true, speed: 40 })], ["lfm:gguf:q4"])).toEqual([["lfm", "You have the 4-bit quant"]])
    expect(notes([offer("cpu", { speed: null }), offer("cpu2", { speed: null })]))
      .toEqual([["cpu", "Fits this computer; its speed can’t be measured here"], ["cpu2", undefined]])
  })
})

describe("the line about quants still being measured", () => {
  it("says how many are left, and nothing once all are settled", () => {
    expect(stillMeasuring({ settled: 29, total: 36 })).toContain("7 more quants")
    expect(stillMeasuring({ settled: 35, total: 36 })).toContain("1 more quant runs")
    expect(stillMeasuring(null)).toBeNull()
  })
})
