import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { RecommendedModel } from "@magnitudedev/sdk"
import { onboardingChoices, recommendationsSettled, recommendedForYourMachine, type OfferedRecommendation, type RecommendationsView } from "./recommendations"
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

describe("when Recommended for your machine shows its list", () => {
  const view = (benchmark: RecommendationsView["benchmark"], ready = true): RecommendationsView => ({ ready, benchmark, offered: [], roomShortfall: null, families: [] })

  it("shows no list while benchmarking or ranking, nor before the recommendations are read", () => {
    expect(recommendationsSettled(view({ title: "Benchmarking your hardware", subtitle: "Testing baseline compute", percent: 12, fraction: 0.12, failed: false }))).toBe(false)
    expect(recommendationsSettled(view(null, false))).toBe(false)
    expect(recommendationsSettled(view(null))).toBe(true)
  })

  it("shows the three smartest that run well, smartest first, once measuring is done", () => {
    const settled = [offer("flash", { floor: true, speed: 78 }), offer("qwen27", { floor: true, speed: 26 }), offer("lfm25", { floor: true, speed: 110 }), offer("lfm3", { floor: true, speed: 160 })]
    expect(recommendedForYourMachine(settled, new Set()).map(entry => [entry.model.key, entry.note]))
      .toEqual([["flash", "Smartest that runs well here"], ["qwen27", undefined], ["lfm25", "Fastest"]])
  })

  it("lists the same models again when the US filter is turned on and off", () => {
    const all = [offer("qwen4", { floor: true, speed: 70 }), offer("lfm25", { floor: true, speed: 110 }), offer("lfm3", { floor: true, speed: 160 })]
    const before = recommendedForYourMachine(all, new Set())
    expect(recommendedForYourMachine(all.filter(entry => entry.model.key !== "qwen4"), new Set()).map(entry => entry.model.key)).toEqual(["lfm25", "lfm3"])
    expect(recommendedForYourMachine(all, new Set())).toEqual(before)
  })
})
