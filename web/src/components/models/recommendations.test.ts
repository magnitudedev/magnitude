import { describe, expect, it } from "vitest"
import { Option } from "effect"
import type { RecommendedModel } from "@magnitudedev/sdk"
import { recommendedForYourMachine, type OfferedRecommendation } from "./recommendations"
import type { OfferedModel } from "./expanding-model"

const offer = (base: string, options: { chosen?: boolean; floor?: boolean; speed?: number | null }): OfferedRecommendation => {
  const key = `${base}:gguf:q4`
  return {
    recommendation: { base, chosen: options.chosen === false ? Option.none() : Option.some(key), meetsOnboardingFloor: options.floor ?? false } as unknown as RecommendedModel,
    model: { key: base, recommended: options.chosen === false ? null : key, quants: [{ key, bits: 4, speed: options.speed ?? null }] } as unknown as OfferedModel,
  }
}

describe("Recommended for your machine", () => {
  const notes = (offered: readonly OfferedRecommendation[], owned: readonly string[] = []) =>
    recommendedForYourMachine(offered, new Set(owned)).map(entry => [entry.model.key, entry.note])

  it("offers only downloadable models fast enough to start with, at most three", () => {
    expect(notes([offer("a", { chosen: false, floor: true }), offer("b", { floor: true, speed: 20 }), offer("c", { floor: true, speed: 30 }),
      offer("d", { floor: true, speed: 10 }), offer("e", { floor: true, speed: 50 })]).map(([key]) => key)).toEqual(["b", "c", "d"])
  })

  it("calls the first the smartest that runs well and names the fastest", () => {
    expect(notes([offer("smart", { floor: true, speed: 20 }), offer("quick", { floor: true, speed: 60 }), offer("other", { floor: true, speed: 30 })]))
      .toEqual([["smart", "Smartest that runs well here"], ["quick", "Fastest"], ["other", undefined]])
  })

  it("says which quant is already here", () => {
    expect(notes([offer("lfm", { floor: true, speed: 40 })], ["lfm:gguf:q4"])).toEqual([["lfm", "You have the 4-bit quant"]])
  })
})
