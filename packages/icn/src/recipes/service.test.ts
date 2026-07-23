import { Option } from "effect"
import { describe, expect, it } from "vitest"
import type { GenerationPerformanceAssessmentSchema } from "../generated/schemas.js"
import {
  exactGenerationEstimate,
  exactRequestedProfileContext,
  recommendationCacheKey,
} from "./service.js"

describe("local model recommendation candidate evidence", () => {
  it("accepts only exact 100K and 200K product contexts", () => {
    expect(Option.getOrUndefined(exactRequestedProfileContext("model:p1:ctx100000", 100_000)))
      .toBe(100_000)
    expect(Option.getOrUndefined(exactRequestedProfileContext("model:p1:ctx200000", 200_000)))
      .toBe(200_000)
    expect(Option.isNone(exactRequestedProfileContext("model:p1:ctx100000", 99_999))).toBe(true)
    expect(Option.isNone(exactRequestedProfileContext("model:p1:ctx64000", 64_000))).toBe(true)
    expect(Option.isNone(exactRequestedProfileContext("model:p1:native", 100_000))).toBe(true)
  })

  it("joins generation evidence only at the exact selected context", () => {
    const performance: GenerationPerformanceAssessmentSchema = {
      status: "estimated",
      method: "test-estimator",
      confidence: "moderate",
      workload: "decode",
      always_active_weight_bytes: 1,
      routed_expert_weight_bytes: 0,
      expert_count: 0,
      expert_used_count: 0,
      cross_memory_domain_placement: false,
      points: [{
        context_tokens: 100_000,
        kv_bytes_read_per_token: 1,
        lower_tokens_per_second: 20,
        expected_tokens_per_second: 25,
        upper_tokens_per_second: 30,
      }],
    }
    expect(Option.getOrUndefined(exactGenerationEstimate(performance, 100_000)))
      .toMatchObject({ contextTokens: 100_000, expectedTokensPerSecond: 25 })
    expect(Option.isNone(exactGenerationEstimate(performance, 200_000))).toBe(true)
  })

  it("does not fabricate generation evidence when estimation is unavailable", () => {
    const performance: GenerationPerformanceAssessmentSchema = {
      status: "unavailable",
      method: "test-estimator",
      code: "unavailable",
      message: "No estimate",
    }
    expect(Option.isNone(exactGenerationEstimate(performance, 100_000))).toBe(true)
  })

  it("invalidates recommendation cache identity when policy changes", () => {
    expect(recommendationCacheKey("policy-v1", "native", "topology", ["commit"]))
      .not.toBe(recommendationCacheKey("policy-v2", "native", "topology", ["commit"]))
  })
})
