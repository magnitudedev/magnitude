import { describe, expect, it } from "vitest"
import { Schema } from "effect"
import {
  LocalInferenceRecommendationState,
  LocalModelFitAssessmentSchema,
} from "./local-inference"

describe("LocalModelFitAssessmentSchema", () => {
  it("preserves the authoritative ICN fit result and per-domain requirements", () => {
    const assessment = Schema.decodeUnknownSync(LocalModelFitAssessmentSchema)({
      _tag: "Assessed",
      requiredTotalBytes: 24,
      domains: [{
        memoryDomainId: "unified",
        requiredBytes: 24,
        stableCapacityBytes: 20,
        marginBytes: -4,
      }],
      result: "does_not_fit",
    })

    expect(assessment).toEqual({
      _tag: "Assessed",
      requiredTotalBytes: 24,
      domains: [{
        memoryDomainId: "unified",
        requiredBytes: 24,
        stableCapacityBytes: 20,
        marginBytes: -4,
      }],
      result: "does_not_fit",
    })
  })

  it("rejects non-ICN assessment variants", () => {
    expect(() => Schema.decodeUnknownSync(LocalModelFitAssessmentSchema)({
      _tag: "Unknown",
      domains: [],
      result: "unknown",
    })).toThrow()
  })

  it("distinguishes recommendation loading from a completed empty result", () => {
    expect(Schema.decodeUnknownSync(LocalInferenceRecommendationState)({
      _tag: "Loading",
    })).toEqual({ _tag: "Loading" })
    expect(Schema.decodeUnknownSync(LocalInferenceRecommendationState)({
      _tag: "Ready",
      recommendations: [],
    })).toEqual({ _tag: "Ready", recommendations: [] })
  })
})
