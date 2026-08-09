import { LocalModelMutationFailed } from "@magnitudedev/acn-protocol"
import { Option } from "effect"
import { describe, expect, it } from "vitest"

import {
  AssessmentEnvironmentIdSchema,
  ModelAssessmentIdSchema,
  ModelOfferingTargetIdSchema,
  ModelServingConfigurationIdSchema,
} from "@magnitudedev/acn-protocol"
import {
  exactTargetTensorStorageBytes,
  localModelRecommendationFailure,
  mergeAssessmentResults,
  needsFallbackAssessment,
} from "./local-model-recommendations"
import type {
  LocalModelAssessment,
  LocalModelAssessmentResult,
} from "./local-model-assessments"

const assessed = (
  assessments: readonly LocalModelAssessment[],
): LocalModelAssessmentResult => ({
  _tag: "Assessed",
  targetId: ModelOfferingTargetIdSchema.make("target"),
  environmentId: AssessmentEnvironmentIdSchema.make("environment"),
  assessments,
})

const doesNotFit = (contextLength: number): LocalModelAssessment => ({
  _tag: "DoesNotFit",
  profile: { contextLength },
  configurationId: ModelServingConfigurationIdSchema.make(`cfg-${contextLength}`),
  assessmentId: ModelAssessmentIdSchema.make(`assessment-${contextLength}`),
  memory: [],
  deficitBytes: 1,
  limitingResource: "device",
})

describe("needsFallbackAssessment", () => {
  it("requests fallbacks only for preferred memory DoesNotFit without Fits", () => {
    expect(needsFallbackAssessment(
      assessed([doesNotFit(100_000)]),
      100_000,
    )).toBe(true)
    expect(needsFallbackAssessment(
      assessed([{
        _tag: "Fits",
        assessment: {
          _tag: "Fits",
          profile: { contextLength: 100_000 },
          configurationId: ModelServingConfigurationIdSchema.make("cfg"),
          assessmentId: ModelAssessmentIdSchema.make("assessment"),
          environmentId: AssessmentEnvironmentIdSchema.make("environment"),
          memory: [],
          performance: [],
        },
      }]),
      100_000,
    )).toBe(false)
    expect(needsFallbackAssessment(
      assessed([doesNotFit(50_000)]),
      100_000,
    )).toBe(false)
  })
})

describe("mergeAssessmentResults", () => {
  it("appends fallback assessments onto the preferred Assessed result", () => {
    const preferred = assessed([doesNotFit(100_000)])
    const fallback = assessed([doesNotFit(75_000), doesNotFit(50_000)])
    expect(mergeAssessmentResults(preferred, fallback)).toMatchObject({
      _tag: "Assessed",
      assessments: [
        { _tag: "DoesNotFit", profile: { contextLength: 100_000 } },
        { _tag: "DoesNotFit", profile: { contextLength: 75_000 } },
        { _tag: "DoesNotFit", profile: { contextLength: 50_000 } },
      ],
    })
  })
})

describe("localModelRecommendationFailure", () => {
  it("preserves typed assessment failure metadata for the public lifecycle", () => {
    expect(localModelRecommendationFailure(new LocalModelMutationFailed({
      code: "planner_timeout",
      message: "Hardware assessment took longer than five minutes.",
      retryable: true,
    }))).toEqual({
      code: "planner_timeout",
      message: "Hardware assessment took longer than five minutes.",
      retryable: true,
    })
  })
})

describe("exactTargetTensorStorageBytes", () => {
  const model = (files: readonly unknown[]) => ({
    target: { _tag: "Package", package: { files } },
  }) as Parameters<typeof exactTargetTensorStorageBytes>[0]

  it("sums exact tensor storage and deduplicates immutable content", () => {
    expect(exactTargetTensorStorageBytes(model([
      { role: "weights", sha256: "a", tensorStorageBytes: Option.some(10) },
      { role: "weights", sha256: "a", tensorStorageBytes: Option.some(10) },
      { role: "weights", sha256: "b", tensorStorageBytes: Option.some(15) },
      { role: "projector", sha256: "c", tensorStorageBytes: Option.some(100) },
    ]))).toEqual(Option.some(25))
  })

  it("declines to reject when any required component is unknown", () => {
    expect(exactTargetTensorStorageBytes(model([
      { role: "weights", sha256: "a", tensorStorageBytes: Option.some(10) },
      { role: "weights", sha256: "b", tensorStorageBytes: Option.none() },
    ]))).toEqual(Option.none())
  })
})
