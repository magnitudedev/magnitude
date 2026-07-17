import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import { ReasoningEffortSchema } from "@magnitudedev/ai"
import {
  LlamaCppReasoningProfileSchema,
  resolveLlamaCppReasoningEffort,
} from "./model-properties"

const mapping = (reasoningEffort: string) => ({
  reasoningEffort: ReasoningEffortSchema.make(reasoningEffort),
  templateOptions: {
    enableThinking: Option.none<boolean>(),
    reasoningEffort: Option.none<string>(),
  },
  thinkingBudget: { _tag: "Disabled" as const },
})

describe("LlamaCppReasoningProfileSchema", () => {
  it("accepts a profile and resolves its exact effort mapping", () => {
    const profile = {
      defaultReasoningEffort: ReasoningEffortSchema.make("high"),
      effortMappings: [mapping("none"), mapping("high")],
    }

    expect(Schema.is(LlamaCppReasoningProfileSchema)(profile)).toBe(true)
    expect(Option.getOrUndefined(resolveLlamaCppReasoningEffort(profile, ReasoningEffortSchema.make("high"))))
      .toEqual(mapping("high"))
    expect(Option.isNone(resolveLlamaCppReasoningEffort(profile, ReasoningEffortSchema.make("medium"))))
      .toBe(true)
  })

  it("rejects duplicate efforts and a missing default effort", () => {
    expect(Schema.is(LlamaCppReasoningProfileSchema)({
      defaultReasoningEffort: ReasoningEffortSchema.make("high"),
      effortMappings: [mapping("high"), mapping("high")],
    })).toBe(false)
    expect(Schema.is(LlamaCppReasoningProfileSchema)({
      defaultReasoningEffort: ReasoningEffortSchema.make("high"),
      effortMappings: [mapping("none")],
    })).toBe(false)
  })
})
