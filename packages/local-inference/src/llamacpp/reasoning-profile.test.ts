import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { buildLlamaCppReasoningProfile } from "./reasoning-profile"
import {
  LlamaCppNativeReasoningEffortSchema,
  llamaCppReasoningDefinitionForNativeValue,
} from "./reasoning-policy"

const definition = (nativeValue: string) => llamaCppReasoningDefinitionForNativeValue(
  LlamaCppNativeReasoningEffortSchema.make(nativeValue),
)

describe("llama.cpp reasoning profile construction", () => {
  it("uses the policy toggle fallbacks when only the boolean control is detected", () => {
    expect(buildLlamaCppReasoningProfile({
      enableThinkingToggle: true,
      symbolicEfforts: [],
    })).toEqual({
      defaultReasoningEffort: "high",
      effortMappings: [
        {
          reasoningEffort: "none",
          templateOptions: { enableThinking: Option.some(false), reasoningEffort: Option.none() },
          thinkingBudget: { _tag: "Disabled" },
        },
        {
          reasoningEffort: "high",
          templateOptions: { enableThinking: Option.some(true), reasoningEffort: Option.none() },
          thinkingBudget: { _tag: "Enabled", tokens: 4_096 },
        },
      ],
    })
  })

  it("preserves verified symbolic options and derives the budget from policy", () => {
    const low = definition("low")
    expect(buildLlamaCppReasoningProfile({
      enableThinkingToggle: false,
      symbolicEfforts: [{
        reasoningEffort: low.reasoningEffort,
        templateOptions: { enableThinking: Option.none(), reasoningEffort: Option.some("low") },
        baselineEquivalent: false,
      }],
    })).toEqual({
      defaultReasoningEffort: "low",
      effortMappings: [{
        reasoningEffort: "low",
        templateOptions: { enableThinking: Option.none(), reasoningEffort: Option.some("low") },
        thinkingBudget: { _tag: "Enabled", tokens: 1_024 },
      }],
    })
  })

  it("adds the disabled toggle fallback without interpreting effort names", () => {
    const max = definition("max")
    expect(buildLlamaCppReasoningProfile({
      enableThinkingToggle: true,
      symbolicEfforts: [{
        reasoningEffort: max.reasoningEffort,
        templateOptions: { enableThinking: Option.some(true), reasoningEffort: Option.some("max") },
        baselineEquivalent: false,
      }],
    })).toEqual({
      defaultReasoningEffort: "none",
      effortMappings: [
        {
          reasoningEffort: "none",
          templateOptions: { enableThinking: Option.some(false), reasoningEffort: Option.none() },
          thinkingBudget: { _tag: "Disabled" },
        },
        {
          reasoningEffort: "max",
          templateOptions: { enableThinking: Option.some(true), reasoningEffort: Option.some("max") },
          thinkingBudget: { _tag: "Disabled" },
        },
      ],
    })
  })

  it("uses disabled semantics for a native disabling alias", () => {
    const disabled = definition("off")
    expect(buildLlamaCppReasoningProfile({
      enableThinkingToggle: true,
      symbolicEfforts: [{
        reasoningEffort: disabled.reasoningEffort,
        templateOptions: { enableThinking: Option.some(false), reasoningEffort: Option.some("off") },
        baselineEquivalent: false,
      }],
    }).effortMappings).toEqual([{
      reasoningEffort: "none",
      templateOptions: { enableThinking: Option.some(false), reasoningEffort: Option.some("off") },
      thinkingBudget: { _tag: "Disabled" },
    }])
  })
})
