import { describe, expect, it } from "vitest"
import {
  LLAMA_CPP_NATIVE_REASONING_EFFORT_CANDIDATES,
  LLAMA_CPP_REASONING_POLICY,
  LlamaCppNativeReasoningEffortSchema,
  llamaCppDisabledReasoningDefinition,
  llamaCppPreferredReasoningDefinition,
  llamaCppReasoningDefinitionForNativeValue,
} from "./reasoning-policy"

describe("llama.cpp reasoning policy", () => {
  it("owns every native value exactly once", () => {
    expect(new Set(LLAMA_CPP_NATIVE_REASONING_EFFORT_CANDIDATES).size)
      .toBe(LLAMA_CPP_NATIVE_REASONING_EFFORT_CANDIDATES.length)
    expect(new Set(LLAMA_CPP_REASONING_POLICY.map(({ reasoningEffort }) => reasoningEffort)).size)
      .toBe(LLAMA_CPP_REASONING_POLICY.length)
  })

  it("resolves native aliases through their single policy definition", () => {
    const aliases = ["extra_high", "extra-high", "xhigh", "very_high"]
      .map((alias) => LlamaCppNativeReasoningEffortSchema.make(alias))
    expect(aliases.map((alias) => llamaCppReasoningDefinitionForNativeValue(alias).reasoningEffort))
      .toEqual(["extra_high", "extra_high", "extra_high", "extra_high"])
  })

  it("identifies disabled and preferred efforts from their semantics", () => {
    expect(llamaCppDisabledReasoningDefinition().semantics).toEqual({ _tag: "Disabled" })
    expect(llamaCppPreferredReasoningDefinition().semantics).toEqual({ _tag: "Budgeted", tokens: 4_096 })
  })
})
