import { describe, expect, it } from "vitest"
import {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  VisionProperty,
  type ModelSummary,
} from "@magnitudedev/sdk"
import { reasoningEffortOptions, reasoningPropertyLabel, visionPropertyLabel } from "./model-properties"

const defaultEffort = ReasoningEffortSchema.make("Default")
const lowEffort = ReasoningEffortSchema.make("Low")

const model = (
  reasoning: ModelSummary["properties"]["reasoning"],
  vision: ModelSummary["properties"]["vision"] = new VisionProperty.states.Deferred({}),
): ModelSummary => ({
  providerId: ProviderIdSchema.make("llamacpp"),
  providerModelId: ProviderModelIdSchema.make("/models/test.gguf"),
  displayName: "Test model",
  contextWindow: 32_768,
  maxOutputTokens: 8_192,
  defaultReasoningEffort: defaultEffort,
  properties: { reasoning, vision },
  availability: { _tag: "Available" },
})

describe("model property presentation", () => {
  it("shows only the provider default before reasoning discovery", () => {
    const value = model(new ReasoningProperty.states.Deferred({}))

    expect(reasoningEffortOptions(value)).toEqual([
      { value: defaultEffort, label: "Default", isDefault: true },
    ])
    expect(reasoningPropertyLabel(value)).toContain("after loading")
  })

  it("renders the exact cached effort list without a global catalog", () => {
    const value = model(new ReasoningProperty.states.Cached({ value: [defaultEffort, lowEffort] }))

    expect(reasoningEffortOptions(value).map((option) => option.value)).toEqual([defaultEffort, lowEffort])
    expect(reasoningPropertyLabel(value)).toContain("cached")
  })

  it("renders resolved vision directly", () => {
    const value = model(
      new ReasoningProperty.states.Resolved({ value: [defaultEffort] }),
      new VisionProperty.states.Resolved({ value: true }),
    )

    expect(visionPropertyLabel(value)).toBe("Vision supported")
  })
})
