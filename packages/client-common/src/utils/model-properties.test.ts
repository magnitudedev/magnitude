import { describe, expect, it } from "vitest"
import {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ModelDiscoveryOperationIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  VisionProperty,
  type ModelSummary,
} from "@magnitudedev/sdk"
import { reasoningEffortControl, reasoningPropertyLabel, visionPropertyLabel } from "./model-properties"

const defaultEffort = ReasoningEffortSchema.make("high")
const lowEffort = ReasoningEffortSchema.make("low")

const model = (
  reasoning: ModelSummary["properties"]["reasoning"],
  vision: ModelSummary["properties"]["vision"] = new VisionProperty.states.Deferred({}),
): ModelSummary => ({
  providerId: ProviderIdSchema.make("local"),
  providerModelId: ProviderModelIdSchema.make("/models/test.gguf"),
  displayName: "Test model",
  contextWindow: 32_768,
  maxOutputTokens: 8_192,
  defaultReasoningEffort: defaultEffort,
  properties: { reasoning, vision },
  availability: { _tag: "Available" },
})

describe("model property presentation", () => {
  it("does not expose the provider fallback before reasoning discovery", () => {
    const value = model(new ReasoningProperty.states.Deferred({}))

    expect(reasoningEffortControl(value)).toEqual({ _tag: "Unavailable", label: "Load to inspect" })
    expect(reasoningPropertyLabel(value)).toContain("after loading")
  })

  it("renders exactly the cached effort list without inserting the default", () => {
    const value = model(new ReasoningProperty.states.Cached({ value: [defaultEffort, lowEffort] }))

    expect(reasoningEffortControl(value)).toEqual({
      _tag: "Available",
      options: [
        { value: defaultEffort, label: "High" },
        { value: lowEffort, label: "Low" },
      ],
    })
    expect(reasoningPropertyLabel(value)).toContain("cached")
  })

  it("represents active inspection without a selectable fallback", () => {
    const value = model(new ReasoningProperty.states.Discovering({ operationId: ModelDiscoveryOperationIdSchema.make("inspection"), phase: "inspecting" }))
    expect(reasoningEffortControl(value)).toEqual({ _tag: "Unavailable", label: "Inspecting…" })
  })

  it("renders resolved vision directly", () => {
    const value = model(
      new ReasoningProperty.states.Resolved({ value: [defaultEffort] }),
      new VisionProperty.states.Resolved({ value: true }),
    )

    expect(visionPropertyLabel(value)).toBe("Vision supported")
  })
})
