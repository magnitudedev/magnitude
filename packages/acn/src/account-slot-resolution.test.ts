import { describe, expect, it } from "vitest"
import {
  AVAILABLE_PROVIDER_MODEL,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  VisionProperty,
  type ProviderModel,
} from "@magnitudedev/sdk"
import { slotStatesFromModels } from "./account"

const providerId = ProviderIdSchema.make("llamacpp")
const providerModelId = ProviderModelIdSchema.make("/models/test.gguf")
const defaultEffort = ReasoningEffortSchema.make("high")
const highEffort = ReasoningEffortSchema.make("max")

const modelWithReasoning = (reasoning: ProviderModel["properties"]["reasoning"]): ProviderModel => ({
  providerId,
  providerModelId,
  displayName: "Test model",
  contextWindow: 32_768,
  maxOutputTokens: 8_192,
  defaultReasoningEffort: defaultEffort,
  properties: {
    vision: new VisionProperty.states.Deferred({}),
    reasoning,
  },
  availability: AVAILABLE_PROVIDER_MODEL,
  pricing: { input: 0, output: 0, cached_input: null },
})

const selectedHigh = {
  slots: {
    primary: { providerId, providerModelId, reasoningEffort: highEffort },
  },
}

describe("slot reasoning-property resolution", () => {
  it("keeps a persisted non-default effort pending while discovery is deferred", () => {
    const slots = slotStatesFromModels(
      [modelWithReasoning(new ReasoningProperty.states.Deferred({}))],
      selectedHigh,
    )

    expect(slots.primary).toMatchObject({
      _tag: "Pending",
      selection: { reasoningEffort: highEffort },
      waitingFor: ["reasoning"],
    })
  })

  it("makes the slot ready when the cached effort list contains the selection", () => {
    const slots = slotStatesFromModels(
      [modelWithReasoning(new ReasoningProperty.states.Cached({ value: [defaultEffort, highEffort] }))],
      selectedHigh,
    )

    expect(slots.primary).toMatchObject({
      _tag: "Ready",
      selection: { reasoningEffort: highEffort },
    })
  })

  it("blocks an unresolved non-default selection after discovery fails", () => {
    const slots = slotStatesFromModels(
      [modelWithReasoning(new ReasoningProperty.states.Failed({
        error: { code: "discovery_failed", message: "template unavailable", retryable: true },
      }))],
      selectedHigh,
    )

    expect(slots.primary).toMatchObject({
      _tag: "Blocked",
      selection: { reasoningEffort: highEffort },
      reason: "property_discovery_failed",
    })
  })

  it("falls back to the provider default after resolved options remove an effort", () => {
    const slots = slotStatesFromModels(
      [modelWithReasoning(new ReasoningProperty.states.Resolved({ value: [defaultEffort] }))],
      selectedHigh,
    )

    expect(slots.primary).toMatchObject({
      _tag: "Ready",
      selection: { reasoningEffort: defaultEffort },
    })
  })
})
