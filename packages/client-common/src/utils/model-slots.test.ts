import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  ModelCatalogReady,
  ModelSlotsReady,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  SlotPending,
  SlotUnassigned,
  VisionProperty,
  type ModelSummary,
} from "@magnitudedev/sdk"
import { selectedSlotModel } from "./model-slots"

const defaultEffort = ReasoningEffortSchema.make("high")

const model = (id: string): ModelSummary => ({
  providerId: ProviderIdSchema.make("llamacpp"),
  providerModelId: ProviderModelIdSchema.make(id),
  displayName: id,
  contextWindow: 32_768,
  maxOutputTokens: 8_192,
  defaultReasoningEffort: defaultEffort,
  properties: {
    reasoning: new ReasoningProperty.states.Resolved({ value: [defaultEffort] }),
    vision: new VisionProperty.states.Deferred({}),
  },
  availability: { _tag: "Available" },
})

const slots = (primary: SlotPending | SlotUnassigned) => new ModelSlotsReady({
  config: { slots: {}, localSlotIntent: {} },
  slots: {
    primary,
    secondary: new SlotUnassigned({ slotId: "secondary", reason: "no_candidate" }),
  },
})

describe("authoritative slot model", () => {
  it("uses the selected provider and model instead of catalog order", () => {
    const selected = model("selected")
    const view = selectedSlotModel(
      new ModelCatalogReady({ models: [model("first"), selected], providers: [] }),
      slots(new SlotPending({
        slotId: "primary",
        selection: {
          providerId: selected.providerId,
          providerModelId: selected.providerModelId,
          reasoningEffort: defaultEffort,
        },
        source: "automatic",
        waitingFor: ["reasoning"],
      })),
      "primary",
    )

    expect(Option.getOrThrow(view).model.providerModelId).toBe(selected.providerModelId)
  })

  it("returns none for an unassigned slot", () => {
    const view = selectedSlotModel(
      new ModelCatalogReady({ models: [model("first")], providers: [] }),
      slots(new SlotUnassigned({ slotId: "primary", reason: "no_candidate" })),
      "primary",
    )

    expect(Option.isNone(view)).toBe(true)
  })
})
