import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelSlotUnassigned,
  ModelSlotConfiguredLocal,
  ModelInstanceIdSchema,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
  type ModelSlotsState,
} from "@magnitudedev/sdk"
import {
  presentedSlotSelection,
  modelLoadIsVisible,
  selectedModelStopIsVisible,
  slotAssignmentIsVisible,
} from "./service"

const authoritative: ModelSlotsState = {
  slots: {
    primary: new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID }),
    secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
  },
  recentModels: { primary: [], secondary: [] },
  favoriteModels: [],
}

const assignment = (providerModelId: string, pending: boolean) => ({
  slotId: PRIMARY_SLOT_ID,
  selection: {
    providerId: ProviderIdSchema.make("test"),
    providerModelId: ProviderModelIdSchema.make(providerModelId),
    reasoningEffort: ReasoningEffortSchema.make("none"),
  },
  pending,
})

describe("model-slot service presentation", () => {
  it("projects the latest pending assignment over authoritative state", () => {
    const presented = presentedSlotSelection(
      authoritative,
      [assignment("first", true), assignment("second", true)],
      PRIMARY_SLOT_ID,
    )

    expect(Option.getOrThrow(presented).providerModelId).toBe("second")
  })

  it("falls back to authoritative state when the assignment is no longer pending", () => {
    const presented = presentedSlotSelection(
      authoritative,
      [assignment("failed", false)],
      PRIMARY_SLOT_ID,
    )

    expect(Option.isNone(presented)).toBe(true)
  })
})

describe("model-slot mutation synchronization", () => {
  const selection = assignment("selected", true).selection
  const configured: ModelSlotsState = {
    ...authoritative,
    slots: {
      ...authoritative.slots,
      primary: new ModelSlotConfiguredLocal({
        slotId: PRIMARY_SLOT_ID,
        selection,
        descriptor: {
          providerId: selection.providerId,
          providerModelId: selection.providerModelId,
          displayName: "Selected",
          variantLabel: Option.none(),
        },
        availability: { _tag: "Available" },
        residency: {
          _tag: "Loading",
          instanceId: ModelInstanceIdSchema.make("instance"),
          stage: "queued",
          progress: Option.none(),
          plannedAllocation: Option.none(),
        },
        actions: ["Stop"],
      }),
    },
  }

  it("requires the exact assigned selection", () => {
    expect(slotAssignmentIsVisible(configured, PRIMARY_SLOT_ID, selection)).toBe(true)
    expect(slotAssignmentIsVisible(
      configured,
      PRIMARY_SLOT_ID,
      assignment("replacement", true).selection,
    )).toBe(false)
  })

  it("synchronizes selected-model control against request and instance state", () => {
    expect(modelLoadIsVisible(configured, PRIMARY_SLOT_ID)).toBe(true)
    expect(selectedModelStopIsVisible(configured, PRIMARY_SLOT_ID)).toBe(false)
    const primary = configured.slots.primary
    if (primary._tag !== "ConfiguredLocal") throw new Error("expected local slot")

    const waiting: ModelSlotsState = {
      ...configured,
      slots: {
        ...configured.slots,
        primary: new ModelSlotConfiguredLocal({
          ...primary,
          residency: { _tag: "Requested" },
        }),
      },
    }
    expect(modelLoadIsVisible(waiting, PRIMARY_SLOT_ID)).toBe(true)
    expect(selectedModelStopIsVisible(waiting, PRIMARY_SLOT_ID)).toBe(false)
    const failed: ModelSlotsState = {
      ...waiting,
      slots: {
        ...waiting.slots,
        primary: new ModelSlotConfiguredLocal({
          ...primary,
          residency: {
            _tag: "Failed",
            failure: { code: "load_failed", message: "failed", retryable: true },
          },
          actions: ["RetryLoad"],
        }),
      },
    }
    expect(modelLoadIsVisible(failed, PRIMARY_SLOT_ID)).toBe(true)
    expect(selectedModelStopIsVisible(authoritative, PRIMARY_SLOT_ID)).toBe(true)
  })
})
