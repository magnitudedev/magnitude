import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelSlotUnassigned,
  ModelSlotConfiguredLocal,
  ModelInstanceIdSchema,
  ModelServingConfigurationIdSchema,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
  type ModelSlotsState,
} from "@magnitudedev/sdk"
import {
  admittedInstanceIsVisible,
  presentedSlotSelection,
  slotAssignmentIsVisible,
  type ModelSlotAssignmentMutationState,
} from "./atoms"

const authoritative: ModelSlotsState = {
  slots: {
    primary: new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID }),
    secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
  },
  recentModels: { primary: [], secondary: [] },
  favoriteModels: [],
}

const assignment = (providerModelId: string, waiting: boolean) => ({
  input: {
    slotId: PRIMARY_SLOT_ID,
    selection: {
      providerId: ProviderIdSchema.make("test"),
      providerModelId: ProviderModelIdSchema.make(providerModelId),
      reasoningEffort: ReasoningEffortSchema.make("none"),
    },
  },
  result: Result.initial(waiting),
}) as ModelSlotAssignmentMutationState

describe("optimistic model-slot presentation", () => {
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
  const selection = assignment("selected", true).input.selection
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
        instance: Option.some({
          id: ModelInstanceIdSchema.make("instance"),
          configurationId: ModelServingConfigurationIdSchema.make("configuration"),
          lifecycle: {
            _tag: "Loading",
            stage: "queued",
            progress: Option.none(),
            plannedAllocation: Option.none(),
          },
        }),
        actions: ["Stop"],
      }),
    },
  }

  it("requires the exact assigned selection", () => {
    expect(slotAssignmentIsVisible(configured, PRIMARY_SLOT_ID, selection)).toBe(true)
    expect(slotAssignmentIsVisible(
      configured,
      PRIMARY_SLOT_ID,
      assignment("replacement", true).input.selection,
    )).toBe(false)
  })

  it("requires the exact admitted instance", () => {
    expect(admittedInstanceIsVisible(
      configured,
      PRIMARY_SLOT_ID,
      ModelInstanceIdSchema.make("instance"),
    )).toBe(true)
    expect(admittedInstanceIsVisible(
      configured,
      PRIMARY_SLOT_ID,
      ModelInstanceIdSchema.make("replacement"),
    )).toBe(false)
  })
})
