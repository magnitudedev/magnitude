import { Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelSlotUnassigned,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
  type ModelSlotsState,
} from "@magnitudedev/sdk"
import {
  presentedSlotSelection,
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
