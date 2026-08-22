import { Option } from "effect"
import {
  PRIMARY_SLOT_ID,
  type ModelSlotsState,
  type SlotId,
  type SlotSelection,
} from "./model-state"

/**
 * Visibility predicates over `ModelSlotsState`: the postconditions a slot
 * command promises once acknowledged. Shared by the contract's mutation
 * synchronization and by client presentation.
 */

export const sameSlotSelection = (left: SlotSelection, right: SlotSelection): boolean =>
  left.providerId === right.providerId
  && left.providerModelId === right.providerModelId
  && left.reasoningEffort === right.reasoningEffort

export const authoritativeSlotSelection = (
  state: ModelSlotsState,
  slotId: SlotId,
): Option.Option<SlotSelection> => {
  const slot = state.slots[slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"]
  return slot._tag === "Unassigned" ? Option.none() : Option.some(slot.selection)
}

export const slotAssignmentIsVisible = (
  state: ModelSlotsState,
  slotId: SlotId,
  selection: SlotSelection,
): boolean => Option.exists(authoritativeSlotSelection(state, slotId), (current) =>
  sameSlotSelection(current, selection))

export const modelLoadIsVisible = (
  state: ModelSlotsState,
  slotId: SlotId,
): boolean => {
  const slot = state.slots[slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"]
  return slot._tag === "ConfiguredLocal"
    && slot.residency._tag !== "Unloaded"
}

export const selectedModelStopIsVisible = (
  state: ModelSlotsState,
  slotId: SlotId,
): boolean => {
  const slot = state.slots[slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"]
  return slot._tag !== "ConfiguredLocal"
    || slot.residency._tag !== "Requested"
      && slot.residency._tag !== "Loading"
      && slot.residency._tag !== "Ready"
}
