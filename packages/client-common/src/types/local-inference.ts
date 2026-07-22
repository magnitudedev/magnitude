import type {
  LocalInferenceHardware,
  LocalModelInventoryState,
  ModelSlotsState,
} from "@magnitudedev/sdk"

/** Pure presentation input composed from the three independent mirrors. */
export interface LocalInferenceView {
  readonly hardware: LocalInferenceHardware
  readonly inventory: LocalModelInventoryState
  readonly slots: ModelSlotsState
}
