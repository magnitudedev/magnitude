import type {
  LocalInferenceHardware,
  LocalModelsState,
  ModelSlotsState,
  ProviderModelCatalogState,
} from "@magnitudedev/sdk"

/** Pure presentation input composed from independent server-owned domains. */
export interface LocalInferenceView {
  readonly hardware: LocalInferenceHardware
  readonly models: LocalModelsState
  readonly catalog: ProviderModelCatalogState
  readonly slots: ModelSlotsState
}
