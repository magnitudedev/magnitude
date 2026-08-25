import { Option } from "effect"
import {
  formatMemorySize,
  formatLocalModelDisplayName,
  modelSlotResidentAllocation,
} from "@magnitudedev/client-common"
import { PRIMARY_SLOT_ID, ProviderIdSchema } from "@magnitudedev/sdk"
import type { LocalModelsState, ModelSlot, ModelSlotsState, ProviderId, SlotId } from "@magnitudedev/sdk"

const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")
export interface LocalInferenceFooterView {
  readonly modelName: string | null
  readonly residency: "loaded" | "loading" | "not_loaded" | null
  readonly memoryLabel: string | null
}

const residentMemoryLabel = (
  slot: ModelSlot,
): string | null =>
  Option.match(modelSlotResidentAllocation(slot), {
    onNone: () => null,
    onSome: ({ memoryDomains }) => {
      const bytes = memoryDomains.reduce(
        (total, domain) => total
          + domain.modelBytes
          + domain.contextBytes
          + domain.computeBytes
          + domain.auxiliaryBytes,
        0,
      )
      return `${formatMemorySize(bytes)} mem`
    },
  })

export const deriveLocalInferenceFooterView = (
  models: LocalModelsState | null,
  slots: ModelSlotsState | null,
  selectedModelName: string | null,
  selectedProviderId: ProviderId | null,
  selectedSlotId: SlotId,
): LocalInferenceFooterView => {
  if (selectedProviderId !== null && selectedProviderId !== LOCAL_PROVIDER_ID) {
    return { modelName: selectedModelName ?? "Cloud model", residency: null, memoryLabel: null }
  }
  if (slots === null) {
    return {
      modelName: selectedModelName,
      residency: selectedProviderId === LOCAL_PROVIDER_ID ? "not_loaded" : null,
      memoryLabel: null,
    }
  }
  const selectedSlot = slots.slots[
    selectedSlotId === PRIMARY_SLOT_ID ? "primary" : "secondary"
  ]
  const slot = selectedSlot._tag !== "Unassigned"
    && selectedSlot.selection.providerId === LOCAL_PROVIDER_ID
    ? selectedSlot
    : undefined
  const activeModel = slot && models !== null
    ? models.models.find((model) => model.servingState._tag === "Assessed"
      && model.servingState.availabilityState._tag === "Selectable"
      && model.servingState.availabilityState.providerModelId === slot.selection.providerModelId)
    : undefined
  const download = models?.models.find(({ acquisitionState }) =>
    acquisitionState._tag === "Installing"
    || acquisitionState._tag === "InstallFailed"
    || acquisitionState._tag === "Updating"
    || acquisitionState._tag === "UpdateFailed")
  const currentResidency = slot?._tag === "ConfiguredLocal"
    ? slot.residency
    : undefined
  const residency = currentResidency?._tag === "Ready"
    ? "loaded" as const
    : currentResidency?._tag === "Requested"
      || currentResidency?._tag === "Loading"
      || currentResidency?._tag === "Stopping"
      ? "loading" as const
      : "not_loaded" as const
  return {
    modelName: selectedModelName
      ?? (activeModel ? formatLocalModelDisplayName(activeModel) : undefined)
      ?? (download ? formatLocalModelDisplayName(download) : undefined)
      ?? null,
    residency,
    memoryLabel: residency === "loaded" && slot ? residentMemoryLabel(slot) : null,
  }
}
