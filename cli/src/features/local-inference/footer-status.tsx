import { deriveHardwareMemoryView, type LocalInferenceView } from "@magnitudedev/client-common"
import { PRIMARY_SLOT_ID, ProviderIdSchema } from "@magnitudedev/sdk"
import type { ProviderId, SlotId } from "@magnitudedev/sdk"

const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")
const compactGiB = (bytes: number): string =>
  (bytes / 1024 ** 3).toFixed(1).replace(/\.0$/, "")

export interface LocalInferenceFooterView {
  readonly modelName: string | null
  /** Resident local memory, shown only while the selected model is loaded. */
  readonly memoryLabel: string | null
}

export const deriveLocalInferenceFooterView = (
  state: LocalInferenceView | null,
  selectedModelName: string | null,
  selectedProviderId: ProviderId | null,
  selectedSlotId: SlotId,
): LocalInferenceFooterView => {
  if (selectedProviderId !== null && selectedProviderId !== LOCAL_PROVIDER_ID) {
    return { modelName: selectedModelName ?? "Cloud model", memoryLabel: null }
  }
  if (state === null) {
    return {
      modelName: selectedModelName,
      memoryLabel: null,
    }
  }
  const selectedSlot = state.slots.slots[
    selectedSlotId === PRIMARY_SLOT_ID ? "primary" : "secondary"
  ]
  const slot = selectedSlot._tag !== "Unassigned"
    && selectedSlot.selection.providerId === LOCAL_PROVIDER_ID
    ? selectedSlot
    : undefined
  const activeModel = slot
    ? state.models.models.find((model) => model.preparation._tag === "Available"
      && model.preparation.providerModelIds.includes(slot.selection.providerModelId))
    : undefined
  const downloadModel = state.models.models.find((model) =>
    model.download._tag === "Downloading" || model.download._tag === "Failed")
  const model = activeModel ?? downloadModel
  const memory = slot?._tag === "Ready"
    ? deriveHardwareMemoryView(state.hardware, { fallbackToAccelerators: true }).compact
    : null
  return {
    modelName: selectedModelName ?? model?.displayName ?? null,
    memoryLabel: memory
      ? `${compactGiB(memory.usedBytes)} / ${compactGiB(memory.totalBytes)} GiB`
      : null,
  }
}
