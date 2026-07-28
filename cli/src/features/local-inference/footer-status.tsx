import { Option } from "effect"
import { modelSlotResidentAllocation } from "@magnitudedev/client-common"
import { PRIMARY_SLOT_ID, ProviderIdSchema } from "@magnitudedev/sdk"
import type { LocalModelsState, ModelSlot, ModelSlotsState, ProviderId, SlotId } from "@magnitudedev/sdk"

const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")
export interface LocalInferenceFooterView {
  readonly modelName: string | null
  readonly residency: "loaded" | "loading" | "not_loaded" | null
  readonly memoryLabel: string | null
}

const compactGiB = (bytes: number): string =>
  (bytes / 1024 ** 3).toFixed(1).replace(/\.0$/, "")

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
      return `${compactGiB(bytes)} GB mem`
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
    ? models.models.find((model) => model.preparation._tag === "Available"
      && model.preparation.providerModelIds.includes(slot.selection.providerModelId))
    : undefined
  const downloadModel = models?.models.find((model) =>
    model.download._tag === "Downloading" || model.download._tag === "Failed")
  const model = activeModel ?? downloadModel
  const lifecycle = slot?._tag === "ConfiguredLocal"
    ? Option.getOrNull(slot.instance)?.lifecycle
    : undefined
  const residency = lifecycle?._tag === "Ready"
    ? "loaded" as const
    : lifecycle?._tag === "Loading" || lifecycle?._tag === "Stopping"
      ? "loading" as const
      : "not_loaded" as const
  return {
    modelName: selectedModelName ?? model?.displayName ?? null,
    residency,
    memoryLabel: residency === "loaded" && slot ? residentMemoryLabel(slot) : null,
  }
}
