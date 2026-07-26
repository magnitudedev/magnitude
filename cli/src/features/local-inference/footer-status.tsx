import { Option } from "effect"
import type { LocalInferenceView } from "@magnitudedev/client-common"
import { PRIMARY_SLOT_ID, ProviderIdSchema } from "@magnitudedev/sdk"
import type { ProviderId, SlotId } from "@magnitudedev/sdk"

const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")
export interface LocalInferenceFooterView {
  readonly modelName: string | null
  readonly residency: "loaded" | "loading" | "not_loaded" | null
  readonly memoryLabel: string | null
}

const compactGiB = (bytes: number): string =>
  (bytes / 1024 ** 3).toFixed(1).replace(/\.0$/, "")

const residentMemoryLabel = (state: LocalInferenceView): string | null =>
  Option.match(state.hardware.residentMemory, {
    onNone: () => null,
    onSome: ({ domains }) => {
      const bytes = domains.reduce(
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
  state: LocalInferenceView | null,
  selectedModelName: string | null,
  selectedProviderId: ProviderId | null,
  selectedSlotId: SlotId,
): LocalInferenceFooterView => {
  if (selectedProviderId !== null && selectedProviderId !== LOCAL_PROVIDER_ID) {
    return { modelName: selectedModelName ?? "Cloud model", residency: null, memoryLabel: null }
  }
  if (state === null) {
    return {
      modelName: selectedModelName,
      residency: selectedProviderId === LOCAL_PROVIDER_ID ? "not_loaded" : null,
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
  const residency = slot?._tag === "Ready"
    ? "loaded" as const
    : slot?._tag === "LoadingLocalModel" || slot?._tag === "UnloadingLocalModel"
      ? "loading" as const
      : "not_loaded" as const
  return {
    modelName: selectedModelName ?? model?.displayName ?? null,
    residency,
    memoryLabel: residency === "loaded" ? residentMemoryLabel(state) : null,
  }
}
