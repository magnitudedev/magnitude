import { Option } from "effect"
import {
  PRIMARY_SLOT_ID,
  ProviderModelCatalogLifecycle,
  type ModelSlot,
  type ModelSlotsState,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
  type SlotId,
} from "@magnitudedev/sdk"

type AssignedSlot = Exclude<ModelSlot, { readonly _tag: "Unassigned" }>

export interface SelectedSlotModel {
  readonly model: ProviderModelCatalogEntry
  readonly slot: AssignedSlot
}

export const isModelSlotUsableForMessages = (slot: ModelSlot): boolean =>
  slot._tag === "UnloadedLocalModel"
  || slot._tag === "LoadingLocalModel"
  || slot._tag === "Ready"

export function selectedSlotModel(
  catalog: ProviderModelCatalogState,
  slots: ModelSlotsState,
  slotId: SlotId,
): Option.Option<SelectedSlotModel> {
  const models = ProviderModelCatalogLifecycle.match(catalog, {
    Loading: () => Option.none<readonly ProviderModelCatalogEntry[]>(),
    Ready: ({ models }) => Option.some(models),
    Refreshing: ({ models }) => Option.some(models),
    Degraded: ({ models }) => Option.some(models),
    Unavailable: () => Option.none<readonly ProviderModelCatalogEntry[]>(),
  })
  const slot = slots.slots[slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"]
  if (slot._tag === "Unassigned") return Option.none()
  return Option.flatMap(models, (catalogModels) => Option.map(
    Option.fromNullable(catalogModels.find((model) => model.providerId === slot.selection.providerId
      && model.providerModelId === slot.selection.providerModelId)),
    (model) => ({ model, slot }),
  ))
}
