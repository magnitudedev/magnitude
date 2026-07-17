import { Option } from "effect"
import {
  ModelCatalogLifecycle,
  ModelSlotsLifecycle,
  type ModelCatalogState,
  type ModelSlotsState,
  type ModelSummary,
  type SlotId,
  type SlotState,
} from "@magnitudedev/sdk"

type AssignedSlotState = Exclude<SlotState, { readonly _tag: "Unassigned" }>

export interface SelectedSlotModel {
  readonly model: ModelSummary
  readonly slot: AssignedSlotState
}

/** Join catalog details to ACN's authoritative slot selection. */
export function selectedSlotModel(
  catalog: ModelCatalogState,
  slots: ModelSlotsState,
  slotId: SlotId,
): Option.Option<SelectedSlotModel> {
  const models = ModelCatalogLifecycle.match(catalog, {
    loading: () => Option.none<readonly ModelSummary[]>(),
    ready: ({ models }) => Option.some(models),
    refreshing: ({ models }) => Option.some(models),
    degraded: ({ models }) => Option.some(models),
    unavailable: () => Option.none<readonly ModelSummary[]>(),
  })
  const slot = ModelSlotsLifecycle.match(slots, {
    loading: () => Option.none<SlotState>(),
    ready: ({ slots }) => Option.some(slots[slotId]),
    refreshing: ({ slots }) => Option.some(slots[slotId]),
    degraded: ({ slots }) => Option.some(slots[slotId]),
    unavailable: ({ slots }) => Option.some(slots[slotId]),
  })
  return Option.flatMap(slot, (selected) => {
    if (selected._tag === "Unassigned") return Option.none()
    return Option.flatMap(models, (catalogModels) => Option.map(
      Option.fromNullable(catalogModels.find((model) => model.providerId === selected.selection.providerId
        && model.providerModelId === selected.selection.providerModelId)),
      (model) => ({ model, slot: selected }),
    ))
  })
}
