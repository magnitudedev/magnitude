import { Option } from "effect"
import {
  PRIMARY_SLOT_ID,
  ProviderModelCatalogLifecycle,
  type ModelSlot,
  type LocalModelCatalogCandidate,
  type ModelSlotsState,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
  type SlotId,
} from "@magnitudedev/sdk"
import type { LocalInferenceView } from "../types/local-inference"

export type SelectedLocalModelSetup = LocalModelCatalogCandidate

export const deriveSelectedLocalModelCandidate = (
  state: LocalInferenceView,
): SelectedLocalModelSetup | null => {
  const primary = state.slots.slots.primary
  if (primary._tag === "Unassigned" || state.models.recommendations._tag !== "Ready") return null
  return state.models.recommendations.catalog.find(({ providerModelId }) =>
    providerModelId === primary.selection.providerModelId)
    ?? null
}

export const deriveSelectedLocalModelSetup = (
  state: LocalInferenceView,
): SelectedLocalModelSetup | null => {
  const candidate = deriveSelectedLocalModelCandidate(state)
  if (!candidate) return null
  if (candidate.download._tag === "Downloading" || candidate.download._tag === "Failed") {
    return candidate
  }
  return candidate.download._tag === "Downloaded" && candidate.preparation._tag === "Calibrating"
    ? candidate
    : null
}

type AssignedSlot = Exclude<ModelSlot, { readonly _tag: "Unassigned" }>

export interface SelectedSlotModel {
  readonly model: ProviderModelCatalogEntry
  readonly slot: AssignedSlot
}

export const formatModelLoadProgress = (percentage: number): string =>
  `Loading model · ${percentage}%`

export function deriveLocalModelLoadActivity(
  slots: ModelSlotsState,
  slotId: SlotId,
) {
  const slot = slots.slots[slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"]
  if (slot._tag === "LoadingLocalModel") return slot
  if (slot._tag === "Blocked"
    && slot.reason._tag === "LocalModelStoppedLowMemory") return slot
  return null
}

export type LocalModelLoadActivity = NonNullable<
  ReturnType<typeof deriveLocalModelLoadActivity>
>

export const isModelSlotConfigured = (slot: ModelSlot): boolean =>
  slot._tag !== "Unassigned"

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
