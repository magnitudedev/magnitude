import { Option } from "effect"
import {
  PRIMARY_SLOT_ID,
  ProviderModelCatalogLifecycle,
  type ModelInstanceAllocation,
  type ModelReleaseReason,
  type ModelSlot,
  type LocalModel,
  type LocalModelsState,
  type ModelSlotsState,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
  type SlotId,
} from "@magnitudedev/sdk"
import { localModelProviderModelId } from "../local-models/projection"
export const deriveSelectedLocalModel = (
  models: LocalModelsState,
  slots: ModelSlotsState,
): LocalModel | null => {
  const primary = slots.slots.primary
  if (primary._tag === "Unassigned") return null
  return models.models.find((model) =>
    Option.contains(localModelProviderModelId(model), primary.selection.providerModelId)) ?? null
}

type AssignedSlot = Extract<ModelSlot, {
  readonly _tag: "ConfiguredRemote" | "ConfiguredLocal"
}>

export interface SelectedSlotModel {
  readonly model: ProviderModelCatalogEntry
  readonly slot: AssignedSlot
}

export const formatModelLoadProgress = (percentage: number): string =>
  `Loading model into memory · ${percentage}%`

export function modelReleaseReasonLabel(reason: ModelReleaseReason): string {
  switch (reason) {
    case "user_stop": return "User requested"
    case "idle_timeout": return "Idle timeout"
    case "replacement": return "Model replacement"
    case "memory_pressure": return "Low memory"
  }
}

export function deriveLocalModelLoadActivity(
  slots: ModelSlotsState,
  slotId: SlotId,
) {
  const slot = slots.slots[slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"]
  if (slot._tag !== "ConfiguredLocal") return null
  return slot.residency._tag === "Requested"
    || slot.residency._tag === "Loading"
    || slot.residency._tag === "Stopping"
    || slot.residency._tag === "Failed" && slot.residency.failure.code === "low_memory"
    ? slot
    : null
}

export type LocalModelLoadActivity = NonNullable<
  ReturnType<typeof deriveLocalModelLoadActivity>
>

export const isModelSlotConfigured = (slot: ModelSlot): slot is AssignedSlot =>
  slot._tag === "ConfiguredRemote" || slot._tag === "ConfiguredLocal"

export const modelSlotResidentAllocation = (
  slot: ModelSlot,
): Option.Option<ModelInstanceAllocation> => {
  if (slot._tag !== "ConfiguredLocal") return Option.none()
  if (slot.residency._tag === "Ready") return Option.some(slot.residency.allocation)
  if (slot.residency._tag === "Stopping" && slot.residency.allocation._tag === "Resident") {
    return Option.some(slot.residency.allocation.allocation)
  }
  return Option.none()
}

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
  if (slot._tag === "Unassigned" || slot._tag === "Resolving") return Option.none()
  return Option.flatMap(models, (catalogModels) => Option.map(
    Option.fromNullable(catalogModels.find((model) =>
      model.providerId === slot.selection.providerId
      && model.providerModelId === slot.selection.providerModelId)),
    (model) => ({ model, slot }),
  ))
}
