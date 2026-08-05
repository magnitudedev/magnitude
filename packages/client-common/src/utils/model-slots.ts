import { Option } from "effect"
import {
  PRIMARY_SLOT_ID,
  ProviderModelCatalogLifecycle,
  type ModelInstanceAllocation,
  type ModelInstanceId,
  type ModelReleaseReason,
  type ModelSlot,
  type LocalModelCatalogCandidate,
  type LocalModelsState,
  type ModelSlotsState,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
  type SlotId,
} from "@magnitudedev/sdk"

export const deriveSelectedLocalModelCandidate = (
  models: LocalModelsState,
  slots: ModelSlotsState,
): LocalModelCatalogCandidate | null => {
  const primary = slots.slots.primary
  if (primary._tag === "Unassigned" || models.recommendations._tag !== "Ready") return null
  const configurationId = models.models
    .flatMap((model) => model.offerings)
    .find(({ providerModelId }) => providerModelId === primary.selection.providerModelId)
    ?.configurationId
  if (configurationId === undefined) return null
  return models.recommendations.catalog.find((candidate) =>
    candidate.configurationId === configurationId)
    ?? null
}

type AssignedSlot = Exclude<ModelSlot, { readonly _tag: "Unassigned" }>

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
  if (slot._tag !== "ConfiguredLocal" || Option.isNone(slot.instance)) return null
  const lifecycle = slot.instance.value.lifecycle
  return lifecycle._tag === "Loading"
    || lifecycle._tag === "Stopping"
    || lifecycle._tag === "Stopped" && lifecycle.reason === "memory_pressure"
    || lifecycle._tag === "Failed" && lifecycle.failure.code === "low_memory"
    ? slot
    : null
}

export type LocalModelLoadActivity = NonNullable<
  ReturnType<typeof deriveLocalModelLoadActivity>
>

export const isModelSlotConfigured = (slot: ModelSlot): slot is AssignedSlot =>
  slot._tag !== "Unassigned"

export const modelSlotInstanceId = (
  slot: ModelSlot,
): Option.Option<ModelInstanceId> =>
  slot._tag === "ConfiguredLocal"
    ? Option.map(slot.instance, (instance) => instance.id)
    : Option.none()

export const modelSlotResidentAllocation = (
  slot: ModelSlot,
): Option.Option<ModelInstanceAllocation> => {
  if (slot._tag !== "ConfiguredLocal" || Option.isNone(slot.instance)) return Option.none()
  const lifecycle = slot.instance.value.lifecycle
  if (lifecycle._tag === "Ready") return Option.some(lifecycle.allocation)
  if (lifecycle._tag === "Stopping" && lifecycle.allocation._tag === "Resident") {
    return Option.some(lifecycle.allocation.allocation)
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
  if (slot._tag === "Unassigned") return Option.none()
  return Option.flatMap(models, (catalogModels) => Option.map(
    Option.fromNullable(catalogModels.find((model) =>
      model.providerId === slot.selection.providerId
      && model.providerModelId === slot.selection.providerModelId)),
    (model) => ({ model, slot }),
  ))
}
