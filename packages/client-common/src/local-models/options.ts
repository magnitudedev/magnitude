import { Option } from "effect"
import { localModelIsAvailable } from "@magnitudedev/sdk"
import type {
  LocalInferenceHardware,
  LocalModel,
  LocalModelsState,
  ModelSlotsState,
  ProviderModelId,
} from "@magnitudedev/sdk"
import { formatLocalModelDisplayName } from "../utils/model-presentation"
import { installedLocalModels, localModelProviderModelId } from "./projection"

export interface LocalModelOption {
  readonly id: string
  readonly kind: "running" | "stored" | "downloadable"
  readonly model: LocalModel
}

const kindOrder: Record<LocalModelOption["kind"], number> = {
  running: 0,
  stored: 1,
  downloadable: 2,
}

export const targetPhysicalMemoryBytes = (hardware: LocalInferenceHardware): number =>
  hardware.memoryDomains.reduce((total, domain) => total + domain.totalBytes, 0)

/** Every installed, fitting model an agent can use, catalog or custom. */
export const localModelConnectionOptions = (
  models: readonly LocalModel[],
): readonly { readonly id: ProviderModelId; readonly label: string }[] => models
  .flatMap(model => Option.toArray(Option.map(localModelProviderModelId(model), id => ({
    id,
    label: formatLocalModelDisplayName(model),
  }))))
  .sort((left, right) => left.label.localeCompare(right.label) || left.id.localeCompare(right.id))

export const localModelOptions = (
  models: LocalModelsState,
  slots: ModelSlotsState,
): readonly LocalModelOption[] => {
  const runningProviderModelIds = new Set([
    slots.slots.primary,
    slots.slots.secondary,
  ].flatMap((slot) => slot._tag === "ConfiguredLocal"
    && slot.residency._tag === "Ready"
    ? [slot.selection.providerModelId]
    : []))
  const installed = installedLocalModels(models).map((model): LocalModelOption => ({
    id: `installed:${model.modelId}`,
    kind: Option.exists(localModelProviderModelId(model), (providerModelId) =>
      runningProviderModelIds.has(providerModelId)) ? "running" : "stored",
    model,
  }))
  const representedModelIds = new Set(installed.map(({ model }) => model.modelId))
  const downloadable = models.models.flatMap((model): readonly LocalModelOption[] => {
    if (model._tag !== "Catalog"
      || !localModelIsAvailable(model)
      || representedModelIds.has(model.modelId)
      || model.servingState._tag !== "Assessed"
      || model.servingState.assessment._tag !== "Fits") return []
    return [{
      id: `downloadable:${model.modelId}`,
      kind: "downloadable",
      model,
    }]
  })
  return [...installed, ...downloadable].sort((left, right) =>
    kindOrder[left.kind] - kindOrder[right.kind]
    || formatLocalModelDisplayName(left.model).localeCompare(formatLocalModelDisplayName(right.model))
    || left.id.localeCompare(right.id))
}
