import type { ProviderModelCatalogEntry, ReasoningEffort } from "@magnitudedev/sdk"

export interface ReasoningEffortOption {
  readonly value: ReasoningEffort
  readonly label: string
}

export type ReasoningEffortControl =
  | { readonly _tag: "Available"; readonly options: readonly ReasoningEffortOption[] }
  | { readonly _tag: "Unavailable"; readonly label: string }

export function formatReasoningEffort(effort: ReasoningEffort): string {
  return String(effort)
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase())
}

export function reasoningEffortControl(model: ProviderModelCatalogEntry): ReasoningEffortControl {
  return model.capabilities.reasoning.supported
    ? {
        _tag: "Available",
        options: model.capabilities.reasoning.efforts.map((value) => ({ value, label: formatReasoningEffort(value) })),
      }
    : { _tag: "Unavailable", label: "Reasoning not supported" }
}

export function reasoningPropertyLabel(model: ProviderModelCatalogEntry): string {
  return model.capabilities.reasoning.supported ? "Reasoning supported" : "Reasoning not supported"
}

export function visionPropertyLabel(model: ProviderModelCatalogEntry): string {
  return model.capabilities.vision ? "Vision supported" : "Vision not supported"
}
