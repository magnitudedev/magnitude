import type { ModelSummary, ReasoningEffort } from "@magnitudedev/sdk"

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

export function reasoningEffortControl(model: ModelSummary): ReasoningEffortControl {
  const property = model.properties.reasoning
  switch (property._tag) {
    case "Cached":
    case "Resolved":
    case "Refreshing":
      return {
        _tag: "Available",
        options: property.value.map((value) => ({ value, label: formatReasoningEffort(value) })),
      }
    case "Deferred": return { _tag: "Unavailable", label: "Load to inspect" }
    case "Discovering": return { _tag: "Unavailable", label: property.phase === "loading" ? "Loading…" : "Inspecting…" }
    case "Failed": return { _tag: "Unavailable", label: "Inspection failed" }
  }
}

export function reasoningPropertyLabel(model: ModelSummary): string {
  const property = model.properties.reasoning
  switch (property._tag) {
    case "Deferred": return "Reasoning options available after loading"
    case "Discovering": return property.phase === "loading" ? "Loading to discover reasoning options" : "Inspecting reasoning options"
    case "Cached": return "Reasoning options cached; they will be verified when used"
    case "Resolved": return "Reasoning options verified"
    case "Refreshing": return "Refreshing reasoning options; cached options remain available"
    case "Failed": return `Reasoning discovery failed: ${property.error.message}`
  }
}

export function visionPropertyLabel(model: ModelSummary): string {
  const property = model.properties.vision
  switch (property._tag) {
    case "Deferred": return "Vision capability available after loading"
    case "Discovering": return property.phase === "loading" ? "Loading to discover vision capability" : "Inspecting vision capability"
    case "Cached": return property.value ? "Vision supported (cached)" : "Vision not supported (cached)"
    case "Resolved": return property.value ? "Vision supported" : "Vision not supported"
    case "Refreshing": return property.value ? "Vision supported (refreshing)" : "Vision not supported (refreshing)"
    case "Failed": return `Vision discovery failed: ${property.error.message}`
  }
}
