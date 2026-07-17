import type { ModelSummary, ReasoningEffort } from "@magnitudedev/sdk"

export interface ReasoningEffortOption {
  readonly value: ReasoningEffort
  readonly label: string
}

export function formatReasoningEffort(effort: ReasoningEffort): string {
  return String(effort)
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase())
}

export function reasoningEffortOptions(model: ModelSummary): readonly ReasoningEffortOption[] {
  const property = model.properties.reasoning
  const discovered = property._tag === "Cached" || property._tag === "Resolved" || property._tag === "Refreshing"
    ? property.value
    : [model.defaultReasoningEffort]
  const efforts = [...new Set([model.defaultReasoningEffort, ...discovered])]
  return efforts.map((value) => ({
    value,
    label: formatReasoningEffort(value),
  }))
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
