import type {
  LocalInferenceState,
  LocalModelChoice,
  LocalModelRecommendation,
} from "@magnitudedev/sdk"

export type LocalInferenceSelection =
  | { readonly kind: "running" | "stored"; readonly id: string; readonly choice: LocalModelChoice }
  | { readonly kind: "recommendation"; readonly id: string; readonly recommendation: LocalModelRecommendation }

export const buildLocalInferenceSelections = (
  state: LocalInferenceState,
): LocalInferenceSelection[] => {
  const choices = state.choices
    .filter((choice) => choice.compatible)
    .map((choice): LocalInferenceSelection => ({
      kind: choice._tag === "RunningExternal" || choice._tag === "RunningManaged" ? "running" : "stored",
      id: choice.choiceId,
      choice,
    }))
  const choiceIds = new Set(choices.map((choice) => choice.id))
  return [
    ...choices,
    ...state.recommendations
      .filter((recommendation) => !choiceIds.has(recommendation.configurationId))
      .map((recommendation): LocalInferenceSelection => ({
        kind: "recommendation",
        id: recommendation.configurationId,
        recommendation,
      })),
  ]
}

export const selectedInferenceIndex = (
  selections: readonly LocalInferenceSelection[],
  selectedId: string | null,
): number => {
  const index = selectedId === null
    ? -1
    : selections.findIndex((selection) => selection.id === selectedId)
  return index >= 0 ? index : 0
}

export const formatBytes = (bytes: number): string => {
  const gib = bytes / 1024 ** 3
  return gib >= 1 ? `${gib.toFixed(gib >= 10 ? 1 : 2)} GiB` : `${(bytes / 1024 ** 2).toFixed(0)} MiB`
}

export const formatContext = (tokens: number): string => tokens < 1_000
  ? String(tokens)
  : tokens % 1_024 === 0
    ? `${tokens / 1_024}K`
    : `${Math.round(tokens / 1_000)}K`

export const selectionTitle = (selection: LocalInferenceSelection): string =>
  selection.kind === "recommendation" ? selection.recommendation.displayName : selection.choice.displayName

export const selectionMetadata = (selection: LocalInferenceSelection): string => {
  const quantization = selection.kind === "recommendation"
    ? selection.recommendation.quantization.format
    : selection.choice.quantization?.format ?? "Quantization unavailable"
  const size = selection.kind === "recommendation"
    ? formatBytes(selection.recommendation.totalDownloadBytes)
    : selection.choice.sizeBytes === undefined
      ? "Size unavailable"
      : formatBytes(selection.choice.sizeBytes)
  const context = selection.kind === "recommendation"
    ? `${formatContext(selection.recommendation.contextTokens)} × ${selection.recommendation.servingProfile.parallelSlots} slot${selection.recommendation.servingProfile.parallelSlots === 1 ? "" : "s"}`
    : selection.choice.contextTokens === undefined
      ? "Context unavailable"
      : `${formatContext(selection.choice.contextTokens)} context`
  return `${quantization} · ${size} · ${context}`
}
