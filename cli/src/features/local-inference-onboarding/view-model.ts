import type {
  LocalInferenceOnboardingSnapshot,
  LocalModelChoice,
  LocalModelRecommendation,
} from "@magnitudedev/sdk"

export type LocalInferenceSelection =
  | { readonly kind: "running" | "downloaded"; readonly id: string; readonly choice: LocalModelChoice }
  | { readonly kind: "recommendation"; readonly id: string; readonly recommendation: LocalModelRecommendation }

export const shouldShowLocalInferenceOnboarding = (
  snapshot: LocalInferenceOnboardingSnapshot,
  forceSetup: boolean,
): boolean => forceSetup || snapshot.onboarding.required

export const buildLocalInferenceSelections = (
  snapshot: LocalInferenceOnboardingSnapshot,
): LocalInferenceSelection[] => [
  ...snapshot.running
    .filter((choice) => choice.compatible)
    .map((choice): LocalInferenceSelection => ({ kind: "running", id: choice.choiceId, choice })),
  ...snapshot.downloaded
    .filter((choice) => choice.compatible)
    .map((choice): LocalInferenceSelection => ({ kind: "downloaded", id: choice.choiceId, choice })),
  ...snapshot.recommendations
    .map((recommendation): LocalInferenceSelection => ({
      kind: "recommendation",
      id: recommendation.configurationId,
      recommendation,
    })),
]

export const formatBytes = (bytes: number): string => {
  const gib = bytes / 1024 ** 3
  if (gib >= 1) return `${gib.toFixed(gib >= 10 ? 1 : 2)} GiB`
  return `${(bytes / 1024 ** 2).toFixed(0)} MiB`
}

export const formatContext = (tokens: number): string =>
  tokens < 1_000
    ? `${tokens}`
    : tokens % 1_024 === 0
      ? `${tokens / 1_024}K`
      : `${Math.round(tokens / 1_000)}K`

export const selectionTitle = (selection: LocalInferenceSelection): string =>
  selection.kind === "recommendation"
    ? selection.recommendation.displayName
    : selection.choice.displayName

export const selectionSubtitle = (selection: LocalInferenceSelection): string => {
  if (selection.kind !== "recommendation") {
    return [
      selection.choice.quantization?.format ?? "Quant unavailable",
      selection.choice.sizeBytes !== undefined ? formatBytes(selection.choice.sizeBytes) : "Size unavailable",
      `${formatContext(selection.choice.contextTokens)} context`,
    ].join(" · ")
  }
  const item = selection.recommendation
  return `${item.quantization.format} · ${formatBytes(item.totalDownloadBytes)} · ${formatContext(item.contextTokens)} context`
}

const formatBillions = (value: number): string =>
  Number.isInteger(value) ? `${value}B` : `${value.toFixed(1)}B`

export const selectionParameters = (selection: LocalInferenceSelection): string | null => {
  const total = selection.kind === "recommendation"
    ? selection.recommendation.totalParametersBillions
    : selection.choice.totalParametersBillions
  const active = selection.kind === "recommendation"
    ? selection.recommendation.activeParametersBillions
    : selection.choice.activeParametersBillions
  if (total === undefined) return null
  return active !== undefined
    ? `${formatBillions(total)} total / ${formatBillions(active)} active`
    : `${formatBillions(total)} parameters`
}
