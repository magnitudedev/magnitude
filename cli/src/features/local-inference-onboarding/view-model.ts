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

export const formatModelSize = (bytes: number): string => {
  const gb = bytes / 1_000_000_000
  return `${gb.toFixed(gb >= 10 ? 1 : 2)} GB`
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

const formatBillions = (value: number): string =>
  Number.isInteger(value) ? `${value}B` : `${value.toFixed(1)}B`

export const selectionMetadata = (selection: LocalInferenceSelection): string => {
  const total = selection.kind === "recommendation"
    ? selection.recommendation.totalParametersBillions
    : selection.choice.totalParametersBillions
  const active = selection.kind === "recommendation"
    ? selection.recommendation.activeParametersBillions
    : selection.choice.activeParametersBillions
  const parameters = total === undefined
    ? null
    : active !== undefined
    ? `${formatBillions(total)} total / ${formatBillions(active)} active`
    : `${formatBillions(total)} parameters`
  const quant = selection.kind === "recommendation"
    ? selection.recommendation.quantization.format
    : selection.choice.quantization?.format ?? "Quant unavailable"
  const size = selection.kind === "recommendation"
    ? formatModelSize(selection.recommendation.totalDownloadBytes)
    : selection.choice.sizeBytes !== undefined
      ? formatModelSize(selection.choice.sizeBytes)
      : "Size unavailable"
  const contextTokens = selection.kind === "recommendation"
    ? selection.recommendation.contextTokens
    : selection.choice.contextTokens
  return [quant, size, parameters, `${formatContext(contextTokens)} context`]
    .filter((value): value is string => value !== null)
    .join(" · ")
}

export const selectionFidelity = (selection: LocalInferenceSelection): string | null =>
  selection.kind === "recommendation"
    ? selection.recommendation.quantization.fidelityLabel
    : null
