import { Option } from "effect"
import { formatMemorySize, localModelSpeculativeMethodLabel } from "@magnitudedev/client-common"
import type { LocalModel } from "@magnitudedev/sdk"
import type { PentagonRadarAxes } from "../../components/pentagon-radar"
import { performanceRangeSpeedLabel } from "./view-model"

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value))

export const normalizeLocalModelRadarSpeed = (tokensPerSecond: number): number => {
  if (tokensPerSecond <= 30) return clamp01(tokensPerSecond / 30) * 0.5
  if (tokensPerSecond <= 100) return 0.5 + ((tokensPerSecond - 30) / 70) * 0.4
  if (tokensPerSecond >= 1_000) return 1
  return 0.9 + 0.1 * Math.log(tokensPerSecond / 100) / Math.log(10)
}

type AssessedModel = Extract<LocalModel["servingState"], { readonly _tag: "Assessed" }>
type ModelAssessment = AssessedModel["assessment"]

const memoryUseRatio = (assessment: ModelAssessment): number => {
  if (assessment._tag !== "Fits") return 1
  return assessment.memory.domains.reduce((highestUse, domain) => {
    const usableCapacityBytes = domain.capacityBytes - domain.compatibilityReserveBytes
    const use = usableCapacityBytes <= 0
      ? (domain.requiredBytes > 0 ? 1 : 0)
      : domain.requiredBytes / usableCapacityBytes
    return Math.max(highestUse, clamp01(use))
  }, 0)
}

const memoryEfficiency = (assessment: ModelAssessment): number => 1 - memoryUseRatio(assessment)

const speculationValue = (model: LocalModel): number => {
  if (model.bundle._tag !== "SpeculativeDecoding") return 0
  switch (model.bundle.method._tag) {
    case "Mtp": return 1 / 3
    case "DFlash": return 2 / 3
    case "DSpark": return 1
  }
}

const accuracyLabel = (rank: number): string => rank >= 75
  ? "Native"
  : rank >= 55
    ? "Very high"
    : rank >= 45 ? "High" : "Reduced"

const shortVariantLabel = (model: LocalModel): string =>
  String(model.presentation.variantLabel).match(/\b(?:IQ|Q)\d+(?:\.\d+)?\b/i)?.[0]
    ?? String(model.presentation.variantLabel).split(/[ _-]/, 1)[0]
    ?? String(model.presentation.variantLabel)

const targetPackage = (model: LocalModel) =>
  model.bundle._tag === "Standalone" ? model.bundle.package : model.bundle.target

const quantizationBits = (model: LocalModel): Option.Option<number> => {
  const target = targetPackage(model)
  for (const candidate of [target.properties.quantization, target.properties.quantizationName]) {
    const bits = candidate.match(/(?:IQ|Q)?(\d+(?:\.\d+)?)\s*(?:[- ]?bit)?/i)?.[1]
    if (bits !== undefined) return Option.some(Number(bits))
  }
  return Option.none()
}

const discoveredAccuracyLabel = (bits: Option.Option<number>): string => Option.match(bits, {
  onNone: () => "Not assessed",
  onSome: (value) => value >= 8
    ? "Native"
    : value >= 6
      ? "Very high"
      : value >= 5 ? "High" : "Reduced",
})

const memoryFootprintLabel = (assessment: ModelAssessment): string => {
  const use = memoryUseRatio(assessment)
  if (use <= 0.2) return "Tiny"
  if (use <= 0.4) return "Light"
  if (use <= 0.6) return "Medium"
  if (use <= 0.8) return "Heavy"
  return "Tight"
}

export const localModelRadarAxes = (model: LocalModel): Option.Option<PentagonRadarAxes> => {
  if (model.servingState._tag !== "Assessed"
    || model.servingState.assessment._tag !== "Fits") return Option.none()

  const assessment = model.servingState.assessment
  const comparisonContext = Math.min(50_000, assessment.profile.contextLength)
  const performance = assessment.performance.reduce((closest, candidate) =>
    Math.abs(candidate.contextTokens - comparisonContext)
      < Math.abs(closest.contextTokens - comparisonContext) ? candidate : closest)
  if (performance === undefined) return Option.none()

  const speculation = Option.getOrElse(localModelSpeculativeMethodLabel(model), () => "None")
  const catalog = model.catalogMembershipState._tag === "InCatalog"
    ? Option.some(model.catalogMembershipState.catalogData)
    : Option.none()
  const bits = quantizationBits(model)
  const quantization = targetPackage(model).properties.quantization
  const axes: PentagonRadarAxes = [
    {
      value: Option.map(catalog, ({ intelligenceScore }) => clamp01(intelligenceScore / 100)),
      label: "INTELLIGENCE",
      detail: Option.match(catalog, {
        onNone: () => "Not assessed",
        onSome: ({ intelligenceScore }) => `${Math.round(intelligenceScore)}%`,
      }),
    },
    {
      value: Option.some(normalizeLocalModelRadarSpeed(performance.estimatedTokensPerSecond)),
      label: "SPEED",
      detail: performanceRangeSpeedLabel(model, "tok/s"),
    },
    {
      value: Option.some(speculationValue(model)),
      label: "SPECULATION",
      detail: speculation,
    },
    {
      value: Option.some(memoryEfficiency(assessment)),
      label: "MEMORY",
      detail: `${memoryFootprintLabel(assessment)} (${formatMemorySize(assessment.memory.totalRequiredBytes)})`,
    },
    {
      value: Option.match(catalog, {
        onNone: () => Option.map(bits, (value) => clamp01(value / 8)),
        onSome: ({ fidelityRank }) => Option.some(clamp01(fidelityRank / 100)),
      }),
      label: "ACCURACY",
      detail: Option.match(catalog, {
        onNone: () => `${discoveredAccuracyLabel(bits)} (${quantization})`,
        onSome: ({ fidelityRank }) => `${accuracyLabel(fidelityRank)} (${shortVariantLabel(model)})`,
      }),
    },
  ]
  return axes.every(({ value }) => Option.isNone(value) || Number.isFinite(value.value))
    ? Option.some(axes)
    : Option.none()
}
