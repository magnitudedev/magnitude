import { Option } from "effect"
import { localModelSpeculativeMethodLabel } from "@magnitudedev/client-common"
import type { LocalModel } from "@magnitudedev/sdk"
import type { PentagonRadarAxes } from "../../components/pentagon-radar"
import { performanceRangeSpeedLabel } from "../local-inference/view-model"

const GIB = 1024 ** 3

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value))

export const normalizeCatalogRadarSpeed = (tokensPerSecond: number): number => {
  if (tokensPerSecond <= 30) return clamp01(tokensPerSecond / 30) * 0.5
  if (tokensPerSecond <= 100) return 0.5 + ((tokensPerSecond - 30) / 70) * 0.4
  if (tokensPerSecond >= 1_000) return 1
  return 0.9 + 0.1 * Math.log(tokensPerSecond / 100) / Math.log(10)
}

const compactMemorySize = (bytes: number): string => {
  const gigabytes = bytes / GIB
  return `${gigabytes.toFixed(gigabytes >= 10 ? 0 : 1)} GB`
}

type AssessedModel = Extract<LocalModel["servingState"], { readonly _tag: "Assessed" }>
type ModelAssessment = AssessedModel["assessment"]

const catalogRadarMemoryUseRatio = (assessment: ModelAssessment): number => {
  if (assessment._tag !== "Fits") return 1
  return assessment.memory.domains.reduce((highestUse, domain) => {
    const usableCapacityBytes = domain.capacityBytes - domain.compatibilityReserveBytes
    const use = usableCapacityBytes <= 0
      ? (domain.requiredBytes > 0 ? 1 : 0)
      : domain.requiredBytes / usableCapacityBytes
    return Math.max(highestUse, clamp01(use))
  }, 0)
}

const catalogRadarMemoryEfficiency = (assessment: ModelAssessment): number =>
  1 - catalogRadarMemoryUseRatio(assessment)

const catalogRadarSpeculationValue = (model: LocalModel): number => {
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

const memoryFootprintLabel = (assessment: ModelAssessment): string => {
  const use = catalogRadarMemoryUseRatio(assessment)
  if (use <= 0.2) return "Light"
  if (use <= 0.4) return "Moderate"
  if (use <= 0.6) return "Substantial"
  if (use <= 0.8) return "Heavy"
  return "Near capacity"
}

export const catalogRadarAxes = (model: LocalModel): Option.Option<PentagonRadarAxes> => {
  if (model.catalogMembershipState._tag !== "InCatalog"
    || model.servingState._tag !== "Assessed"
    || model.servingState.assessment._tag !== "Fits") return Option.none()

  const catalog = model.catalogMembershipState.catalogData
  const assessment = model.servingState.assessment
  const comparisonContext = Math.min(50_000, assessment.profile.contextLength)
  const performance = assessment.performance.find(({ contextTokens }) =>
    contextTokens === comparisonContext)
  if (performance === undefined) return Option.none()

  const speculation = Option.getOrElse(localModelSpeculativeMethodLabel(model), () => "None")
  const axes: PentagonRadarAxes = [
    {
      value: clamp01(catalog.intelligenceScore / 100),
      label: "INTELLIGENCE",
      detail: `${Math.round(catalog.intelligenceScore)}%`,
    },
    {
      value: normalizeCatalogRadarSpeed(performance.estimatedTokensPerSecond),
      label: "SPEED",
      detail: performanceRangeSpeedLabel(model, "tok/s"),
    },
    {
      value: catalogRadarSpeculationValue(model),
      label: "SPECULATION",
      detail: speculation,
    },
    {
      value: catalogRadarMemoryEfficiency(assessment),
      label: "MEMORY",
      detail: `${memoryFootprintLabel(assessment)} (${compactMemorySize(assessment.memory.totalRequiredBytes)})`,
    },
    {
      value: clamp01(catalog.fidelityRank / 100),
      label: "ACCURACY",
      detail: `${accuracyLabel(catalog.fidelityRank)} (${shortVariantLabel(model)})`,
    },
  ]
  return axes.every(({ value }) => Number.isFinite(value)) ? Option.some(axes) : Option.none()
}
