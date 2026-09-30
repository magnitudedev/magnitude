import { Option } from "effect"
import {
  localModelServingState,
  type GenerationPerformanceSamples,
  type LocalModel,
  type LocalModelServingState,
} from "@magnitudedev/sdk"
import { formatMemorySize } from "./format-bytes"
import { localModelSpeculativeMethodLabel } from "./model-presentation"

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value))

export interface LocalModelRadarAxis {
  readonly value: Option.Option<number>
  readonly label: string
  readonly detail: string
}

export type LocalModelRadarAxes = readonly [
  LocalModelRadarAxis,
  LocalModelRadarAxis,
  LocalModelRadarAxis,
  LocalModelRadarAxis,
  LocalModelRadarAxis,
]

export const normalizeLocalModelRadarSpeed = (tokensPerSecond: number): number => {
  if (tokensPerSecond <= 30) return clamp01(tokensPerSecond / 30) * 0.5
  if (tokensPerSecond <= 100) return 0.5 + ((tokensPerSecond - 30) / 70) * 0.4
  if (tokensPerSecond >= 1_000) return 1
  return 0.9 + (0.1 * Math.log(tokensPerSecond / 100)) / Math.log(10)
}

type AssessedModel = Extract<LocalModelServingState, {
  readonly _tag: "Assessed"
}>
type ModelAssessment = AssessedModel["assessment"]

const memoryUseRatio = (assessment: ModelAssessment): number => {
  if (assessment._tag !== "Fits") return 1
  return assessment.memory.domains.reduce((highestUse, domain) => {
    const usableCapacityBytes =
      domain.capacityBytes - domain.compatibilityReserveBytes
    const use =
      usableCapacityBytes <= 0
        ? domain.requiredBytes > 0
          ? 1
          : 0
        : domain.requiredBytes / usableCapacityBytes
    return Math.max(highestUse, clamp01(use))
  }, 0)
}

const speculationValue = (model: LocalModel): number => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  if (serving?._tag !== "Assessed") return 0
  return Option.match(serving.speculativeMethod, {
    onNone: () => 0,
    onSome: (method) => {
      switch (method._tag) {
        case "Mtp": return 1 / 4
        case "DFlash": return 2 / 4
        case "DSpark": return 3 / 4
        case "DFlash2": return 1
      }
    },
  })
}

const fidelityLabel = (rank: number): string =>
  rank >= 70 ? "Very high" : rank >= 45 ? "High" : rank >= 30 ? "Medium" : "Reduced"

const shortVariantLabel = (model: LocalModel): string =>
  String(model.presentation.variantLabel).match(/\b(?:IQ|Q)\d+(?:\.\d+)?\b/i)?.[0] ??
  String(model.presentation.variantLabel).split(/[ _-]/, 1)[0] ??
  String(model.presentation.variantLabel)

const quantizationBits = (model: LocalModel): Option.Option<number> => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  if (serving?._tag !== "Assessed") return Option.none()
  for (const candidate of [
    serving.metadata.quantization,
    serving.metadata.quantizationName,
  ]) {
    const bits = candidate.match(/(?:IQ|Q)?(\d+(?:\.\d+)?)\s*(?:[- ]?bit)?/i)?.[1]
    if (bits !== undefined) return Option.some(Number(bits))
  }
  return Option.none()
}

const discoveredFidelityLabel = (bits: Option.Option<number>): string =>
  Option.match(bits, {
    onNone: () => "Not assessed",
    onSome: (value) =>
      value >= 8 ? "Very high" : value >= 5 ? "High" : value >= 4 ? "Medium" : "Reduced",
  })

const memoryFitLabel = (assessment: ModelAssessment): string => {
  const use = memoryUseRatio(assessment)
  if (use <= 0.2) return "Spacious"
  if (use <= 0.4) return "Roomy"
  if (use <= 0.6) return "Comfortable"
  if (use <= 0.8) return "Snug"
  return "Tight"
}

/** How to read an estimated speed: arithmetic over bandwidth, not a benchmark. */
export const localModelSpeedNote = "Rough estimate. Does not account for speculative decoding."

/** The explanation behind an estimated speed's info icon. */
export const localModelSpeedTooltip =
  "Estimated from your machine's memory bandwidth. Actual performance will differ. Does not include speculative decoding."

export const performanceRangeSpeedLabel = (
  samples: GenerationPerformanceSamples,
  contextLength: number
): string => {
  const lowerContext = Math.min(25_000, contextLength)
  const upperContext = Math.min(75_000, contextLength)
  const lowerSample = samples.find(
    ({ contextTokens }) => contextTokens === lowerContext
  )
  const upperSample = samples.find(
    ({ contextTokens }) => contextTokens === upperContext
  )
  if (lowerSample === undefined || upperSample === undefined) return "Not assessed"
  const lower = Math.min(
    lowerSample.estimatedTokensPerSecond,
    upperSample.estimatedTokensPerSecond
  )
  const upper = Math.max(
    lowerSample.estimatedTokensPerSecond,
    upperSample.estimatedTokensPerSecond
  )
  return Math.round(lower) === Math.round(upper)
    ? `~${Math.round(lower)} tok/s`
    : `~${Math.round(lower)}–${Math.round(upper)} tok/s`
}

export const localModelRadarAxes = (
  model: LocalModel
): Option.Option<LocalModelRadarAxes> => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  if (serving?._tag !== "Assessed" || serving.assessment._tag !== "Fits") {
    return Option.none()
  }

  const assessment = serving.assessment
  const comparisonContext = Math.min(50_000, assessment.profile.contextLength)
  const samples = assessment.performance
  const catalog = model._tag === "Catalog" ? Option.some(model.catalogData) : Option.none()
  const speculation = Option.getOrElse(localModelSpeculativeMethodLabel(model), () => "None")
  const bits = quantizationBits(model)
  const quantization = serving.metadata.quantization
  const axes: LocalModelRadarAxes = [
    {
      value: Option.map(catalog, ({ intelligence }) =>
        clamp01(intelligence / 100)
      ),
      label: "INTELLIGENCE",
      detail: Option.match(catalog, {
        onNone: () => "Not assessed",
        onSome: ({ intelligence }) => `${intelligence}%`,
      }),
    },
    {
      value: Option.some(normalizeLocalModelRadarSpeed(samples.reduce((closest, candidate) =>
        Math.abs(candidate.contextTokens - comparisonContext) <
        Math.abs(closest.contextTokens - comparisonContext)
          ? candidate
          : closest
      ).estimatedTokensPerSecond)),
      label: "SPEED",
      detail: performanceRangeSpeedLabel(samples, assessment.profile.contextLength),
    },
    {
      value: Option.some(speculationValue(model)),
      label: "SPECULATION",
      detail: speculation,
    },
    {
      value: Option.some(1 - memoryUseRatio(assessment)),
      label: "FIT",
      detail: `${memoryFitLabel(assessment)} (${formatMemorySize(
        assessment.memory.totalRequiredBytes
      )})`,
    },
    {
      value: Option.match(catalog, {
        onNone: () => Option.map(bits, (value) => clamp01(value / 8)),
        onSome: ({ fidelityRank }) => Option.some(clamp01(fidelityRank / 100)),
      }),
      label: "FIDELITY",
      detail: Option.match(catalog, {
        onNone: () => `${discoveredFidelityLabel(bits)} (${quantization})`,
        onSome: ({ fidelityRank }) =>
          `${fidelityLabel(fidelityRank)} (${shortVariantLabel(model)})`,
      }),
    },
  ]
  return axes.every(
    ({ value }) => Option.isNone(value) || Number.isFinite(value.value)
  )
    ? Option.some(axes)
    : Option.none()
}
