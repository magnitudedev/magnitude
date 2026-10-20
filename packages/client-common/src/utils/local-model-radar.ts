import { Option } from "effect"
import {
  localModelServingState,
  type GenerationPerformanceEvidence,
  type GenerationPerformanceSamples,
  type LocalModel,
  type LocalModelPerformance,
  type LocalModelServingState,
  type MachineMeasurement,
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

const shortVariantLabel = (label: string): string =>
  label.match(/\b(?:IQ|Q)\d+(?:\.\d+)?\b/i)?.[0] ?? label.split(/[ _-]/, 1)[0] ?? label

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

/** A custom model's fidelity rank on the catalog's scale, where a variant's rank is ten per weight bit (Q4 is 40, Q8 is 80). */
const customFidelityRank = (model: LocalModel): Option.Option<number> =>
  Option.map(quantizationBits(model), (bits) => bits * 10)

const memoryFitLabel = (assessment: ModelAssessment): string => {
  const use = memoryUseRatio(assessment)
  if (use <= 0.2) return "Spacious"
  if (use <= 0.4) return "Roomy"
  if (use <= 0.6) return "Comfortable"
  if (use <= 0.8) return "Snug"
  return "Tight"
}

/** How to read an estimated speed: arithmetic over this machine's measurements, not a benchmark. */
export const localModelSpeedNote = "Rough estimate. Does not account for speculative decoding."

/** The explanation behind an estimated speed's info icon. */
export const localModelSpeedTooltip =
  "Estimated from measurements of your machine running Magnitude's own kernels. Actual performance will differ. Does not include speculative decoding."

/** The estimate whose context depth is nearest `contextTokens`. */
const nearestSample = (
  samples: GenerationPerformanceSamples,
  contextTokens: number,
): GenerationPerformanceEvidence => samples.reduce((closest, candidate) =>
  Math.abs(candidate.contextTokens - contextTokens) < Math.abs(closest.contextTokens - contextTokens)
    ? candidate
    : closest)

const measurementLabel = (measurement: MachineMeasurement): string =>
  measurement._tag === "Failed"
    ? `Measuring this machine failed; retrying at ${new Date(measurement.retryAt).toLocaleTimeString([], {
        hour: "numeric",
        minute: "2-digit",
      })}`
    : "Measuring this machine…"

export const performanceRangeSpeedLabel = (
  performance: LocalModelPerformance,
  contextLength: number
): string => {
  switch (performance._tag) {
    case "Pending": return measurementLabel(performance.measurement)
    case "Unavailable": return performance.failure.message
    case "Estimated": {
      const speeds = [
        nearestSample(performance.samples, Math.min(25_000, contextLength)),
        nearestSample(performance.samples, Math.min(75_000, contextLength)),
      ].map(({ estimatedTokensPerSecond }) => Math.round(estimatedTokensPerSecond))
      const lower = Math.min(...speeds)
      const upper = Math.max(...speeds)
      return lower === upper ? `~${lower} tok/s` : `~${lower}–${upper} tok/s`
    }
  }
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
  const catalog = model._tag === "Catalog" ? Option.some(model.catalogData) : Option.none()
  const speculation = Option.getOrElse(localModelSpeculativeMethodLabel(model), () => "None")
  const variant = String(model.presentation.variantLabel)
  const fidelityRank = Option.match(catalog, {
    onNone: () => customFidelityRank(model),
    onSome: ({ fidelityRank }) => Option.some(fidelityRank),
  })
  const axes: LocalModelRadarAxes = [
    {
      value: Option.map(catalog, ({ intelligence }) =>
        clamp01(intelligence / 100)
      ),
      label: "INTELLIGENCE",
      detail: Option.match(catalog, {
        onNone: () => "Not rated",
        onSome: ({ intelligence }) => `${intelligence}%`,
      }),
    },
    {
      // The estimate nearest the comparison depth; none while speed is pending or unavailable.
      value: assessment.performance._tag === "Estimated"
        ? Option.some(normalizeLocalModelRadarSpeed(
            nearestSample(assessment.performance.samples, comparisonContext).estimatedTokensPerSecond,
          ))
        : Option.none(),
      label: "SPEED",
      detail: performanceRangeSpeedLabel(assessment.performance, assessment.profile.contextLength),
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
      value: Option.map(fidelityRank, (rank) => clamp01(rank / 100)),
      label: "FIDELITY",
      detail: Option.match(fidelityRank, {
        onNone: () => `Not rated (${variant})`,
        onSome: (rank) => `${fidelityLabel(rank)} (${shortVariantLabel(variant)})`,
      }),
    },
  ]
  return axes.every(
    ({ value }) => Option.isNone(value) || Number.isFinite(value.value)
  )
    ? Option.some(axes)
    : Option.none()
}

/** How much of this computer's usable memory the model needs when it fits, from 0 to 1. */
export const localModelMemoryUse = (model: LocalModel): Option.Option<number> => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  return serving?._tag === "Assessed" && serving.assessment._tag === "Fits" ? Option.some(memoryUseRatio(serving.assessment)) : Option.none()
}

/** What a model card states beyond its fit: catalog intelligence and the estimated speed range. */
export interface LocalModelFacts {
  readonly intelligence: Option.Option<number>
  readonly speed: Option.Option<string>
}

export const localModelFacts = (model: LocalModel): LocalModelFacts => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  const fits = serving?._tag === "Assessed" && serving.assessment._tag === "Fits" ? serving.assessment : undefined
  return {
    intelligence: model._tag === "Catalog" ? Option.some(model.catalogData.intelligence) : Option.none(),
    speed: fits === undefined ? Option.none() : Option.some(performanceRangeSpeedLabel(fits.performance, fits.profile.contextLength)),
  }
}
