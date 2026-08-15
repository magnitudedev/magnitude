import { Option } from "effect"
import {
  RecommendationIdSchema,
  type FitsModelAssessment,
  type RecommendableModel,
  type ServingProfile,
} from "@magnitudedev/acn-protocol"
import { localCatalogProviderModelId } from "./local-provider-model-id"

const COMPARISON_CONTEXT_LENGTH = 50_000
const LINEAR_SPEED_UTILITY_LIMIT = 40
const SPEED_UTILITY_CEILING = 100

export type RecommendationIntent = RecommendationSelection["intent"]

export interface RecommendationWeights {
  readonly capability: number
  readonly speed: number
  readonly fidelity: number
  readonly memory: number
}

export const recommendationIntentWeights: Readonly<Record<
  RecommendationIntent,
  RecommendationWeights
>> = {
  balanced: { capability: 0.4, speed: 0.3, fidelity: 0.2, memory: 0.1 },
  smartest: { capability: 0.6, speed: 0.1, fidelity: 0.3, memory: 0 },
  fastest: { capability: 0.3, speed: 0.6, fidelity: 0.05, memory: 0.05 },
  lightweight: { capability: 0.1, speed: 0.1, fidelity: 0.1, memory: 0.7 },
}

const recommendationIntents = [
  "balanced",
  "smartest",
  "fastest",
  "lightweight",
] as const satisfies readonly RecommendationIntent[]

export interface RecommendationCandidate {
  readonly model: RecommendableModel
  readonly profile: ServingProfile
  readonly assessment: FitsModelAssessment
  readonly artifactId: string
  readonly catalogModelId: string
  readonly capabilityScore: number
  readonly fidelityRank: number
  readonly quantizationAware: boolean
  readonly estimatedLoadedBytes: number
  readonly stableCapacityBudgetBytes: number
}

export interface RecommendationSelection {
  readonly id: ReturnType<typeof RecommendationIdSchema.make>
  readonly configurationId: FitsModelAssessment["configurationId"]
  readonly recommendableModelId: string
  readonly displayName: string
  readonly intent: "balanced" | "smartest" | "fastest" | "lightweight"
  readonly explanation: string
}

const comparisonGenerationFor = (candidate: RecommendationCandidate) => {
  const comparisonContext = Math.min(
    COMPARISON_CONTEXT_LENGTH,
    candidate.profile.contextLength,
  )
  return candidate.assessment.performance.find(({ contextTokens }) =>
    contextTokens === comparisonContext)!
}

const stableCompare = (
  left: RecommendationCandidate,
  right: RecommendationCandidate,
): number =>
  String(left.assessment.configurationId).localeCompare(
    String(right.assessment.configurationId),
  )

const clamp = (value: number): number => Math.max(0, Math.min(1, value))

export const speedUtility = (tokensPerSecond: number): number => {
  const boundedSpeed = Math.max(0, Math.min(SPEED_UTILITY_CEILING, tokensPerSecond))
  const rawUtility = boundedSpeed <= LINEAR_SPEED_UTILITY_LIMIT
    ? boundedSpeed / LINEAR_SPEED_UTILITY_LIMIT
    : 1 + Math.log(boundedSpeed / LINEAR_SPEED_UTILITY_LIMIT)
  const maximumUtility = 1 + Math.log(
    SPEED_UTILITY_CEILING / LINEAR_SPEED_UTILITY_LIMIT,
  )
  return rawUtility / maximumUtility
}

export const recommendationUtility = (
  candidate: RecommendationCandidate,
  weights: RecommendationWeights,
): number => {
  const generation = comparisonGenerationFor(candidate)
  const capability = clamp(candidate.capabilityScore / 100)
  const speed = speedUtility(generation.estimatedTokensPerSecond)
  const memory = clamp(1 - candidate.estimatedLoadedBytes
    / Math.max(1, candidate.stableCapacityBudgetBytes))
  const fidelity = clamp(candidate.fidelityRank / 100)
  return capability ** weights.capability
    * speed ** weights.speed
    * fidelity ** weights.fidelity
    * memory ** weights.memory
}

export const intentUtility = (
  candidate: RecommendationCandidate,
  intent: RecommendationIntent,
): number => recommendationUtility(candidate, recommendationIntentWeights[intent])

export const balancedUtility = (candidate: RecommendationCandidate): number =>
  intentUtility(candidate, "balanced")

export const smartestUtility = (candidate: RecommendationCandidate): number =>
  intentUtility(candidate, "smartest")

const compareForIntent = (
  left: RecommendationCandidate,
  right: RecommendationCandidate,
  intent: RecommendationIntent,
): number => intentUtility(right, intent) - intentUtility(left, intent)
  || stableCompare(left, right)

/** Compatible catalog candidates in the same general-purpose order used by Balanced. */
export const rankCatalogCandidates = (
  input: readonly RecommendationCandidate[],
): readonly RecommendationCandidate[] =>
  [...input]
    .sort((left, right) => compareForIntent(left, right, "balanced"))

export const assembleRecommendationCatalogCandidates = (
  input: readonly RecommendationCandidate[],
  recommendations: readonly RecommendationSelection[],
): readonly RecommendationCandidate[] => {
  const candidatesByConfiguration = new Map(
    input.map((candidate) => [candidate.assessment.configurationId, candidate]),
  )
  const selected = recommendations.flatMap((recommendation) => {
    const candidate = candidatesByConfiguration.get(recommendation.configurationId)
    return candidate ? [candidate] : []
  }).filter((candidate, index, candidates) => candidates.findIndex(({ artifactId }) =>
    artifactId === candidate.artifactId) === index)
  const selectedArtifactIds = new Set(
    selected.map(({ artifactId }) => artifactId),
  )
  return [
    ...selected,
    ...rankCatalogCandidates(input)
      .filter((candidate) => !selectedArtifactIds.has(candidate.artifactId)),
  ]
}

const percentDifference = (value: number, reference: number): number => Math.round(
  Math.abs(value / Math.max(1, reference) - 1) * 100,
)

const wholeSpeed = (tokensPerSecond: number): number => Math.round(tokensPerSecond)

const qualitySummary = (candidate: RecommendationCandidate): string =>
  candidate.quantizationAware
    ? "retains very high output quality with minimal loss"
    : candidate.fidelityRank >= 75 ? "preserves nearly all of the original model's quality"
    : candidate.fidelityRank >= 55 ? "retains very high quality with minimal loss"
    : candidate.fidelityRank >= 45 ? "retains high quality with only minor loss"
    : "retains good quality with some possible loss"

const qualitySentence = (candidate: RecommendationCandidate): string => {
  const summary = qualitySummary(candidate)
  return `${summary.charAt(0).toUpperCase()}${summary.slice(1)}.`
}

const shorterContextTradeoff = (
  candidate: RecommendationCandidate,
  balanced: RecommendationCandidate,
): string => candidate.profile.contextLength < balanced.profile.contextLength
  ? candidate.profile.contextLength * 2 === balanced.profile.contextLength
    ? " It handles half as much context at once."
    : ` It handles ${percentDifference(candidate.profile.contextLength, balanced.profile.contextLength)}% less context at once.`
  : ""

const describeBalanced = (candidate: RecommendationCandidate): string => {
  const generation = comparisonGenerationFor(candidate)
  return `Best overall mix of coding ability, speed, and memory use. Runs at ~${wholeSpeed(generation.estimatedTokensPerSecond)} tok/s at ${Math.round(generation.contextTokens / 1_000)}K context, supports up to ${Math.round(candidate.profile.contextLength / 1_000)}K context, and ${qualitySummary(candidate)}.`
}

const describeSmartest = (
  candidate: RecommendationCandidate,
  balanced: RecommendationCandidate,
): string => {
  const generation = comparisonGenerationFor(candidate)
  const capabilityGain = candidate.capabilityScore - balanced.capabilityScore
  const reason = capabilityGain >= 5
    ? "Offers stronger performance on difficult coding tasks. "
    : ""
  const memoryChange = percentDifference(
    candidate.estimatedLoadedBytes,
    balanced.estimatedLoadedBytes,
  )
  const memoryTradeoff = memoryChange >= 5
    ? ` It uses about ${memoryChange}% more memory than Balanced.`
    : ""
  const speed = generation.estimatedTokensPerSecond
  const balancedSpeed = comparisonGenerationFor(balanced).estimatedTokensPerSecond
  const speedTradeoff = speed < balancedSpeed * 0.95
    ? ` It is about ${percentDifference(speed, balancedSpeed)}% slower than Balanced.`
    : " It runs at nearly the same speed as Balanced."
  return `${reason}${qualitySentence(candidate)}${memoryTradeoff}${speedTradeoff}`
}

const describeFastest = (
  candidate: RecommendationCandidate,
  balanced: RecommendationCandidate,
): string => {
  const generation = comparisonGenerationFor(candidate)
  const balancedSpeed = comparisonGenerationFor(balanced).estimatedTokensPerSecond
  const speedGain = generation.estimatedTokensPerSecond >= balancedSpeed * 1.05
    ? `About ${percentDifference(generation.estimatedTokensPerSecond, balancedSpeed)}% faster than Balanced, at ~${wholeSpeed(generation.estimatedTokensPerSecond)} tok/s at ${Math.round(generation.contextTokens / 1_000)}K context.`
    : `Prioritizes responsiveness at ~${wholeSpeed(generation.estimatedTokensPerSecond)} tok/s at ${Math.round(generation.contextTokens / 1_000)}K context.`
  const capabilityTradeoff = candidate.capabilityScore < balanced.capabilityScore
    ? " It is less capable on difficult coding tasks."
    : ""
  return `${speedGain}${capabilityTradeoff}${shorterContextTradeoff(candidate, balanced)} ${qualitySentence(candidate)}`
}

const describeLightweight = (
  candidate: RecommendationCandidate,
  balanced: RecommendationCandidate,
): string => {
  const generation = comparisonGenerationFor(candidate)
  const loadedMemoryReduction = Math.max(0, Math.round(
    (1 - candidate.estimatedLoadedBytes / balanced.estimatedLoadedBytes) * 100,
  ))
  const balancedSpeed = comparisonGenerationFor(balanced).estimatedTokensPerSecond
  const speedTradeoff = generation.estimatedTokensPerSecond < balancedSpeed * 0.95
    ? ` It is about ${percentDifference(generation.estimatedTokensPerSecond, balancedSpeed)}% slower than Balanced.`
    : generation.estimatedTokensPerSecond > balancedSpeed * 1.05
      ? ` It is about ${percentDifference(generation.estimatedTokensPerSecond, balancedSpeed)}% faster than Balanced.`
      : " It runs at about the same speed as Balanced."
  const capabilityTradeoff = candidate.capabilityScore < balanced.capabilityScore
    ? " It is less capable on difficult coding tasks."
    : ""
  const memory = Math.round(candidate.estimatedLoadedBytes / 1024 ** 3)
  const memorySummary = loadedMemoryReduction > 0
    ? `Uses ${loadedMemoryReduction}% less memory while loaded than Balanced`
    : `Prioritizes low loaded memory at about ${memory} GiB`
  return `${memorySummary} and is easier to keep on this machine.${capabilityTradeoff}${speedTradeoff}${shorterContextTradeoff(candidate, balanced)} ${qualitySentence(candidate)}`
}

const toRecommendation = (
  candidate: RecommendationCandidate,
  intent: RecommendationSelection["intent"],
  balanced: RecommendationCandidate,
): RecommendationSelection => ({
  id: RecommendationIdSchema.make(`${candidate.assessment.configurationId}:${intent}`),
  configurationId: candidate.assessment.configurationId,
  recommendableModelId: localCatalogProviderModelId(candidate.model),
  displayName: candidate.model.displayName,
  intent,
  explanation: intent === "balanced" ? describeBalanced(candidate)
    : intent === "smartest" ? describeSmartest(candidate, balanced)
    : intent === "fastest" ? describeFastest(candidate, balanced)
    : describeLightweight(candidate, balanced),
})

export const selectRecommendationPortfolio = (
  input: readonly RecommendationCandidate[],
): readonly RecommendationSelection[] => {
  if (input.length === 0) return []

  const balanced = [...input].sort((left, right) =>
    compareForIntent(left, right, "balanced")).at(0)
  if (!balanced) return []

  const selectedConfigurationIds = new Set<string>()
  return recommendationIntents.flatMap((intent): readonly RecommendationSelection[] => {
    const candidate = [...input]
      .sort((left, right) => compareForIntent(left, right, intent))
      .find(({ assessment }) => !selectedConfigurationIds.has(assessment.configurationId))
    if (!candidate) return []
    selectedConfigurationIds.add(candidate.assessment.configurationId)
    return [toRecommendation(candidate, intent, balanced)]
  })
}
