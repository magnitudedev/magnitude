import { Option } from "effect"
import {
  RecommendationIdSchema,
  type FitsModelAssessment,
  type Recommendation,
  type RecommendableModel,
  type ServingProfile,
} from "@magnitudedev/acn-protocol"

export const MINIMUM_EXPECTED_TOKENS_PER_SECOND = 8

const MAX_RECOMMENDATIONS = 4
const COMPARISON_CONTEXT_LENGTH = 50_000
const SPEED_UTILITY_CEILING = 60
const DOWNLOAD_UTILITY_BYTES = 16 * 1024 ** 3
const LIGHTWEIGHT_CAPACITY_RATIO = 0.2
const LIGHTWEIGHT_BALANCED_MEMORY_RATIO = 0.8

export interface RecommendationCandidate {
  readonly model: RecommendableModel
  readonly profile: ServingProfile
  readonly assessment: FitsModelAssessment
  readonly artifactId: string
  readonly checkpointId: string
  readonly capability:
    | {
        readonly score: number
        readonly provenance: string
      }
    | undefined
  readonly fidelityRank: number
  readonly quantizationAware: boolean
  readonly estimatedLoadedBytes: number
  readonly stableCapacityBudgetBytes: number
  readonly totalDownloadBytes: number
}

const fullContextGenerationFor = (candidate: RecommendationCandidate) =>
  candidate.assessment.performance.at(-1)!

const comparisonGenerationFor = (candidate: RecommendationCandidate) => {
  const comparisonContext = Math.min(
    COMPARISON_CONTEXT_LENGTH,
    candidate.profile.contextLength,
  )
  return candidate.assessment.performance.find(({ contextTokens }) =>
    contextTokens === comparisonContext)!
}

export const conservativeGenerationSpeed = (
  candidate: RecommendationCandidate,
): number => {
  const generation = comparisonGenerationFor(candidate)
  if (generation.confidence === "high") return generation.estimatedTokensPerSecond
  if (generation.confidence === "moderate") {
    return (generation.lowerTokensPerSecond + generation.estimatedTokensPerSecond) / 2
  }
  return generation.lowerTokensPerSecond
}

const capabilityScore = (candidate: RecommendationCandidate): number | undefined =>
  candidate.capability?.score

const measuredCapability = (candidate: RecommendationCandidate): boolean =>
  candidate.capability?.provenance === "measured_terminal_bench_2.1"

const meetsUsabilityFloor = (tokensPerSecond: number): boolean =>
  Math.round(tokensPerSecond * 10) / 10 >= MINIMUM_EXPECTED_TOKENS_PER_SECOND

const stableCompare = (
  left: RecommendationCandidate,
  right: RecommendationCandidate,
): number =>
  String(left.assessment.configurationId).localeCompare(
    String(right.assessment.configurationId),
  )

const usable = (candidate: RecommendationCandidate): boolean => {
  const generation = fullContextGenerationFor(candidate)
  return generation.contextTokens === candidate.profile.contextLength
    && meetsUsabilityFloor(generation.estimatedTokensPerSecond)
}

const preferScoredCandidates = (
  candidates: readonly RecommendationCandidate[],
): readonly RecommendationCandidate[] =>
  candidates.some((candidate) => capabilityScore(candidate) !== undefined)
    ? candidates.filter((candidate) => capabilityScore(candidate) !== undefined)
    : candidates

const capabilityFloor = (
  candidates: readonly RecommendationCandidate[],
  maximumLoss: number,
  minimumRetention: number,
): number => {
  const scores = candidates.flatMap((candidate) => {
    const score = capabilityScore(candidate)
    return score === undefined ? [] : [score]
  })
  if (scores.length === 0) return Number.NEGATIVE_INFINITY
  const ceiling = Math.max(...scores)
  return Math.max(ceiling - maximumLoss, ceiling * minimumRetention)
}

const withinCapabilityGuard = (
  candidates: readonly RecommendationCandidate[],
  maximumLoss: number,
  minimumRetention: number,
): readonly RecommendationCandidate[] => {
  const floor = capabilityFloor(candidates, maximumLoss, minimumRetention)
  return candidates.filter((candidate) => (capabilityScore(candidate) ?? floor) >= floor)
}

const clamp = (value: number): number => Math.max(0, Math.min(1, value))

const speedUtility = (tokensPerSecond: number): number => clamp(
  Math.log(tokensPerSecond / MINIMUM_EXPECTED_TOKENS_PER_SECOND)
    / Math.log(SPEED_UTILITY_CEILING / MINIMUM_EXPECTED_TOKENS_PER_SECOND),
)

export const balancedUtility = (candidate: RecommendationCandidate): number => {
  const generation = comparisonGenerationFor(candidate)
  const capability = (capabilityScore(candidate) ?? 50) / 100
  const memory = clamp(1 - candidate.estimatedLoadedBytes
    / Math.max(1, candidate.stableCapacityBudgetBytes))
  const download = DOWNLOAD_UTILITY_BYTES
    / (DOWNLOAD_UTILITY_BYTES + candidate.totalDownloadBytes)
  return capability * 0.4
    + speedUtility(generation.estimatedTokensPerSecond) * 0.3
    + memory * 0.15
    + clamp(candidate.fidelityRank / 100) * 0.1
    + download * 0.05
}

const compareBalanced = (
  left: RecommendationCandidate,
  right: RecommendationCandidate,
): number => balancedUtility(right) - balancedUtility(left)
  || stableCompare(left, right)

/** Compatible catalog candidates in the same general-purpose order used by Balanced. */
export const rankCatalogCandidates = (
  input: readonly RecommendationCandidate[],
): readonly RecommendationCandidate[] =>
  [...input]
    .sort((left, right) =>
      Number(usable(right)) - Number(usable(left))
        || (usable(left) && usable(right) ? compareBalanced(left, right) : 0)
        || (capabilityScore(right) ?? 0) - (capabilityScore(left) ?? 0)
        || right.fidelityRank - left.fidelityRank
        || stableCompare(left, right))

export const assembleRecommendationCatalogCandidates = (
  input: readonly RecommendationCandidate[],
  recommendations: readonly Recommendation[],
): readonly RecommendationCandidate[] => {
  const candidatesByConfiguration = new Map(
    input.map((candidate) => [candidate.assessment.configurationId, candidate]),
  )
  const selected = recommendations.flatMap((recommendation) => {
    const candidate = candidatesByConfiguration.get(recommendation.configuration.id)
    return candidate ? [candidate] : []
  })
  const selectedArtifactIds = new Set(
    selected.map(({ artifactId }) => artifactId),
  )
  return [
    ...selected,
    ...rankCatalogCandidates(input)
      .filter((candidate) => !selectedArtifactIds.has(candidate.artifactId)),
  ]
}

const compareBestQuality = (
  left: RecommendationCandidate,
  right: RecommendationCandidate,
): number => (capabilityScore(right) ?? 0) - (capabilityScore(left) ?? 0)
  || Number(measuredCapability(right)) - Number(measuredCapability(left))
  || right.fidelityRank - left.fidelityRank
  || right.profile.contextLength - left.profile.contextLength
  || comparisonGenerationFor(right).estimatedTokensPerSecond
    - comparisonGenerationFor(left).estimatedTokensPerSecond
  || stableCompare(left, right)

const sameConfiguration = (
  left: RecommendationCandidate,
  right: RecommendationCandidate,
): boolean => left.assessment.configurationId === right.assessment.configurationId

const materiallyLighterThan = (
  candidate: RecommendationCandidate,
  reference: RecommendationCandidate,
  ratio: number,
): boolean => candidate.estimatedLoadedBytes <= reference.estimatedLoadedBytes * ratio
  || candidate.totalDownloadBytes <= reference.totalDownloadBytes * ratio

const lightweightMemoryShare = (
  candidate: RecommendationCandidate,
): number => Math.max(
  0,
  ...candidate.assessment.memory.map((domain) => domain.requiredBytes
    / Math.max(1, domain.capacityBytes - domain.compatibilityReserveBytes)),
)

const withinLightweightMemoryTier = (
  candidate: RecommendationCandidate,
  balanced: RecommendationCandidate,
): boolean => lightweightMemoryShare(candidate) <= LIGHTWEIGHT_CAPACITY_RATIO
  && candidate.estimatedLoadedBytes
    <= balanced.estimatedLoadedBytes * LIGHTWEIGHT_BALANCED_MEMORY_RATIO

const compareLightweight = (
  left: RecommendationCandidate,
  right: RecommendationCandidate,
): number => (capabilityScore(right) ?? 0) - (capabilityScore(left) ?? 0)
  || left.estimatedLoadedBytes - right.estimatedLoadedBytes
  || right.fidelityRank - left.fidelityRank
  || conservativeGenerationSpeed(right) - conservativeGenerationSpeed(left)
  || left.totalDownloadBytes - right.totalDownloadBytes
  || stableCompare(left, right)

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

const describeBestQuality = (
  candidate: RecommendationCandidate,
  balanced: RecommendationCandidate,
): string => {
  const generation = comparisonGenerationFor(candidate)
  const capabilityGain = (capabilityScore(candidate) ?? 0)
    - (capabilityScore(balanced) ?? 0)
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
  const capabilityTradeoff = (capabilityScore(candidate) ?? 0)
      < (capabilityScore(balanced) ?? 0)
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
  const downloadReduction = Math.max(0, Math.round(
    (1 - candidate.totalDownloadBytes / balanced.totalDownloadBytes) * 100,
  ))
  const reduction = loadedMemoryReduction >= downloadReduction
    ? `${loadedMemoryReduction}% less memory while loaded`
    : `${downloadReduction}% less disk space`
  const balancedSpeed = comparisonGenerationFor(balanced).estimatedTokensPerSecond
  const speedTradeoff = generation.estimatedTokensPerSecond < balancedSpeed * 0.95
    ? ` It is about ${percentDifference(generation.estimatedTokensPerSecond, balancedSpeed)}% slower than Balanced.`
    : generation.estimatedTokensPerSecond > balancedSpeed * 1.05
      ? ` It is about ${percentDifference(generation.estimatedTokensPerSecond, balancedSpeed)}% faster than Balanced.`
      : " It runs at about the same speed as Balanced."
  const capabilityTradeoff = (capabilityScore(candidate) ?? 0)
      < (capabilityScore(balanced) ?? 0)
    ? " It is less capable on difficult coding tasks."
    : ""
  return `Uses ${reduction} than Balanced and is easier to keep on this machine.${capabilityTradeoff}${speedTradeoff}${shorterContextTradeoff(candidate, balanced)} ${qualitySentence(candidate)}`
}

const toRecommendation = (
  candidate: RecommendationCandidate,
  intent: Recommendation["intent"],
  balanced: RecommendationCandidate,
): Recommendation => ({
  id: RecommendationIdSchema.make(`${candidate.assessment.configurationId}:${intent}`),
  targetId: candidate.model.targetId,
  recommendableModelId: candidate.model.id,
  displayName: candidate.model.displayName,
  description: candidate.model.description,
  configuration: {
    id: candidate.assessment.configurationId,
    target: candidate.model.target,
    profile: candidate.profile,
  },
  assessment: candidate.assessment,
  intent,
  explanation: intent === "balanced" ? describeBalanced(candidate)
    : intent === "best_quality" ? describeBestQuality(candidate, balanced)
    : intent === "fastest" ? describeFastest(candidate, balanced)
    : describeLightweight(candidate, balanced),
})

const preferNewCheckpointWithin = (
  candidates: readonly RecommendationCandidate[],
  usedCheckpointIds: ReadonlySet<string>,
): RecommendationCandidate | undefined => candidates.find((candidate) =>
  !usedCheckpointIds.has(candidate.checkpointId)) ?? candidates.at(0)

export const selectRecommendationPortfolio = (
  input: readonly RecommendationCandidate[],
): readonly Recommendation[] => {
  const feasible = preferScoredCandidates(input.filter(usable))
  if (feasible.length === 0) return []

  const bestQuality = [...feasible].sort(compareBestQuality).at(0)
  if (!bestQuality) return []

  const balancedCapable = withinCapabilityGuard(feasible, 20, 0.7)
  const bestFidelity = Math.max(...balancedCapable.map(({ fidelityRank }) => fidelityRank))
  const balancedCandidates = balancedCapable
    .filter(({ fidelityRank }) => fidelityRank >= bestFidelity - 20)
    .sort(compareBalanced)
  let balanced = balancedCandidates.at(0)
  if (!balanced) return []

  if (sameConfiguration(balanced, bestQuality)) {
    const lighterSameCheckpoint = balancedCandidates
      .filter((candidate) => candidate.checkpointId === bestQuality.checkpointId
        && !sameConfiguration(candidate, bestQuality)
        && candidate.fidelityRank >= bestQuality.fidelityRank - 20
        && materiallyLighterThan(candidate, bestQuality, 0.9))
      .sort(compareBalanced)
      .at(0)
    if (lighterSameCheckpoint) balanced = lighterSameCheckpoint
  }

  const selected: Array<{
    readonly candidate: RecommendationCandidate
    readonly intent: Recommendation["intent"]
  }> = [{ candidate: balanced, intent: "balanced" }]
  const selectedConfigurations = new Set([balanced.assessment.configurationId])
  const usedCheckpointIds = new Set([balanced.checkpointId])

  const bestQualityCapabilityGain = (capabilityScore(bestQuality) ?? 0)
    - (capabilityScore(balanced) ?? 0)
  const bestQualityFidelityGain = bestQuality.fidelityRank - balanced.fidelityRank
  if (!selectedConfigurations.has(bestQuality.assessment.configurationId)
    && (bestQualityCapabilityGain >= 5 || bestQualityFidelityGain >= 10)) {
    selected.push({ candidate: bestQuality, intent: "best_quality" })
    selectedConfigurations.add(bestQuality.assessment.configurationId)
    usedCheckpointIds.add(bestQuality.checkpointId)
  }

  const fastestCapable = withinCapabilityGuard(feasible, 35, 0.5)
    .filter((candidate) =>
      !selectedConfigurations.has(candidate.assessment.configurationId))
    .sort((left, right) => conservativeGenerationSpeed(right)
      - conservativeGenerationSpeed(left)
      || stableCompare(left, right))
  const fastestRate = fastestCapable.length > 0
    ? Math.max(...fastestCapable.map(conservativeGenerationSpeed))
    : 0
  const nearFastest = fastestCapable.filter((candidate) =>
    conservativeGenerationSpeed(candidate) >= fastestRate * 0.9)
  const fastest = preferNewCheckpointWithin(nearFastest, usedCheckpointIds)
  if (fastest
    && conservativeGenerationSpeed(fastest)
      >= conservativeGenerationSpeed(balanced) * 1.15) {
    selected.push({ candidate: fastest, intent: "fastest" })
    selectedConfigurations.add(fastest.assessment.configurationId)
    usedCheckpointIds.add(fastest.checkpointId)
  }

  const lightweightCapable = feasible
    .filter((candidate) =>
      !selectedConfigurations.has(candidate.assessment.configurationId)
      && withinLightweightMemoryTier(candidate, balanced))
    .sort(compareLightweight)
  const lightweight = lightweightCapable.at(0)
  if (lightweight) {
    selected.push({ candidate: lightweight, intent: "lightweight" })
  }

  return selected.slice(0, MAX_RECOMMENDATIONS)
    .map(({ candidate, intent }) => toRecommendation(candidate, intent, balanced))
}
