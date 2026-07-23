import { Option } from "effect"
import type { RecipeBenchmarkEvidence } from "./types.js"
import type {
  ModelRecipeGenerationEstimate,
  ModelRecipeRecommendation,
  ModelRecipeRecommendationIntent,
} from "./schema.js"

export const RECOMMENDATION_POLICY_VERSION = "local-model-multicriteria-v6-user-facing-copy"
export const MINIMUM_EXPECTED_TOKENS_PER_SECOND = 15

const MAX_RECOMMENDATIONS = 4
const SPEED_UTILITY_CEILING = 60
const DOWNLOAD_UTILITY_BYTES = 16 * 1024 ** 3

export interface RecommendationCandidate {
  readonly value: ModelRecipeRecommendation
  readonly artifactId: string
  readonly checkpointId: string
  readonly capability: RecipeBenchmarkEvidence | undefined
  readonly fidelityRank: number
}

const generationFor = (
  candidate: RecommendationCandidate,
): ModelRecipeGenerationEstimate | undefined => Option.getOrUndefined(candidate.value.estimatedGeneration)

export const conservativeGenerationSpeed = (candidate: RecommendationCandidate): number => {
  const generation = generationFor(candidate)
  if (!generation) return 0
  if (generation.confidence === "high") return generation.expectedTokensPerSecond
  if (generation.confidence === "moderate") {
    return (generation.lowerTokensPerSecond + generation.expectedTokensPerSecond) / 2
  }
  return generation.lowerTokensPerSecond
}

const capabilityScore = (candidate: RecommendationCandidate): number | undefined =>
  candidate.capability?.score

const measuredCapability = (candidate: RecommendationCandidate): boolean =>
  candidate.capability?.provenance === "measured_terminal_bench_2.1"

const stableCompare = (left: RecommendationCandidate, right: RecommendationCandidate): number =>
  String(left.value.configurationId).localeCompare(String(right.value.configurationId))

const usable = (candidate: RecommendationCandidate): boolean => {
  const generation = generationFor(candidate)
  return generation !== undefined
    && generation.contextTokens === candidate.value.contextWindow
    && generation.expectedTokensPerSecond >= MINIMUM_EXPECTED_TOKENS_PER_SECOND
    && (candidate.value.contextWindow === 100_000 || candidate.value.contextWindow === 200_000)
}

const preferScoredCandidates = (
  candidates: readonly RecommendationCandidate[],
): readonly RecommendationCandidate[] => candidates.some((candidate) => capabilityScore(candidate) !== undefined)
  ? candidates.filter((candidate) => capabilityScore(candidate) !== undefined)
  : candidates

const collapseLargestContext = (
  candidates: readonly RecommendationCandidate[],
): readonly RecommendationCandidate[] => {
  const byArtifact = new Map<string, RecommendationCandidate>()
  for (const candidate of candidates) {
    const current = byArtifact.get(candidate.artifactId)
    if (!current
      || candidate.value.contextWindow > current.value.contextWindow
      || (candidate.value.contextWindow === current.value.contextWindow
        && stableCompare(candidate, current) < 0)) {
      byArtifact.set(candidate.artifactId, candidate)
    }
  }
  return [...byArtifact.values()]
}

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
  const generation = generationFor(candidate)
  if (!generation) return Number.NEGATIVE_INFINITY
  const capability = (capabilityScore(candidate) ?? 50) / 100
  const memory = clamp(1 - candidate.value.estimatedRuntimeBytes
    / Math.max(1, candidate.value.stableCapacityBudgetBytes))
  const download = DOWNLOAD_UTILITY_BYTES
    / (DOWNLOAD_UTILITY_BYTES + candidate.value.totalDownloadBytes)
  return capability * 0.4
    + speedUtility(generation.expectedTokensPerSecond) * 0.3
    + memory * 0.15
    + clamp(candidate.fidelityRank / 100) * 0.1
    + download * 0.05
}

const compareBalanced = (
  left: RecommendationCandidate,
  right: RecommendationCandidate,
): number => balancedUtility(right) - balancedUtility(left)
  || stableCompare(left, right)

const compareBestQuality = (
  left: RecommendationCandidate,
  right: RecommendationCandidate,
): number => (capabilityScore(right) ?? 0) - (capabilityScore(left) ?? 0)
  || Number(measuredCapability(right)) - Number(measuredCapability(left))
  || right.fidelityRank - left.fidelityRank
  || right.value.contextWindow - left.value.contextWindow
  || (generationFor(right)?.expectedTokensPerSecond ?? 0)
    - (generationFor(left)?.expectedTokensPerSecond ?? 0)
  || stableCompare(left, right)

const sameConfiguration = (
  left: RecommendationCandidate,
  right: RecommendationCandidate,
): boolean => left.value.configurationId === right.value.configurationId

const materiallyLighterThan = (
  candidate: RecommendationCandidate,
  reference: RecommendationCandidate,
  ratio: number,
): boolean => candidate.value.estimatedRuntimeBytes <= reference.value.estimatedRuntimeBytes * ratio
  || candidate.value.totalDownloadBytes <= reference.value.totalDownloadBytes * ratio

const percentDifference = (value: number, reference: number): number => Math.round(
  Math.abs(value / Math.max(1, reference) - 1) * 100,
)

const qualitySummary = (candidate: RecommendationCandidate): string =>
  candidate.value.quantization.quantAwareCheckpoint
    ? "retains very high output quality with minimal loss"
    : candidate.fidelityRank >= 75 ? "preserves nearly all of the original model's quality"
    : candidate.fidelityRank >= 55 ? "retains very high quality with minimal loss"
    : candidate.fidelityRank >= 45 ? "retains high quality with only minor loss"
    : "uses substantial compression with some possible quality loss"

const qualitySentence = (candidate: RecommendationCandidate): string => {
  const summary = qualitySummary(candidate)
  return `${summary.charAt(0).toUpperCase()}${summary.slice(1)}.`
}

const shorterContextTradeoff = (
  candidate: RecommendationCandidate,
  balanced: RecommendationCandidate,
): string => candidate.value.contextWindow < balanced.value.contextWindow
  ? candidate.value.contextWindow * 2 === balanced.value.contextWindow
    ? " It handles half as much code and conversation history at once."
    : ` It handles ${percentDifference(candidate.value.contextWindow, balanced.value.contextWindow)}% less code and conversation history at once.`
  : ""

const describeBalanced = (candidate: RecommendationCandidate): string => {
  const generation = generationFor(candidate)!
  return `Best overall mix of coding ability, speed, and memory use. Runs at about ${generation.expectedTokensPerSecond.toFixed(1)} tokens/sec with ${Math.round(candidate.value.contextWindow / 1_000)}K context and ${qualitySummary(candidate)}.`
}

const describeBestQuality = (
  candidate: RecommendationCandidate,
  balanced: RecommendationCandidate,
): string => {
  const generation = generationFor(candidate)!
  const capabilityGain = (capabilityScore(candidate) ?? 0) - (capabilityScore(balanced) ?? 0)
  const reason = capabilityGain >= 5
    ? "Offers stronger performance on difficult coding tasks. "
    : ""
  const memoryChange = percentDifference(
    candidate.value.estimatedRuntimeBytes,
    balanced.value.estimatedRuntimeBytes,
  )
  const memoryTradeoff = memoryChange >= 5
    ? ` It uses about ${memoryChange}% more memory than Balanced.`
    : ""
  const speed = generation.expectedTokensPerSecond
  const balancedSpeed = generationFor(balanced)!.expectedTokensPerSecond
  const speedTradeoff = speed < balancedSpeed * 0.95
    ? ` It is about ${percentDifference(speed, balancedSpeed)}% slower.`
    : " It runs at nearly the same speed as Balanced."
  return `${reason}${qualitySentence(candidate)}${memoryTradeoff}${speedTradeoff}`
}

const describeFastest = (
  candidate: RecommendationCandidate,
  balanced: RecommendationCandidate,
): string => {
  const generation = generationFor(candidate)!
  const balancedSpeed = generationFor(balanced)!.expectedTokensPerSecond
  const speedGain = generation.expectedTokensPerSecond >= balancedSpeed * 1.05
    ? `About ${percentDifference(generation.expectedTokensPerSecond, balancedSpeed)}% faster than Balanced, at roughly ${generation.expectedTokensPerSecond.toFixed(1)} tokens/sec.`
    : `Prioritizes responsiveness at roughly ${generation.expectedTokensPerSecond.toFixed(1)} tokens/sec.`
  const capabilityTradeoff = (capabilityScore(candidate) ?? 0) < (capabilityScore(balanced) ?? 0)
    ? " It is less capable on difficult coding tasks."
    : ""
  return `${speedGain}${capabilityTradeoff}${shorterContextTradeoff(candidate, balanced)} ${qualitySentence(candidate)}`
}

const describeLightweight = (
  candidate: RecommendationCandidate,
  balanced: RecommendationCandidate,
): string => {
  const generation = generationFor(candidate)!
  const runtimeReduction = Math.max(0, Math.round(
    (1 - candidate.value.estimatedRuntimeBytes / balanced.value.estimatedRuntimeBytes) * 100,
  ))
  const downloadReduction = Math.max(0, Math.round(
    (1 - candidate.value.totalDownloadBytes / balanced.value.totalDownloadBytes) * 100,
  ))
  const reduction = runtimeReduction >= downloadReduction
    ? `${runtimeReduction}% less runtime memory`
    : `${downloadReduction}% less disk space`
  const balancedSpeed = generationFor(balanced)!.expectedTokensPerSecond
  const speedTradeoff = generation.expectedTokensPerSecond < balancedSpeed * 0.95
    ? ` It is about ${percentDifference(generation.expectedTokensPerSecond, balancedSpeed)}% slower on this machine.`
    : generation.expectedTokensPerSecond > balancedSpeed * 1.05
      ? ` It is about ${percentDifference(generation.expectedTokensPerSecond, balancedSpeed)}% faster on this machine.`
      : " It runs at about the same speed on this machine."
  const capabilityTradeoff = (capabilityScore(candidate) ?? 0) < (capabilityScore(balanced) ?? 0)
    ? " It is less capable on difficult coding tasks."
    : ""
  return `Uses ${reduction} than Balanced and is easier to keep on this machine.${capabilityTradeoff}${speedTradeoff}${shorterContextTradeoff(candidate, balanced)} ${qualitySentence(candidate)}`
}

const withIntent = (
  candidate: RecommendationCandidate,
  intent: ModelRecipeRecommendationIntent,
  balanced: RecommendationCandidate,
): ModelRecipeRecommendation => ({
  ...candidate.value,
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
): readonly ModelRecipeRecommendation[] => {
  const feasible = preferScoredCandidates(input.filter(usable))
  if (feasible.length === 0) return []
  const largestContexts = collapseLargestContext(feasible)

  const bestQuality = [...largestContexts].sort(compareBestQuality).at(0)
  if (!bestQuality) return []

  const balancedCapable = withinCapabilityGuard(largestContexts, 20, 0.7)
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
    readonly intent: ModelRecipeRecommendationIntent
  }> = [{ candidate: balanced, intent: "balanced" }]
  const selectedConfigurations = new Set([balanced.value.configurationId])
  const usedCheckpointIds = new Set([balanced.checkpointId])

  const bestQualityCapabilityGain = (capabilityScore(bestQuality) ?? 0)
    - (capabilityScore(balanced) ?? 0)
  const bestQualityFidelityGain = bestQuality.fidelityRank - balanced.fidelityRank
  if (!selectedConfigurations.has(bestQuality.value.configurationId)
    && (bestQualityCapabilityGain >= 5 || bestQualityFidelityGain >= 10)) {
    selected.push({ candidate: bestQuality, intent: "best_quality" })
    selectedConfigurations.add(bestQuality.value.configurationId)
    usedCheckpointIds.add(bestQuality.checkpointId)
  }

  const fastestCapable = withinCapabilityGuard(feasible, 25, 0.6)
    .filter((candidate) => !selectedConfigurations.has(candidate.value.configurationId))
    .sort((left, right) => conservativeGenerationSpeed(right) - conservativeGenerationSpeed(left)
      || stableCompare(left, right))
  const fastestRate = fastestCapable.length > 0
    ? Math.max(...fastestCapable.map(conservativeGenerationSpeed))
    : 0
  const nearFastest = fastestCapable.filter((candidate) =>
    conservativeGenerationSpeed(candidate) >= fastestRate * 0.9)
  const fastest = preferNewCheckpointWithin(nearFastest, usedCheckpointIds)
  if (fastest
    && conservativeGenerationSpeed(fastest) >= conservativeGenerationSpeed(balanced) * 1.15) {
    selected.push({ candidate: fastest, intent: "fastest" })
    selectedConfigurations.add(fastest.value.configurationId)
    usedCheckpointIds.add(fastest.checkpointId)
  }

  const lightweightCapable = withinCapabilityGuard(largestContexts, 25, 0.45)
    .filter((candidate) => !selectedConfigurations.has(candidate.value.configurationId))
    .sort((left, right) => left.value.estimatedRuntimeBytes - right.value.estimatedRuntimeBytes
      || left.value.totalDownloadBytes - right.value.totalDownloadBytes
      || (capabilityScore(right) ?? 0) - (capabilityScore(left) ?? 0)
      || right.fidelityRank - left.fidelityRank
      || stableCompare(left, right))
  const lightestRuntime = lightweightCapable.at(0)?.value.estimatedRuntimeBytes ?? 0
  const nearLightest = lightweightCapable.filter((candidate) =>
    candidate.value.estimatedRuntimeBytes <= lightestRuntime * 1.15)
  const lightweight = preferNewCheckpointWithin(nearLightest, usedCheckpointIds)
  if (lightweight && materiallyLighterThan(lightweight, balanced, 0.8)) {
    selected.push({ candidate: lightweight, intent: "lightweight" })
  }

  return selected.slice(0, MAX_RECOMMENDATIONS)
    .map(({ candidate, intent }) => withIntent(candidate, intent, balanced))
}
