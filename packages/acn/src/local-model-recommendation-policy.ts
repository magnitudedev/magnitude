import { Option } from "effect"
import {
  RecommendationIdSchema,
  type FitsOfferingAssessment,
  type Recommendation,
  type RecommendableModel,
  type ServingProfile,
} from "@magnitudedev/protocol"

export const RECOMMENDATION_POLICY_VERSION =
  "local-model-multicriteria-v7-workstation-intents"
export const MINIMUM_EXPECTED_TOKENS_PER_SECOND = 15

const MAX_RECOMMENDATIONS = 4
const SPEED_UTILITY_CEILING = 60
const DOWNLOAD_UTILITY_BYTES = 16 * 1024 ** 3

export interface RecommendationCandidate {
  readonly model: RecommendableModel
  readonly profile: ServingProfile
  readonly assessment: FitsOfferingAssessment
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
  readonly estimatedRuntimeBytes: number
  readonly stableCapacityBudgetBytes: number
  readonly totalDownloadBytes: number
}

const generationFor = (candidate: RecommendationCandidate) =>
  Option.getOrUndefined(candidate.assessment.performance)

export const conservativeGenerationSpeed = (
  candidate: RecommendationCandidate,
): number => {
  const generation = generationFor(candidate)
  if (!generation) return 0
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

const stableCompare = (
  left: RecommendationCandidate,
  right: RecommendationCandidate,
): number =>
  String(left.assessment.configurationId).localeCompare(
    String(right.assessment.configurationId),
  )

const usable = (candidate: RecommendationCandidate): boolean => {
  const generation = generationFor(candidate)
  return generation !== undefined
    && generation.contextTokens === candidate.profile.contextLength
    && generation.estimatedTokensPerSecond >= MINIMUM_EXPECTED_TOKENS_PER_SECOND
    && (candidate.profile.contextLength === 100_000
      || candidate.profile.contextLength === 200_000)
}

const preferScoredCandidates = (
  candidates: readonly RecommendationCandidate[],
): readonly RecommendationCandidate[] =>
  candidates.some((candidate) => capabilityScore(candidate) !== undefined)
    ? candidates.filter((candidate) => capabilityScore(candidate) !== undefined)
    : candidates

const collapseLargestContext = (
  candidates: readonly RecommendationCandidate[],
): readonly RecommendationCandidate[] => {
  const byArtifact = new Map<string, RecommendationCandidate>()
  for (const candidate of candidates) {
    const current = byArtifact.get(candidate.artifactId)
    if (
      !current
      || candidate.profile.contextLength > current.profile.contextLength
      || (candidate.profile.contextLength === current.profile.contextLength
        && stableCompare(candidate, current) < 0)
    ) {
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

const lightweightCapabilityGuard = (
  candidates: readonly RecommendationCandidate[],
): readonly RecommendationCandidate[] => {
  const guarded = withinCapabilityGuard(candidates, 45, 0.3)
    .filter((candidate) => (capabilityScore(candidate) ?? 20) >= 20)
  return guarded.length > 0 ? guarded : candidates
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
  const memory = clamp(1 - candidate.estimatedRuntimeBytes
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

const compareBestQuality = (
  left: RecommendationCandidate,
  right: RecommendationCandidate,
): number => (capabilityScore(right) ?? 0) - (capabilityScore(left) ?? 0)
  || Number(measuredCapability(right)) - Number(measuredCapability(left))
  || right.fidelityRank - left.fidelityRank
  || right.profile.contextLength - left.profile.contextLength
  || (generationFor(right)?.estimatedTokensPerSecond ?? 0)
    - (generationFor(left)?.estimatedTokensPerSecond ?? 0)
  || stableCompare(left, right)

const sameConfiguration = (
  left: RecommendationCandidate,
  right: RecommendationCandidate,
): boolean => left.assessment.configurationId === right.assessment.configurationId

const materiallyLighterThan = (
  candidate: RecommendationCandidate,
  reference: RecommendationCandidate,
  ratio: number,
): boolean => candidate.estimatedRuntimeBytes <= reference.estimatedRuntimeBytes * ratio
  || candidate.totalDownloadBytes <= reference.totalDownloadBytes * ratio

const percentDifference = (value: number, reference: number): number => Math.round(
  Math.abs(value / Math.max(1, reference) - 1) * 100,
)

const qualitySummary = (candidate: RecommendationCandidate): string =>
  candidate.quantizationAware
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
): string => candidate.profile.contextLength < balanced.profile.contextLength
  ? candidate.profile.contextLength * 2 === balanced.profile.contextLength
    ? " It handles half as much code and conversation history at once."
    : ` It handles ${percentDifference(candidate.profile.contextLength, balanced.profile.contextLength)}% less code and conversation history at once.`
  : ""

const describeBalanced = (candidate: RecommendationCandidate): string => {
  const generation = generationFor(candidate)!
  return `Best overall mix of coding ability, speed, and memory use. Runs at about ${generation.estimatedTokensPerSecond.toFixed(1)} tokens/sec with ${Math.round(candidate.profile.contextLength / 1_000)}K context and ${qualitySummary(candidate)}.`
}

const describeBestQuality = (
  candidate: RecommendationCandidate,
  balanced: RecommendationCandidate,
): string => {
  const generation = generationFor(candidate)!
  const capabilityGain = (capabilityScore(candidate) ?? 0)
    - (capabilityScore(balanced) ?? 0)
  const reason = capabilityGain >= 5
    ? "Offers stronger performance on difficult coding tasks. "
    : ""
  const memoryChange = percentDifference(
    candidate.estimatedRuntimeBytes,
    balanced.estimatedRuntimeBytes,
  )
  const memoryTradeoff = memoryChange >= 5
    ? ` It uses about ${memoryChange}% more memory than Balanced.`
    : ""
  const speed = generation.estimatedTokensPerSecond
  const balancedSpeed = generationFor(balanced)!.estimatedTokensPerSecond
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
  const balancedSpeed = generationFor(balanced)!.estimatedTokensPerSecond
  const speedGain = generation.estimatedTokensPerSecond >= balancedSpeed * 1.05
    ? `About ${percentDifference(generation.estimatedTokensPerSecond, balancedSpeed)}% faster than Balanced, at roughly ${generation.estimatedTokensPerSecond.toFixed(1)} tokens/sec.`
    : `Prioritizes responsiveness at roughly ${generation.estimatedTokensPerSecond.toFixed(1)} tokens/sec.`
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
  const generation = generationFor(candidate)!
  const runtimeReduction = Math.max(0, Math.round(
    (1 - candidate.estimatedRuntimeBytes / balanced.estimatedRuntimeBytes) * 100,
  ))
  const downloadReduction = Math.max(0, Math.round(
    (1 - candidate.totalDownloadBytes / balanced.totalDownloadBytes) * 100,
  ))
  const reduction = runtimeReduction >= downloadReduction
    ? `${runtimeReduction}% less runtime memory`
    : `${downloadReduction}% less disk space`
  const balancedSpeed = generationFor(balanced)!.estimatedTokensPerSecond
  const speedTradeoff = generation.estimatedTokensPerSecond < balancedSpeed * 0.95
    ? ` It is about ${percentDifference(generation.estimatedTokensPerSecond, balancedSpeed)}% slower on this machine.`
    : generation.estimatedTokensPerSecond > balancedSpeed * 1.05
      ? ` It is about ${percentDifference(generation.estimatedTokensPerSecond, balancedSpeed)}% faster on this machine.`
      : " It runs at about the same speed on this machine."
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
  modelId: candidate.model.targetId,
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

  const lightweightCapable = lightweightCapabilityGuard(largestContexts)
    .filter((candidate) =>
      !selectedConfigurations.has(candidate.assessment.configurationId))
    .sort((left, right) => left.estimatedRuntimeBytes - right.estimatedRuntimeBytes
      || left.totalDownloadBytes - right.totalDownloadBytes
      || (capabilityScore(right) ?? 0) - (capabilityScore(left) ?? 0)
      || right.fidelityRank - left.fidelityRank
      || stableCompare(left, right))
  const lightestRuntime = lightweightCapable.at(0)?.estimatedRuntimeBytes ?? 0
  const nearLightest = lightweightCapable.filter((candidate) =>
    candidate.estimatedRuntimeBytes <= lightestRuntime * 1.15)
  const lightweight = preferNewCheckpointWithin(nearLightest, usedCheckpointIds)
  if (lightweight && materiallyLighterThan(lightweight, balanced, 0.8)) {
    selected.push({ candidate: lightweight, intent: "lightweight" })
  }

  return selected.slice(0, MAX_RECOMMENDATIONS)
    .map(({ candidate, intent }) => toRecommendation(candidate, intent, balanced))
}
