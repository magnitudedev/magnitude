import { Option } from "effect"
import type {
  GenerationPerformanceSamples,
  LocalModelRankingScores,
  ServingProfile,
} from "@magnitudedev/acn-protocol"

const COMPARISON_CONTEXT_LENGTH = 50_000
const LINEAR_SPEED_SCORE_LIMIT = 40
const SPEED_SCORE_CEILING = 100

export interface LocalModelRankingCandidate {
  readonly intelligenceScore: number
  readonly fidelityRank: number
  readonly profile: ServingProfile
  readonly performance: GenerationPerformanceSamples
}

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value))

export const normalizedModelSpeedScore = (tokensPerSecond: number): number => {
  const boundedSpeed = Math.max(0, Math.min(SPEED_SCORE_CEILING, tokensPerSecond))
  const rawScore = boundedSpeed <= LINEAR_SPEED_SCORE_LIMIT
    ? boundedSpeed / LINEAR_SPEED_SCORE_LIMIT
    : 1 + Math.log(boundedSpeed / LINEAR_SPEED_SCORE_LIMIT)
  const maximumScore = 1 + Math.log(SPEED_SCORE_CEILING / LINEAR_SPEED_SCORE_LIMIT)
  return rawScore / maximumScore
}

export const modelRankingScores = (
  candidate: LocalModelRankingCandidate,
): Option.Option<LocalModelRankingScores> => {
  const comparisonContext = Math.min(
    COMPARISON_CONTEXT_LENGTH,
    candidate.profile.contextLength,
  )
  const generation = candidate.performance.find(({ contextTokens }) =>
    contextTokens === comparisonContext)
  if (generation === undefined) return Option.none()
  return Option.some({
    intelligence: clamp01(candidate.intelligenceScore / 100),
    speed: normalizedModelSpeedScore(generation.estimatedTokensPerSecond),
    fidelity: clamp01(candidate.fidelityRank / 100),
  })
}
