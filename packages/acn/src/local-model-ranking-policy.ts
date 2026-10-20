import { Option } from "effect"
import {
  type LocalModelPerformance,
  type LocalModelRankingScores,
  type ServingProfile,
} from "@magnitudedev/acn-protocol"

const COMPARISON_CONTEXT_LENGTH = 50_000
const LINEAR_SPEED_SCORE_LIMIT = 40
const SPEED_SCORE_CEILING = 100

export interface LocalModelRankingCandidate {
  readonly intelligenceScore: number
  readonly fidelityRank: number
  readonly profile: ServingProfile
  readonly performance: LocalModelPerformance
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

/** The speed term is the estimate at the comparison depth; none while speed is not estimated. */
export const modelRankingScores = (
  candidate: LocalModelRankingCandidate,
): LocalModelRankingScores => {
  const comparisonContext = Math.min(
    COMPARISON_CONTEXT_LENGTH,
    candidate.profile.contextLength,
  )
  const speed = candidate.performance._tag === "Estimated"
    ? Option.map(
        Option.fromNullable(candidate.performance.samples.find(({ contextTokens }) =>
          contextTokens === comparisonContext)),
        ({ estimatedTokensPerSecond }) => normalizedModelSpeedScore(estimatedTokensPerSecond),
      )
    : Option.none()
  return {
    intelligence: clamp01(candidate.intelligenceScore / 100),
    speed,
    fidelity: clamp01(candidate.fidelityRank / 100),
  }
}
