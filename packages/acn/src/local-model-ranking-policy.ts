import { Data, Effect } from "effect"
import type {
  FitsModelAssessment,
  LocalModelRankingScores,
  RecommendableModel,
  ServingProfile,
} from "@magnitudedev/acn-protocol"

const COMPARISON_CONTEXT_LENGTH = 50_000
const LINEAR_SPEED_SCORE_LIMIT = 40
const SPEED_SCORE_CEILING = 100

export interface LocalModelRankingCandidate {
  readonly model: RecommendableModel
  readonly profile: ServingProfile
  readonly assessment: FitsModelAssessment
}

export class LocalModelRankingSampleMissing extends Data.TaggedError(
  "LocalModelRankingSampleMissing",
)<{
  readonly modelId: string
  readonly contextTokens: number
}> {}

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
): Effect.Effect<LocalModelRankingScores, LocalModelRankingSampleMissing> => {
  const comparisonContext = Math.min(
    COMPARISON_CONTEXT_LENGTH,
    candidate.profile.contextLength,
  )
  const generation = candidate.assessment.performance.find(({ contextTokens }) =>
    contextTokens === comparisonContext)
  if (generation === undefined) {
    return Effect.fail(new LocalModelRankingSampleMissing({
      modelId: candidate.model.modelId,
      contextTokens: comparisonContext,
    }))
  }
  return Effect.succeed({
    intelligence: clamp01(candidate.model.qualityScore / 100),
    speed: normalizedModelSpeedScore(generation.estimatedTokensPerSecond),
    quality: clamp01(candidate.model.fidelityRank / 100),
  })
}
