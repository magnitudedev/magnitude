import { Effect } from 'effect'
import type { TurnCompleted } from '../events'
import { getAgentDefinition, isValidVariant, type AgentVariant } from '../agents'
import { CanonicalTurnProjection } from '../projections/canonical-turn'
import { AgentStatusProjection, getAgentByForkId } from '../projections/agent-status'

export const buildInterruptedTurnCompleted = (params: {
  forkId: string | null
  turnId: string
  chainId: string | null
}) => Effect.gen(function* () {
  const { forkId, turnId, chainId } = params

  const canonicalProjection = yield* CanonicalTurnProjection.Tag
  const agentProjection = yield* AgentStatusProjection.Tag

  yield* canonicalProjection.getFork(forkId)
  const agentState = yield* agentProjection.get

  const variant: AgentVariant = forkId
    ? (() => {
        const role = getAgentByForkId(agentState, forkId)?.role
        return role && isValidVariant(role) ? role : 'builder'
      })()
    : 'lead'

  getAgentDefinition(variant)

  const event: TurnCompleted = {
    type: 'turn_completed',
    forkId,
    turnId,
    chainId: chainId ?? '',
    strategyId: 'xml-act',
    result: { success: false, error: 'Interrupted', cancelled: true },
    inputTokens: null,
    outputTokens: null,
    cacheReadTokens: null,
    cacheWriteTokens: null,
    providerId: null,
    modelId: null,
  }

  return event
})