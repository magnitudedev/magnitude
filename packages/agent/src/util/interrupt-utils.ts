import { Effect } from 'effect'
import type { TurnOutcomeEvent } from '../events'
import { isValidVariant, type AgentVariant } from '../agents/variants'
import { getAgentDefinition } from '../agents/registry'
import { CanonicalTurnProjection } from '../projections/canonical-turn'
import { AgentStatusProjection, getAgentByForkId } from '../projections/agent-status'

export const buildInterruptedTurnOutcome = (params: {
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
        return role && isValidVariant(role) ? role : 'worker'
      })()
    : 'lead'

  getAgentDefinition(variant)

  const event: TurnOutcomeEvent = {
    type: 'turn_outcome',
    forkId,
    turnId,
    chainId: chainId ?? '',
    strategyId: 'xml-act',
    outcome: { _tag: 'Cancelled', reason: { _tag: 'UserInterrupt' } },
    inputTokens: null,
    outputTokens: null,
    cacheReadTokens: null,
    cacheWriteTokens: null,
    providerId: null,
    modelId: null,
  }

  return event
})