import { Effect } from 'effect'
import type { TurnOutcomeEvent } from '../events'
import { isRoleId, type RoleId } from '../agents/role-validation'
import { getAgentDefinition } from '../agents/registry'
import { HarnessStateProjection } from '../projections/harness-state'
import { AgentLifecycleProjection, getAgentByForkId } from '../projections/agent-lifecycle'

export const buildInterruptedTurnOutcome = (params: {
  forkId: string | null
  turnId: string
  chainId: string | null
}) => Effect.gen(function* () {
  const { forkId, turnId, chainId } = params

  const harnessStateProjection = yield* HarnessStateProjection.Tag
  const agentProjection = yield* AgentLifecycleProjection.Tag

  yield* harnessStateProjection.getFork(forkId)
  const agentState = yield* agentProjection.get

  const roleId: RoleId = forkId
    ? (() => {
        const role = getAgentByForkId(agentState, forkId)?.role
        return role && isRoleId(role) ? role : 'engineer'
      })()
    : 'leader'

  getAgentDefinition(roleId)

  const event: TurnOutcomeEvent = {
    type: 'turn_outcome',
    forkId,
    turnId,
    chainId: chainId ?? '',
    strategyId: 'native',
    outcome: { _tag: 'Cancelled', reason: { _tag: 'UserInterrupt' }, requestId: null },
    commitPolicy: { _tag: 'commitErrorOnly' },
    inputTokens: null,
    outputTokens: null,
    cacheReadTokens: null,
    cacheWriteTokens: null,
    cost: null,
    providerId: null,
    modelId: null,
  }

  return event
})
