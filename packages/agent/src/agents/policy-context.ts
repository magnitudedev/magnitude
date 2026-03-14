/**
 * PolicyContext provider factory.
 *
 * Reads from projections to build the PolicyContext snapshot
 * that turn policies and permission policies use for decision-making.
 */

import { Effect } from 'effect'
import type { Projection } from '@magnitudedev/event-core'
import type { AgentRoutingState, AgentInstance } from '../projections/agent-routing'
import type { AgentStatusState } from '../projections/agent-status'
import type { ForkWorkingState } from '../projections/working-state'

import type { PolicyContext, PolicyContextProvider } from './types'

/** Build a PolicyContextProvider that reads from the given projections. */
export function createPolicyContextProvider(
  forkId: string | null,
  cwd: string,
  agentProjection: Projection.ProjectionInstance<AgentRoutingState>,
  agentStatusProjection: Projection.ProjectionInstance<AgentStatusState>,
  workingStateProjection: Projection.ForkedProjectionInstance<ForkWorkingState>,
): PolicyContextProvider {
  return {
    get: Effect.gen(function* () {
      const agentState = yield* agentProjection.get
      const agentStatuses = yield* agentStatusProjection.get
      const forkWorkingState = yield* workingStateProjection.getFork(forkId)

      return {
        forkId,
        cwd,
        activeAgentCount: [...agentState.agents.values()].filter((a: AgentInstance) => agentStatuses.statuses.get(a.agentId) === 'working').length,
        userMessagePending: forkWorkingState.hasQueuedMessages,
        agents: [...agentState.agents.values()].map((a: AgentInstance) => ({
          agentId: a.agentId,
          type: a.role,
          status: agentStatuses.statuses.get(a.agentId) ?? (a.dismissed ? 'dismissed' : 'starting')
        })),
      }
    })
  }
}
