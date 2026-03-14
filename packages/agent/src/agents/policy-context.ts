/**
 * PolicyContext provider factory.
 *
 * Reads from projections to build the PolicyContext snapshot
 * that turn policies and permission policies use for decision-making.
 */

import { Effect } from 'effect'
import type { Projection } from '@magnitudedev/event-core'
import type { AgentStatusState, AgentInfo } from '../projections/agent-status'
import type { ForkWorkingState } from '../projections/working-state'

import type { PolicyContext, PolicyContextProvider } from './types'

/** Build a PolicyContextProvider that reads from the given projections. */
export function createPolicyContextProvider(
  forkId: string | null,
  cwd: string,
  agentStatusProjection: Projection.ProjectionInstance<AgentStatusState>,
  workingStateProjection: Projection.ForkedProjectionInstance<ForkWorkingState>,
): PolicyContextProvider {
  return {
    get: Effect.gen(function* () {
      const agentStatuses = yield* agentStatusProjection.get
      const forkWorkingState = yield* workingStateProjection.getFork(forkId)

      return {
        forkId,
        cwd,
        activeAgentCount: [...agentStatuses.agents.values()].filter((a: AgentInfo) => a.status === 'working').length,
        userMessagePending: forkWorkingState.hasQueuedMessages,
        agents: [...agentStatuses.agents.values()].map((a: AgentInfo) => ({
          agentId: a.agentId,
          type: a.role,
          status: a.status,
        })),
      }
    })
  }
}
