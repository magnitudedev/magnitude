/**
 * PolicyContext provider factory.
 *
 * Reads from projections to build the PolicyContext snapshot
 * that turn policies and tool policies use for decision-making.
 */

import { Effect } from 'effect'
import type { Projection } from '@magnitudedev/event-core'
import type { AgentStatusState, AgentInfo } from '../projections/agent-status'
import type { ForkTurnState } from '../projections/turn'

import type { EphemeralSessionContext, PolicyContext, PolicyContextProvider } from './types'

/** Build a PolicyContextProvider that reads from the given projections. */
export function createPolicyContextProvider(
  forkId: string | null,
  cwd: string,
  workspacePath: string,
  ephemeralSessionContext: EphemeralSessionContext,
  agentStatusProjection: Projection.ProjectionInstance<AgentStatusState>,
  turnProjection: Projection.ForkedProjectionInstance<ForkTurnState>,
): PolicyContextProvider {
  return {
    get: Effect.gen(function* () {
      const agentStatuses = yield* agentStatusProjection.get
      const forkTurnState = yield* turnProjection.getFork(forkId)

      return {
        forkId,
        cwd,
        workspacePath,
        activeAgentCount: [...agentStatuses.agents.values()].filter((a: AgentInfo) => a.status === 'working').length,
        userMessagePending: forkTurnState.triggers.some((trigger) => trigger._tag === 'user_message'),
        disableShellSafeguards: ephemeralSessionContext.disableShellSafeguards,
        disableCwdSafeguards: ephemeralSessionContext.disableCwdSafeguards,
        agents: [...agentStatuses.agents.values()].map((a: AgentInfo) => ({
          agentId: a.agentId,
          type: a.role,
          status: a.status,
        })),
      }
    })
  }
}
