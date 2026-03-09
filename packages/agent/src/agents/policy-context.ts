/**
 * PolicyContext provider factory.
 *
 * Reads from projections to build the PolicyContext snapshot
 * that turn policies and permission policies use for decision-making.
 */

import { Effect } from 'effect'
import type { Projection } from '@magnitudedev/event-core'
import type { AgentRegistryState } from '../projections/agent-registry'
import type { ForkWorkingState } from '../projections/working-state'

import type { PolicyContext, PolicyContextProvider } from './types'

/** Build a PolicyContextProvider that reads from the given projections. */
export function createPolicyContextProvider(
  forkId: string | null,
  cwd: string,
  agentRegistryProjection: Projection.ProjectionInstance<AgentRegistryState>,
  workingStateProjection: Projection.ForkedProjectionInstance<ForkWorkingState>,
): PolicyContextProvider {
  return {
    get: Effect.gen(function* () {
      const registryState = yield* agentRegistryProjection.get
      const forkWorkingState = yield* workingStateProjection.getFork(forkId)

      return {
        forkId,
        cwd,
        activeAgentCount: [...registryState.agents.values()].filter(a => a.status === 'running').length,
        userMessagePending: forkWorkingState.hasQueuedMessages,

        agents: [...registryState.agents.values()].map(a => ({ agentId: a.agentId, type: a.type, status: a.status })),
      }
    })
  }
}
