/**
 * PolicyContext provider factory.
 *
 * Reads from projections to build the PolicyContext snapshot
 * that turn policies and tool policies use for decision-making.
 */

import { Effect } from 'effect'
import type { Projection } from '@magnitudedev/event-core'
import { AgentLifecycleProjection, type AgentInfo } from '../projections/agent-lifecycle'
import { TurnProjection } from '../projections/turn'

import type { PolicyContext, PolicyContextProvider } from './types'
import { SessionOptionsAmbient, type SessionOptions } from '../ambient/session-ambient'

/** Build a PolicyContextProvider that reads from the given projections. */
export function createPolicyContextProvider(
  forkId: string | null,
  cwd: string,
  scratchpadPath: string,
  sessionOptions: SessionOptions,
  agentLifecycleProjection: Projection.ProjectionInstance<typeof AgentLifecycleProjection.stateSchema>,
  turnProjection: Projection.ForkedProjectionInstance<typeof TurnProjection.forkStateSchema>,
): PolicyContextProvider {
  return {
    get: Effect.gen(function* () {
      const agentLifecycleState = yield* agentLifecycleProjection.get
      const forkTurnState = yield* turnProjection.getFork(forkId)

      return {
        forkId,
        cwd,
        scratchpadPath,
        activeAgentCount: [...agentLifecycleState.agents.values()].filter((a: AgentInfo) => a.status === 'working').length,
        userMessagePending: forkTurnState.pendingInboundCommunications.some(
          (message) => message.source === 'user'
        ),
        disableShellSafeguards: sessionOptions.disableShellSafeguards,
        disableCwdSafeguards: sessionOptions.disableCwdSafeguards,
        agents: [...agentLifecycleState.agents.values()].map((a: AgentInfo) => ({
          agentId: a.agentId,
          type: a.role,
          status: a.status,
        })),
      }
    })
  }
}
