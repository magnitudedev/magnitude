/**
 * Magnitude-specific types for agent definitions.
 */

import { Context, Effect } from 'effect'

/** Framework state provided to agent policies (permission + turn) for decision-making. */
export interface PolicyContext {
  readonly forkId: string | null
  readonly cwd: string
  readonly activeAgentCount: number
  readonly userMessagePending: boolean

  readonly agents: readonly { agentId: string; type: string; status: string }[]
}

/** Service that reads current PolicyContext for a fork from projections. */
export interface PolicyContextProvider {
  readonly get: Effect.Effect<PolicyContext>
}

export class PolicyContextProviderTag extends Context.Tag('PolicyContextProvider')<
  PolicyContextProviderTag,
  PolicyContextProvider
>() {}
