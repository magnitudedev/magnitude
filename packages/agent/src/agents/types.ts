/**
 * Magnitude-specific types for agent definitions.
 */

import { Context, Effect } from 'effect'

export interface EphemeralSessionContext {
  readonly disableShellSafeguards: boolean
  readonly disableCwdSafeguards: boolean
}

export class EphemeralSessionContextTag extends Context.Tag('EphemeralSessionContext')<
  EphemeralSessionContextTag,
  EphemeralSessionContext
>() {}

/** Framework state provided to agent policies (tool + turn) for decision-making. */
export interface PolicyContext {
  readonly forkId: string | null
  readonly cwd: string
  readonly workspacePath: string
  readonly activeAgentCount: number
  readonly userMessagePending: boolean
  readonly disableShellSafeguards: boolean
  readonly disableCwdSafeguards: boolean

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
