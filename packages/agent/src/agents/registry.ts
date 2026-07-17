/**
 * Agent Registry
 *
 * All agent definitions, accessible by RoleId.
 */

import { createRoles, isRoleId, type RoleId, type RoleDefinition } from '@magnitudedev/roles'

import type { Effect, Layer } from 'effect'
import type { AgentLifecycleState } from '../projections/agent-lifecycle'
import type { ObservableConfig, ForkSetupContext } from '../observables/types'

/** Default agent display name when no explicit name is available. */
export const DEFAULT_AGENT_NAME = "magnitude" as const

// Agent-package-level role definition.
// Extends the roles-package RoleDefinition with agent-level concerns:
// observables and fork lifecycle hooks.
export interface AgentRoleDefinition extends RoleDefinition {
  /** Observable state feeds scoped to this role's fork. */
  readonly observables?: readonly ObservableConfig<never>[]

  /** Called during fork setup to provide additional layers. */
  readonly setup?: (ctx: ForkSetupContext) => Effect.Effect<Layer.Layer<unknown>>

  /** Called during fork disposal for cleanup. */
  readonly teardown?: (ctx: ForkSetupContext) => Effect.Effect<void>
}

const BASE_ROLES = createRoles()
const ROLES: Record<RoleId, AgentRoleDefinition> = Object.fromEntries(
  Object.entries(BASE_ROLES).map(([id, def]) => [
    id,
    def,
  ])
) as Record<RoleId, AgentRoleDefinition>

const _overrides = new Map<string, AgentRoleDefinition>()

export function registerAgentDefinition(roleId: string, def: AgentRoleDefinition): void {
  _overrides.set(roleId, def)
}

export function clearAgentOverrides(): void {
  _overrides.clear()
}

export function getAgentDefinition(roleId: RoleId): AgentRoleDefinition {
  const override = _overrides.get(roleId)
  if (override) return override
  return ROLES[roleId]
}

/**
 * Resolve role for a fork.
 * Returns null when the agent is missing (e.g. already killed).
 * Callers must bail out when null is returned.
 */
export function getForkInfo(
  agentStatus: AgentLifecycleState,
  forkId: string | null
): { roleId: RoleId } | null {
  if (forkId === null) {
    return { roleId: 'leader' }
  }
  const agentId = agentStatus.agentByForkId.get(forkId)
  const agent = agentId ? agentStatus.agents.get(agentId) : undefined
  if (!agent) return null
  const role = agent.role
  if (!isRoleId(role)) return null
  return { roleId: role as RoleId }
}
