/**
 * Agent Registry
 *
 * All agent definitions, accessible by variant.
 */

import type { RoleDefinition } from '@magnitudedev/roles'

import type { MagnitudeSlot } from '../model-slots'
import type { AgentStatusState } from '../projections/agent-status'
import type { AgentVariant } from './variants'

import { leadRole } from './lead'
import { leadOneshotRole } from './lead-oneshot'
import { workerRole } from './worker'

type MagnitudeRoleDef = RoleDefinition

const AGENTS: Record<AgentVariant, MagnitudeRoleDef> = {
  lead: leadRole,
  'lead-oneshot': leadOneshotRole,
  worker: workerRole,
}

const _overrides = new Map<string, MagnitudeRoleDef>()

export function registerAgentDefinition(name: string, def: MagnitudeRoleDef): void {
  _overrides.set(name, def)
}

export function clearAgentOverrides(): void {
  _overrides.clear()
}

export function getAgentDefinition(variant: AgentVariant): MagnitudeRoleDef {
  const override = _overrides.get(variant)
  if (override) return override
  return AGENTS[variant]
}

export function getAgentSlot(variant: AgentVariant): MagnitudeSlot {
  return getAgentDefinition(variant).slot as MagnitudeSlot
}

/**
 * Resolve variant and slot for a fork.
 * Returns null when the agent is missing (e.g. already killed).
 * Callers must bail out when null is returned.
 */
export function getForkInfo(
  agentStatus: AgentStatusState,
  forkId: string | null
): { variant: AgentVariant; slot: MagnitudeSlot } | null {
  if (forkId === null) {
    return { variant: 'lead', slot: getAgentSlot('lead') }
  }
  const agentId = agentStatus.agentByForkId.get(forkId)
  const agent = agentId ? agentStatus.agents.get(agentId) : undefined
  if (!agent) return null
  return { variant: agent.role, slot: getAgentSlot(agent.role) }
}
