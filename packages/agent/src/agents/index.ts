/**
 * Agent Registry
 *
 * All agent definitions, accessible by variant.
 */

import type { RoleDefinition, ToolSet } from '@magnitudedev/roles'
import type { PolicyContext } from './types'
import type { MagnitudeSlot } from '../model-slots'
import { leadRole } from './lead'
import { leadOneshotRole } from './lead-oneshot'
import { builderRole } from './builder'
import { explorerRole } from './explorer'
import { plannerRole } from './planner'
import { debuggerRole } from './debugger'
import { reviewerRole } from './reviewer'
import { browserRole } from './browser'

type MagnitudeRoleDef = RoleDefinition<ToolSet, MagnitudeSlot, PolicyContext>

const AGENTS: Record<string, MagnitudeRoleDef> = {
  lead: leadRole,
  'lead-oneshot': leadOneshotRole,
  builder: builderRole,
  explorer: explorerRole,
  planner: plannerRole,
  debugger: debuggerRole,
  reviewer: reviewerRole,
  browser: browserRole,
}

export type AgentVariant = 'lead' | 'lead-oneshot' | 'builder' | 'explorer' | 'planner' | 'debugger' | 'reviewer' | 'browser'

export function isValidVariant(v: string): v is AgentVariant {
  return Object.hasOwn(AGENTS, v)
}

export function getSpawnableVariants(): AgentVariant[] {
  return (Object.entries(AGENTS) as [AgentVariant, MagnitudeRoleDef][])
    .filter(([, role]) => role.spawnable)
    .map(([id]) => id)
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
  return getAgentDefinition(variant).slot
}
