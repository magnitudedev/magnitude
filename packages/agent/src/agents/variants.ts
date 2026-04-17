/**
 * Agent Variants
 *
 * Static variant type and validation — leaf module with no heavy imports.
 * Exists to break the circular dependency: catalog → agent-tools → agents → lead-shared → catalog
 */

export type AgentVariant = 'lead' | 'lead-oneshot' | 'worker'

const VARIANT_SET: ReadonlySet<string> = new Set<AgentVariant>(['lead', 'lead-oneshot', 'worker'])

const SPAWNABLE_VARIANTS: readonly AgentVariant[] = ['worker']

export function isValidVariant(v: string): v is AgentVariant {
  return VARIANT_SET.has(v)
}

export function getSpawnableVariants(): AgentVariant[] {
  return [...SPAWNABLE_VARIANTS]
}
