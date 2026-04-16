/**
 * Agents — external entry point
 *
 * Internal files import from ./registry or ./variants directly.
 */

export type { AgentVariant } from './variants'
export { isValidVariant, getSpawnableVariants } from './variants'
export { getAgentDefinition, getAgentSlot, getForkInfo, registerAgentDefinition, clearAgentOverrides } from './registry'
