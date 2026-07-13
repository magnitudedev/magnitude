/**
 * Agents — external entry point
 *
 * Internal files import from ./registry or ./role-validation directly.
 */

export type { RoleId } from './role-validation'
export { isRoleId, isSpawnableRole, getSpawnableRoles, ROLE_IDS } from './role-validation'
export type { AgentRoleDefinition } from './registry'
export { getAgentDefinition, getForkInfo, registerAgentDefinition, clearAgentOverrides } from './registry'
