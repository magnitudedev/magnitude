import { isRoleId, type RoleId, ROLE_IDS } from '@magnitudedev/roles'

// Roles that can be spawned via spawn-worker tool.
// Excludes leader (not spawnable) and advisor (peer-only, created via messageAdvisor).
const SPAWNABLE_ROLES: ReadonlySet<RoleId> = new Set<RoleId>(
  ['scout', 'architect', 'engineer', 'critic', 'scientist', 'artisan']
)

export function isSpawnableRole(value: string): value is RoleId {
  return isRoleId(value) && SPAWNABLE_ROLES.has(value as RoleId)
}

export function getSpawnableRoles(): RoleId[] {
  return [...SPAWNABLE_ROLES]
}

export { isRoleId, type RoleId, ROLE_IDS }
