import type { AuthApplicator, BoundModel, ModelSpec } from '@magnitudedev/ai'
import type { RoleId } from '@magnitudedev/magnitude-client'
import { createRoleSpec } from '@magnitudedev/magnitude-client'
import type { Slot } from './types'

/** Maps role slot to magnitude-client RoleId. */
const SLOT_TO_ROLE: Record<Slot, RoleId> = {
  leader: 'leader',
  scout: 'scout',
  architect: 'architect',
  engineer: 'engineer',
  critic: 'critic',
  scientist: 'scientist',
  artisan: 'artisan',
  advisor: 'advisor',
}

/**
 * A model override entry with an explicit auth applicator.
 */
export interface ModelOverrideEntry {
  readonly spec: ModelSpec<any, any, any>
  readonly auth?: AuthApplicator
}

/**
 * Per-slot model overrides. Each entry can be a bare ModelSpec (uses defaultAuth)
 * or a ModelOverrideEntry with its own auth.
 */
export type ModelOverrides = Partial<Record<Slot, ModelSpec<any, any, any> | ModelOverrideEntry>>

function isOverrideEntry(value: ModelSpec<any, any, any> | ModelOverrideEntry): value is ModelOverrideEntry {
  return typeof value === 'object' && value !== null && 'spec' in value
}

/**
 * Resolve a bound model for a given slot.
 *
 * Priority:
 * 1. Override for the slot (if provided)
 * 2. Default magnitude-client role spec
 */
export function resolveModel(
  slot: Slot,
  endpoint: string,
  auth: AuthApplicator,
  overrides?: ModelOverrides,
): BoundModel<any, any, any> {
  if (overrides?.[slot]) {
    const override = overrides[slot]!
    if (isOverrideEntry(override)) {
      return override.spec.bind({ auth: override.auth ?? auth })
    }
    // Bare ModelSpec
    return override.bind({ auth })
  }

  const roleId = SLOT_TO_ROLE[slot]
  const spec = createRoleSpec(roleId, endpoint)
  return spec.bind({ auth })
}
