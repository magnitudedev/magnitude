import type { AuthApplicator, BoundModel } from '@magnitudedev/ai'
import type { MagnitudeModelSpec, MagnitudeConnectionError, MagnitudeStreamError, RoleId } from '@magnitudedev/magnitude-client'
import { createRoleSpec } from '@magnitudedev/magnitude-client'
import type { Slot } from './types'

/** Maps role slot to magnitude-client role ID. */
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
 * A model override entry with explicit auth.
 */
export interface ModelOverrideEntry {
  readonly spec: MagnitudeModelSpec
  readonly auth?: AuthApplicator
}

/**
 * Per-slot model overrides. Each entry can be a bare MagnitudeModelSpec (uses default auth)
 * or a ModelOverrideEntry with its own auth.
 */
export type ModelOverrides = Partial<Record<Slot, MagnitudeModelSpec | ModelOverrideEntry>>

function isOverrideEntry(value: MagnitudeModelSpec | ModelOverrideEntry): value is ModelOverrideEntry {
  return typeof value === 'object' && value !== null && 'spec' in value
}

/**
 * Resolve a bound model for a given slot.
 */
export function resolveModel(
  slot: Slot,
  endpoint: string,
  auth: AuthApplicator,
  overrides?: ModelOverrides,
): BoundModel<{}, MagnitudeConnectionError, MagnitudeStreamError> {
  if (overrides?.[slot]) {
    const override = overrides[slot]!
    if (isOverrideEntry(override)) {
      return override.spec.bind({ auth: override.auth ?? auth })
    }
    return override.bind({ auth })
  }

  const roleId = SLOT_TO_ROLE[slot]
  const spec = createRoleSpec(roleId, endpoint)
  return spec.bind({ auth })
}
