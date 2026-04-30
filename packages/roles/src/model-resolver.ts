import type { AuthApplicator, BoundModel } from '@magnitudedev/ai'
import type { MagnitudeModelSpec, MagnitudeConnectionError, MagnitudeStreamError, ModelProfile, RoleId } from '@magnitudedev/magnitude-client'
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
 * A model override entry with explicit auth and metadata.
 */
export interface ModelOverrideEntry {
  readonly spec: MagnitudeModelSpec
  readonly profile: ModelProfile
  readonly auth?: AuthApplicator
}

/**
 * Per-slot model overrides.
 */
export type ModelOverrides = Partial<Record<Slot, ModelOverrideEntry>>

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
    return override.spec.bind({ auth: override.auth ?? auth })
  }

  const roleId = SLOT_TO_ROLE[slot]
  const spec = createRoleSpec(roleId, endpoint)
  return spec.bind({ auth })
}
