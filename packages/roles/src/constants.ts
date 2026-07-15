

// ---------------------------------------------------------------------------
// Slot identifiers — application-level concept (not provider-level)
// ---------------------------------------------------------------------------

export type SlotId = 'primary' | 'secondary'

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------

export type RoleId = 'leader' | 'scout' | 'architect' | 'engineer' | 'critic' | 'scientist' | 'artisan' | 'advisor'

export const ROLE_IDS: readonly RoleId[] = ['leader', 'scout', 'architect', 'engineer', 'critic', 'scientist', 'artisan', 'advisor'] as const

export function isRoleId(value: string): value is RoleId {
  return ROLE_IDS.some((roleId) => roleId === value)
}

// ---------------------------------------------------------------------------
// Slot-based model configuration
// ---------------------------------------------------------------------------

/** Maps each role to its slot (product decision, not user-configurable). */
export const ROLE_TO_SLOT: Readonly<Record<RoleId, SlotId>> = {
  leader: 'primary',
  architect: 'primary',
  scientist: 'primary',
  advisor: 'primary',
  engineer: 'secondary',
  critic: 'secondary',
  artisan: 'secondary',
  scout: 'secondary',
}

/** Default reasoning effort per slot (hardcoded product decision). */
export const DEFAULT_REASONING_EFFORT: Readonly<Record<SlotId, string>> = {
  primary: 'high',
  secondary: 'medium',
}

/** All slot IDs in canonical order. */
export const SLOT_IDS = ['primary', 'secondary'] as const

/** User-facing display names (capitalized slot IDs). */
export const SLOT_DISPLAY_NAMES: Readonly<Record<SlotId, string>> = {
  primary: 'Primary',
  secondary: 'Secondary',
}

/** Help text shown under each slot card in the settings UI. */
export const SLOT_DESCRIPTIONS: Readonly<Record<SlotId, string>> = {
  primary: 'The model you chat with and which delegates tasks to worker models.',
  secondary: 'Used for all worker tasks — implementation, review, exploration, and creative work.',
}
