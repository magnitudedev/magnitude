import type { MagnitudeSlot } from '@magnitudedev/agent'
import type { ProviderClient } from '@magnitudedev/providers'

// Temporary compatibility layer: maps the current 3-row UI model picker
// to the new per-role slot system. Remove this when UI supports per-slot config.

// Goal is to eliminate usage of this and use the appropriate per-role (orchestrator, subagents) directly instead

export type LegacySlotGroup = 'primary' | 'secondary' | 'browser'

const SLOT_GROUPS: Record<LegacySlotGroup, readonly MagnitudeSlot[]> = {
  primary: ['orchestrator', 'planner', 'reviewer'],
  secondary: ['explorer', 'builder', 'debugger'],
  browser: ['browser'],
}

// The "display" slot for each group — used to peek current model
const DISPLAY_SLOT: Record<LegacySlotGroup, MagnitudeSlot> = {
  primary: 'orchestrator',
  secondary: 'explorer',
  browser: 'browser',
}

export function getDisplaySlot(group: LegacySlotGroup): MagnitudeSlot {
  return DISPLAY_SLOT[group]
}

export function getSlotsForGroup(group: LegacySlotGroup): readonly MagnitudeSlot[] {
  return SLOT_GROUPS[group]
}

export function roleToSlotGroup(role: string): LegacySlotGroup {
  if (role === 'browser') return 'browser'
  if (role === 'orchestrator' || role === 'orchestrator-oneshot' || role === 'planner' || role === 'reviewer') return 'primary'
  return 'secondary'
}

// Set a model selection across all slots in a group
export async function setGroupSelection(
  runtime: ProviderClient<MagnitudeSlot>,
  group: LegacySlotGroup,
  providerId: string,
  modelId: string,
  auth: any,
  options?: { persist?: boolean },
) {
  for (const slot of SLOT_GROUPS[group]) {
    await runtime.state.setSelection(slot, providerId, modelId, auth, options)
  }
}
