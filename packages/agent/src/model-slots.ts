export const MAGNITUDE_SLOTS = ['lead', 'explorer', 'planner', 'builder', 'debugger', 'reviewer', 'browser'] as const

export type MagnitudeSlot = typeof MAGNITUDE_SLOTS[number]

export function isMagnitudeSlot(s: string): s is MagnitudeSlot {
  return (MAGNITUDE_SLOTS as readonly string[]).includes(s)
}
