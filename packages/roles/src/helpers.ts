import type { TurnResult } from './types'

// =============================================================================
// Turn helpers
// =============================================================================

export function observe(reminder: string | null = null): TurnResult {
  return reminder ? { action: 'continue', reminder } : { action: 'continue' }
}

export function idle(reminder: string | null = null): TurnResult {
  return reminder ? { action: 'idle', reminder } : { action: 'idle' }
}

export function finish(reminder: string | null = null): TurnResult {
  return reminder ? { action: 'finish', reminder } : { action: 'finish' }
}
