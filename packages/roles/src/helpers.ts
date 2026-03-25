import type { TurnResult } from './types'

// =============================================================================
// Turn helpers
// =============================================================================

export function continue_(reminder: string | null = null): TurnResult {
  return reminder ? { action: 'continue', reminder } : { action: 'continue' }
}

export function yield_(reminder: string | null = null): TurnResult {
  return reminder ? { action: 'yield', reminder } : { action: 'yield' }
}

export function finish(reminder: string | null = null): TurnResult {
  return reminder ? { action: 'finish', reminder } : { action: 'finish' }
}
