import type { PermissionResult, TurnResult } from './types'

// =============================================================================
// Permission helpers
// =============================================================================

export function allow(): PermissionResult {
  return { decision: 'allow' }
}

export function approve(reason?: string): PermissionResult {
  return reason ? { decision: 'approve', reason } : { decision: 'approve' }
}

export function reject(reason?: string): PermissionResult {
  return { decision: 'reject', reason }
}

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
