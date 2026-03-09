/**
 * Modular Agent Definition — Convenience Constructors
 */

import type { PermissionResult, PermissionPreview, TurnResult, DisplayResult, DisplayOptions } from './types'

// =============================================================================
// Permission helpers
// =============================================================================

export function allow(): PermissionResult {
  return { decision: 'allow' }
}

export function approve(opts?: { reason?: string; preview?: PermissionPreview }): PermissionResult {
  return { decision: 'approve', ...opts }
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

// =============================================================================
// Display helpers
// =============================================================================

export function hidden(): DisplayResult {
  return { action: 'hidden' }
}

export function visible(options?: DisplayOptions): DisplayResult {
  return { action: 'visible', options }
}
