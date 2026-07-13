/**
 * Observer escalation system types.
 */

import type { ObserverJustification } from './justifications'

// =============================================================================
// ObserverTurnData — compact shape for parent inbox XML rendering
// =============================================================================

export interface ObserverTurnData {
  readonly justification: ObserverJustification | null
}

// =============================================================================
// Events
// =============================================================================

export interface ObserverOutcome {
  readonly type: 'observer_outcome'
  readonly forkId: string | null
  readonly observedTurnId: string
  readonly observerTurnId: string
  readonly chainId: string
  readonly escalate: boolean
  readonly justification: ObserverJustification | null
  readonly reasoning: string
}
