/**
 * Compaction signals — defined separately to avoid circular imports.
 * CompactionProjection reads WindowProjection, so WindowProjection
 * cannot import CompactionProjection. These signals are shared between both.
 */

import { Signal } from '@magnitudedev/event-core'
import type { CompletedTurn } from '../window/types'
import type { CompactionOutcome, SessionContext } from '../events'

export interface CompactionInjectedSignal {
  readonly forkId: string | null
  readonly turn: CompletedTurn
  readonly compactionOutcome: CompactionOutcome
  readonly compactedMessageCount: number
  readonly inputTokens: number | null
  readonly outputTokens: number | null
  readonly refreshedContext: SessionContext | null
}

export const compactionSignals = {
  shouldCompactChanged: Signal.create<{ forkId: string | null; shouldCompact: boolean }>('Compaction/shouldCompactChanged'),
  compactionBlockingChanged: Signal.create<{ forkId: string | null; blocking: boolean }>('Compaction/compactionBlockingChanged'),
  contextLimitBlockedChanged: Signal.create<{ forkId: string | null; blocked: boolean }>('Compaction/contextLimitBlockedChanged'),
  compactionInjected: Signal.create<CompactionInjectedSignal>('Compaction/compactionInjected'),
}
