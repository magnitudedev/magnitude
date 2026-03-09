/**
 * Fork Services
 *
 * ForkStateReaderTag — service for tools to access fork projection state
 */

import { Effect, Deferred, Context } from 'effect'
import type { ForkState } from '../projections/fork'

// =============================================================================
// Fork State Reader Service (for tools to access fork projection state)
// =============================================================================

export interface ForkStateReader {
  readonly getForkState: () => Effect.Effect<ForkState>
  readonly registerBlocking: (forkId: string, deferred: Deferred.Deferred<unknown, never>) => void
  readonly resolveBlocking: (forkId: string, result: unknown) => Effect.Effect<void>
}

export class ForkStateReaderTag extends Context.Tag('ForkStateReader')<
  ForkStateReaderTag,
  ForkStateReader
>() {}
