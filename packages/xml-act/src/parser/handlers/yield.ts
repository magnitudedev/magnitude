/**
 * Yield self-close handler.
 */

import type { TurnEngineEvent } from '../../types'
import type { Op } from '../../machine'
import type { Frame } from '../types'

export function handleYield(
  variant: string | undefined,
  endCurrentProse: () => void,
  apply: (ops: Op<Frame, TurnEngineEvent>[]) => void,
  setDeferredYield: (target: 'user' | 'invoke' | 'worker' | 'parent') => void,
): void {
  const target = (variant as 'user' | 'invoke' | 'worker' | 'parent') ?? 'user'
  endCurrentProse()
  setDeferredYield(target)
  apply([{ type: 'observe' }])
}
