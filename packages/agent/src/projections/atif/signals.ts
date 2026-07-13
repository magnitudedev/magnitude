/**
 * ATIF projection signals
 */

import { Signal } from '@magnitudedev/event-core'
import type { AtifStep } from './types'

export const atifSignals = {
  /** Emitted when a new ATIF step is added to any fork */
  stepAdded: Signal.create<{ forkId: string | null; step: AtifStep; stepIndex: number }>('Atif/stepAdded'),

  /** Emitted when a fork trajectory is completed */
  forkCompleted: Signal.create<{ forkId: string | null; stepCount: number }>('Atif/forkCompleted'),
}
