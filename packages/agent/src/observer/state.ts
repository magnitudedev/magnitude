/**
 * Observer worker state.
 *
 * Effect-managed Ref scoped to the worker fiber lifecycle.
 * No module-level mutable state.
 */

import { Context, Data, Ref, Layer, Fiber } from 'effect'
import { FSM } from '@magnitudedev/utils'
import type { AppEvent } from '../events'

const { defineFSM } = FSM

export type ObserverTurnOutcomeEvent = Extract<AppEvent, { type: 'turn_outcome' }>

export class ObserverIdle extends Data.TaggedClass('idle')<{
  /** Latest turn_outcome deferred while an advisor-required escalation is pending. */
  readonly pendingEvent: ObserverTurnOutcomeEvent | null
}> {}

export class ObserverRunning extends Data.TaggedClass('running')<{
  readonly runId: string
  /** Latest turn_outcome that arrived while this observer run was active. */
  readonly pendingEvent: ObserverTurnOutcomeEvent | null
}> {}

export const ObserverRunLifecycle = defineFSM(
  {
    idle: ObserverIdle,
    running: ObserverRunning,
  },
  {
    idle: ['running'],
    running: ['idle'],
  },
)

export type ObserverRunState =
  | ObserverIdle
  | ObserverRunning

export interface ObserverForkState {
  readonly observer: ObserverRunState
  readonly fiber: Fiber.RuntimeFiber<void, never> | null
}

export function initialObserverForkState(): ObserverForkState {
  return {
    observer: new ObserverIdle({ pendingEvent: null }),
    fiber: null,
  }
}

export const ObserverStateTag = Context.GenericTag<Ref.Ref<Map<string | null, ObserverForkState>>>(
  'ObserverState',
)

export const ObserverStateLive = Layer.scoped(
  ObserverStateTag,
  Ref.make(new Map<string | null, ObserverForkState>()),
)
