/**
 * TurnController Worker
 *
 * Evaluates turn readiness after projections settle and publishes turn_started.
 */

import { Effect, Option } from 'effect'
import { Worker, type PublishFn } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import {
  TurnProjection,
  type PendingInboundCommunication,
  type TurnTrigger,
  type ForkTurnState,
} from '../projections/turn'
import { CompactionProjection } from '../projections/compaction'
import { createId } from '../util/id'

function resolveChainId(triggers: readonly TurnTrigger[]): string | null {
  for (const t of triggers) {
    if (t._tag === 'chain_continue') return t.chainId
  }
  return null
}

/**
 * A trigger is "due" if it can fire right now. chain_continue triggers may
 * carry a notBefore timestamp (for retry backoff) — those are pending until
 * the wall clock catches up. All other triggers are always due.
 */
function isTriggerDue(trigger: TurnTrigger, now: number): boolean {
  if (trigger._tag !== 'chain_continue') return true
  return Option.isNone(trigger.notBefore) || trigger.notBefore.value <= now
}

function startTurnForFork(
  forkId: string | null,
  turnFork: ForkTurnState,
  publish: PublishFn<AppEvent>,
) {
  return Effect.gen(function* () {
    const turnId = createId()
    const chainId = resolveChainId(turnFork.triggers) ?? createId()

    yield* publish({
      type: 'turn_started',
      forkId,
      turnId,
      chainId,
    })
  })
}

export const TurnController = Worker.define<AppEvent>()({
  name: 'TurnController',

  onProjectionsSettled: ({ publish, read }) =>
    Effect.gen(function* () {
      const turnForks = yield* read.allForks(TurnProjection)
      const compactionForks = yield* read.allForks(CompactionProjection)
      const now = Date.now()

      for (const [forkId, turnFork] of turnForks) {
        const compactionFork: import('../projections/compaction').CompactionState | undefined = compactionForks.get(forkId)
        const hasDueTrigger = turnFork.triggers.some((t) => isTriggerDue(t, now))
        const isTurnIdle = turnFork._tag === 'idle'
        const contextLimitBlocked = compactionFork?.contextLimitBlocked === true

        const canStart =
          hasDueTrigger &&
          isTurnIdle &&
          !contextLimitBlocked

        if (canStart) {
          yield* startTurnForFork(forkId, turnFork, publish)
        }
      }
    }),
})
