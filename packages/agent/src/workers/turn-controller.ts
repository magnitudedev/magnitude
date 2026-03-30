/**
 * TurnController Worker
 *
 * Evaluates turn readiness after projections settle and publishes turn_started.
 */

import { Effect } from 'effect'
import { Worker, type PublishFn } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import {
  TurnProjection,
  type PendingInboundCommunication,
  type TurnTrigger,
  type ForkTurnState,
} from '../projections/turn'
import { CompactionProjection, type CompactionState } from '../projections/compaction'
import { WorkflowProjection, type WorkflowCriteriaState, type WorkflowCriterion } from '../projections/workflow'
import { createId } from '../util/id'

function resolveChainId(triggers: readonly TurnTrigger[]): string | null {
  for (const t of triggers) {
    if (t._tag === 'chain_continue') return t.chainId
  }
  return null
}

function startTurnForFork(
  forkId: string | null,
  turnFork: ForkTurnState,
  publish: PublishFn<AppEvent>,
) {
  return Effect.gen(function* () {
    const currentTurnAllowsDirectUserReply = turnFork.pendingInboundCommunications.some(
      (message: PendingInboundCommunication) =>
        message.source === 'user' && message.replyPolicy === 'user_reply_once'
    )

    const turnId = createId()
    const chainId = resolveChainId(turnFork.triggers) ?? createId()

    yield* publish({
      type: 'turn_started',
      forkId,
      turnId,
      chainId,
      currentTurnAllowsDirectUserReply,
    })
  })
}

export const TurnController = Worker.define<AppEvent>()({
  name: 'TurnController',

  onProjectionsSettled: ({ publish, read }) =>
    Effect.gen(function* () {
      const turnForks = yield* read.allForks(TurnProjection)
      const compactionForks = yield* read.allForks(CompactionProjection)
      const workflowForks = yield* read.allForks(WorkflowProjection)

      for (const [forkId, turnFork] of turnForks) {
        const compactionFork: CompactionState | undefined = compactionForks.get(forkId)
        const workflowFork: WorkflowCriteriaState | undefined = workflowForks.get(forkId)

        const hasTrigger = turnFork.triggers.length > 0
        const isTurnIdle = turnFork._tag === 'idle'
        const isCompactionIdle = compactionFork === undefined || compactionFork._tag === 'idle'
        const contextLimitBlocked = compactionFork?.contextLimitBlocked === true
        const hasPendingVerdict = (workflowFork?.criteria ?? []).some(
          (criterion: WorkflowCriterion) =>
            criterion.lifecycle._tag === 'pending' || criterion.lifecycle._tag === 'running'
        )

        const canStart =
          hasTrigger &&
          isTurnIdle &&
          isCompactionIdle &&
          !contextLimitBlocked &&
          !hasPendingVerdict

        if (canStart) {
          yield* startTurnForFork(forkId, turnFork, publish)
        }
      }
    }),
})
