import { Effect } from 'effect'
import { Fork, WorkerBusTag } from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { TaskGraphStateReaderTag } from '../../tools/task-reader'
import { taskNotFound } from './errors'
import { buildTaskCancelledValidated } from './builders'
import type { TaskDirectiveContext, TaskDirectiveResult } from './handler'

const { ForkContext } = Fork

export interface CancelDirective {
  readonly kind: 'cancel'
  readonly taskId: string
}

export const handleCancelDirective = (directive: CancelDirective, _context: TaskDirectiveContext) =>
  Effect.gen(function* () {
    const taskReader = yield* TaskGraphStateReaderTag
    const target = yield* taskReader.getTask(directive.taskId)
    if (!target) {
      const err = taskNotFound(directive.taskId)
      return { success: false, code: err.code, error: err.message } as const
    }

    const subtree = yield* taskReader.getSubtree(directive.taskId)
    const { forkId: parentForkId } = yield* ForkContext
    const bus = yield* WorkerBusTag<AppEvent>()
    const killedWorkers: Array<{ agentId: string; forkId: string }> = []

    for (const task of subtree) {
      if (!task.worker) continue
      killedWorkers.push({ agentId: task.worker.agentId, forkId: task.worker.forkId })
      yield* bus.publish({
        type: 'agent_killed',
        forkId: task.worker.forkId,
        parentForkId,
        agentId: task.worker.agentId,
        reason: `Task "${task.id}" cancelled`,
      })
    }

    yield* bus.publish(buildTaskCancelledValidated({
      taskId: directive.taskId,
      cancelledSubtree: subtree.map((t) => t.id),
      killedWorkers,
    }, { forkId: parentForkId, timestamp: Date.now(), graph: { tasks: new Map() } }))

    return { success: true } as const
  })
