import { Effect } from 'effect'
import { Fork, WorkerBusTag } from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { TaskGraphStateReaderTag } from '../../tools/task-reader'
import { buildTaskAssignedValidated } from './builders'
import { taskNotFound, workerNotFound } from './errors'
import type { TaskDirectiveContext } from './handler'

const { ForkContext } = Fork

export interface KillWorkerDirective {
  readonly kind: 'kill_worker'
  readonly id: string
}

export const handleKillWorkerDirective = (directive: KillWorkerDirective, _context: TaskDirectiveContext) =>
  Effect.gen(function* () {
    const taskReader = yield* TaskGraphStateReaderTag
    const task = yield* taskReader.getTask(directive.id)
    if (!task) {
      const err = taskNotFound(directive.id)
      return { success: false, code: err.code, error: err.message } as const
    }

    if (!task.worker) {
      const err = workerNotFound(directive.id)
      return { success: false, code: err.code, error: err.message } as const
    }

    const bus = yield* WorkerBusTag<AppEvent>()
    const { forkId: parentForkId } = yield* ForkContext
    const timestamp = Date.now()
    const replacedWorker = { agentId: task.worker.agentId, forkId: task.worker.forkId }

    yield* bus.publish({
      type: 'agent_killed',
      forkId: task.worker.forkId,
      parentForkId,
      agentId: task.worker.agentId,
      reason: `Killed for task "${directive.id}"`,
    })

    yield* bus.publish(buildTaskAssignedValidated({
      taskId: directive.id,
      assignee: task.assignee ?? 'user',
      workerRole: undefined,
      message: '',
      workerInfo: undefined,
      replacedWorker,
    }, { forkId: parentForkId, timestamp, graph: { tasks: new Map() } }))

    return { success: true } as const
  })
