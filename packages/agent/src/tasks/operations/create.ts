import { Effect } from 'effect'
import { WorkerBusTag } from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { TaskGraphStateReaderTag } from '../../tools/task-reader'
import { buildTaskCreatedValidated, buildTaskStatusChangedValidated } from './builders'
import { duplicateTaskId, parentNotFound } from './errors'
import type { TaskDirectiveContext, TaskDirectiveResult } from './handler'

export interface CreateDirective {
  readonly kind: 'create'
  readonly taskId: string
  readonly taskType: string
  readonly parentId: string | null
  readonly after?: string | null
  readonly title: string
}

export const handleCreateDirective = (directive: CreateDirective, context: TaskDirectiveContext) =>
  Effect.gen(function* () {
    const normalizedType = directive.taskType ? directive.taskType.trim().toLowerCase() : ''
    const taskReader = yield* TaskGraphStateReaderTag
    const existingTask = yield* taskReader.getTask(directive.taskId)
    if (existingTask) {
      const err = duplicateTaskId(directive.taskId)
      return { success: false, code: err.code, error: err.message } as const
    }

    if (directive.parentId) {
      const parentTask = yield* taskReader.getTask(directive.parentId)
      if (!parentTask) {
        const err = parentNotFound(directive.taskId, directive.parentId)
        return { success: false, code: err.code, error: err.message } as const
      }
    }

    const events: AppEvent[] = []
    if (directive.parentId) {
      const parentTask = yield* taskReader.getTask(directive.parentId)
      if (parentTask && parentTask.status === 'completed') {
        const reopen = buildTaskStatusChangedValidated(
          {
            taskId: parentTask.id,
            patch: { status: 'pending' },
          },
          { forkId: context.forkId, timestamp: context.timestamp, graph: context.graph },
          parentTask.status,
        )
        if (reopen.success === false) {
          return { success: false, code: reopen.code, error: reopen.message } as const
        }
        events.push(reopen.event)
      }
    }

    events.push(
      buildTaskCreatedValidated({
        taskId: directive.taskId,
        title: directive.title.trim(),
        taskType: normalizedType,
        parentId: directive.parentId,
        after: directive.after ?? undefined,
      }, { forkId: context.forkId, timestamp: context.timestamp, graph: context.graph }),
    )

    const bus = yield* WorkerBusTag<AppEvent>()
    for (const event of events) {
      yield* bus.publish(event)
    }

    return { success: true } as const
  })
