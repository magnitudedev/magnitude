import { Effect } from 'effect'
import { WorkerBusTag } from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { TaskGraphStateReaderTag } from '../../tools/task-reader'
import { buildTaskStatusChangedValidated, buildTaskUpdatedValidated } from './builders'
import { completionBlocked, emptyUpdatePatch, parentNotFound, taskNotFound } from './errors'
import type { TaskDirectiveContext, TaskDirectiveResult } from './handler'
import type { TaskDirectiveStatus } from './types'

type UpdateDirectiveStatus = Exclude<TaskDirectiveStatus, 'cancelled'>

export interface UpdateDirective {
  readonly kind: 'update'
  readonly taskId: string
  readonly parent?: string | null
  readonly after?: string | null
  readonly status?: UpdateDirectiveStatus
  readonly title?: string | null
}

export const handleUpdateDirective = (directive: UpdateDirective, context: TaskDirectiveContext) =>
  Effect.gen(function* () {
    const taskReader = yield* TaskGraphStateReaderTag
    const task = yield* taskReader.getTask(directive.taskId)
    if (!task) {
      const err = taskNotFound(directive.taskId)
      return { success: false, code: err.code, error: err.message } as const
    }

    if (directive.parent !== undefined && directive.parent !== null && directive.parent !== '') {
      const parentTask = yield* taskReader.getTask(directive.parent)
      if (!parentTask) {
        const err = parentNotFound(directive.taskId, directive.parent)
        return { success: false, code: err.code, error: err.message } as const
      }
    }

    if (directive.status === 'completed') {
      const canComplete = yield* taskReader.canComplete(directive.taskId)
      if (!canComplete) {
        const err = completionBlocked(directive.taskId)
        return { success: false, code: err.code, error: err.message } as const
      }
    }

    if (directive.status !== undefined) {
      const validated = buildTaskStatusChangedValidated({
        taskId: directive.taskId,
        patch: { status: directive.status },
      }, { forkId: context.forkId, timestamp: context.timestamp, graph: context.graph }, task.status)
      if (validated.success === false) {
        return { success: false, code: validated.code, error: validated.message } as const
      }

      const bus = yield* WorkerBusTag<AppEvent>()
      yield* bus.publish(validated.event)

      const patch: {
        title?: string
        parentId?: string | null
        after?: string
      } = {}

      if (directive.title !== undefined && directive.title !== null && directive.title.trim() !== '') {
        patch.title = directive.title
      }
      if (directive.parent !== undefined) {
        patch.parentId = directive.parent === '' ? null : directive.parent
      }
      if (directive.after !== undefined && directive.after !== null) {
        patch.after = directive.after
      }

      if (Object.keys(patch).length > 0) {
        yield* bus.publish(buildTaskUpdatedValidated({
          taskId: directive.taskId,
          patch,
        }, { forkId: context.forkId, timestamp: context.timestamp, graph: context.graph }))
      }

      return { success: true } as const
    }

    const patch: {
      title?: string
      parentId?: string | null
      after?: string
    } = {}

    if (directive.title !== undefined && directive.title !== null && directive.title.trim() !== '') {
      patch.title = directive.title
    }
    if (directive.parent !== undefined) {
      patch.parentId = directive.parent === '' ? null : directive.parent
    }
    if (directive.after !== undefined && directive.after !== null) {
      patch.after = directive.after
    }

    if (Object.keys(patch).length === 0) {
      const err = emptyUpdatePatch(directive.taskId)
      return { success: false, code: err.code, error: err.message } as const
    }

    const bus = yield* WorkerBusTag<AppEvent>()
    yield* bus.publish(buildTaskUpdatedValidated({
      taskId: directive.taskId,
      patch,
    }, { forkId: context.forkId, timestamp: context.timestamp, graph: context.graph }))

    return { success: true } as const
  })
