import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema } from '@magnitudedev/tools'
import { defineXmlBinding } from '@magnitudedev/xml-act'
import { Fork, WorkerBusTag } from '@magnitudedev/event-core'
import { ConversationStateReaderTag } from './memory-reader'
import { TaskGraphStateReaderTag } from './task-reader'
import type { TaskStatus } from '../projections/task-graph'
import { buildAgentContext, buildConversationSummary } from '../prompts'
import { formatTaskTypeGuidanceForTool, isTaskAssigneeAllowed, isValidTaskType } from '../tasks'
import { getSpawnableVariants, type AgentVariant } from '../agents'
import type { AppEvent } from '../events'

const { ForkContext } = Fork
const TaskErrorSchema = ToolErrorSchema('TaskError', {})

export const createTaskTool = defineTool({
  name: 'create-task' as const,
  group: 'task' as const,
  description: 'Create a task with a type, optional parent, and title. Returns strategic guidance based on the task type that should be followed.',
  inputSchema: Schema.Struct({
    taskId: Schema.String,
    type: Schema.String,
    parent: Schema.optional(Schema.String),
    after: Schema.optional(Schema.String),
    title: Schema.String,
  }),
  outputSchema: Schema.String,
  errorSchema: TaskErrorSchema,
  execute: ({ taskId, type, parent, after, title }) => Effect.gen(function* () {
    const normalizedType = type.trim().toLowerCase()

    if (!isValidTaskType(normalizedType)) {
      return yield* Effect.fail({
        _tag: 'TaskError' as const,
        message: `Invalid task type "${type}".`,
      })
    }

    const taskReader = yield* TaskGraphStateReaderTag

    if (parent) {
      const parentTask = yield* taskReader.getTask(parent)
      if (!parentTask) {
        return yield* Effect.fail({
          _tag: 'TaskError' as const,
          message: `Cannot create task "${taskId}": parent "${parent}" does not exist.`,
        })
      }
    }

    const bus = yield* WorkerBusTag<AppEvent>()
    const { forkId } = yield* ForkContext

    yield* bus.publish({
      type: 'task_created',
      forkId,
      taskId,
      title: title.trim(),
      taskType: normalizedType,
      parentId: parent ?? null,
      after,
      timestamp: Date.now(),
    })

    return formatTaskTypeGuidanceForTool(normalizedType)
  }),
  label: (input) => input.taskId ? `Creating task ${input.taskId}` : 'Creating task…',
})

export const createTaskXmlBinding = defineXmlBinding(createTaskTool, {
  input: {
    attributes: [
      { field: 'taskId', attr: 'id' },
      { field: 'type', attr: 'type' },
      { field: 'parent', attr: 'parent' },
      { field: 'after', attr: 'after' },
    ],
    body: 'title',
  },
  output: {},
} as const)

export const updateTaskTool = defineTool({
  name: 'update-task' as const,
  group: 'task' as const,
  description: 'Rename, reparent, and/or update task status.',
  inputSchema: Schema.Struct({
    taskId: Schema.String,
    parent: Schema.optional(Schema.String),
    after: Schema.optional(Schema.String),
    status: Schema.optional(Schema.String),
    title: Schema.optional(Schema.String),
  }),
  outputSchema: Schema.Struct({
    taskId: Schema.String,
  }),
  errorSchema: TaskErrorSchema,
  execute: ({ taskId, parent, after, status, title }) => Effect.gen(function* () {
    const hasMutation =
      parent !== undefined || after !== undefined || status !== undefined || title !== undefined
    if (!hasMutation) {
      return yield* Effect.fail({
        _tag: 'TaskError' as const,
        message: 'update-task requires at least one mutation: parent, after, status, or title.',
      })
    }

    const taskReader = yield* TaskGraphStateReaderTag
    const task = yield* taskReader.getTask(taskId)

    if (!task) {
      return yield* Effect.fail({
        _tag: 'TaskError' as const,
        message: `Unknown task "${taskId}".`,
      })
    }

    if (parent !== undefined && parent !== '') {
      const parentTask = yield* taskReader.getTask(parent)
      if (!parentTask) {
        return yield* Effect.fail({
          _tag: 'TaskError' as const,
          message: `Cannot move "${taskId}": parent "${parent}" does not exist.`,
        })
      }
    }

    const bus = yield* WorkerBusTag<AppEvent>()
    const { forkId } = yield* ForkContext
    const timestamp = Date.now()

    const patch: {
      title?: string
      parentId?: string | null
      after?: string
      status?: TaskStatus
    } = {}

    if (title !== undefined && title.trim() !== '') {
      patch.title = title
    }

    if (parent !== undefined) {
      patch.parentId = parent === '' ? null : parent
    }

    if (after !== undefined) {
      patch.after = after
    }

    if (status !== undefined) {
      if (status !== 'pending' && status !== 'completed' && status !== 'archived') {
        return yield* Effect.fail({
          _tag: 'TaskError' as const,
          message: `Invalid status "${status}". Allowed statuses: pending, completed, archived.`,
        })
      }

      if (status === 'completed') {
        if (task.status !== 'pending' && task.status !== 'working' && task.status !== 'archived') {
          return yield* Effect.fail({
            _tag: 'TaskError' as const,
            message: `Task "${taskId}" cannot transition from "${task.status}" to "completed".`,
          })
        }

        const canComplete = yield* taskReader.canComplete(taskId)
        if (!canComplete) {
          return yield* Effect.fail({
            _tag: 'TaskError' as const,
            message: `Task "${taskId}" cannot be completed because it has incomplete child tasks.`,
          })
        }
      }

      if (status === 'archived' && task.status !== 'completed') {
        return yield* Effect.fail({
          _tag: 'TaskError' as const,
          message: `Task "${taskId}" can only be archived from completed status.`,
        })
      }

      if (status === 'pending' && task.status !== 'completed' && task.status !== 'archived') {
        return yield* Effect.fail({
          _tag: 'TaskError' as const,
          message: `Task "${taskId}" can only transition to pending from completed or archived status.`,
        })
      }

      patch.status = status
    }

    if (Object.keys(patch).length > 0) {
      yield* bus.publish({
        type: 'task_updated',
        forkId,
        taskId,
        patch,
        timestamp,
      })
    }

    return { taskId }
  }),
  label: (input) => input.taskId ? `Updating task ${input.taskId}` : 'Updating task…',
})

export const updateTaskXmlBinding = defineXmlBinding(updateTaskTool, {
  input: {
    attributes: [
      { field: 'taskId', attr: 'id' },
      { field: 'parent', attr: 'parent' },
      { field: 'after', attr: 'after' },
      { field: 'status', attr: 'status' },
    ],
    body: 'title',
  },
  output: {
    childTags: [{ field: 'taskId', tag: 'taskId' }],
  },
} as const)

export const assignTaskTool = defineTool({
  name: 'assign-task' as const,
  group: 'task' as const,
  description: 'Assign a task to self or a worker role. Assigning to a worker starts execution.',
  inputSchema: Schema.Struct({
    taskId: Schema.String,
    assignee: Schema.String,
    message: Schema.optional(Schema.String),
  }),
  outputSchema: Schema.Struct({
    taskId: Schema.String,
    agentId: Schema.optional(Schema.String),
    forkId: Schema.optional(Schema.String),
  }),
  errorSchema: TaskErrorSchema,
  execute: ({ taskId, assignee, message }) => Effect.gen(function* () {
    const normalizedAssignee = assignee.trim().toLowerCase()

    const taskReader = yield* TaskGraphStateReaderTag
    const task = yield* taskReader.getTask(taskId)
    if (!task) {
      return yield* Effect.fail({
        _tag: 'TaskError' as const,
        message: `Unknown task "${taskId}".`,
      })
    }

    if (normalizedAssignee !== 'self' && normalizedAssignee !== 'user') {
      const spawnable = getSpawnableVariants()
      if (!spawnable.includes(normalizedAssignee as AgentVariant)) {
        return yield* Effect.fail({
          _tag: 'TaskError' as const,
          message: `Assignee "${assignee}" is not a valid worker role.`,
        })
      }
    }

    if (!isTaskAssigneeAllowed(task.taskType, normalizedAssignee as 'self' | AgentVariant)) {
      return yield* Effect.fail({
        _tag: 'TaskError' as const,
        message: `Assignee "${assignee}" is not allowed for task type "${task.taskType}".`,
      })
    }

    const bus = yield* WorkerBusTag<AppEvent>()
    const { forkId: parentForkId } = yield* ForkContext
    const timestamp = Date.now()

    let replacedWorker: { agentId: string; forkId: string } | undefined = undefined
    if (task.worker) {
      replacedWorker = { agentId: task.worker.agentId, forkId: task.worker.forkId }
      yield* bus.publish({
        type: 'agent_killed',
        forkId: task.worker.forkId,
        parentForkId,
        agentId: task.worker.agentId,
        reason: `Reassigned via assign-task for task "${taskId}"`,
      })
    }

    if (normalizedAssignee === 'self' || normalizedAssignee === 'user') {
      yield* bus.publish({
        type: 'task_assigned',
        forkId: parentForkId,
        taskId,
        assignee: normalizedAssignee,
        workerRole: undefined,
        message: '',
        workerInfo: undefined,
        replacedWorker,
        timestamp,
      })

      return { taskId }
    }

    const trimmedMessage = message?.trim()
    if (!trimmedMessage) {
      return yield* Effect.fail({
        _tag: 'TaskError' as const,
        message: `assign-task requires instructions in the body when assigning to "${normalizedAssignee}".`,
      })
    }

    const conversationReader = yield* ConversationStateReaderTag
    const conversationState = yield* conversationReader.getState()
    const summary = buildConversationSummary(conversationState.entries) ?? ''
    const prompt = buildAgentContext(task.title, trimmedMessage, summary)

    const { ExecutionManager } = yield* Effect.tryPromise({
      try: () => import('../execution/execution-manager'),
      catch: (e) => ({
        _tag: 'TaskError' as const,
        message: e instanceof Error ? e.message : String(e),
      }),
    })

    const executionManager = yield* ExecutionManager
    const role = normalizedAssignee as AgentVariant
    const agentId = `${role}-${taskId}`

    const forkId = yield* executionManager.fork({
      parentForkId,
      name: task.title,
      agentId,
      prompt,
      message: trimmedMessage,
      mode: 'spawn',
      role,
      taskId,
    })

    yield* bus.publish({
      type: 'task_assigned',
      forkId: parentForkId,
      taskId,
      assignee: role,
      workerRole: role,
      message: trimmedMessage,
      workerInfo: {
        agentId,
        forkId,
        role,
      },
      replacedWorker,
      timestamp,
    })

    return { taskId, agentId, forkId }
  }),
  label: (input) => input.taskId ? `Assigning task ${input.taskId}` : 'Assigning task…',
})

export const assignTaskXmlBinding = defineXmlBinding(assignTaskTool, {
  input: {
    attributes: [
      { field: 'taskId', attr: 'id' },
      { field: 'assignee', attr: 'assignee' },
    ],
    body: 'message',
  },
  output: {
    childTags: [
      { field: 'taskId', tag: 'taskId' },
      { field: 'agentId', tag: 'agentId' },
      { field: 'forkId', tag: 'forkId' },
    ],
  },
} as const)

export const cancelTaskTool = defineTool({
  name: 'cancel-task' as const,
  group: 'task' as const,
  description: 'Cancel a task subtree and kill any linked workers.',
  inputSchema: Schema.Struct({
    taskId: Schema.String,
  }),
  outputSchema: Schema.Struct({
    taskId: Schema.String,
    cancelledCount: Schema.Number,
    workersKilled: Schema.Number,
  }),
  errorSchema: TaskErrorSchema,
  execute: ({ taskId }) => Effect.gen(function* () {
    const taskReader = yield* TaskGraphStateReaderTag
    const target = yield* taskReader.getTask(taskId)

    if (!target) {
      return yield* Effect.fail({
        _tag: 'TaskError' as const,
        message: `Unknown task "${taskId}".`,
      })
    }

    const subtree = yield* taskReader.getSubtree(taskId)
    const { forkId: parentForkId } = yield* ForkContext
    const bus = yield* WorkerBusTag<AppEvent>()

    const killedWorkers: Array<{ agentId: string; forkId: string }> = []

    for (const task of subtree) {
      if (!task.worker) continue
      killedWorkers.push({
        agentId: task.worker.agentId,
        forkId: task.worker.forkId,
      })

      yield* bus.publish({
        type: 'agent_killed',
        forkId: task.worker.forkId,
        parentForkId,
        agentId: task.worker.agentId,
        reason: `Task "${task.id}" cancelled`,
      })
    }

    yield* bus.publish({
      type: 'task_cancelled',
      forkId: parentForkId,
      taskId,
      cancelledSubtree: subtree.map((t) => t.id),
      killedWorkers,
      timestamp: Date.now(),
    })

    return {
      taskId,
      cancelledCount: subtree.length,
      workersKilled: killedWorkers.length,
    }
  }),
  label: (input) => input.taskId ? `Cancelling task ${input.taskId}` : 'Cancelling task…',
})

export const cancelTaskXmlBinding = defineXmlBinding(cancelTaskTool, {
  input: {
    attributes: [{ field: 'taskId', attr: 'id' }],
  },
  output: {
    childTags: [
      { field: 'taskId', tag: 'taskId' },
      { field: 'cancelledCount', tag: 'cancelledCount' },
      { field: 'workersKilled', tag: 'workersKilled' },
    ],
  },
} as const)

export const taskTools = [createTaskTool, updateTaskTool, assignTaskTool, cancelTaskTool] as const
