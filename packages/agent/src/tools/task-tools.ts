import { Effect } from 'effect'
import { Fork, WorkerBusTag } from '@magnitudedev/event-core'
import { ConversationStateReaderTag } from './memory-reader'
import { TaskGraphStateReaderTag } from './task-reader'
import type { TaskStatus } from '../projections/task-graph'
import {
  buildAgentContext,
  buildConversationSummary,
  formatDuplicateTaskIdError,
  formatInvalidAssigneeError,
  formatInvalidTaskTypeError,
  formatMissingAssignmentMessageError,
  formatTaskCompletionBlockedError,
  formatTaskNotFoundError,
  formatTaskParentNotFoundError,
} from '../prompts'
import {
  getTaskTypeDefinition,
  isTaskAssigneeAllowed,
  isValidTaskType,
  parseTaskAssignee,
  type TaskTypeId,
  type WorkerAssignee,
} from '../tasks'
import { getSpawnableVariants } from '../agents'
import type { AppEvent } from '../events'

const { ForkContext } = Fork

export interface CreateTaskOperationInput {
  readonly taskId: string
  readonly type: string
  readonly parent: string | null
  readonly after?: string | null
  readonly title: string
}

export type TaskOpResult =
  | { success: true }
  | { success: false; error: string }

export const createTaskOperation = (input: CreateTaskOperationInput) => Effect.gen(function* () {
  const normalizedType = input.type.trim().toLowerCase()

  if (!isValidTaskType(normalizedType)) {
    return { success: false, error: formatInvalidTaskTypeError(input.taskId, input.type) } as const
  }

  const taskReader = yield* TaskGraphStateReaderTag
  const existingTask = yield* taskReader.getTask(input.taskId)
  if (existingTask) {
    return { success: false, error: formatDuplicateTaskIdError(input.taskId) } as const
  }

  if (input.parent) {
    const parentTask = yield* taskReader.getTask(input.parent)
    if (!parentTask) {
      return { success: false, error: formatTaskParentNotFoundError(input.taskId, input.parent) } as const
    }
  }

  const bus = yield* WorkerBusTag<AppEvent>()
  const { forkId } = yield* ForkContext

  yield* bus.publish({
    type: 'task_created',
    forkId,
    taskId: input.taskId,
    title: input.title.trim(),
    taskType: normalizedType,
    parentId: input.parent ?? null,
    after: input.after ?? undefined,
    timestamp: Date.now(),
  })

  return { success: true } as const
})

export interface UpdateTaskOperationInput {
  readonly taskId: string
  readonly parent?: string | null
  readonly after?: string | null
  readonly status?: TaskStatus
  readonly title?: string | null
}

export const updateTaskOperation = (input: UpdateTaskOperationInput) => Effect.gen(function* () {
  const taskReader = yield* TaskGraphStateReaderTag
  const task = yield* taskReader.getTask(input.taskId)
  if (!task) {
    return { success: false, error: formatTaskNotFoundError(input.taskId) } as const
  }

  if (input.parent !== undefined && input.parent !== null && input.parent !== '') {
    const parentTask = yield* taskReader.getTask(input.parent)
    if (!parentTask) {
      return { success: false, error: formatTaskParentNotFoundError(input.taskId, input.parent) } as const
    }
  }

  if (input.status === 'completed') {
    const canComplete = yield* taskReader.canComplete(input.taskId)
    if (!canComplete) {
      return { success: false, error: formatTaskCompletionBlockedError(input.taskId) } as const
    }
  }

  const patch: {
    title?: string
    parentId?: string | null
    after?: string
    status?: TaskStatus
  } = {}

  if (input.title !== undefined && input.title !== null && input.title.trim() !== '') {
    patch.title = input.title
  }
  if (input.parent !== undefined) {
    patch.parentId = input.parent === '' ? null : input.parent
  }
  if (input.after !== undefined && input.after !== null) {
    patch.after = input.after
  }
  if (input.status !== undefined) {
    patch.status = input.status
  }

  if (Object.keys(patch).length === 0) {
    return { success: true } as const
  }

  const bus = yield* WorkerBusTag<AppEvent>()
  const { forkId } = yield* ForkContext

  yield* bus.publish({
    type: 'task_updated',
    forkId,
    taskId: input.taskId,
    patch,
    timestamp: Date.now(),
  })

  return { success: true } as const
})

export interface AssignTaskOperationInput {
  readonly taskId: string
  readonly assignee: string
  readonly message?: string
  readonly spawnWorker: (params: {
    parentForkId: string | null
    name: string
    agentId: string
    prompt: string
    message: string
    role: WorkerAssignee
    taskId: string
  }) => Effect.Effect<string, never, any>
}

export const assignTaskOperation = (input: AssignTaskOperationInput) => Effect.gen(function* () {
  const normalizedAssignee = input.assignee.trim().toLowerCase()

  const taskReader = yield* TaskGraphStateReaderTag
  const task = yield* taskReader.getTask(input.taskId)
  if (!task) {
    return { success: false, error: formatTaskNotFoundError(input.taskId) } as const
  }

  const parsedAssignee = parseTaskAssignee(normalizedAssignee)
  if (!parsedAssignee) {
    return { success: false, error: formatInvalidAssigneeError(input.taskId, input.assignee) } as const
  }

  if (parsedAssignee !== 'user') {
    const spawnable = getSpawnableVariants()
    if (!spawnable.includes(parsedAssignee)) {
      return { success: false, error: formatInvalidAssigneeError(input.taskId, input.assignee) } as const
    }
  }

  if (!isTaskAssigneeAllowed(task.taskType, parsedAssignee)) {
    return { success: false, error: formatInvalidAssigneeError(input.taskId, input.assignee) } as const
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
      reason: `Reassigned for task "${input.taskId}"`,
    })
  }

  if (parsedAssignee === 'user') {
    yield* bus.publish({
      type: 'task_assigned',
      forkId: parentForkId,
      taskId: input.taskId,
      assignee: 'user',
      workerRole: undefined,
      message: '',
      workerInfo: undefined,
      replacedWorker,
      timestamp,
    })
    return { success: true } as const
  }

  const role: WorkerAssignee = parsedAssignee
  const trimmedMessage = input.message?.trim()
  if (!trimmedMessage) {
    return { success: false, error: formatMissingAssignmentMessageError(input.taskId) } as const
  }

  const conversationReader = yield* ConversationStateReaderTag
  const conversationState = yield* conversationReader.getState()
  const summary = buildConversationSummary(conversationState.entries) ?? ''

  const taskTypeDef = getTaskTypeDefinition(task.taskType as TaskTypeId)
  let taskContract: string | undefined
  if (taskTypeDef.kind === 'leaf') {
    taskContract = [taskTypeDef.workerGuidance, taskTypeDef.criteria].filter(Boolean).join('\n\n')
  } else if (taskTypeDef.kind === 'generic' && taskTypeDef.workerGuidance) {
    taskContract = [taskTypeDef.workerGuidance, taskTypeDef.criteria].filter(Boolean).join('\n\n')
  }

  const prompt = buildAgentContext(task.title, trimmedMessage, summary, input.taskId, taskContract)
  const agentId = `${role}-${input.taskId}`

  const forkId = yield* input.spawnWorker({
    parentForkId,
    name: task.title,
    agentId,
    prompt,
    message: trimmedMessage,
    role,
    taskId: input.taskId,
  })

  yield* bus.publish({
    type: 'task_assigned',
    forkId: parentForkId,
    taskId: input.taskId,
    assignee: role,
    workerRole: role,
    message: trimmedMessage,
    workerInfo: { agentId, forkId, role },
    replacedWorker,
    timestamp,
  })

  return { success: true } as const
})

export interface CancelTaskOperationInput {
  readonly taskId: string
}

export const cancelTaskOperation = (input: CancelTaskOperationInput) => Effect.gen(function* () {
  const taskReader = yield* TaskGraphStateReaderTag
  const target = yield* taskReader.getTask(input.taskId)
  if (!target) {
    return { success: false, error: formatTaskNotFoundError(input.taskId) } as const
  }

  const subtree = yield* taskReader.getSubtree(input.taskId)
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

  yield* bus.publish({
    type: 'task_cancelled',
    forkId: parentForkId,
    taskId: input.taskId,
    cancelledSubtree: subtree.map((t) => t.id),
    killedWorkers,
    timestamp: Date.now(),
  })

  return { success: true } as const
})
