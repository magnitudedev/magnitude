import type { TaskAssigned, TaskCancelled, TaskCreated, TaskUpdated } from '../../events'
import type { TaskStatus } from '../../projections/task-graph'
import { invalidStatusTransition } from './errors'
import type { Validated } from './events'
import { canTransition, hasIncompleteChildren, isValidStatus } from './rules'
import type {
  AssignTaskDirectiveInput,
  CancelTaskDirectiveInput,
  CreateTaskDirectiveInput,
  RouteTaskMessageDirectiveInput,
  TaskOperationContext,
  TaskOperationResult,
  UpdateTaskDirectiveInput,
} from './types'

function baseEvent(context: TaskOperationContext) {
  return {
    forkId: context.forkId,
    timestamp: context.timestamp,
  } as const
}

export function buildTaskCreatedValidated(
  input: CreateTaskDirectiveInput,
  context: TaskOperationContext,
): Validated<TaskCreated> {
  return {
    type: 'task_created',
    ...baseEvent(context),
    taskId: input.taskId,
    title: input.title,
    parentId: input.parentId,
    after: input.after,
  } as Validated<TaskCreated>
}

export function buildTaskStatusChangedValidated(
  input: UpdateTaskDirectiveInput,
  context: TaskOperationContext,
  currentStatus: string,
): TaskOperationResult<Validated<TaskUpdated>> {
  const requestedStatus = input.patch.status
  if (!requestedStatus || !isValidStatus(requestedStatus) || !isValidStatus(currentStatus)) {
    const err = invalidStatusTransition(input.taskId, currentStatus, String(requestedStatus))
    return { success: false, code: err.code, message: err.message }
  }

  if (!canTransition(currentStatus, requestedStatus)) {
    const err = invalidStatusTransition(input.taskId, currentStatus, requestedStatus)
    return { success: false, code: err.code, message: err.message }
  }

  if (requestedStatus === 'completed' && hasIncompleteChildren(input.taskId, context.graph)) {
    const err = invalidStatusTransition(input.taskId, currentStatus, requestedStatus)
    return { success: false, code: err.code, message: err.message }
  }

  const event: Validated<TaskUpdated> = {
    type: 'task_updated',
    ...baseEvent(context),
    taskId: input.taskId,
    patch: {
      status: requestedStatus as TaskStatus,
    },
  } as Validated<TaskUpdated>

  return { success: true, event }
}

export function buildTaskUpdatedValidated(
  input: UpdateTaskDirectiveInput,
  context: TaskOperationContext,
): Validated<TaskUpdated> {
  return {
    type: 'task_updated',
    ...baseEvent(context),
    taskId: input.taskId,
    patch: input.patch,
  } as Validated<TaskUpdated>
}

export function buildTaskAssignedValidated(
  input: AssignTaskDirectiveInput,
  context: TaskOperationContext,
): Validated<TaskAssigned> {
  return {
    type: 'task_assigned',
    ...baseEvent(context),
    taskId: input.taskId,
    assignee: input.assignee,
    workerRole: input.workerRole,
    message: input.message,
    workerInfo: input.workerInfo,
    replacedWorker: input.replacedWorker,
  } as Validated<TaskAssigned>
}

export function buildTaskCancelledValidated(
  input: CancelTaskDirectiveInput,
  context: TaskOperationContext,
): Validated<TaskCancelled> {
  return {
    type: 'task_cancelled',
    ...baseEvent(context),
    taskId: input.taskId,
    cancelledSubtree: input.cancelledSubtree,
    killedWorkers: input.killedWorkers,
  } as Validated<TaskCancelled>
}

type TaskMessageRoutedEvent = {
  readonly type: 'task_message_routed'
  readonly forkId: string | null
  readonly taskId: string
  readonly destinationAgentId: string
  readonly timestamp: number
}

export function buildTaskMessageRoutedValidated(
  input: RouteTaskMessageDirectiveInput,
  context: TaskOperationContext,
): Validated<TaskMessageRoutedEvent> {
  return {
    type: 'task_message_routed',
    ...baseEvent(context),
    taskId: input.taskId,
    destinationAgentId: input.destinationAgentId,
  } as Validated<TaskMessageRoutedEvent>
}
