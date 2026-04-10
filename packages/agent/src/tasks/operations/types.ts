import type { Fork, WorkerBusService } from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import type { ConversationStateReaderTag } from '../../tools/memory-reader'
import type { TaskGraphStateReaderTag } from '../../tools/task-reader'
import type { TaskAssignee } from '../types'
import type { TaskTypeId } from '../registry'
import type { TaskStatus } from '../../projections/task-graph'

export interface TaskOperationTaskSnapshot {
  readonly id: string
  readonly status: TaskStatus
  readonly parentId: string | null
  readonly childIds: readonly string[]
  readonly worker: {
    readonly agentId: string
    readonly forkId: string
    readonly role: string
  } | null
}

export interface TaskOperationGraphSnapshot {
  readonly tasks: ReadonlyMap<string, TaskOperationTaskSnapshot>
}

export interface TaskOperationContext {
  readonly forkId: string | null
  readonly timestamp: number
  readonly graph: TaskOperationGraphSnapshot
}

export type TaskOperationEnv =
  | Fork.ForkContextService
  | TaskGraphStateReaderTag
  | ConversationStateReaderTag
  | WorkerBusService<AppEvent>

export interface CreateTaskDirectiveInput {
  readonly taskId: string
  readonly title: string
  readonly taskType: TaskTypeId
  readonly parentId: string | null
  readonly after?: string
}

export type TaskDirectiveStatus = 'pending' | 'completed' | 'cancelled'

export function isTaskDirectiveStatus(value: string): value is TaskDirectiveStatus {
  return value === 'pending' || value === 'completed' || value === 'cancelled'
}

export interface UpdateTaskDirectiveInput {
  readonly taskId: string
  readonly patch: {
    readonly title?: string
    readonly parentId?: string | null
    readonly after?: string
    readonly status?: Exclude<TaskDirectiveStatus, 'cancelled'>
  }
}

export interface AssignTaskDirectiveInput {
  readonly taskId: string
  readonly assignee: TaskAssignee
  readonly message: string
  readonly workerRole?: string
  readonly workerInfo?: {
    readonly agentId: string
    readonly forkId: string
    readonly role: string
  }
  readonly replacedWorker?: {
    readonly agentId: string
    readonly forkId: string
  }
}

export interface CancelTaskDirectiveInput {
  readonly taskId: string
  readonly cancelledSubtree: readonly string[]
  readonly killedWorkers: readonly {
    readonly agentId: string
    readonly forkId: string
  }[]
}

export interface RouteTaskMessageDirectiveInput {
  readonly taskId: string
  readonly destinationAgentId: string
}

export interface TaskOperationErrorEnvelope {
  readonly success: false
  readonly code: string
  readonly message: string
}

export interface TaskOperationSuccessEnvelope<TEvent> {
  readonly success: true
  readonly event: TEvent
}

export type TaskOperationResult<TEvent> =
  | TaskOperationSuccessEnvelope<TEvent>
  | TaskOperationErrorEnvelope
