import type { AgentVariant } from '../agents'

export type WorkerAssignee = Exclude<AgentVariant, 'lead' | 'lead-oneshot'>
export type TaskAssignee = 'user' | WorkerAssignee
export type TaskTypeKind = 'leaf' | 'composite' | 'user' | 'generic'

interface TaskTypeBase<TId extends string = string, TKind extends TaskTypeKind = TaskTypeKind> {
  readonly id: TId
  readonly kind: TKind
  readonly label: string
  readonly description: string
  readonly allowedAssignees: readonly TaskAssignee[]
  readonly leadGuidance: string
  readonly criteria: string
}

export interface LeafTaskType<TId extends string = string> extends TaskTypeBase<TId, 'leaf'> {
  readonly allowedAssignees: readonly [WorkerAssignee, ...WorkerAssignee[]]
  readonly workerGuidance: string
}

export interface CompositeTaskType<TId extends string = string> extends TaskTypeBase<TId, 'composite'> {
  readonly allowedAssignees: readonly []
  readonly workerGuidance?: never
}

export interface UserTaskType<TId extends string = string> extends TaskTypeBase<TId, 'user'> {
  readonly allowedAssignees: readonly ['user', ...Array<'user'>]
  readonly workerGuidance?: never
}

export interface GenericTaskType<TId extends string = string> extends TaskTypeBase<TId, 'generic'> {
  readonly allowedAssignees: readonly ['user' | WorkerAssignee, ...Array<'user' | WorkerAssignee>]
  readonly workerGuidance?: string
}

export type TaskTypeDefinition<TId extends string = string> =
  | LeafTaskType<TId>
  | CompositeTaskType<TId>
  | UserTaskType<TId>
  | GenericTaskType<TId>

export function isLeafTaskType(taskType: TaskTypeDefinition): taskType is LeafTaskType {
  return taskType.kind === 'leaf'
}

export function isCompositeTaskType(taskType: TaskTypeDefinition): taskType is CompositeTaskType {
  return taskType.kind === 'composite'
}

export function isUserTaskType(taskType: TaskTypeDefinition): taskType is UserTaskType {
  return taskType.kind === 'user'
}

export function isGenericTaskType(taskType: TaskTypeDefinition): taskType is GenericTaskType {
  return taskType.kind === 'generic'
}
