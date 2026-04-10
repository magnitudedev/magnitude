import { Option } from 'effect'
import type {
  GenericTaskType,
  CompositeTaskType,
  LeafTaskType,
  TaskAssignee,
  TaskTypeDefinition,
  TaskTypeKind,
  UserTaskType,
} from './types'
import {
  isGenericTaskType,
  isCompositeTaskType,
  isLeafTaskType,
  isUserTaskType,
} from './types'
import {
  approveTaskType,
  bugTaskType,
  diagnoseTaskType,
  featureTaskType,
  groupTaskType,
  ideateTaskType,
  implementTaskType,
  otherTaskType,
  planTaskType,
  refactorTaskType,
  researchTaskType,
  reviewTaskType,
  scanTaskType,
  webTestTaskType,
} from './definitions'

export const TASK_TYPES = {
  feature: featureTaskType,
  bug: bugTaskType,
  refactor: refactorTaskType,
  research: researchTaskType,
  plan: planTaskType,
  implement: implementTaskType,
  review: reviewTaskType,
  approve: approveTaskType,
  group: groupTaskType,
  other: otherTaskType,
  scan: scanTaskType,
  diagnose: diagnoseTaskType,
  'web-test': webTestTaskType,
  ideate: ideateTaskType,
} satisfies Record<string, TaskTypeDefinition>

export type TaskTypeId = keyof typeof TASK_TYPES

export function isValidTaskType(value: string): value is TaskTypeId {
  return Object.hasOwn(TASK_TYPES, value)
}

export function getTaskTypeDefinition(taskType: TaskTypeId): TaskTypeDefinition {
  return TASK_TYPES[taskType]
}

export function listTaskTypeDefinitions(): readonly TaskTypeDefinition[] {
  return Object.values(TASK_TYPES)
}

export function getLeafTaskTypeDefinition(taskType: TaskTypeId): Option.Option<LeafTaskType> {
  const def = TASK_TYPES[taskType]
  return isLeafTaskType(def) ? Option.some(def) : Option.none()
}

export function getCompositeTaskTypeDefinition(taskType: TaskTypeId): Option.Option<CompositeTaskType> {
  const def = TASK_TYPES[taskType]
  return isCompositeTaskType(def) ? Option.some(def) : Option.none()
}

export function getUserTaskTypeDefinition(taskType: TaskTypeId): Option.Option<UserTaskType> {
  const def = TASK_TYPES[taskType]
  return isUserTaskType(def) ? Option.some(def) : Option.none()
}

export function getGenericTaskTypeDefinition(taskType: TaskTypeId): Option.Option<GenericTaskType> {
  const def = TASK_TYPES[taskType]
  return isGenericTaskType(def) ? Option.some(def) : Option.none()
}


export function listLeafTaskTypeDefinitions(): readonly LeafTaskType[] {
  return Object.values(TASK_TYPES).filter(isLeafTaskType)
}

export function listCompositeTaskTypeDefinitions(): readonly CompositeTaskType[] {
  return Object.values(TASK_TYPES).filter(isCompositeTaskType)
}

export function listUserTaskTypeDefinitions(): readonly UserTaskType[] {
  return Object.values(TASK_TYPES).filter(isUserTaskType)
}

export function listGenericTaskTypeDefinitions(): readonly GenericTaskType[] {
  return Object.values(TASK_TYPES).filter(isGenericTaskType)
}

export function isTaskTypeKind(taskType: TaskTypeId, kind: TaskTypeKind): boolean {
  return TASK_TYPES[taskType].kind === kind
}

export function isTaskAssigneeAllowed(taskType: TaskTypeId, assignee: TaskAssignee): boolean {
  const allowedAssignees = TASK_TYPES[taskType].allowedAssignees as readonly TaskAssignee[]
  return allowedAssignees.includes(assignee)
}
