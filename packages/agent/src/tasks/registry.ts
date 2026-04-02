import type { TaskAssignee, TaskTypeDefinition } from './types'
import {
  approveTaskType,
  bugTaskType,
  featureTaskType,
  groupTaskType,
  implementTaskType,
  otherTaskType,
  planTaskType,
  refactorTaskType,
  researchTaskType,
  reviewTaskType,
} from './definitions'

export type TaskTypeId = 'feature' | 'bug' | 'refactor' | 'research' | 'plan' | 'implement' | 'review' | 'approve' | 'group' | 'other'

export const TASK_TYPES: Record<TaskTypeId, TaskTypeDefinition> = {
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
}

export function isValidTaskType(value: string): value is TaskTypeId {
  return Object.hasOwn(TASK_TYPES, value)
}

export function getTaskTypeDefinition(taskType: TaskTypeId): TaskTypeDefinition {
  return TASK_TYPES[taskType]
}

export function listTaskTypeDefinitions(): readonly TaskTypeDefinition[] {
  return Object.values(TASK_TYPES)
}

export function isTaskAssigneeAllowed(taskType: TaskTypeId, assignee: TaskAssignee): boolean {
  return TASK_TYPES[taskType].allowedAssignees.includes(assignee)
}

export function getTaskTypeStrategy(taskType: TaskTypeId): string {
  return TASK_TYPES[taskType].strategy
}
