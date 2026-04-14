import { Context, Effect } from 'effect'
import { type TaskAssignee } from '../tasks'
import {
  canCompleteTask,
  collectSubtreeTaskIds,
  type TaskGraphState,
  type TaskRecord,
} from '../projections/task-graph'

export interface TaskGraphStateReader {
  readonly getTask: (id: string) => Effect.Effect<TaskRecord | undefined>
  readonly getState: () => Effect.Effect<TaskGraphState>
  readonly getChildren: (id: string) => Effect.Effect<readonly TaskRecord[]>
  readonly canComplete: (id: string) => Effect.Effect<boolean>
  readonly canAssign: (id: string, assignee: TaskAssignee) => Effect.Effect<boolean>
  readonly getSubtree: (id: string) => Effect.Effect<readonly TaskRecord[]>
}

export function getChildRecords(state: TaskGraphState, id: string): readonly TaskRecord[] {
  const task = state.tasks.get(id)
  if (!task) return []
  return task.childIds
    .map((childId) => state.tasks.get(childId))
    .filter((child): child is TaskRecord => child !== undefined)
}

export function canCompleteRecord(state: TaskGraphState, id: string): boolean {
  const task = state.tasks.get(id)
  if (!task) return false
  return canCompleteTask(state, id)
}

export function canAssignRecord(state: TaskGraphState, id: string, _assignee: TaskAssignee): boolean {
  return state.tasks.has(id)
}

export function collectSubtreeRecords(state: TaskGraphState, id: string): readonly TaskRecord[] {
  return collectSubtreeTaskIds(state, id)
    .map((taskId) => state.tasks.get(taskId))
    .filter((task): task is TaskRecord => task !== undefined)
}

export class TaskGraphStateReaderTag extends Context.Tag('TaskGraphStateReader')<
  TaskGraphStateReaderTag,
  TaskGraphStateReader
>() {}