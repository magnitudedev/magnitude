import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { Schema } from 'effect'

export const TaskStatusSchema = Schema.Literal('pending', 'completed')
export type TaskStatus = typeof TaskStatusSchema.Type

export const TaskWorkerInfoSchema = Schema.Struct({
  agentId: Schema.String,
  forkId: Schema.String,
  role: Schema.String,
  message: Schema.NullOr(Schema.String),
})
export type TaskWorkerInfo = typeof TaskWorkerInfoSchema.Type

const TaskAssigneeSchema = Schema.Literal('user', 'worker')

export const TaskRecordSchema = Schema.Struct({
  id: Schema.String,
  title: Schema.String,
  parentId: Schema.NullOr(Schema.String),
  childIds: Schema.Array(Schema.String),
  assignee: Schema.NullOr(TaskAssigneeSchema),
  worker: Schema.NullOr(TaskWorkerInfoSchema),
  status: TaskStatusSchema,
  createdAt: Schema.Number,
  updatedAt: Schema.Number,
  completedAt: Schema.NullOr(Schema.Number),
})
export type TaskRecord = typeof TaskRecordSchema.Type

export const TaskGraphStateSchema = Schema.Struct({
  tasks: Schema.ReadonlyMap({ key: Schema.String, value: TaskRecordSchema }),
  rootTaskIds: Schema.Array(Schema.String),
})
export type TaskGraphState = typeof TaskGraphStateSchema.Type

export interface TaskCreatedSignal {
  readonly taskId: string
  readonly parentId: string | null
  readonly timestamp: number
}

export interface TaskCompletedSignal {
  readonly taskId: string
  readonly timestamp: number
}

export interface TaskCancelledSignal {
  readonly taskId: string
  readonly cancelledSubtree: readonly string[]
  readonly timestamp: number
}

export interface TaskStatusChangedSignal {
  readonly taskId: string
  readonly previous: TaskStatus
  readonly next: TaskStatus
  readonly timestamp: number
}

function getTask(state: TaskGraphState, taskId: string): TaskRecord {
  return state.tasks.get(taskId)!
}

export function collectSubtreeTaskIds(state: TaskGraphState, rootTaskId: string): string[] {
  const result: string[] = []
  const stack = [rootTaskId]

  while (stack.length > 0) {
    const taskId = stack.pop()
    if (!taskId) continue

    const task = state.tasks.get(taskId)
    if (!task) continue

    result.push(taskId)
    for (const childId of task.childIds) stack.push(childId)
  }

  return result
}

export function canCompleteTask(state: TaskGraphState, taskId: string): boolean {
  const task = getTask(state, taskId)
  return task.childIds.every((childId) => {
    const childStatus = getTask(state, childId).status
    return childStatus === 'completed'
  })
}

export function patchTask(
  state: TaskGraphState,
  taskId: string,
  updater: (task: TaskRecord) => TaskRecord,
): TaskGraphState {
  const existing = getTask(state, taskId)
  const nextTask = updater(existing)
  const nextTasks = new Map(state.tasks)
  nextTasks.set(taskId, nextTask)
  return { ...state, tasks: nextTasks }
}

function removeFromParentOrRoots(state: TaskGraphState, task: TaskRecord, timestamp: number): TaskGraphState {
  if (task.parentId === null) {
    return {
      ...state,
      rootTaskIds: state.rootTaskIds.filter((id) => id !== task.id),
    }
  }

  return patchTask(state, task.parentId, (parent) => ({
    ...parent,
    childIds: parent.childIds.filter((id) => id !== task.id),
    updatedAt: timestamp,
  }))
}

function insertIntoOrderedIds(ids: readonly string[], taskId: string, after?: string): readonly string[] {
  if (after === undefined) {
    return [...ids, taskId]
  }
  if (after === '') {
    return [taskId, ...ids]
  }
  const idx = ids.indexOf(after)
  if (idx === -1) {
    return [...ids, taskId]
  }
  const result = [...ids]
  result.splice(idx + 1, 0, taskId)
  return result
}

function removeId(ids: readonly string[], taskId: string): readonly string[] {
  return ids.filter((id) => id !== taskId)
}

function addToParentOrRoots(
  state: TaskGraphState,
  taskId: string,
  parentId: string | null,
  timestamp: number,
  after?: string,
): TaskGraphState {
  if (parentId === null) {
    return {
      ...state,
      rootTaskIds: insertIntoOrderedIds(state.rootTaskIds, taskId, after),
    }
  }

  return patchTask(state, parentId, (parent) => ({
    ...parent,
    childIds: insertIntoOrderedIds(parent.childIds, taskId, after),
    updatedAt: timestamp,
  }))
}

export function reparentTask(
  state: TaskGraphState,
  taskId: string,
  nextParentId: string | null,
  timestamp: number,
): TaskGraphState {
  const task = getTask(state, taskId)

  if (nextParentId === taskId) {
    return state
  }

  if (nextParentId !== null) {
    const nextParent = state.tasks.get(nextParentId)
    if (!nextParent) {
      return state
    }

    const subtreeIds = new Set(collectSubtreeTaskIds(state, taskId))
    if (subtreeIds.has(nextParentId)) {
      return state
    }
  }

  let nextState = removeFromParentOrRoots(state, task, timestamp)
  nextState = addToParentOrRoots(nextState, taskId, nextParentId, timestamp, undefined)

  return patchTask(nextState, taskId, (current) => ({
    ...current,
    parentId: nextParentId,
    updatedAt: timestamp,
  }))
}

function findTaskByWorkerAgentId(state: TaskGraphState, agentId: string): TaskRecord | undefined {
  for (const task of state.tasks.values()) {
    if (task.worker?.agentId === agentId) return task
  }
  return undefined
}

export function isTaskStatus(value: string): value is TaskStatus {
  return value === 'pending' || value === 'completed'
}

export function canTransition(current: TaskStatus, requested: TaskStatus): boolean {
  if (current === requested) return false

  switch (current) {
    case 'pending':
      return requested === 'completed'
    case 'completed':
      return requested === 'pending'
  }
}

export const TaskGraphProjection = Projection.define<AppEvent>()(({
  name: 'TaskGraph',
  state: TaskGraphStateSchema,

  initial: {
    tasks: new Map<string, TaskRecord>(),
    rootTaskIds: [],
  },

  reads: [],

  signals: {
    taskCreated: Signal.create<TaskCreatedSignal>('TaskGraph/taskCreated'),
    taskCompleted: Signal.create<TaskCompletedSignal>('TaskGraph/taskCompleted'),
    taskCancelled: Signal.create<TaskCancelledSignal>('TaskGraph/taskCancelled'),
    taskStatusChanged: Signal.create<TaskStatusChangedSignal>('TaskGraph/taskStatusChanged'),
  },

  eventHandlers: {
    task_created: ({ event, state, emit }) => {
      if (state.tasks.has(event.taskId)) {
        return state
      }

      if (event.parentId !== null && !state.tasks.has(event.parentId)) {
        return state
      }

      const task: TaskRecord = {
        id: event.taskId,
        title: event.title,
        parentId: event.parentId,
        childIds: [],
        assignee: null,
        worker: null,
        status: 'pending',
        createdAt: event.timestamp,
        updatedAt: event.timestamp,
        completedAt: null,
      }

      let nextState: TaskGraphState = {
        ...state,
        tasks: new Map(state.tasks).set(event.taskId, task),
      }

      if (event.parentId === null) {
        nextState = {
          ...nextState,
          rootTaskIds: insertIntoOrderedIds(nextState.rootTaskIds, event.taskId, event.after),
        }
      } else {
        nextState = patchTask(nextState, event.parentId, (parent) => ({
          ...parent,
          childIds: insertIntoOrderedIds(parent.childIds, event.taskId, event.after),
          status: parent.status === 'completed' ? 'pending' : parent.status,
          completedAt: parent.status === 'completed' ? null : parent.completedAt,
          updatedAt: event.timestamp,
        }))
      }

      emit.taskCreated({
        taskId: event.taskId,
        parentId: event.parentId,
        timestamp: event.timestamp,
      })

      return nextState
    },

    task_updated: ({ event, state, emit }) => {
      const existing = getTask(state, event.taskId)
      let nextState = state

      if (event.patch.parentId !== undefined && event.patch.parentId !== existing.parentId) {
        // Reparent: remove from old parent, add to new with after
        nextState = removeFromParentOrRoots(nextState, existing, event.timestamp)
        nextState = patchTask(nextState, event.taskId, (task) => ({
          ...task,
          parentId: event.patch.parentId!,
          updatedAt: event.timestamp,
        }))
        nextState = addToParentOrRoots(nextState, event.taskId, event.patch.parentId!, event.timestamp, event.patch.after)
      } else if (event.patch.after !== undefined) {
        // Reorder within same parent
        const parentId = existing.parentId
        if (parentId === null) {
          nextState = {
            ...nextState,
            rootTaskIds: insertIntoOrderedIds(removeId(nextState.rootTaskIds, event.taskId), event.taskId, event.patch.after),
          }
        } else {
          nextState = patchTask(nextState, parentId, (parent) => ({
            ...parent,
            childIds: insertIntoOrderedIds(removeId(parent.childIds, event.taskId), event.taskId, event.patch.after),
            updatedAt: event.timestamp,
          }))
        }
      }

      if (event.patch.title !== undefined) {
        nextState = patchTask(nextState, event.taskId, (task) => ({
          ...task,
          title: event.patch.title ?? task.title,
          updatedAt: event.timestamp,
        }))
      }

      if (event.patch.status !== undefined) {
        if (!isTaskStatus(event.patch.status)) {
          return nextState
        }

        const requestedStatus = event.patch.status
        const current = getTask(nextState, event.taskId)
        const previousStatus = current.status

        if (previousStatus === requestedStatus) {
          return nextState
        }

        if (!canTransition(previousStatus, requestedStatus)) {
          return nextState
        }

        switch (requestedStatus) {
          case 'completed': {
            nextState = patchTask(nextState, event.taskId, (task) => ({
              ...task,
              status: 'completed',
              completedAt: event.timestamp,
              updatedAt: event.timestamp,
            }))

            emit.taskCompleted({
              taskId: event.taskId,
              timestamp: event.timestamp,
            })

            emit.taskStatusChanged({
              taskId: event.taskId,
              previous: previousStatus,
              next: 'completed',
              timestamp: event.timestamp,
            })

            return nextState
          }

          case 'pending': {
            nextState = patchTask(nextState, event.taskId, (task) => ({
              ...task,
              status: 'pending',
              completedAt: null,
              updatedAt: event.timestamp,
            }))

            emit.taskStatusChanged({
              taskId: event.taskId,
              previous: previousStatus,
              next: 'pending',
              timestamp: event.timestamp,
            })

            return nextState
          }

        }
      }

      return nextState
    },

    task_assigned: ({ event, state, emit }) => {
      const current = getTask(state, event.taskId)

      const worker: TaskWorkerInfo | null = event.workerInfo
        ? {
            agentId: event.workerInfo.agentId,
            forkId: event.workerInfo.forkId,
            role: event.workerInfo.role,
            message: event.message,
          }
        : null

      const next = patchTask(state, event.taskId, (task) => ({
        ...task,
        assignee: event.assignee,
        worker,
        updatedAt: event.timestamp,
      }))

      return next
    },

    agent_task_changed: ({ event, state, emit }) => {
      const oldTask = state.tasks.get(event.oldTaskId)
      const newTask = state.tasks.get(event.newTaskId)
      if (!oldTask || !newTask) return state

      const workerInfo = oldTask.worker
      if (!workerInfo || workerInfo.agentId !== event.agentId) return state

      // Remove worker from old task
      let nextState = patchTask(state, event.oldTaskId, (task) => ({
        ...task,
        worker: null,
        updatedAt: event.timestamp,
      }))

      // Assign worker to new task
      const newWorker: TaskWorkerInfo = {
        agentId: workerInfo.agentId,
        forkId: workerInfo.forkId,
        role: workerInfo.role,
        message: workerInfo.message,
      }
      nextState = patchTask(nextState, event.newTaskId, (task) => ({
        ...task,
        worker: newWorker,
        assignee: 'worker' as const,
        updatedAt: event.timestamp,
      }))

      return nextState
    },

    task_cancelled: ({ event, state, emit }) => {
      const target = getTask(state, event.taskId)
      const subtree = new Set(event.cancelledSubtree)
      if (!subtree.has(event.taskId)) {
        subtree.add(event.taskId)
      }

      const nextTasks = new Map(state.tasks)

      for (const taskId of subtree) {
        nextTasks.delete(taskId)
      }

      let nextRootTaskIds = state.rootTaskIds.filter((id) => !subtree.has(id))

      if (target.parentId !== null && !subtree.has(target.parentId) && nextTasks.has(target.parentId)) {
        const parent = nextTasks.get(target.parentId)
        if (parent) {
          nextTasks.set(target.parentId, {
            ...parent,
            childIds: parent.childIds.filter((id) => !subtree.has(id)),
            updatedAt: event.timestamp,
          })
        }
      }

      for (const [id, task] of nextTasks) {
        const filtered = task.childIds.filter((childId) => !subtree.has(childId))
        if (filtered.length !== task.childIds.length) {
          nextTasks.set(id, { ...task, childIds: filtered, updatedAt: event.timestamp })
        }
      }

      nextRootTaskIds = nextRootTaskIds.filter((id) => nextTasks.has(id))

      emit.taskCancelled({
        taskId: event.taskId,
        cancelledSubtree: [...subtree],
        timestamp: event.timestamp,
      })

      return {
        tasks: nextTasks,
        rootTaskIds: nextRootTaskIds,
      }
    },
  },

  signalHandlers: () => [],
}))
