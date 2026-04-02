import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { AgentStatusProjection, type AgentStatusState } from './agent-status'
import { isTaskAssigneeAllowed, type TaskAssignee, type TaskTypeId } from '../tasks'

export type TaskStatus = 'pending' | 'working' | 'completed'

export interface TaskWorkerInfo {
  readonly agentId: string
  readonly forkId: string
  readonly role: string
  readonly message: string
}

export interface TaskRecord {
  readonly id: string
  readonly title: string
  readonly taskType: TaskTypeId
  readonly parentId: string | null
  readonly childIds: readonly string[]
  readonly assignee: TaskAssignee | null
  readonly worker: TaskWorkerInfo | null
  readonly status: TaskStatus
  readonly createdAt: number
  readonly updatedAt: number
  readonly completedAt: number | null
}

export interface TaskGraphState {
  readonly tasks: ReadonlyMap<string, TaskRecord>
  readonly rootTaskIds: readonly string[]
}

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
  readonly reason: 'assignment' | 'worker-working' | 'worker-idle' | 'completion' | 'cancel'
  readonly timestamp: number
}

function getTask(state: TaskGraphState, taskId: string): TaskRecord {
  const task = state.tasks.get(taskId)
  if (!task) throw new Error(`[TaskGraphProjection] Task not found: ${taskId}`)
  return task
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
  return task.childIds.every((childId) => getTask(state, childId).status === 'completed')
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
    throw new Error(`[TaskGraphProjection] Cannot parent task ${taskId} to itself`)
  }

  if (nextParentId !== null) {
    const nextParent = state.tasks.get(nextParentId)
    if (!nextParent) {
      throw new Error(`[TaskGraphProjection] Parent task not found: ${nextParentId}`)
    }

    const subtreeIds = new Set(collectSubtreeTaskIds(state, taskId))
    if (subtreeIds.has(nextParentId)) {
      throw new Error(`[TaskGraphProjection] Cannot reparent ${taskId} under descendant ${nextParentId}`)
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

function deriveStatusFromWorker(task: TaskRecord, agentState: AgentStatusState): TaskStatus {
  if (task.completedAt !== null) return 'completed'
  if (!task.worker) return 'pending'

  const linkedAgentId = agentState.agentByForkId.get(task.worker.forkId)
  if (!linkedAgentId) return 'pending'
  const agent = agentState.agents.get(linkedAgentId)
  if (!agent) return 'pending'

  return agent.status === 'working' || agent.status === 'starting' ? 'working' : 'pending'
}

function findTaskByWorkerAgentId(state: TaskGraphState, agentId: string): TaskRecord | undefined {
  for (const task of state.tasks.values()) {
    if (task.worker?.agentId === agentId) return task
  }
  return undefined
}

export const TaskGraphProjection = Projection.define<AppEvent, TaskGraphState>()(({
  name: 'TaskGraph',

  initial: {
    tasks: new Map(),
    rootTaskIds: [],
  },

  reads: [AgentStatusProjection],

  signals: {
    taskCreated: Signal.create<TaskCreatedSignal>('TaskGraph/taskCreated'),
    taskCompleted: Signal.create<TaskCompletedSignal>('TaskGraph/taskCompleted'),
    taskCancelled: Signal.create<TaskCancelledSignal>('TaskGraph/taskCancelled'),
    taskStatusChanged: Signal.create<TaskStatusChangedSignal>('TaskGraph/taskStatusChanged'),
  },

  eventHandlers: {
    task_created: ({ event, state, emit }) => {
      if (state.tasks.has(event.taskId)) {
        throw new Error(`[TaskGraphProjection] Task already exists: ${event.taskId}`)
      }

      if (event.parentId !== null && !state.tasks.has(event.parentId)) {
        throw new Error(`[TaskGraphProjection] Parent task not found: ${event.parentId}`)
      }

      const task: TaskRecord = {
        id: event.taskId,
        title: event.title,
        taskType: event.taskType,
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

    task_updated: ({ event, state }) => {
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

      return nextState
    },

    task_assigned: ({ event, state, read, emit }) => {
      const current = getTask(state, event.taskId)

      if (!isTaskAssigneeAllowed(current.taskType, event.assignee)) {
        throw new Error(
          `[TaskGraphProjection] Assignee "${event.assignee}" is not allowed for task type "${current.taskType}"`,
        )
      }

      const agentState = read(AgentStatusProjection)
      const worker: TaskWorkerInfo | null = event.assignee === 'self'
        ? null
        : event.workerInfo
          ? {
              agentId: event.workerInfo.agentId,
              forkId: event.workerInfo.forkId,
              role: event.workerInfo.role,
              message: event.message,
            }
          : current.worker

      const nextStatus = event.assignee === 'self'
        ? 'pending'
        : deriveStatusFromWorker({ ...current, worker, completedAt: null }, agentState)

      const next = patchTask(state, event.taskId, (task) => ({
        ...task,
        assignee: event.assignee,
        worker,
        status: nextStatus,
        completedAt: null,
        updatedAt: event.timestamp,
      }))

      if (current.status !== nextStatus) {
        emit.taskStatusChanged({
          taskId: event.taskId,
          previous: current.status,
          next: nextStatus,
          reason: 'assignment',
          timestamp: event.timestamp,
        })
      }

      return next
    },

    task_completed: ({ event, state, emit }) => {
      const current = getTask(state, event.taskId)

      if (!canCompleteTask(state, event.taskId)) {
        throw new Error(`[TaskGraphProjection] Cannot complete task ${event.taskId}: incomplete child tasks`)
      }

      if (current.status === 'completed') return state

      const next = patchTask(state, event.taskId, (task) => ({
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
        previous: current.status,
        next: 'completed',
        reason: 'completion',
        timestamp: event.timestamp,
      })

      return next
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

  signalHandlers: (on) => [
    on(AgentStatusProjection.signals.agentBecameWorking, ({ value, state, read, emit }) => {
      const task = findTaskByWorkerAgentId(state, value.agentId)
      if (!task || task.status === 'completed') return state

      const agentState = read(AgentStatusProjection)
      const nextStatus = deriveStatusFromWorker(task, agentState)

      if (nextStatus === task.status) return state

      const next = patchTask(state, task.id, (record) => ({
        ...record,
        status: nextStatus,
        updatedAt: value.timestamp,
      }))

      emit.taskStatusChanged({
        taskId: task.id,
        previous: task.status,
        next: nextStatus,
        reason: 'worker-working',
        timestamp: value.timestamp,
      })

      return next
    }),

    on(AgentStatusProjection.signals.agentBecameIdle, ({ value, state, read, emit }) => {
      const task = findTaskByWorkerAgentId(state, value.agentId)
      if (!task || task.status === 'completed') return state

      const agentState = read(AgentStatusProjection)
      const nextStatus = deriveStatusFromWorker(task, agentState)

      if (nextStatus === task.status) return state

      const next = patchTask(state, task.id, (record) => ({
        ...record,
        status: nextStatus,
        updatedAt: value.timestamp,
      }))

      emit.taskStatusChanged({
        taskId: task.id,
        previous: task.status,
        next: nextStatus,
        reason: 'worker-idle',
        timestamp: value.timestamp,
      })

      return next
    }),
  ],
}))