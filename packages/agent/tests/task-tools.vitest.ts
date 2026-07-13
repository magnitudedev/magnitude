import { describe, expect, it } from '@effect/vitest'
import { Effect, Stream } from 'effect'
import { Fork, WorkerBusTag, type WorkerBusService } from '@magnitudedev/event-core'
import type { AppEvent } from '../src/events'
import type { ConversationState } from '../src/projections/conversation'
import { type TaskGraphState, type TaskRecord } from '../src/projections/task-graph'
import { ConversationStateReaderTag } from '../src/tools/memory-reader'
import { TaskGraphStateReaderTag, type TaskGraphStateReader } from '../src/tools/task-reader'
import {
  executeKillWorker,
  executeUpdateTask,
  executeCancelTask,
} from '../src/tools/task-tools'

const mkTaskState = (overrides?: Partial<TaskGraphState>): TaskGraphState => ({
  tasks: new Map(),
  rootTaskIds: [],
  ...overrides,
})

const mkTaskReader = (state: TaskGraphState): TaskGraphStateReader => ({
  getTask: (id) => Effect.succeed(state.tasks.get(id)),
  getState: () => Effect.succeed(state),
  getChildren: (id) => {
    const task = state.tasks.get(id)
    if (!task) return Effect.succeed([])
    return Effect.succeed(
      task.childIds.map((childId) => state.tasks.get(childId)).filter((t): t is NonNullable<typeof t> => t !== undefined),
    )
  },
  canComplete: (id) => {
    const task = state.tasks.get(id)
    if (!task) return Effect.succeed(false)
    return Effect.succeed(task.childIds.every((childId) => {
      const child = state.tasks.get(childId)
      return child && child.status === 'completed'
    }))
  },
  canAssign: () => Effect.succeed(true),
  getSubtree: (id) => {
    // DFS traversal matching the real collectSubtreeRecords implementation
    const result: TaskRecord[] = []
    const stack = [id]
    while (stack.length > 0) {
      const taskId = stack.pop()
      if (!taskId) continue
      const task = state.tasks.get(taskId)
      if (!task) continue
      result.push(task)
      for (const childId of task.childIds) stack.push(childId)
    }
    return Effect.succeed(result)
  },
})

function runOp<A, E>(
  effect: Effect.Effect<A, E, Fork.ForkContextService | TaskGraphStateReaderTag | ConversationStateReaderTag | WorkerBusService<AppEvent>>,
  state: TaskGraphState,
  published: AppEvent[] = [],
) {
  const bus: WorkerBusService<AppEvent> = {
    publish: (event) => Effect.sync(() => {
      published.push(event)
    }),
    subscribeToTypes: () => Stream.empty,
    stream: Stream.empty,
    subscribe: () => Effect.succeed(Stream.empty),
  }

  return Effect.runPromise(
    effect.pipe(
      Effect.provideService(Fork.ForkContext, { forkId: null, roleId: 'leader' }),
      Effect.provideService(TaskGraphStateReaderTag, mkTaskReader(state)),
      Effect.provideService(ConversationStateReaderTag, {
        getState: () => Effect.succeed({
          entries: [],
          pendingProse: '',
          userMessageIds: new Set(),
        } satisfies ConversationState),
      }),
      Effect.provideService(WorkerBusTag<AppEvent>(), bus),
    ),
  )
}

type TaskToolError = { readonly _tag: 'TaskToolError'; readonly message: string }

function isTaskToolError(value: unknown): value is TaskToolError {
  return typeof value === 'object' && value !== null && '_tag' in value && (value as TaskToolError)._tag === 'TaskToolError'
}

describe('task tool execute functions', () => {
  it('kill_worker on task without worker errors', async () => {
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t4', {
          id: 't4', title: 'Task', status: 'pending',
          parentId: null, childIds: [], assignee: null, worker: null,
          createdAt: 0, updatedAt: 0, completedAt: null,
        }],
      ]),
      rootTaskIds: ['t4'],
    })

    const result = await runOp(
      executeKillWorker({ taskId: 't4' }).pipe(Effect.catchAll((e) => Effect.succeed(e))),
      state,
    )

    expect(isTaskToolError(result)).toBe(true)
    if (isTaskToolError(result)) {
      expect(result.message).toContain('no active worker')
    }
  })

  it('kill_worker kills existing worker and clears assignment', async () => {
    const published: AppEvent[] = []
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t5', {
          id: 't5', title: 'Task', status: 'pending',
          parentId: null, childIds: [], assignee: 'worker',
          worker: { agentId: 'worker-5', forkId: 'fork-5', role: 'engineer' as const, message: 'Existing' },
          createdAt: 0, updatedAt: 0, completedAt: null,
        }],
      ]),
      rootTaskIds: ['t5'],
    })

    const result = await runOp(executeKillWorker({ taskId: 't5' }), state, published)

    expect(result).toEqual({ taskId: 't5' })
    expect(published.some((e) => e.type === 'agent_killed')).toBe(true)
    expect(published.some((e) => e.type === 'task_assigned')).toBe(true)
  })

  it('empty update patch returns error', async () => {
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t1', { id: 't1', title: 'Task', status: 'pending', parentId: null, childIds: [], assignee: null, worker: null, createdAt: 0, updatedAt: 0, completedAt: null }],
      ]),
      rootTaskIds: ['t1'],
    })

    const result = await runOp(
      executeUpdateTask({ taskId: 't1' }).pipe(Effect.catchAll((e) => Effect.succeed(e))),
      state,
    )

    expect(isTaskToolError(result)).toBe(true)
    if (isTaskToolError(result)) {
      expect(result.message).toContain('no changes provided')
    }
  })

  it('invalid status transition returns error', async () => {
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t1', { id: 't1', title: 'Task', status: 'pending', parentId: null, childIds: [], assignee: null, worker: null, createdAt: 0, updatedAt: 0, completedAt: null }],
      ]),
      rootTaskIds: ['t1'],
    })

    const result = await runOp(
      executeUpdateTask({ taskId: 't1', status: 'pending' }).pipe(Effect.catchAll((e) => Effect.succeed(e))),
      state,
    )

    expect(isTaskToolError(result)).toBe(true)
    if (isTaskToolError(result)) {
      expect(result.message).toContain('cannot transition')
    }
  })

  it('cancel on nonexistent task errors', async () => {
    const state = mkTaskState()

    const result = await runOp(
      executeCancelTask({ taskId: 'nonexistent' }).pipe(Effect.catchAll((e) => Effect.succeed(e))),
      state,
    )

    expect(isTaskToolError(result)).toBe(true)
    if (isTaskToolError(result)) {
      expect(result.message).toContain('does not exist')
    }
  })

  it('cancel kills workers in subtree and publishes events', async () => {
    const published: AppEvent[] = []
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t1', {
          id: 't1', title: 'Parent', status: 'pending',
          parentId: null, childIds: ['t2'], assignee: null, worker: null,
          createdAt: 0, updatedAt: 0, completedAt: null,
        }],
        ['t2', {
          id: 't2', title: 'Child', status: 'pending',
          parentId: 't1', childIds: [], assignee: 'worker',
          worker: { agentId: 'w1', forkId: 'f1', role: 'engineer' as const, message: 'Work' },
          createdAt: 0, updatedAt: 0, completedAt: null,
        }],
      ]),
      rootTaskIds: ['t1'],
    })

    const result = await runOp(executeCancelTask({ taskId: 't1' }), state, published)

    expect(result).toEqual({ taskId: 't1', status: 'cancelled' })
    expect(published.some((e) => e.type === 'agent_killed')).toBe(true)
    expect(published.some((e) => e.type === 'task_cancelled')).toBe(true)
  })

  it('update_task with status=cancelled delegates to cancel', async () => {
    const published: AppEvent[] = []
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t1', {
          id: 't1', title: 'Task', status: 'pending',
          parentId: null, childIds: [], assignee: null, worker: null,
          createdAt: 0, updatedAt: 0, completedAt: null,
        }],
      ]),
      rootTaskIds: ['t1'],
    })

    const result = await runOp(
      executeUpdateTask({ taskId: 't1', status: 'cancelled' }),
      state,
      published,
    )

    expect(result).toEqual({ taskId: 't1', status: 'cancelled' })
    expect(published.some((e) => e.type === 'task_cancelled')).toBe(true)
  })
})
