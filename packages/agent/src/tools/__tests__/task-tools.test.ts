import { describe, expect, test } from 'bun:test'
import { Effect, Stream } from 'effect'
import { Fork, WorkerBusTag, type WorkerBusService } from '@magnitudedev/event-core'
import { createTaskOperation, updateTaskOperation, assignTaskOperation, type TaskOpResult } from '../task-tools'
import { TaskGraphStateReaderTag, type TaskGraphStateReader } from '../task-reader'
import { ConversationStateReaderTag } from '../memory-reader'
import type { TaskGraphState, TaskRecord } from '../../projections/task-graph'
import type { ConversationState } from '../../projections/conversation'
import type { AppEvent } from '../../events'

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
      task.childIds
        .map((childId) => state.tasks.get(childId))
        .filter((t): t is NonNullable<typeof t> => t !== undefined),
    )
  },
  canComplete: (id) => {
    const task = state.tasks.get(id)
    if (!task) return Effect.succeed(false)
    const hasIncompleteChild = task.childIds.some((childId) => {
      const child = state.tasks.get(childId)
      return !child || child.status !== 'completed'
    })
    return Effect.succeed(!hasIncompleteChild)
  },
  canAssign: () => Effect.succeed(true),
  getSubtree: (id) => {
    const task = state.tasks.get(id)
    return Effect.succeed(task ? [task] : [])
  },
})

const runOp = <A>(
  effect: Effect.Effect<A, never, Fork.ForkContextService | TaskGraphStateReaderTag | ConversationStateReaderTag | WorkerBusService<AppEvent>>,
  state: TaskGraphState,
) =>
  Effect.runPromise(
    effect.pipe(
      Effect.provideService(Fork.ForkContext, { forkId: null }),
      Effect.provideService(TaskGraphStateReaderTag, mkTaskReader(state)),
      Effect.provideService(ConversationStateReaderTag, {
        getState: () => Effect.succeed<ConversationState>({
          entries: [],
          pendingProse: '',
          userMessageIds: new Set(),
        }),
      }),
      Effect.provideService(WorkerBusTag<AppEvent>(), {
        publish: () => Effect.void,
        subscribeToTypes: () => Stream.empty,
        stream: Stream.empty,
        subscribe: () => Effect.succeed(Stream.empty),
      }),
    ),
  )

describe('task-tools error results', () => {
  test('createTaskOperation errors on invalid task type', async () => {
    const result = await runOp(
      createTaskOperation({
        taskId: 't1',
        type: 'not-a-type',
        parent: null,
        title: 'Task',
      }),
      mkTaskState(),
    )

    expect((result as TaskOpResult).success).toBe(false)
    if (!result.success) {
      expect(result.error).toContain('invalid type')
    }
  })

  test('createTaskOperation errors on duplicate task id', async () => {
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t1', { id: 't1', title: 'Existing', taskType: 'scan', status: 'pending', parentId: null, childIds: [], assignee: null, worker: null, createdAt: 0, updatedAt: 0, completedAt: null }],
      ]),
      rootTaskIds: ['t1'],
    })

    const result = await runOp(
      createTaskOperation({
        taskId: 't1',
        type: 'scan',
        parent: null,
        title: 'Duplicate',
      }),
      state,
    )

    expect(result.success).toBe(false)
    if (!result.success) {
      expect(result.error).toContain('already exists')
    }
  })

  test('createTaskOperation errors on missing parent', async () => {
    const result = await runOp(
      createTaskOperation({
        taskId: 'child',
        type: 'scan',
        parent: 'missing-parent',
        title: 'Child',
      }),
      mkTaskState(),
    )

    expect(result.success).toBe(false)
    if (!result.success) {
      expect(result.error).toContain('parent task')
    }
  })

  test('updateTaskOperation errors when completing task with incomplete children', async () => {
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['parent', { id: 'parent', title: 'Parent', taskType: 'implement', status: 'pending', parentId: null, childIds: ['child'], assignee: null, worker: null, createdAt: 0, updatedAt: 0, completedAt: null }],
        ['child', { id: 'child', title: 'Child', taskType: 'scan', status: 'pending', parentId: 'parent', childIds: [], assignee: null, worker: null, createdAt: 0, updatedAt: 0, completedAt: null }],
      ]),
      rootTaskIds: ['parent'],
    })

    const result = await runOp(
      updateTaskOperation({
        taskId: 'parent',
        status: 'completed',
      }),
      state,
    )

    expect(result.success).toBe(false)
    if (!result.success) {
      expect(result.error).toContain('cannot mark task "parent" as completed')
    }
  })

  test('assignTaskOperation errors when task does not exist', async () => {
    const result = await runOp(
      assignTaskOperation({
        taskId: 'missing',
        assignee: 'builder',
        message: 'Please do this',
        spawnWorker: () => Effect.succeed('fork-id'),
      }),
      mkTaskState(),
    )

    expect(result.success).toBe(false)
    if (!result.success) {
      expect(result.error).toContain('does not exist')
    }
  })
})
