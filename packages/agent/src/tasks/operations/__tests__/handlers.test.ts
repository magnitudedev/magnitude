import { describe, expect, test } from 'bun:test'
import { Effect, Stream } from 'effect'
import { Fork, WorkerBusTag, type WorkerBusService } from '@magnitudedev/event-core'
import type { AppEvent } from '../../../events'
import type { ConversationState } from '../../../projections/conversation'
import type { TaskGraphState, TaskRecord } from '../../../projections/task-graph'
import { ConversationStateReaderTag } from '../../../tools/memory-reader'
import { TaskGraphStateReaderTag, type TaskGraphStateReader } from '../../../tools/task-reader'
import { handleMessageDirective } from '../message'
import { handleUpdateDirective } from '../update'
import { handleSpawnWorkerDirective } from '../spawn-worker'
import { handleKillWorkerDirective } from '../kill-worker'

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
    const task = state.tasks.get(id)
    return Effect.succeed(task ? [task] : [])
  },
})

function runOp<A>(
  effect: Effect.Effect<
    A,
    never,
    Fork.ForkContextService | TaskGraphStateReaderTag | ConversationStateReaderTag | WorkerBusService<AppEvent>
  >,
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
      Effect.provideService(Fork.ForkContext, { forkId: null }),
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

describe('task operation handlers validation', () => {
  test('top-level message respects parent default destination', async () => {
    const result = await runOp(handleMessageDirective({
      kind: 'message',
      defaultTopLevelDestination: 'parent',
      allowSingleUserReplyThisTurn: false,
      directUserRepliesSent: 0,
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), mkTaskState())

    expect(result.success).toBe(true)
    if (result.success) {
      expect(result.destination).toBe('parent')
      expect(result.directUserRepliesSent).toBe(0)
    }
  })

  test('top-level message routes user reply to parent after first direct reply', async () => {
    const result = await runOp(handleMessageDirective({
      kind: 'message',
      defaultTopLevelDestination: 'user',
      allowSingleUserReplyThisTurn: true,
      directUserRepliesSent: 1,
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), mkTaskState())

    expect(result.success).toBe(true)
    if (result.success) {
      expect(result.destination).toBe('parent')
      expect(result.directUserRepliesSent).toBe(1)
    }
  })

  test('spawn-worker on unassigned task spawns worker and publishes task_assigned', async () => {
    const published: AppEvent[] = []
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t2', {
          id: 't2', title: 'New Task', taskType: 'implement', status: 'pending',
          parentId: null, childIds: [], assignee: null, worker: null,
          createdAt: 0, updatedAt: 0, completedAt: null,
        }],
      ]),
      rootTaskIds: ['t2'],
    })

    const result = await runOp(handleSpawnWorkerDirective({
      kind: 'spawn-worker',
      id: 't2',
      role: 'builder',
      spawnWorker: () => Effect.succeed('fork-2'),
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state, published)

    expect(result.success).toBe(true)
    expect(published.some((e) => e.type === 'task_assigned')).toBe(true)
    expect(published.some((e) => e.type === 'agent_killed')).toBe(false)
  })

  test('spawn-worker on assigned task kills existing worker first', async () => {
    const published: AppEvent[] = []
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t3', {
          id: 't3', title: 'Task', taskType: 'implement', status: 'pending',
          parentId: null, childIds: [], assignee: 'builder',
          worker: { agentId: 'worker-3', forkId: 'fork-3', role: 'builder' as const, message: 'Existing' },
          createdAt: 0, updatedAt: 0, completedAt: null,
        }],
      ]),
      rootTaskIds: ['t3'],
    })

    const result = await runOp(handleSpawnWorkerDirective({
      kind: 'spawn-worker',
      id: 't3',
      role: 'builder',
      spawnWorker: () => Effect.succeed('fork-3-new'),
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state, published)

    expect(result.success).toBe(true)
    expect(published.some((e) => e.type === 'agent_killed')).toBe(true)
    expect(published.some((e) => e.type === 'task_assigned')).toBe(true)
  })

  test('kill-worker on task without worker errors', async () => {
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t4', {
          id: 't4', title: 'Task', taskType: 'implement', status: 'pending',
          parentId: null, childIds: [], assignee: null, worker: null,
          createdAt: 0, updatedAt: 0, completedAt: null,
        }],
      ]),
      rootTaskIds: ['t4'],
    })

    const result = await runOp(handleKillWorkerDirective({
      kind: 'kill-worker',
      id: 't4',
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state)

    expect(result.success).toBe(false)
    if (!result.success) {
      expect(result.code).toBe('worker_not_found')
    }
  })

  test('kill-worker kills existing worker and clears assignment', async () => {
    const published: AppEvent[] = []
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t5', {
          id: 't5', title: 'Task', taskType: 'implement', status: 'pending',
          parentId: null, childIds: [], assignee: 'builder',
          worker: { agentId: 'worker-5', forkId: 'fork-5', role: 'builder' as const, message: 'Existing' },
          createdAt: 0, updatedAt: 0, completedAt: null,
        }],
      ]),
      rootTaskIds: ['t5'],
    })

    const result = await runOp(handleKillWorkerDirective({
      kind: 'kill-worker',
      id: 't5',
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state, published)

    expect(result.success).toBe(true)
    expect(published.some((e) => e.type === 'agent_killed')).toBe(true)
    expect(published.some((e) => e.type === 'task_assigned')).toBe(true)
  })

  test('empty update patch returns error', async () => {
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t1', { id: 't1', title: 'Task', taskType: 'implement', status: 'pending', parentId: null, childIds: [], assignee: null, worker: null, createdAt: 0, updatedAt: 0, completedAt: null }],
      ]),
      rootTaskIds: ['t1'],
    })

    const result = await runOp(handleUpdateDirective({
      kind: 'update',
      taskId: 't1',
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state)

    expect(result.success).toBe(false)
    if (!result.success) {
      expect(result.code).toBe('empty_update_patch')
    }
  })

  test('invalid status transition returns error', async () => {
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t1', { id: 't1', title: 'Task', taskType: 'implement', status: 'pending', parentId: null, childIds: [], assignee: null, worker: null, createdAt: 0, updatedAt: 0, completedAt: null }],
      ]),
      rootTaskIds: ['t1'],
    })

    const result = await runOp(handleUpdateDirective({
      kind: 'update',
      taskId: 't1',
      status: 'pending',
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state)

    expect(result.success).toBe(false)
    if (!result.success) {
      expect(result.code).toBe('invalid_status_transition')
    }
  })
})
