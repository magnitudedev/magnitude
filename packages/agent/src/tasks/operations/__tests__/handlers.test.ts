import { describe, expect, test } from 'bun:test'
import { Effect, Stream } from 'effect'
import { Fork, WorkerBusTag, type WorkerBusService } from '@magnitudedev/event-core'
import type { AppEvent } from '../../../events'
import type { ConversationState } from '../../../projections/conversation'
import type { TaskGraphState, TaskRecord } from '../../../projections/task-graph'
import { ConversationStateReaderTag } from '../../../tools/memory-reader'
import { TaskGraphStateReaderTag, type TaskGraphStateReader } from '../../../tools/task-reader'
import { handleAssignDirective } from '../assign'
import { handleMessageDirective } from '../message'
import { handleUpdateDirective } from '../update'

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
      return child && (child.status === 'completed' || child.status === 'archived')
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
  test('task-scope message to missing task returns explicit error', async () => {
    const result = await runOp(handleMessageDirective({
      kind: 'message',
      scope: 'task',
      taskId: 'missing-task',
      defaultTopLevelDestination: 'parent',
      allowSingleUserReplyThisTurn: false,
      directUserRepliesSent: 0,
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), mkTaskState())

    expect(result.success).toBe(false)
    if (!result.success) {
      expect(result.code).toBe('task_message_route_failed')
    }
  })

  test('task-scope message to task with no worker returns explicit error', async () => {
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t1', { id: 't1', title: 'Task', taskType: 'implement', status: 'pending', parentId: null, childIds: [], assignee: null, worker: null, createdAt: 0, updatedAt: 0, completedAt: null }],
      ]),
      rootTaskIds: ['t1'],
    })

    const result = await runOp(handleMessageDirective({
      kind: 'message',
      scope: 'task',
      taskId: 't1',
      defaultTopLevelDestination: 'parent',
      allowSingleUserReplyThisTurn: false,
      directUserRepliesSent: 0,
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state)

    expect(result.success).toBe(false)
    if (!result.success) {
      expect(result.code).toBe('invalid_task_message_route')
    }
  })

  test('assign with missing message does not kill existing worker', async () => {
    const published: AppEvent[] = []
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t1', {
          id: 't1',
          title: 'Task',
          taskType: 'implement',
          status: 'pending',
          parentId: null,
          childIds: [],
          assignee: 'builder',
          worker: { agentId: 'worker-1', forkId: 'fork-1', role: 'builder' as const, message: 'Existing assignment' },
          createdAt: 0,
          updatedAt: 0,
          completedAt: null,
        }],
      ]),
      rootTaskIds: ['t1'],
    })

    const result = await runOp(handleAssignDirective({
      kind: 'assign',
      taskId: 't1',
      assignee: 'builder',
      message: '   ',
      spawnWorker: () => Effect.succeed('fork-new'),
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state, published)

    expect(result.success).toBe(false)
    if (!result.success) {
      expect(result.code).toBe('missing_assignment_message')
    }
    expect(published.some((e) => e.type === 'agent_killed')).toBe(false)
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
