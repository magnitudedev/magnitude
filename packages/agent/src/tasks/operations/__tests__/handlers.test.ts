import { describe, expect, test } from 'bun:test'
import { Effect, Stream } from 'effect'
import { Fork, WorkerBusTag, type WorkerBusService } from '@magnitudedev/event-core'
import type { AppEvent } from '../../../events'
import type { ConversationState } from '../../../projections/conversation'
import type { TaskGraphState, TaskRecord } from '../../../projections/task-graph'
import { ConversationStateReaderTag } from '../../../tools/memory-reader'
import { TaskGraphStateReaderTag, type TaskGraphStateReader } from '../../../tools/task-reader'
import { handleAssignDirective } from '../assign'
import { handleReassignDirective } from '../reassign'
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

  test('assign on unassigned task with role spawns worker', async () => {
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

    const result = await runOp(handleAssignDirective({
      kind: 'assign',
      taskId: 't2',
      assignee: 'builder',
      message: 'Do the work',
      spawnWorker: () => Effect.succeed('fork-2'),
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state, published)

    expect(result.success).toBe(true)
    expect(published.some((e) => e.type === 'task_assigned')).toBe(true)
    expect(published.some((e) => e.type === 'agent_killed')).toBe(false)
  })

  test('assign on task with active worker returns success without kill or spawn', async () => {
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

    const result = await runOp(handleAssignDirective({
      kind: 'assign',
      taskId: 't3',
      assignee: null,
      message: 'Continue working',
      spawnWorker: () => Effect.succeed('should-not-spawn'),
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state, published)

    expect(result.success).toBe(true)
    expect(published.some((e) => e.type === 'agent_killed')).toBe(false)
    expect(published.some((e) => e.type === 'task_assigned')).toBe(false)
  })

  test('assign on active worker with same role succeeds', async () => {
    const published: AppEvent[] = []
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t4', {
          id: 't4', title: 'Task', taskType: 'implement', status: 'pending',
          parentId: null, childIds: [], assignee: 'builder',
          worker: { agentId: 'worker-4', forkId: 'fork-4', role: 'builder' as const, message: 'Existing' },
          createdAt: 0, updatedAt: 0, completedAt: null,
        }],
      ]),
      rootTaskIds: ['t4'],
    })

    const result = await runOp(handleAssignDirective({
      kind: 'assign',
      taskId: 't4',
      assignee: 'builder',
      message: 'Keep going',
      spawnWorker: () => Effect.succeed('should-not-spawn'),
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state, published)

    expect(result.success).toBe(true)
    expect(published.some((e) => e.type === 'agent_killed')).toBe(false)
    expect(published.some((e) => e.type === 'task_assigned')).toBe(false)
  })

  test('assign on active worker with different role returns mismatch error', async () => {
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

    const result = await runOp(handleAssignDirective({
      kind: 'assign',
      taskId: 't5',
      assignee: 'reviewer',
      message: 'Switch role',
      spawnWorker: () => Effect.succeed('should-not-spawn'),
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state)

    expect(result.success).toBe(false)
    if (!result.success) {
      expect(result.error).toBe(
        'Task assignment rejected: task "t5" already has active worker role "builder". Use <reassign> to replace the worker with role "reviewer".',
      )
    }
  })

  test('assign on unassigned task without role returns missing-role error', async () => {
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t6', {
          id: 't6', title: 'Task', taskType: 'implement', status: 'pending',
          parentId: null, childIds: [], assignee: null, worker: null,
          createdAt: 0, updatedAt: 0, completedAt: null,
        }],
      ]),
      rootTaskIds: ['t6'],
    })

    const result = await runOp(handleAssignDirective({
      kind: 'assign',
      taskId: 't6',
      assignee: null,
      message: 'Need worker',
      spawnWorker: () => Effect.succeed('should-not-spawn'),
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state)

    expect(result.success).toBe(false)
    if (!result.success) {
      expect(result.error).toBe(
        'Task assignment rejected: role is required when task "t6" has no active worker.',
      )
    }
  })

  test('reassign kills old worker and spawns new worker', async () => {
    const published: AppEvent[] = []
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t7', {
          id: 't7', title: 'Task', taskType: 'implement', status: 'pending',
          parentId: null, childIds: [], assignee: 'builder',
          worker: { agentId: 'worker-7', forkId: 'fork-7', role: 'builder' as const, message: 'Existing' },
          createdAt: 0, updatedAt: 0, completedAt: null,
        }],
      ]),
      rootTaskIds: ['t7'],
    })

    const result = await runOp(handleReassignDirective({
      kind: 'reassign',
      taskId: 't7',
      assignee: 'builder',
      message: 'Replace worker',
      spawnWorker: () => Effect.succeed('fork-7-new'),
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state, published)

    expect(result.success).toBe(true)
    expect(published.some((e) => e.type === 'agent_killed')).toBe(true)
    expect(published.some((e) => e.type === 'task_assigned')).toBe(true)
  })

  test('reassign without role returns missing-role error', async () => {
    const state = mkTaskState({
      tasks: new Map<string, TaskRecord>([
        ['t8', {
          id: 't8', title: 'Task', taskType: 'implement', status: 'pending',
          parentId: null, childIds: [], assignee: 'builder',
          worker: { agentId: 'worker-8', forkId: 'fork-8', role: 'builder' as const, message: 'Existing' },
          createdAt: 0, updatedAt: 0, completedAt: null,
        }],
      ]),
      rootTaskIds: ['t8'],
    })

    const result = await runOp(handleReassignDirective({
      kind: 'reassign',
      taskId: 't8',
      assignee: null,
      message: 'Replace worker',
      spawnWorker: () => Effect.succeed('should-not-spawn'),
    }, { forkId: null, timestamp: Date.now(), graph: { tasks: new Map() } }), state)

    expect(result.success).toBe(false)
    if (!result.success) {
      expect(result.error).toBe(
        'Task reassignment rejected: role is required for <reassign> on task "t8".',
      )
    }
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
