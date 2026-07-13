import { Effect, Option, Schema } from 'effect'
import { defineHarnessTool, StreamValidationError } from '@magnitudedev/harness'
import { Fork, WorkerBusTag } from '@magnitudedev/event-core'
import { ExecutionManager } from '../execution/types'
import { TaskGraphStateReaderTag } from './task-reader'
import { AgentStateReaderTag } from './fork'
import { formatTaskOutsideSubtreeError } from '../prompts/error-states'
import { canTransition, isTaskStatus, type TaskStatus } from '../projections/task-graph'
import { buildAgentContext, buildConversationSummary } from '../prompts'
import { ConversationStateReaderTag } from './memory-reader'
import { isSpawnableRole, getSpawnableRoles, type RoleId } from '../agents/role-validation'
import { ToolErrorSchema } from './errors'
import type { AppEvent } from '../events'

const TaskToolErrorSchema = ToolErrorSchema('TaskToolError', {})

const { ForkContext } = Fork

// ── Error message helpers (single source of truth) ─────────────────

const taskNotFound = (taskId: string) =>
  `Task operation rejected: task "${taskId}" does not exist.`

const parentNotFound = (taskId: string, parentId: string) =>
  `Task operation rejected: parent task "${parentId}" was not found for task "${taskId}".`

const duplicateTaskId = (taskId: string) =>
  `Task creation rejected: task "${taskId}" already exists. Task IDs must be unique.`

const completionBlocked = (taskId: string) =>
  `Task update rejected: cannot mark task "${taskId}" as completed while child tasks are incomplete.`

const invalidStatusTransition = (taskId: string, from: string, to: string) =>
  `Task update rejected: cannot transition task "${taskId}" from "${from}" to "${to}".`

const emptyUpdatePatch = (taskId: string) =>
  `Task update rejected: no changes provided for task "${taskId}".`

const workerNotFound = (taskId: string) =>
  `Worker operation rejected: task "${taskId}" has no active worker.`

const taskHasWorker = (taskId: string) =>
  `Task "${taskId}" already has a worker assigned. Only one worker may be assigned to a task. To work in parallel, create another task. To replace the worker, kill the existing one first.`

const agentNotFound = (agentId: string) =>
  `Agent operation rejected: no agent found with ID "${agentId}".`

// ── Worker subtree guard ────────────────────────────────────────────

const isTaskInAssignedSubtree = (
  tasks: ReadonlyMap<string, { parentId: string | null }>,
  candidateParentId: string,
  assignedTaskId: string,
): boolean => {
  let current: string | null = candidateParentId
  while (current !== null) {
    if (current === assignedTaskId) return true
    current = tasks.get(current)?.parentId ?? null
  }
  return false
}

// ── Tool failure helper ─────────────────────────────────────────────

const taskFail = (message: string) =>
  Effect.fail({ _tag: 'TaskToolError' as const, message })

// ── Execute functions (exported for testability) ────────────────────

export const executeCreateTask = (input: {
  taskId: string
  parentId: string | null
  title: string
  after?: string | null
}) =>
  Effect.gen(function* () {
    const taskReader = yield* TaskGraphStateReaderTag
    const { forkId } = yield* ForkContext

    // Worker subtree guard — workers can only create subtasks under their assigned task
    if (forkId !== null) {
      const agentStateReader = yield* AgentStateReaderTag
      const agentState = yield* agentStateReader.getAgentState()
      const agentId = agentState.agentByForkId.get(forkId)
      const assignedTaskId = agentId ? agentState.agents.get(agentId)?.taskId?.trim() : null

      if (assignedTaskId) {
        const parentId = input.parentId ?? null
        const allowed =
          parentId !== null && isTaskInAssignedSubtree((yield* taskReader.getState()).tasks, parentId, assignedTaskId)

        if (!allowed) {
          const attemptedParent = parentId ?? '(none)'
          return yield* taskFail(formatTaskOutsideSubtreeError(input.taskId, attemptedParent, assignedTaskId))
        }
      }
    }

    // Validate: no duplicate task ID
    const existingTask = yield* taskReader.getTask(input.taskId)
    if (existingTask) {
      return yield* taskFail(duplicateTaskId(input.taskId))
    }

    // Validate: parent exists
    if (input.parentId) {
      const parentTask = yield* taskReader.getTask(input.parentId)
      if (!parentTask) {
        return yield* taskFail(parentNotFound(input.taskId, input.parentId))
      }

      // Reopen completed parent when adding a child
      if (parentTask.status === 'completed') {
        const bus = yield* WorkerBusTag<AppEvent>()
        const timestamp = Date.now()
        yield* bus.publish({
          type: 'task_updated',
          forkId,
          taskId: parentTask.id,
          patch: { status: 'pending' as TaskStatus },
          timestamp,
        })
      }
    }

    // Publish task_created
    const bus = yield* WorkerBusTag<AppEvent>()
    const timestamp = Date.now()
    yield* bus.publish({
      type: 'task_created',
      forkId,
      taskId: input.taskId,
      title: input.title.trim(),
      parentId: input.parentId,
      after: input.after ?? undefined,
      timestamp,
    })

    return { taskId: input.taskId }
  })

export const executeUpdateTask = (input: {
  taskId: string
  status?: 'pending' | 'completed' | 'cancelled'
  parent?: string | null
  after?: string | null
  title?: string | null
}) =>
  Effect.gen(function* () {
    // Cancel is a special case — delegates to cancel logic
    if (input.status === 'cancelled') {
      return yield* executeCancelTask({ taskId: input.taskId })
    }

    const taskReader = yield* TaskGraphStateReaderTag
    const task = yield* taskReader.getTask(input.taskId)
    if (!task) {
      return yield* taskFail(taskNotFound(input.taskId))
    }

    // Validate parent if specified
    if (input.parent !== undefined && input.parent !== null && input.parent !== '') {
      const parentTask = yield* taskReader.getTask(input.parent)
      if (!parentTask) {
        return yield* taskFail(parentNotFound(input.taskId, input.parent))
      }
    }

    const { forkId } = yield* ForkContext
    const bus = yield* WorkerBusTag<AppEvent>()
    const timestamp = Date.now()

    // Status transition
    if (input.status !== undefined) {
      const requestedStatus = input.status

      // Validate transition is legal
      if (!isTaskStatus(requestedStatus) || !isTaskStatus(task.status)) {
        return yield* taskFail(invalidStatusTransition(input.taskId, task.status, String(requestedStatus)))
      }

      if (!canTransition(task.status, requestedStatus)) {
        return yield* taskFail(invalidStatusTransition(input.taskId, task.status, requestedStatus))
      }

      // Completed: check children are all done
      if (requestedStatus === 'completed') {
        const canComplete = yield* taskReader.canComplete(input.taskId)
        if (!canComplete) {
          return yield* taskFail(completionBlocked(input.taskId))
        }
      }

      yield* bus.publish({
        type: 'task_updated',
        forkId,
        taskId: input.taskId,
        patch: { status: requestedStatus as TaskStatus },
        timestamp,
      })
    }

    // Non-status fields
    const patch: {
      title?: string
      parentId?: string | null
      after?: string
    } = {}

    if (input.title !== undefined && input.title !== null && input.title.trim() !== '') {
      patch.title = input.title
    }
    if (input.parent !== undefined) {
      patch.parentId = input.parent === '' ? null : input.parent
    }
    if (input.after !== undefined && input.after !== null) {
      patch.after = input.after
    }

    if (Object.keys(patch).length > 0) {
      yield* bus.publish({
        type: 'task_updated',
        forkId,
        taskId: input.taskId,
        patch,
        timestamp,
      })
    }

    // Reject if nothing changed at all
    if (input.status === undefined && Object.keys(patch).length === 0) {
      return yield* taskFail(emptyUpdatePatch(input.taskId))
    }

    return { taskId: input.taskId, status: input.status ?? task.status }
  })

export const executeCancelTask = (input: { taskId: string }) =>
  Effect.gen(function* () {
    const taskReader = yield* TaskGraphStateReaderTag
    const target = yield* taskReader.getTask(input.taskId)
    if (!target) {
      return yield* taskFail(taskNotFound(input.taskId))
    }

    const subtree = yield* taskReader.getSubtree(input.taskId)
    const { forkId: parentForkId } = yield* ForkContext
    const bus = yield* WorkerBusTag<AppEvent>()
    const killedWorkers: Array<{ agentId: string; forkId: string }> = []

    // Kill all workers in the subtree
    for (const task of subtree) {
      if (!task.worker) continue
      killedWorkers.push({ agentId: task.worker.agentId, forkId: task.worker.forkId })
      yield* bus.publish({
        type: 'agent_killed',
        forkId: task.worker.forkId,
        parentForkId,
        agentId: task.worker.agentId,
        reason: `Task "${task.id}" cancelled`,
      })
    }

    yield* bus.publish({
      type: 'task_cancelled',
      forkId: parentForkId,
      taskId: input.taskId,
      cancelledSubtree: subtree.map((t) => t.id),
      killedWorkers,
      timestamp: Date.now(),
    })

    return { taskId: input.taskId, status: 'cancelled' as const }
  })

export const executeSpawnWorker = (input: {
  taskId: string
  agentId: string
  message: string
  role: RoleId
  yield?: boolean
}) =>
  Effect.gen(function* () {
    const taskReader = yield* TaskGraphStateReaderTag
    const task = yield* taskReader.getTask(input.taskId)
    if (!task) {
      return yield* taskFail(taskNotFound(input.taskId))
    }

    if (task.worker) {
      return yield* taskFail(taskHasWorker(input.taskId))
    }

    // Build prompt for the forked agent
    const conversationReader = yield* ConversationStateReaderTag
    const conversationState = yield* conversationReader.getState()
    const summary = buildConversationSummary(conversationState.entries)
    const prompt = buildAgentContext(task.title, summary, input.taskId)

    // Fork the worker
    const execManager = yield* ExecutionManager
    const { forkId: parentForkId } = yield* ForkContext
    const forkId = yield* execManager.fork({
      parentForkId,
      name: task.title,
      agentId: input.agentId,
      prompt,
      message: input.message,
      mode: 'spawn',
      role: input.role,
      taskId: input.taskId,
    })

    // Publish assignment event
    const bus = yield* WorkerBusTag<AppEvent>()
    const timestamp = Date.now()
    yield* bus.publish({
      type: 'task_assigned',
      forkId: parentForkId,
      taskId: input.taskId,
      assignee: 'worker',
      workerRole: input.role,
      message: input.message,
      workerInfo: { agentId: input.agentId, forkId, role: input.role },
      timestamp,
    })

    return {
      taskId: input.taskId,
      agentId: input.agentId,
      title: task.title,
      yield: input.yield ? Option.some(true) : Option.none<boolean>(),
    }
  })

export const executeKillWorker = (input: { taskId: string }) =>
  Effect.gen(function* () {
    const taskReader = yield* TaskGraphStateReaderTag
    const task = yield* taskReader.getTask(input.taskId)
    if (!task) {
      return yield* taskFail(taskNotFound(input.taskId))
    }

    if (!task.worker) {
      return yield* taskFail(workerNotFound(input.taskId))
    }

    const bus = yield* WorkerBusTag<AppEvent>()
    const { forkId: parentForkId } = yield* ForkContext
    const timestamp = Date.now()

    // Kill the agent
    yield* bus.publish({
      type: 'agent_killed',
      forkId: task.worker.forkId,
      parentForkId,
      agentId: task.worker.agentId,
      reason: `Killed for task "${input.taskId}"`,
    })

    // Clear the assignment
    yield* bus.publish({
      type: 'task_assigned',
      forkId: parentForkId,
      taskId: input.taskId,
      assignee: task.assignee ?? 'user',
      workerRole: undefined,
      message: '',
      workerInfo: undefined,
      replacedWorker: { agentId: task.worker.agentId, forkId: task.worker.forkId },
      timestamp,
    })

    return { taskId: input.taskId }
  })

export const executeReassignWorker = (input: {
  agentId: string
  targetTaskId: string
}) =>
  Effect.gen(function* () {
    const agentStateReader = yield* AgentStateReaderTag
    const agentState = yield* agentStateReader.getAgentState()

    // Validate agent exists
    const agent = agentState.agents.get(input.agentId)
    if (!agent) {
      return yield* taskFail(agentNotFound(input.agentId))
    }

    // Validate current task exists
    const taskReader = yield* TaskGraphStateReaderTag
    const currentTask = yield* taskReader.getTask(agent.taskId)
    if (!currentTask) {
      return yield* taskFail(taskNotFound(agent.taskId))
    }

    // Validate target task exists
    const targetTask = yield* taskReader.getTask(input.targetTaskId)
    if (!targetTask) {
      return yield* taskFail(taskNotFound(input.targetTaskId))
    }

    // Validate target task has no worker already
    if (targetTask.worker) {
      return yield* taskFail(taskHasWorker(input.targetTaskId))
    }

    const bus = yield* WorkerBusTag<AppEvent>()
    yield* bus.publish({
      type: 'agent_task_changed',
      forkId: agent.forkId,
      agentId: input.agentId,
      oldTaskId: agent.taskId,
      newTaskId: input.targetTaskId,
    })

    return { agentId: input.agentId, taskId: input.targetTaskId }
  })

// ── Tool definitions ─────────────────────────────────────────────────

const UpdateTaskStatusSchema = Schema.Literal('pending', 'completed', 'cancelled')

export const createTaskTool = defineHarnessTool({
  definition: {
    name: 'create_task',
    description: 'Create a task.',
    inputSchema: Schema.Struct({
      taskId: Schema.String.annotations({ description: 'Unique task identifier' }),
      title: Schema.String.annotations({ description: 'Task title' }),
      parent: Schema.optionalWith(Schema.String.annotations({ description: 'Parent task ID to nest under; omit if no parent' }), { as: 'Option', exact: true }),
      after: Schema.optionalWith(Schema.String.annotations({ description: 'Task ID to insert after; for ordering among siblings' }), { as: 'Option', exact: true }),
    }),
    outputSchema: Schema.Struct({ taskId: Schema.String }),
  },
  errorSchema: TaskToolErrorSchema,
  stream: {
    initial: {},
    onInput: (input, _state, _ctx) => Effect.gen(function* () {
      const parent = input.parent?.value
      if (!input.parent?.isFinal || typeof parent !== 'string' || parent.length === 0) return {}
      const taskReader = yield* TaskGraphStateReaderTag
      const graphState = yield* taskReader.getState()
      if (!graphState.tasks.has(parent)) {
        const validIds = [...graphState.tasks.keys()].slice(0, 20).join(', ')
        return yield* new StreamValidationError({
          message: `Parent task not found: ${parent}. Valid IDs: ${validIds}`,
        })
      }
      return {}
    }),
  },
  execute: (input, _ctx) =>
    executeCreateTask({
      taskId: input.taskId,
      parentId: Option.match(input.parent, { onNone: () => null, onSome: (parent) => parent.trim() || null }),
      title: input.title,
      after: Option.match(input.after, { onNone: () => null, onSome: (after) => after.trim() || null }),
    }),
})

export const updateTaskTool = defineHarnessTool({
  definition: {
    name: 'update_task',
    description: 'Update task status.',
    inputSchema: Schema.Struct({
      taskId: Schema.String.annotations({ description: 'Task ID to update' }),
      status: UpdateTaskStatusSchema.annotations({ description: 'New status: pending, completed, or cancelled' }),
    }),
    outputSchema: Schema.Struct({
      taskId: Schema.String,
      status: UpdateTaskStatusSchema,
    }),
  },
  errorSchema: TaskToolErrorSchema,
  stream: {
    initial: {},
    onInput: (input, _state, _ctx) => Effect.gen(function* () {
      if (!input.taskId?.isFinal) return {}
      const taskReader = yield* TaskGraphStateReaderTag
      const graphState = yield* taskReader.getState()
      if (!graphState.tasks.has(input.taskId.value)) {
        const validIds = [...graphState.tasks.keys()].slice(0, 20).join(', ')
        return yield* new StreamValidationError({
          message: `Task not found: ${input.taskId.value}. Valid IDs: ${validIds}`,
        })
      }
      return {}
    }),
  },
  execute: (input, _ctx) =>
    executeUpdateTask({
      taskId: input.taskId,
      status: input.status,
    }),
})

export const spawnWorkerTool = defineHarnessTool({
  definition: {
    name: 'spawn_worker',
    description: 'Spawn a worker with a given role. Must be attached to a task. Only one worker can be assigned per task. Create another task for parallel work.',
    inputSchema: Schema.Struct({
      taskId: Schema.String.annotations({ description: 'Task ID to spawn a worker for' }),
      role: Schema.String.annotations({ description: 'Worker role (e.g., engineer, scout, architect, critic, scientist, artisan).' }),
      agentId: Schema.String.annotations({ description: 'Unique agent ID for this worker. Use this ID to message or reassign the worker later.' }),
      message: Schema.String.annotations({ description: 'Initial instruction message for the worker' }),
      yield: Schema.optionalWith(Schema.Boolean.annotations({ description: 'Set true to wait for this worker to respond before doing anything else.' }), { as: 'Option', exact: true }),
    }),
    outputSchema: Schema.Struct({
      taskId: Schema.String,
      agentId: Schema.String,
      title: Schema.String,
      yield: Schema.optionalWith(Schema.Boolean, { as: 'Option', exact: true }),
    }),
  },
  errorSchema: TaskToolErrorSchema,
  stream: {
    initial: {},
    onInput: (input, _state, _ctx) => Effect.gen(function* () {
      if (!input.taskId?.isFinal) return {}

      const taskReader = yield* TaskGraphStateReaderTag
      const graphState = yield* taskReader.getState()

      const task = graphState.tasks.get(input.taskId.value)
      if (!task) {
        const validIds = [...graphState.tasks.keys()].slice(0, 20).join(', ')
        return yield* new StreamValidationError({
          message: `Task not found: ${input.taskId.value}. Valid IDs: ${validIds}`,
        })
      }

      if (task.worker) {
        return yield* new StreamValidationError({
          message: taskHasWorker(input.taskId.value),
        })
      }

      if (input.agentId?.isFinal) {
        const agentStateReader = yield* AgentStateReaderTag
        const agentState = yield* agentStateReader.getAgentState()
        if (agentState.agents.has(input.agentId.value)) {
          return yield* new StreamValidationError({
            message: `Agent ${input.agentId.value} already exists. Use a unique agentId.`,
          })
        }
      }

      return {}
    }),
  },
  execute: (input, _ctx) =>
    Effect.gen(function* () {
      if (!isSpawnableRole(input.role)) {
        return yield* taskFail(
          `Invalid worker role "${input.role}". Valid roles: ${getSpawnableRoles().join(', ')}`,
        )
      }
      return yield* executeSpawnWorker({
        taskId: input.taskId,
        agentId: input.agentId,
        message: input.message,
        role: input.role as RoleId,
        ...Option.match(input.yield, {
          onNone: () => ({}),
          onSome: (yieldValue) => ({ yield: yieldValue }),
        }),
      })
    }),
})

export const killWorkerTool = defineHarnessTool({
  definition: {
    name: 'kill_worker',
    description: 'Kill worker for a task id.',
    inputSchema: Schema.Struct({
      taskId: Schema.String.annotations({ description: 'Task ID whose worker to kill' }),
    }),
    outputSchema: Schema.Struct({
      taskId: Schema.String,
    }),
  },
  errorSchema: TaskToolErrorSchema,
  stream: {
    initial: {},
    onInput: (input, _state, _ctx) => Effect.gen(function* () {
      if (!input.taskId?.isFinal) return {}
      const taskReader = yield* TaskGraphStateReaderTag
      const graphState = yield* taskReader.getState()
      if (!graphState.tasks.has(input.taskId.value)) {
        const validIds = [...graphState.tasks.keys()].slice(0, 20).join(', ')
        return yield* new StreamValidationError({
          message: `Task not found: ${input.taskId.value}. Valid IDs: ${validIds}`,
        })
      }
      return {}
    }),
  },
  execute: (input, _ctx) => executeKillWorker({ taskId: input.taskId }),
})

export const reassignWorkerTool = defineHarnessTool({
  definition: {
    name: 'reassign_worker',
    description: 'Reassign a worker from its current task to a different task. The worker keeps its identity and conversation history.',
    inputSchema: Schema.Struct({
      agentId: Schema.String.annotations({ description: 'Agent ID of the worker to reassign' }),
      taskId: Schema.String.annotations({ description: 'Task ID to reassign the worker to' }),
    }),
    outputSchema: Schema.Struct({
      agentId: Schema.String,
      taskId: Schema.String,
    }),
  },
  errorSchema: TaskToolErrorSchema,
  stream: {
    initial: {},
    onInput: (input, _state, _ctx) => Effect.gen(function* () {
      if (input.agentId?.isFinal) {
        const agentStateReader = yield* AgentStateReaderTag
        const agentState = yield* agentStateReader.getAgentState()
        if (!agentState.agents.has(input.agentId.value)) {
          const validIds = [...agentState.agents.keys()].slice(0, 20).join(', ')
          return yield* new StreamValidationError({
            message: `Agent not found: ${input.agentId.value}. Valid IDs: ${validIds}`,
          })
        }
      }

      if (input.taskId?.isFinal) {
        const taskReader = yield* TaskGraphStateReaderTag
        const graphState = yield* taskReader.getState()
        if (!graphState.tasks.has(input.taskId.value)) {
          const validIds = [...graphState.tasks.keys()].slice(0, 20).join(', ')
          return yield* new StreamValidationError({
            message: `Task not found: ${input.taskId.value}. Valid IDs: ${validIds}`,
          })
        }
      }

      return {}
    }),
  },
  execute: (input, _ctx) =>
    executeReassignWorker({
      agentId: input.agentId,
      targetTaskId: input.taskId,
    }),
})
