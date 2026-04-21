import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema } from '@magnitudedev/tools'
import { Fork } from '@magnitudedev/event-core'
import { ExecutionManager } from '../execution/types'
import { TaskGraphStateReaderTag } from './task-reader'
import { AgentStateReaderTag } from './fork'
import { handleTaskDirective } from '../tasks/operations'
import type { TaskOperationGraphSnapshot } from '../tasks/operations/types'
import { formatTaskOutsideSubtreeError } from '../prompts/error-states'
import type { TaskRecord } from '../projections/task-graph'
import { AmbientServiceTag } from '@magnitudedev/event-core'
import { SkillsAmbient } from '../ambient/skills-ambient'

const TaskToolErrorSchema = ToolErrorSchema('TaskToolError', {})

const { ForkContext } = Fork

const toGraphSnapshot = (tasks: ReadonlyMap<string, TaskRecord>): TaskOperationGraphSnapshot => ({
  tasks: new Map(
    [...tasks.entries()].map(([id, task]) => [id, {
      id,
      status: task.status,
      parentId: task.parentId,
      childIds: task.childIds,
      worker: task.worker
        ? { agentId: task.worker.agentId, forkId: task.worker.forkId, role: task.worker.role }
        : null,
    }]),
  ),
})

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

const runDirective = (directive: Parameters<typeof handleTaskDirective>[0]) =>
  Effect.gen(function* () {
    const taskReader = yield* TaskGraphStateReaderTag
    const state = yield* taskReader.getState()
    const { forkId } = yield* ForkContext

    // Worker subtree guard for task creation
    if (directive.kind === 'create' && forkId !== null) {
      const agentStateReader = yield* AgentStateReaderTag
      const agentState = yield* agentStateReader.getAgentState()
      const agentId = agentState.agentByForkId.get(forkId)
      const assignedTaskId = agentId ? agentState.agents.get(agentId)?.taskId?.trim() : null

      if (assignedTaskId) {
        const parentId = directive.parentId ?? null
        const allowed =
          parentId !== null && isTaskInAssignedSubtree(state.tasks, parentId, assignedTaskId)

        if (!allowed) {
          const attemptedParent = parentId ?? '(none)'
          return yield* Effect.fail({
            _tag: 'TaskToolError' as const,
            message: formatTaskOutsideSubtreeError(directive.taskId, attemptedParent, assignedTaskId),
          })
        }
      }
    }

    const ambientService = yield* AmbientServiceTag
    const skills = ambientService.getValue(SkillsAmbient)
    const result = yield* handleTaskDirective(directive, {
      forkId,
      timestamp: Date.now(),
      graph: toGraphSnapshot(state.tasks),
      skills,
    })

    if (result.success === false) {
      return yield* Effect.fail({
        _tag: 'TaskToolError' as const,
        message: result.error,
      })
    }

    return result
  })

const UpdateTaskStatusSchema = Schema.Literal('pending', 'completed', 'cancelled')

export const createTaskTool = defineTool({
  name: 'create-task' as const,
  group: 'task' as const,
  description: 'Create a task.',
  inputSchema: Schema.Struct({
    id: Schema.String.annotations({ description: 'Unique task identifier' }),
    title: Schema.String.annotations({ description: 'Task title' }),
    parent: Schema.optional(Schema.String.annotations({ description: 'Parent task ID to nest under; omit if no parent' })),
  }),
  outputSchema: Schema.Struct({ id: Schema.String }),
  errorSchema: TaskToolErrorSchema,
  execute: (input, _ctx) =>
    Effect.gen(function* () {
      yield* runDirective({
        kind: 'create',
        taskId: input.id,
        parentId: input.parent?.trim() || null,
        title: input.title,
      })
      return { id: input.id }
    }),
  label: (input) => input.id ? `Creating task ${input.id}` : 'Creating task…',
})

export const updateTaskTool = defineTool({
  name: 'update-task' as const,
  group: 'task' as const,
  description: 'Update task status.',
  inputSchema: Schema.Struct({
    id: Schema.String.annotations({ description: 'Task ID to update' }),
    status: UpdateTaskStatusSchema.annotations({ description: 'New status: pending, completed, or cancelled' }),
  }),
  outputSchema: Schema.Struct({
    id: Schema.String,
    status: UpdateTaskStatusSchema,
  }),
  errorSchema: TaskToolErrorSchema,
  execute: (input, _ctx) =>
    Effect.gen(function* () {
      yield* runDirective({
        kind: 'update',
        taskId: input.id,
        status: input.status,
      })
      return { id: input.id, status: input.status }
    }),
  label: (input) => input.id ? `Updating task ${input.id}` : 'Updating task…',
})

export const spawnWorkerTool = defineTool({
  name: 'spawn-worker' as const,
  group: 'task' as const,
  description: 'Spawn a worker for a task id. The body is the worker\'s initial instruction (same mechanics as a normal message). Use <message to="task-id"> for follow-up communications. Only use spawn-worker to create a new worker or replace the current one.',
  inputSchema: Schema.Struct({
    id: Schema.String.annotations({ description: 'Task ID to spawn a worker for' }),
    message: Schema.String.annotations({ description: 'Initial instruction message for the worker' }),
  }),
  outputSchema: Schema.Struct({
    id: Schema.String,
  }),
  errorSchema: TaskToolErrorSchema,
  execute: (input, _ctx) =>
    Effect.gen(function* () {
      const execManager = yield* ExecutionManager
      yield* runDirective({
        kind: 'spawn-worker',
        id: input.id,
        message: input.message,
        spawnWorker: (params): ReturnType<typeof execManager.fork> =>
          execManager.fork({
            parentForkId: params.parentForkId,
            name: params.name,
            agentId: params.agentId,
            prompt: params.prompt,
            message: params.message,
            mode: 'spawn',
            role: 'worker',
            taskId: params.taskId,
          }),
      })
      return { id: input.id }
    }),
  label: (input) => input.id ? `Spawning worker for ${input.id}` : 'Spawning worker…',
})

export const killWorkerTool = defineTool({
  name: 'kill-worker' as const,
  group: 'task' as const,
  description: 'Kill worker for a task id.',
  inputSchema: Schema.Struct({
    id: Schema.String.annotations({ description: 'Task ID whose worker to kill' }),
  }),
  outputSchema: Schema.Struct({
    id: Schema.String,
  }),
  errorSchema: TaskToolErrorSchema,
  execute: (input, _ctx) =>
    Effect.gen(function* () {
      yield* runDirective({
        kind: 'kill-worker',
        id: input.id,
      })
      return { id: input.id }
    }),
  label: (input) => input.id ? `Killing worker for ${input.id}` : 'Killing worker…',
})
