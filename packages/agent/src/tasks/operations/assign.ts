import { Effect } from 'effect'
import { Fork, WorkerBusTag } from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { ConversationStateReaderTag } from '../../tools/memory-reader'
import { TaskGraphStateReaderTag } from '../../tools/task-reader'
import {
  buildAgentContext,
  buildConversationSummary,
} from '../../prompts'
import {
  getTaskTypeDefinition,
  isTaskAssigneeAllowed,
  parseTaskAssignee,
  type TaskTypeId,
  type WorkerAssignee,
} from '../index'
import { getSpawnableVariants } from '../../agents'
import { invalidAssignee, missingAssignmentMessage, taskNotFound } from './errors'
import { buildTaskAssignedValidated } from './builders'
import type { TaskDirectiveContext, TaskDirectiveResult } from './handler'

const { ForkContext } = Fork

export interface AssignDirective {
  readonly kind: 'assign'
  readonly taskId: string
  readonly assignee: string
  readonly message?: string
  readonly spawnWorker: (params: {
    parentForkId: string | null
    name: string
    agentId: string
    prompt: string
    message: string
    role: WorkerAssignee
    taskId: string
  }) => Effect.Effect<string, never, any>
}

export const handleAssignDirective = (directive: AssignDirective, _context: TaskDirectiveContext) =>
  Effect.gen(function* () {
    const normalizedAssignee = directive.assignee.trim().toLowerCase()
    const taskReader = yield* TaskGraphStateReaderTag
    const task = yield* taskReader.getTask(directive.taskId)
    if (!task) {
      const err = taskNotFound(directive.taskId)
      return { success: false, code: err.code, error: err.message } as const
    }

    const parsedAssignee = parseTaskAssignee(normalizedAssignee)
    if (!parsedAssignee) {
      const err = invalidAssignee(directive.taskId, directive.assignee)
      return { success: false, code: err.code, error: err.message } as const
    }

    if (parsedAssignee !== 'user') {
      const spawnable = getSpawnableVariants()
      if (!spawnable.includes(parsedAssignee)) {
        const err = invalidAssignee(directive.taskId, directive.assignee)
        return { success: false, code: err.code, error: err.message } as const
      }
    }

    if (!isTaskAssigneeAllowed(task.taskType, parsedAssignee)) {
      const err = invalidAssignee(directive.taskId, directive.assignee)
      return { success: false, code: err.code, error: err.message } as const
    }

    let trimmedMessage = ''
    if (parsedAssignee !== 'user') {
      trimmedMessage = directive.message?.trim() ?? ''
      if (!trimmedMessage) {
        const err = missingAssignmentMessage(directive.taskId)
        return { success: false, code: err.code, error: err.message } as const
      }
    }

    const bus = yield* WorkerBusTag<AppEvent>()
    const { forkId: parentForkId } = yield* ForkContext
    const timestamp = Date.now()

    let replacedWorker: { agentId: string; forkId: string } | undefined
    if (task.worker) {
      replacedWorker = { agentId: task.worker.agentId, forkId: task.worker.forkId }
      yield* bus.publish({
        type: 'agent_killed',
        forkId: task.worker.forkId,
        parentForkId,
        agentId: task.worker.agentId,
        reason: `Reassigned for task "${directive.taskId}"`,
      })
    }

    if (parsedAssignee === 'user') {
      yield* bus.publish(buildTaskAssignedValidated({
        taskId: directive.taskId,
        assignee: 'user',
        workerRole: undefined,
        message: '',
        workerInfo: undefined,
        replacedWorker,
      }, { forkId: parentForkId, timestamp, graph: { tasks: new Map() } }))
      return { success: true } as const
    }

    const conversationReader = yield* ConversationStateReaderTag
    const conversationState = yield* conversationReader.getState()
    const summary = buildConversationSummary(conversationState.entries) ?? ''

    const taskTypeDef = getTaskTypeDefinition(task.taskType as TaskTypeId)
    let taskContract: string | undefined
    if (taskTypeDef.kind === 'leaf') {
      taskContract = [taskTypeDef.workerGuidance, taskTypeDef.criteria].filter(Boolean).join('\n\n')
    } else if (taskTypeDef.kind === 'generic' && taskTypeDef.workerGuidance) {
      taskContract = [taskTypeDef.workerGuidance, taskTypeDef.criteria].filter(Boolean).join('\n\n')
    }

    const prompt = buildAgentContext(task.title, trimmedMessage, summary, directive.taskId, taskContract)
    const agentId = directive.taskId
    const forkId = yield* directive.spawnWorker({
      parentForkId,
      name: task.title,
      agentId,
      prompt,
      message: trimmedMessage,
      role: parsedAssignee,
      taskId: directive.taskId,
    })

    yield* bus.publish(buildTaskAssignedValidated({
      taskId: directive.taskId,
      assignee: parsedAssignee,
      workerRole: parsedAssignee,
      message: trimmedMessage,
      workerInfo: { agentId, forkId, role: parsedAssignee },
      replacedWorker,
    }, { forkId: parentForkId, timestamp, graph: { tasks: new Map() } }))

    return { success: true } as const
  })
