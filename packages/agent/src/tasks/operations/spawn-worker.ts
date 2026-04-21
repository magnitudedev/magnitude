import { Effect } from 'effect'
import { Fork, WorkerBusTag, type WorkerBusService } from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { ConversationStateReaderTag } from '../../tools/memory-reader'
import { TaskGraphStateReaderTag } from '../../tools/task-reader'
import { buildAgentContext, buildConversationSummary } from '../../prompts'
import { buildTaskAssignedValidated } from './builders'
import { taskNotFound } from './errors'
import type { TaskDirectiveContext } from './handler'

const { ForkContext } = Fork

export interface SpawnWorkerDirective<R = never> {
  readonly kind: 'spawn-worker'
  readonly id: string
  readonly message: string
  readonly spawnWorker: (params: {
    parentForkId: string | null
    name: string
    agentId: string
    prompt: string
    message: string
    taskId: string
  }) => Effect.Effect<string, never, R>
}

export const handleSpawnWorkerDirective = <R>(
  directive: SpawnWorkerDirective<R>,
  context: TaskDirectiveContext,
): Effect.Effect<
  | { readonly success: true; readonly title: string }
  | { readonly success: false; readonly code: string; readonly error: string },
  never,
  TaskGraphStateReaderTag
  | ConversationStateReaderTag
  | WorkerBusService<AppEvent>
  | Fork.ForkContextService
  | R

> =>
  Effect.gen(function* () {
    const taskReader = yield* TaskGraphStateReaderTag
    const task = yield* taskReader.getTask(directive.id)
    if (!task) {
      const err = taskNotFound(directive.id)
      return { success: false, code: err.code, error: err.message } as const
    }

    const parsedAssignee = 'worker' as const

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
        reason: `Respawned for task "${directive.id}"`,
      })
    }

    const conversationReader = yield* ConversationStateReaderTag
    const conversationState = yield* conversationReader.getState()
    const summary = buildConversationSummary(conversationState.entries)

    const taskContract: string | undefined = undefined

    const agentId = directive.id
    const prompt = buildAgentContext(task.title, summary, directive.id, taskContract)
    const forkId = yield* directive.spawnWorker({
      parentForkId,
      name: task.title,
      agentId,
      prompt,
      message: directive.message,
      taskId: directive.id,
    })

    yield* bus.publish(buildTaskAssignedValidated({
      taskId: directive.id,
      assignee: parsedAssignee,
      workerRole: parsedAssignee,
      message: directive.message,
      workerInfo: { agentId, forkId, role: parsedAssignee },
      replacedWorker,
    }, { forkId: parentForkId, timestamp, graph: { tasks: new Map() } }))

    return { success: true, title: task.title } as const
  })
