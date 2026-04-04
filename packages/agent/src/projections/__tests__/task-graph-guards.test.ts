import { describe, expect, it } from 'bun:test'
import { Effect, Layer } from 'effect'
import {
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
  makeProjectionBusLayer,
  ProjectionBusTag,
} from '@magnitudedev/event-core'
import type { AppEvent, TaskAssigned, TaskCreated, TaskUpdated } from '../../events'
import { AgentStatusProjection } from '../agent-status'
import { TaskGraphProjection, type TaskGraphState } from '../task-graph'

const ts = (n: number) => 1_700_300_000_000 + n

const makeRuntimeLayer = () => {
  const projectionBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )

  return Layer.mergeAll(
    FrameworkErrorPubSubLive,
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    projectionBusLayer,
    Layer.provide(AgentStatusProjection.Layer, projectionBusLayer),
    Layer.provide(TaskGraphProjection.Layer, projectionBusLayer),
  )
}

const runEvents = async (events: readonly (TaskCreated | TaskUpdated | TaskAssigned)[]): Promise<TaskGraphState> => {
  const runtimeLayer = makeRuntimeLayer()

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* TaskGraphProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event as unknown as AppEvent & { readonly timestamp: number })
    }

    return yield* projection.get
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as any)
}

describe('TaskGraphProjection defensive guards', () => {
  it('ignores invalid status in task_updated patch', async () => {
    const state = await runEvents([
      {
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 't1',
        title: 'Task 1',
        taskType: 'implement',
        parentId: null,
      } satisfies TaskCreated,
      {
        type: 'task_updated',
        timestamp: ts(2),
        forkId: null,
        taskId: 't1',
        patch: { status: 'not-a-status' },
      } satisfies TaskUpdated,
    ])

    expect(state.tasks.get('t1')?.status).toBe('pending')
  })

  it('ignores disallowed status transition', async () => {
    const state = await runEvents([
      {
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 't2',
        title: 'Task 2',
        taskType: 'implement',
        parentId: null,
      } satisfies TaskCreated,
      {
        type: 'task_updated',
        timestamp: ts(2),
        forkId: null,
        taskId: 't2',
        patch: { status: 'archived' },
      } satisfies TaskUpdated,
      {
        type: 'task_updated',
        timestamp: ts(3),
        forkId: null,
        taskId: 't2',
        patch: { status: 'working' },
      } satisfies TaskUpdated,
    ])

    expect(state.tasks.get('t2')?.status).toBe('archived')
  })

  it('ignores invalid assignee for task type', async () => {
    const state = await runEvents([
      {
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 't3',
        title: 'Task 3',
        taskType: 'implement',
        parentId: null,
      } satisfies TaskCreated,
      {
        type: 'task_assigned',
        timestamp: ts(2),
        forkId: null,
        taskId: 't3',
        assignee: 'user',
        workerInfo: undefined,
        message: 'please do it',
      } satisfies TaskAssigned,
    ])

    const task = state.tasks.get('t3')
    expect(task?.assignee).toBeNull()
    expect(task?.worker).toBeNull()
    expect(task?.status).toBe('pending')
  })
})
