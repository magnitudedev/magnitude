import { describe, expect, it } from 'bun:test'
import { Effect, Layer } from 'effect'
import {
  ProjectionBusTag,
  makeProjectionBusLayer,
  makeAmbientServiceLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import type { AppEvent, TurnOutcome } from '../../events'
import { TurnProjection } from '../turn'
import { AgentRoutingProjection } from '../agent-routing'
import { AgentStatusProjection, type AgentStatusState } from '../agent-status'
import { DisplayProjection, type DisplayState } from '../display'
import { TaskGraphProjection } from '../task-graph'
import { ToolStateProjection } from '../tool-state'
import { TaskWorkerProjection, type TaskWorkerState } from '../task-worker'

const ts = (n: number) => 1_700_100_000_000 + n

interface ProjectionSnapshot {
  agentStatus: AgentStatusState
  display: DisplayState
  taskWorker: TaskWorkerState
}

const makeSnapshot = async (events: AppEvent[]): Promise<ProjectionSnapshot> => {
  const baseBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )
  const baseLayer = Layer.provideMerge(
    makeAmbientServiceLayer<AppEvent>(),
    baseBusLayer,
  )

  const runtimeLayer = Layer.mergeAll(
    FrameworkErrorPubSubLive,
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    baseLayer,
    Layer.provide(TurnProjection.Layer, baseLayer),
    Layer.provide(AgentRoutingProjection.Layer, baseLayer),
    Layer.provide(AgentStatusProjection.Layer, baseLayer),
    Layer.provide(TaskGraphProjection.Layer, baseLayer),
    Layer.provide(ToolStateProjection.Layer, baseLayer),
    Layer.provide(TaskWorkerProjection.Layer, baseLayer),
    Layer.provide(DisplayProjection.Layer, baseLayer),
  )

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const agentStatus = yield* AgentStatusProjection.Tag
    const display = yield* DisplayProjection.Tag
    const taskWorker = yield* TaskWorkerProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event as any)
    }

    return {
      agentStatus: yield* agentStatus.get,
      display: yield* display.getFork(null),
      taskWorker: yield* taskWorker.get,
    }
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as Effect.Effect<ProjectionSnapshot>)
}

/** Build a base event sequence for a subagent with one turn, then append a turn_outcome with the given outcome. */
const subagentScenario = (outcome: TurnOutcome): AppEvent[] => [
  // Create a task for the subagent
  {
    type: 'task_created',
    timestamp: ts(1),
    forkId: null,
    taskId: 'task-1',
    title: 'Test task',
    parentId: null,
  } as any,

  // Create the subagent
  {
    type: 'agent_created',
    timestamp: ts(2),
    forkId: 'fork-sub',
    parentForkId: null,
    agentId: 'agent-sub',
    role: 'builder',
    name: 'Builder',
    context: 'ctx',
    mode: 'spawn',
    taskId: 'task-1',
    message: null,
  } as any,

  // Assign the task to the subagent
  {
    type: 'task_assigned',
    timestamp: ts(3),
    forkId: null,
    taskId: 'task-1',
    assignee: 'worker',
    workerRole: 'builder',
    message: '',
    workerInfo: {
      agentId: 'agent-sub',
      forkId: 'fork-sub',
      role: 'builder',
    },
  } as any,

  // Turn starts — agent becomes working
  {
    type: 'turn_started',
    timestamp: ts(5),
    forkId: 'fork-sub',
    turnId: 't-sub-1',
    chainId: 'c-sub-1',
  } as any,

  // Turn ends with the test outcome
  {
    type: 'turn_outcome',
    timestamp: ts(10),
    forkId: 'fork-sub',
    turnId: 't-sub-1',
    chainId: 'c-sub-1',
    strategyId: 'xml-act',
    outcome,
    inputTokens: null,
    outputTokens: null,
    cacheReadTokens: null,
    cacheWriteTokens: null,
    providerId: null,
    modelId: null,
  } as any,
]

function getSubagentStatus(state: AgentStatusState): string | undefined {
  const agentId = state.agentByForkId.get('fork-sub')
  if (!agentId) return undefined
  return state.agents.get(agentId)?.status
}

function getSubagentWorkerState(state: TaskWorkerState): string | undefined {
  const snapshot = state.snapshots.get('task-1')
  if (!snapshot) return undefined
  return snapshot.workerState.status
}

function getSubagentFinishedSteps(display: DisplayState): any[] {
  const allSteps = display.messages.flatMap(m => m.type === 'think_block' ? m.steps : [])
  return allSteps.filter((s: any) => s.type === 'subagent_finished')
}

describe('chain-continue false-idle bug', () => {
  it('ConnectionFailure keeps agent working, no subagent_finished step, no worker idle', async () => {
    const snapshot = await makeSnapshot(subagentScenario({
      _tag: 'ConnectionFailure',
      detail: { _tag: 'TransportError' },
    }))

    expect(getSubagentStatus(snapshot.agentStatus)).toBe('working')
    expect(getSubagentFinishedSteps(snapshot.display).length).toBe(0)
    expect(getSubagentWorkerState(snapshot.taskWorker)).toBe('working')
  })

  it('ParseFailure keeps agent working, no subagent_finished step, no worker idle', async () => {
    const snapshot = await makeSnapshot(subagentScenario({
      _tag: 'ParseFailure',
      error: { _tag: 'StructuralParseError', error: 'test', remainingText: '' } as any,
    }))

    expect(getSubagentStatus(snapshot.agentStatus)).toBe('working')
    expect(getSubagentFinishedSteps(snapshot.display).length).toBe(0)
    expect(getSubagentWorkerState(snapshot.taskWorker)).toBe('working')
  })

  it('ContextWindowExceeded keeps agent working, no subagent_finished step, no worker idle', async () => {
    const snapshot = await makeSnapshot(subagentScenario({
      _tag: 'ContextWindowExceeded',
    }))

    expect(getSubagentStatus(snapshot.agentStatus)).toBe('working')
    expect(getSubagentFinishedSteps(snapshot.display).length).toBe(0)
    expect(getSubagentWorkerState(snapshot.taskWorker)).toBe('working')
  })

  it('Completed + invoke stays working (regression)', async () => {
    const snapshot = await makeSnapshot(subagentScenario({
      _tag: 'Completed',
      completion: { yieldTarget: 'invoke', feedback: [] },
    }))

    expect(getSubagentStatus(snapshot.agentStatus)).toBe('working')
    expect(getSubagentFinishedSteps(snapshot.display).length).toBe(0)
    expect(getSubagentWorkerState(snapshot.taskWorker)).toBe('working')
  })

  it('Completed + user goes idle with subagent_finished step (regression)', async () => {
    const events = subagentScenario({
      _tag: 'Completed',
      completion: { yieldTarget: 'user', feedback: [] },
    })
    // Add a follow-up task_updated to force TaskWorkerProjection to rebuild with fresh reads
    events.push({
      type: 'task_updated',
      timestamp: ts(15),
      forkId: null,
      taskId: 'task-1',
      patch: { title: 'Updated title' },
    } as any)

    const snapshot = await makeSnapshot(events)

    expect(getSubagentStatus(snapshot.agentStatus)).toBe('idle')
    expect(getSubagentFinishedSteps(snapshot.display).length).toBe(1)
    expect(getSubagentWorkerState(snapshot.taskWorker)).toBe('idle')
  })
})
