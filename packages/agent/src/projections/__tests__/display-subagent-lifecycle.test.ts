import { describe, expect, it } from 'bun:test'
import { Effect, Layer } from 'effect'
import {
  Addressed,
  ProjectionBusTag,
  makeProjectionBusLayer,
  makeAmbientServiceLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { TurnProjection } from '../turn'
import { AgentRoutingProjection } from '../agent-routing'
import { AgentLifecycleProjection } from '../agent-lifecycle'
import { GoalProjection } from '../goal'
import { UserMessageResolutionProjection } from '../user-message-resolution'
import { HarnessStateProjection } from '../harness-state'
import { DisplayTimelineProjection, type DisplayTimeline, type DisplayMessage } from '../../display'

// Materialize timeline messages for assertions — accepts the normalized
// byId/order display form or a plain array (addressed readAll results).
const listMessages = <M,>(
  m: readonly M[] | { readonly byId: { readonly [id: string]: M }; readonly order: readonly string[] },
): readonly M[] => ('order' in m ? m.order.map((id) => m.byId[id]!) : m)


const ts = (n: number) => 1_700_100_000_000 + n
const InMemoryAddressedEntryStoreLive = Addressed.makeInMemoryAddressedEntryStoreLayer()

const makeRootDisplay = async (events: AppEvent[]) => {
  const projectionBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )
  const baseLayer = Layer.provideMerge(
    makeAmbientServiceLayer<AppEvent>(),
    projectionBusLayer,
  )

  const withGoal = Layer.provideMerge(GoalProjection.Layer, baseLayer)
  const withAgentRouting = Layer.provideMerge(AgentRoutingProjection.Layer, withGoal)
  const withAgentLifecycle = Layer.provideMerge(AgentLifecycleProjection.Layer, withAgentRouting)
  const withUserMessageResolution = Layer.provideMerge(UserMessageResolutionProjection.Layer, withAgentLifecycle)
  const withHarnessState = Layer.provideMerge(HarnessStateProjection.Layer, withUserMessageResolution)
  const withTurn = Layer.provideMerge(TurnProjection.Layer, withHarnessState)
  const runtimeLayer = Layer.provideMerge(
    Layer.provide(DisplayTimelineProjection.Layer, InMemoryAddressedEntryStoreLive),
    withTurn
  )

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* DisplayTimelineProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event as any)
    }

    const fork = yield* projection.getFork(null)
    const messages = yield* projection.addressed.forFork(null).messages.readAll(fork.messages)
    return { ...fork, messages }
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as unknown as Effect.Effect<DisplayTimeline>)
}

describe('display subagent lifecycle think steps', () => {
  it('adds root turn-block started/finished steps with cumulative resumed semantics', async () => {
    const rootDisplay = await makeRootDisplay([
      {
        type: 'turn_started',
        timestamp: ts(1),
        forkId: null,
        turnId: 't-root',
        chainId: 'c-root',
      } as any,

      {
        type: 'agent_created',
        timestamp: ts(2),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        role: 'engineer',
        name: 'Builder',
        context: 'ctx',
        mode: 'spawn',
        taskId: 'task-1',
        message: null,
      } as any,

      {
        type: 'turn_started',
        timestamp: ts(5),
        forkId: 'fork-sub',
        turnId: 't-sub-1',
        chainId: 'c-sub-1',
      } as any,
      {
        type: 'tool_event',
        timestamp: ts(6),
        forkId: 'fork-sub',
        turnId: 't-sub-1',
        toolCallId: 'call-1',
        providerToolCallId: 'call-1',
        toolKey: 'shell',
        event: { _tag: 'ToolInputStarted', toolCallId: 'call-1', providerToolCallId: 'call-1', toolName: 'shell', toolKey: 'shell' },
      } as any,
      {
        type: 'turn_outcome',
        timestamp: ts(10),
        forkId: 'fork-sub',
        turnId: 't-sub-1',
        chainId: 'c-sub-1',
        strategyId: 'native',
        outcome: {
          _tag: 'Completed',
          completion: {
            toolCallsCount: 0,
            finishReason: 'stop',
            feedback: [],
            yieldTarget: null,
          },
          requestId: null,
        },
        commitPolicy: { _tag: 'commitCleanTurn' },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        cost: null,
        providerId: 'test',
        modelId: 'role/worker',
      } as any,

      {
        type: 'turn_started',
        timestamp: ts(15),
        forkId: 'fork-sub',
        turnId: 't-sub-2',
        chainId: 'c-sub-2',
      } as any,
      {
        type: 'tool_event',
        timestamp: ts(16),
        forkId: 'fork-sub',
        turnId: 't-sub-2',
        toolCallId: 'call-2',
        providerToolCallId: 'call-2',
        toolKey: 'fileRead',
        event: { _tag: 'ToolInputStarted', toolCallId: 'call-2', providerToolCallId: 'call-2', toolName: 'fileRead', toolKey: 'fileRead' },
      } as any,
      {
        type: 'turn_outcome',
        timestamp: ts(20),
        forkId: 'fork-sub',
        turnId: 't-sub-2',
        chainId: 'c-sub-2',
        strategyId: 'native',
        outcome: {
          _tag: 'Completed',
          completion: {
            toolCallsCount: 0,
            finishReason: 'stop',
            feedback: [],
            yieldTarget: null,
          },
          requestId: null,
        },
        commitPolicy: { _tag: 'commitCleanTurn' },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        cost: null,
        providerId: 'test',
        modelId: 'role/worker',
      } as any,
    ])

    const allSteps = listMessages(rootDisplay.messages).filter((m: any) => m.type === "worker_resumed" || m.type === "worker_finished" || m.type === "worker_killed" || m.type === "worker_user_killed")

    const forkActivity = listMessages(rootDisplay.messages).filter(
      (m): m is Extract<DisplayMessage, { type: 'fork_activity' }> =>
        m.type === 'fork_activity' && m.forkId === 'fork-sub'
    )

    expect(forkActivity.length).toBe(2)
    expect(forkActivity[0]).toMatchObject({
      status: 'completed',
      createdAt: ts(2),
      activeSince: ts(2),
      completedAt: ts(10),
      accumulatedActiveMs: 8,
      resumeCount: 0,
      toolCounts: { commands: 1 },
    })
    expect(forkActivity[1]).toMatchObject({
      status: 'completed',
      createdAt: ts(15),
      activeSince: ts(15),
      completedAt: ts(20),
      accumulatedActiveMs: 13,
      resumeCount: 1,
      toolCounts: { commands: 1, reads: 1 },
    })
    expect(forkActivity[0].id).not.toBe(forkActivity[1].id)

    const resumed = allSteps.filter((s: any) => s.type === 'worker_resumed')
    const finished = allSteps.filter((s: any) => s.type === 'worker_finished')

    // Initial creation doesn't get a WorkerResumedStep — spawnWorker tool step covers it
    expect(resumed.length).toBe(1)
    expect(resumed[0]).toMatchObject({
      workerRole: 'engineer',
      workerId: 'agent-sub',
      title: 'Builder',
    })

    expect(finished.length).toBe(2)
    expect(finished[0]).toMatchObject({
      workerRole: 'engineer',
      workerId: 'agent-sub',
      cumulativeTotalTimeMs: 8,
      cumulativeTotalToolsUsed: 1,
      resumed: false,
    })
    expect(finished[1]).toMatchObject({
      workerRole: 'engineer',
      workerId: 'agent-sub',
      cumulativeTotalTimeMs: 13,
      cumulativeTotalToolsUsed: 2,
      resumed: true,
    })
  })

  it('adds worker_killed step (without worker_finished) and removes fork activity for agent_killed', async () => {
    const rootDisplay = await makeRootDisplay([
      {
        type: 'agent_created',
        timestamp: ts(1),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        role: 'engineer',
        name: 'Builder',
        context: 'ctx',
        mode: 'spawn',
        taskId: 'task-1',
        message: null,
      } as any,
      {
        type: 'agent_killed',
        timestamp: ts(2),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        reason: 'no longer needed',
      } as any,
    ])

    const forkActivity = listMessages(rootDisplay.messages).filter((m: any) => m.type === 'fork_activity' && m.forkId === 'fork-sub')
    expect(forkActivity.length).toBe(0)

    const allSteps = listMessages(rootDisplay.messages).filter((m: any) => m.type === "worker_resumed" || m.type === "worker_finished" || m.type === "worker_killed" || m.type === "worker_user_killed")
    const finished = allSteps.filter((s: any) => s.type === 'worker_finished')
    expect(finished.length).toBe(0)

    const killed = allSteps.filter((s: any) => s.type === 'worker_killed')
    expect(killed.length).toBe(1)
    expect(killed[0]).toMatchObject({
      workerRole: 'engineer',
      workerId: 'agent-sub',
      title: 'Builder',
    })
  })

  it('removes fork activity and adds worker_user_killed step for worker_user_killed', async () => {
    const rootDisplay = await makeRootDisplay([
      {
        type: 'agent_created',
        timestamp: ts(1),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        role: 'engineer',
        name: 'Builder',
        context: 'ctx',
        mode: 'spawn',
        taskId: 'task-1',
        message: null,
      } as any,
      {
        type: 'worker_user_killed',
        timestamp: ts(2),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        source: 'tab_close_confirm',
      } as any,
    ])

    const forkActivity = listMessages(rootDisplay.messages).filter((m: any) => m.type === 'fork_activity' && m.forkId === 'fork-sub')
    expect(forkActivity.length).toBe(0)

    const allSteps = listMessages(rootDisplay.messages).filter((m: any) => m.type === "worker_resumed" || m.type === "worker_finished" || m.type === "worker_killed" || m.type === "worker_user_killed")
    const userKilled = allSteps.filter((s: any) => s.type === 'worker_user_killed')
    expect(userKilled.length).toBe(1)
    expect(userKilled[0]).toMatchObject({
      workerRole: 'engineer',
      workerId: 'agent-sub',
      title: 'Builder',
    })
  })

  it('removes fork activity and does not add worker steps for worker_idle_closed', async () => {
    const rootDisplay = await makeRootDisplay([
      {
        type: 'agent_created',
        timestamp: ts(1),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        role: 'engineer',
        name: 'Builder',
        context: 'ctx',
        mode: 'spawn',
        taskId: 'task-1',
        message: null,
      } as any,
      {
        type: 'worker_idle_closed',
        timestamp: ts(2),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        source: 'idle_tab_close',
      } as any,
    ])

    const forkActivity = listMessages(rootDisplay.messages).filter((m: any) => m.type === 'fork_activity' && m.forkId === 'fork-sub')
    expect(forkActivity.length).toBe(0)

    const allSteps = listMessages(rootDisplay.messages).filter((m: any) => m.type === "worker_resumed" || m.type === "worker_finished" || m.type === "worker_killed" || m.type === "worker_user_killed")
    const userKilled = allSteps.filter((s: any) => s.type === 'worker_user_killed')
    const killed = allSteps.filter((s: any) => s.type === 'worker_killed')
    expect(userKilled.length).toBe(0)
    expect(killed.length).toBe(0)
  })
})
