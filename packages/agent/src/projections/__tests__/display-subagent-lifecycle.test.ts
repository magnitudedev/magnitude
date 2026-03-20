import { describe, expect, it } from 'bun:test'
import { Effect, Layer } from 'effect'
import {
  ProjectionBusTag,
  makeProjectionBusLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { WorkingStateProjection } from '../working-state'
import { AgentRoutingProjection } from '../agent-routing'
import { AgentStatusProjection } from '../agent-status'
import { DisplayProjection } from '../display'

const ts = (n: number) => 1_700_100_000_000 + n

const makeRootDisplay = async (events: AppEvent[]) => {
  const projectionBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )

  const runtimeLayer = Layer.mergeAll(
    FrameworkErrorPubSubLive,
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    projectionBusLayer,
    Layer.provide(WorkingStateProjection.Layer, projectionBusLayer),
    Layer.provide(AgentRoutingProjection.Layer, projectionBusLayer),
    Layer.provide(AgentStatusProjection.Layer, projectionBusLayer),
    Layer.provide(DisplayProjection.Layer, projectionBusLayer),
  )

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* DisplayProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event)
    }

    return yield* projection.getFork(null)
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as any)
}

describe('display subagent lifecycle think steps', () => {
  it('adds root think-block started/finished steps with cumulative resumed semantics', async () => {
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
        role: 'builder',
        name: 'Builder',
        context: 'ctx',
        mode: 'spawn',
        taskId: 'task-1',
        message: '',
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
        toolKey: 'shell',
        event: { _tag: 'ToolInputStarted' },
      } as any,
      {
        type: 'turn_completed',
        timestamp: ts(10),
        forkId: 'fork-sub',
        turnId: 't-sub-1',
        result: { success: true, turnDecision: 'yield', output: '' },
        toolCalls: [{ id: 'call-1', toolName: 'shell', input: {}, result: { _tag: 'Success', output: {} } }],
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
        toolKey: 'fileRead',
        event: { _tag: 'ToolInputStarted' },
      } as any,
      {
        type: 'turn_completed',
        timestamp: ts(20),
        forkId: 'fork-sub',
        turnId: 't-sub-2',
        result: { success: true, turnDecision: 'yield', output: '' },
        toolCalls: [{ id: 'call-2', toolName: 'fileRead', input: {}, result: { _tag: 'Success', output: {} } }],
      } as any,
    ])

    const allSteps = rootDisplay.messages.flatMap(m => m.type === 'think_block' ? m.steps : [])

    const started = allSteps.filter((s: any) => s.type === 'subagent_started')
    const finished = allSteps.filter((s: any) => s.type === 'subagent_finished')

    expect(started.length).toBe(2)
    expect(started[0]).toMatchObject({
      subagentType: 'builder',
      subagentId: 'agent-sub',
      title: 'Builder',
      resumed: false,
    })
    expect(started[1]).toMatchObject({
      subagentId: 'agent-sub',
      resumed: true,
    })

    expect(finished.length).toBe(2)
    expect(finished[0]).toMatchObject({
      subagentId: 'agent-sub',
      cumulativeTotalTimeMs: 5,
      cumulativeTotalToolsUsed: 1,
      resumed: false,
    })
    expect(finished[1]).toMatchObject({
      subagentId: 'agent-sub',
      cumulativeTotalTimeMs: 10,
      cumulativeTotalToolsUsed: 2,
      resumed: true,
    })
  })

  it('adds subagent_killed step (without subagent_finished) and removes fork activity for agent_killed', async () => {
    const rootDisplay = await makeRootDisplay([
      {
        type: 'agent_created',
        timestamp: ts(1),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        role: 'builder',
        name: 'Builder',
        context: 'ctx',
        mode: 'spawn',
        taskId: 'task-1',
        message: '',
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

    const forkActivity = rootDisplay.messages.filter((m: any) => m.type === 'fork_activity' && m.forkId === 'fork-sub')
    expect(forkActivity.length).toBe(0)

    const allSteps = rootDisplay.messages.flatMap(m => m.type === 'think_block' ? m.steps : [])
    const finished = allSteps.filter((s: any) => s.type === 'subagent_finished')
    expect(finished.length).toBe(0)

    const killed = allSteps.filter((s: any) => s.type === 'subagent_killed')
    expect(killed.length).toBe(1)
    expect(killed[0]).toMatchObject({
      subagentId: 'agent-sub',
      title: 'Builder',
    })
  })

  it('removes fork activity and adds subagent_user_killed think step for subagent_user_killed', async () => {
    const rootDisplay = await makeRootDisplay([
      {
        type: 'agent_created',
        timestamp: ts(1),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        role: 'builder',
        name: 'Builder',
        context: 'ctx',
        mode: 'spawn',
        taskId: 'task-1',
        message: '',
      } as any,
      {
        type: 'subagent_user_killed',
        timestamp: ts(2),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        source: 'tab_close_confirm',
      } as any,
    ])

    const forkActivity = rootDisplay.messages.filter((m: any) => m.type === 'fork_activity' && m.forkId === 'fork-sub')
    expect(forkActivity.length).toBe(0)

    const allSteps = rootDisplay.messages.flatMap(m => m.type === 'think_block' ? m.steps : [])
    const userKilled = allSteps.filter((s: any) => s.type === 'subagent_user_killed')
    expect(userKilled.length).toBe(1)
    expect(userKilled[0]).toMatchObject({
      subagentId: 'agent-sub',
      title: 'Builder',
    })
  })

  it('removes fork activity and does not add subagent_user_killed/subagent_killed think steps for subagent_idle_closed', async () => {
    const rootDisplay = await makeRootDisplay([
      {
        type: 'agent_created',
        timestamp: ts(1),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        role: 'builder',
        name: 'Builder',
        context: 'ctx',
        mode: 'spawn',
        taskId: 'task-1',
        message: '',
      } as any,
      {
        type: 'subagent_idle_closed',
        timestamp: ts(2),
        forkId: 'fork-sub',
        parentForkId: null,
        agentId: 'agent-sub',
        source: 'idle_tab_close',
      } as any,
    ])

    const forkActivity = rootDisplay.messages.filter((m: any) => m.type === 'fork_activity' && m.forkId === 'fork-sub')
    expect(forkActivity.length).toBe(0)

    const allSteps = rootDisplay.messages.flatMap(m => m.type === 'think_block' ? m.steps : [])
    const userKilled = allSteps.filter((s: any) => s.type === 'subagent_user_killed')
    const killed = allSteps.filter((s: any) => s.type === 'subagent_killed')
    expect(userKilled.length).toBe(0)
    expect(killed.length).toBe(0)
  })
})
