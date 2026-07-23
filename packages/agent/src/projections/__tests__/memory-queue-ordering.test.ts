import { describe, expect, it } from 'bun:test'
import { Effect, Layer } from 'effect'
import {
  ProjectionBusTag,
  makeProjectionBusLayer,
  makeAmbientServiceLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import { UserBashCommandId, type AppEvent } from '../../events'
import { AgentLifecycleProjection } from '../agent-lifecycle'
import { GoalProjection } from '../goal'
import { WindowProjection, type ForkWindowState } from '../../window'
import { WorkerActivityProjection } from '../worker-activity'
import { OutboundMessagesProjection } from '../outbound-messages'
import { UserMessageResolutionProjection } from '../user-message-resolution'
import { TaskGraphProjection } from '../task-graph'
import { HarnessStateProjection } from '../harness-state'
import { TaskAssignmentProjection } from '../task-assignment'

const ts = (n: number) => 1_700_100_000_000 + n

const makeRuntimeLayer = () => {
  const projectionBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )
  const baseLayer = Layer.provideMerge(
    makeAmbientServiceLayer<AppEvent>(),
    projectionBusLayer,
  )

  const withAgentLifecycle = Layer.provideMerge(AgentLifecycleProjection.Layer, baseLayer)
  const withGoal = Layer.provideMerge(GoalProjection.Layer, withAgentLifecycle)
  const withWorkerActivity = Layer.provideMerge(WorkerActivityProjection.Layer, withGoal)
  const withOutboundMessages = Layer.provideMerge(OutboundMessagesProjection.Layer, withWorkerActivity)
  const withUserMessageResolution = Layer.provideMerge(UserMessageResolutionProjection.Layer, withOutboundMessages)
  const withHarnessState = Layer.provideMerge(HarnessStateProjection.Layer, withUserMessageResolution)
  const withTaskGraph = Layer.provideMerge(TaskGraphProjection.Layer, withHarnessState)
  const withTaskWorker = Layer.provideMerge(TaskAssignmentProjection.Layer, withTaskGraph)
  return Layer.provideMerge(WindowProjection.Layer, withTaskWorker)
}

const runEvents = async (events: AppEvent[]) => {
  const runtimeLayer = makeRuntimeLayer()

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* WindowProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event as any)
    }

    return yield* projection.getFork(null)
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as any) as Promise<ForkWindowState>
}

const getLastContext = (fork: ForkWindowState) => {
  const ctx = [...fork.messages].reverse().find(m => m.type === 'context')
  expect(ctx).toBeTruthy()
  return ctx as Extract<ForkWindowState['messages'][number], { type: 'context' }>
}

describe('WindowProjection queue ordering regressions', () => {
  it('skill_activated on root idle queues only and flushes in order on turn_started', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
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
      {
        type: 'skill_activated',
        timestamp: ts(3),
        forkId: null,
        source: 'user',
        skillName: 'debug',
        message: 'issue',
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(4),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
    ])

    expect(rootFork.queuedTimeline).toEqual([])
    const ctx = getLastContext(rootFork)
    expect(ctx.timeline.map(e => e.kind)).toEqual(['lifecycle_hook', 'worker_user_killed', 'user_message'])
    const user = ctx.timeline.find(e => e.kind === 'user_message')
    expect(user).toBeTruthy()
    expect((user as any).text).toBe('/debug issue')
  })

  it('user_bash_command on root idle queues only and flushes on turn_started', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
      {
        type: 'user_bash_command',
        commandId: UserBashCommandId('bash-1'),
        timestamp: ts(3),
        forkId: null,
        command: 'pwd',
        cwd: '/tmp',
        exitCode: 0,
        stdout: '/tmp',
        stderr: '',
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(4),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
    ])

    expect(rootFork.queuedTimeline).toEqual([])
    const ctx = getLastContext(rootFork)
    expect(ctx.timeline.map(e => e.kind)).toEqual(['user_bash_command'])
    const cmd = ctx.timeline[0] as any
    expect(cmd.command).toBe('pwd')
    expect(cmd.stdout).toBe('/tmp')
  })

  it('userMessageResolved on root idle queues only and flushes in order on turn_started', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
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
      {
        type: 'user_message',
        timestamp: ts(3),
        forkId: null,
        messageId: 'm1',
        content: [{ type: 'text', text: 'hello root' }],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      } as any,
      {
        type: 'user_message_ready',
        timestamp: ts(3),
        messageId: 'm1',
        mentionResolutions: [],
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(4),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
    ])

    expect(rootFork.queuedTimeline).toEqual([])
    const ctx = getLastContext(rootFork)
    expect(ctx.timeline.map(e => e.kind)).toEqual(['lifecycle_hook', 'worker_user_killed', 'user_message'])
    const user = ctx.timeline.find(e => e.kind === 'user_message')
    expect(user).toBeTruthy()
    expect((user as any).text).toBe('hello root')
  })

  it('observations_captured queues + flushes immediately and respects timestamp/seq ordering', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(10),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
      {
        type: 'skill_activated',
        timestamp: ts(20),
        forkId: null,
        source: 'user',
        skillName: 'plan',
        message: undefined,
      } as any,
      {
        type: 'observations_captured',
        timestamp: ts(20),
        turnId: 'turn-1',
        forkId: null,
        parts: [{ type: 'text', text: 'obs text' }],
      } as any,
    ])

    const ctx = getLastContext(rootFork)
    expect(ctx.timeline.map(e => e.kind)).toEqual(['user_message', 'observation'])
    expect((ctx.timeline[0] as any).text).toBe('/plan')
    expect((ctx.timeline[1] as any).parts[0].text).toBe('obs text')
    expect(rootFork.queuedTimeline).toEqual([])
  })

  it('turn_outcome queues + flushes immediately and clears currentTurnId', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(10),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
      {
        type: 'skill_activated',
        timestamp: ts(20),
        forkId: null,
        source: 'user',
        skillName: 'review',
        message: undefined,
      } as any,
      {
        type: 'turn_outcome',
        timestamp: ts(20),
        forkId: null,
        turnId: 'turn-1',
        message: 'boom',
      } as any,
    ])

    const ctx = getLastContext(rootFork)
    // Tool results now live on CompletedTurn, not context entries
    expect(ctx.timeline.map(e => e.kind)).toEqual(['user_message'])
    expect((ctx.timeline[0] as any).text).toBe('/review')
    expect(rootFork.queuedTimeline).toEqual([])
    expect(rootFork.currentTurnId).toBeNull()
  })

  it('flushes mixed result/timeline entries by timestamp+seq and clears queue', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(10),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
      {
        type: 'skill_activated',
        timestamp: ts(20),
        forkId: null,
        source: 'user',
        skillName: 'plan',
        message: undefined,
      } as any,
      {
        type: 'turn_outcome',
        timestamp: ts(20),
        forkId: null,
        turnId: 'turn-1',
        strategyId: 'lead',
        result: { success: false, error: 'turn failed', cancelled: false },
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(30),
        turnId: 'turn-2',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
    ])

    const ctx = getLastContext(rootFork)
    // Tool results now live on CompletedTurn, not context entries
    expect(ctx.timeline.map(e => e.kind)).toEqual(['user_message'])
    expect((ctx.timeline[0] as any).text).toBe('/plan')
    expect(rootFork.queuedTimeline).toEqual([])
  })
})
