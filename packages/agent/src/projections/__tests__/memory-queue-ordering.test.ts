import { describe, expect, it } from 'bun:test'
import { Effect, Layer } from 'effect'
import {
  ProjectionBusTag,
  makeProjectionBusLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { AgentStatusProjection } from '../agent-status'
import { MemoryProjection, type ForkMemoryState } from '../memory'
import { SubagentActivityProjection } from '../subagent-activity'
import { CanonicalTurnProjection } from '../canonical-turn'
import { UserPresenceProjection } from '../user-presence'
import { OutboundMessagesProjection } from '../outbound-messages'
import { UserMessageResolutionProjection } from '../user-message-resolution'
import { TaskGraphProjection } from '../task-graph'

const ts = (n: number) => 1_700_100_000_000 + n

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
    Layer.provide(SubagentActivityProjection.Layer, projectionBusLayer),
    Layer.provide(CanonicalTurnProjection.Layer, projectionBusLayer),
    Layer.provide(UserPresenceProjection.Layer, projectionBusLayer),
    Layer.provide(OutboundMessagesProjection.Layer, projectionBusLayer),
    Layer.provide(UserMessageResolutionProjection.Layer, projectionBusLayer),
    Layer.provide(TaskGraphProjection.Layer, projectionBusLayer),
    Layer.provide(MemoryProjection.Layer, projectionBusLayer),
  )
}

const runEvents = async (events: AppEvent[]) => {
  const runtimeLayer = makeRuntimeLayer()

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* MemoryProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event as any)
    }

    return yield* projection.getFork(null)
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as any) as Promise<ForkMemoryState>
}

const getLastInbox = (fork: ForkMemoryState) => {
  const inbox = [...fork.messages].reverse().find(m => m.type === 'inbox')
  expect(inbox).toBeTruthy()
  return inbox as Extract<ForkMemoryState['messages'][number], { type: 'inbox' }>
}

describe('MemoryProjection queue ordering regressions', () => {
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
        role: 'builder',
        name: 'Builder',
        context: 'ctx',
        mode: 'spawn',
        taskId: 'task-1',
        message: null,
      } as any,
      {
        type: 'subagent_user_killed',
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

    expect(rootFork.queuedEntries).toEqual([])
    const inbox = getLastInbox(rootFork)
    expect(inbox.timeline.map(e => e.kind)).toEqual(['lifecycle_hook', 'subagent_user_killed', 'user_message'])
    const user = inbox.timeline.find(e => e.kind === 'user_message')
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

    expect(rootFork.queuedEntries).toEqual([])
    const inbox = getLastInbox(rootFork)
    expect(inbox.timeline.map(e => e.kind)).toEqual(['user_bash_command'])
    const cmd = inbox.timeline[0] as any
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
        role: 'builder',
        name: 'Builder',
        context: 'ctx',
        mode: 'spawn',
        taskId: 'task-1',
        message: null,
      } as any,
      {
        type: 'subagent_user_killed',
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
        resolvedMentions: [],
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

    expect(rootFork.queuedEntries).toEqual([])
    const inbox = getLastInbox(rootFork)
    expect(inbox.timeline.map(e => e.kind)).toEqual(['lifecycle_hook', 'subagent_user_killed', 'user_message'])
    const user = inbox.timeline.find(e => e.kind === 'user_message')
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

    const inbox = getLastInbox(rootFork)
    expect(inbox.timeline.map(e => e.kind)).toEqual(['user_message', 'observation'])
    expect((inbox.timeline[0] as any).text).toBe('/plan')
    expect((inbox.timeline[1] as any).parts[0].text).toBe('obs text')
    expect(rootFork.queuedEntries).toEqual([])
  })

  it('turn_unexpected_error queues + flushes immediately and clears currentTurnId', async () => {
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
        type: 'turn_unexpected_error',
        timestamp: ts(20),
        forkId: null,
        turnId: 'turn-1',
        message: 'boom',
      } as any,
    ])

    const inbox = getLastInbox(rootFork)
    expect(inbox.results.map(r => r.kind)).toEqual(['error'])
    expect((inbox.results[0] as any).message).toBe('boom')
    expect(inbox.timeline.map(e => e.kind)).toEqual(['user_message'])
    expect((inbox.timeline[0] as any).text).toBe('/review')
    expect(rootFork.queuedEntries).toEqual([])
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
        type: 'turn_completed',
        timestamp: ts(20),
        forkId: null,
        turnId: 'turn-1',
        strategyId: 'lead',
        responseParts: [{ type: 'text', content: 'done' }],
        toolCalls: [],
        observedResults: [],
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

    const inbox = getLastInbox(rootFork)
    expect(inbox.results.map(r => r.kind)).toEqual(['error'])
    expect((inbox.results[0] as any).message).toBe('turn failed')
    expect(inbox.timeline.map(e => e.kind)).toEqual(['user_message'])
    expect((inbox.timeline[0] as any).text).toBe('/plan')
    expect(rootFork.queuedEntries).toEqual([])
  })
})
