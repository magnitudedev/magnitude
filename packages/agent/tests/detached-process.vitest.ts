/**
 * DetachedProcessProjection, TurnProjection wake trigger, and WindowProjection
 * global handler tests.
 */

import { describe, it, expect } from 'vitest'
import { Effect, Layer } from 'effect'
import {
  ProjectionBusTag,
  makeProjectionBusLayer,
  makeAmbientServiceLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import type { AppEvent } from '../src/events'
import { DetachedProcessProjection, type DetachedProcessState } from '../src/projections/detached-process'
import { TurnProjection, type TurnIdle, type TurnLifecycleState } from '../src/projections/turn'
import { AgentRoutingProjection } from '../src/projections/agent-routing'
import { AgentLifecycleProjection } from '../src/projections/agent-lifecycle'
import { GoalProjection } from '../src/projections/goal'
import { UserMessageResolutionProjection } from '../src/projections/user-message-resolution'
import { WindowProjection } from '../src/window/projection'
import type { ForkWindowState } from '../src/window/types'
import { WorkerActivityProjection } from '../src/projections/worker-activity'
import { OutboundMessagesProjection } from '../src/projections/outbound-messages'
import { TaskGraphProjection } from '../src/projections/task-graph'
import { TaskAssignmentProjection } from '../src/projections/task-assignment'
import { HarnessStateProjection } from '../src/projections/harness-state'
import { SkillsAmbient } from '../src/ambient/skills-ambient'
import { ToolUniverseSourceLive } from '../src/tools/tool-universe-live'

const ts = (n: number) => 1_700_400_000_000 + n

// ── Helpers ──────────────────────────────────────────────────────────────

const makeBaseLayer = () => {
  const busLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )
  return Layer.merge(
    Layer.provideMerge(makeAmbientServiceLayer<AppEvent>(), busLayer),
    ToolUniverseSourceLive,
  )
}

const makeDetachedProcessState = async (events: AppEvent[]): Promise<DetachedProcessState> => {
  const base = makeBaseLayer()
  const runtime = Layer.provideMerge(DetachedProcessProjection.Layer, base)

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* DetachedProcessProjection.Tag
    for (const event of events) {
      yield* bus.processEvent(event as any)
    }
    return yield* projection.getFork(null)
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtime)) as any)
}

const makeTurnState = async (events: AppEvent[], forkId: string | null = null): Promise<TurnLifecycleState> => {
  const base = makeBaseLayer()
  const runtime = Layer.provideMerge(
    Layer.mergeAll(
      UserMessageResolutionProjection.Layer,
      AgentRoutingProjection.Layer,
      AgentLifecycleProjection.Layer,
      GoalProjection.Layer,
      TurnProjection.Layer,
    ),
    base,
  )

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* TurnProjection.Tag
    for (const event of events) {
      yield* bus.processEvent(event as any)
    }
    return yield* projection.getFork(forkId)
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtime)) as any)
}

const makeWindowState = async (events: AppEvent[]): Promise<ForkWindowState> => {
  const base = makeBaseLayer()
  const runtime = Layer.provideMerge(
    Layer.mergeAll(
      UserMessageResolutionProjection.Layer,
      AgentRoutingProjection.Layer,
      AgentLifecycleProjection.Layer,
      GoalProjection.Layer,
      TurnProjection.Layer,
      WorkerActivityProjection.Layer,
      OutboundMessagesProjection.Layer,
      TaskGraphProjection.Layer,
      TaskAssignmentProjection.Layer,
      HarnessStateProjection.Layer,
      DetachedProcessProjection.Layer,
      WindowProjection.Layer,
    ),
    base,
  )

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* WindowProjection.Tag

    const ambientService = yield* (yield* Effect.promise(() => import('@magnitudedev/event-core'))).AmbientServiceTag
    yield* ambientService.register(SkillsAmbient)
    yield* ambientService.update(SkillsAmbient, new Map())

    for (const event of events) {
      yield* bus.processEvent(event as any)
    }
    return yield* projection.getFork(null)
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtime)) as any)
}

const registeredEvent = (opts: { forkId?: string | null; pid?: number; command?: string; ownerAgentId?: string } = {}): AppEvent => ({
  type: 'shell_process_registered',
  forkId: opts.forkId ?? null,
  pid: opts.pid ?? 12345,
  command: opts.command ?? 'sleep 30',
  ownerAgentId: opts.ownerAgentId ?? undefined,
  startedAt: ts(1),
  stdoutPath: '/tmp/out.log',
  stderrPath: '/tmp/err.log',
  timestamp: ts(1),
} as any)

const exitedEvent = (opts: { forkId?: string | null; pid?: number; command?: string; exitCode?: number } = {}): AppEvent => ({
  type: 'shell_process_exited',
  forkId: opts.forkId ?? null,
  pid: opts.pid ?? 12345,
  command: opts.command ?? 'sleep 30',
  exitCode: opts.exitCode ?? 0,
  timestamp: ts(2),
} as any)

const completedEvent = (opts: { forkId?: string | null; pid?: number; command?: string; exitCode?: number } = {}): AppEvent => ({
  type: 'shell_completed',
  forkId: opts.forkId ?? null,
  pid: opts.pid ?? 12345,
  command: opts.command ?? 'sleep 30',
  exitCode: opts.exitCode ?? 0,
  timestamp: ts(2),
} as any)

// ── A. DetachedProcessProjection handlers ──────────────────────────────

describe('DetachedProcessProjection', () => {
  it('shell_process_registered adds a running process to state', async () => {
    const fork = await makeDetachedProcessState([registeredEvent()])
    expect(fork.processes.size).toBe(1)
    const proc = fork.processes.get(12345)
    expect(proc).toBeDefined()
    expect(proc!.status).toBe('running')
    expect(proc!.command).toBe('sleep 30')
    expect(proc!.exitCode).toBeNull()
  })

  it('shell_process_exited marks it completed with correct exit code', async () => {
    const fork = await makeDetachedProcessState([
      registeredEvent(),
      exitedEvent({ exitCode: 42 }),
    ])
    const proc = fork.processes.get(12345)
    expect(proc).toBeDefined()
    expect(proc!.status).toBe('completed')
    expect(proc!.exitCode).toBe(42)
  })

  it('shell_process_exited with kill-exit-code marks it killed', async () => {
    const fork = await makeDetachedProcessState([
      registeredEvent(),
      exitedEvent({ exitCode: 137 }),
    ])
    const proc = fork.processes.get(12345)
    expect(proc!.status).toBe('killed')
    expect(proc!.exitCode).toBe(137)
  })

  it('turn_outcome clears non-running processes', async () => {
    const fork = await makeDetachedProcessState([
      registeredEvent(),
      exitedEvent({ exitCode: 0 }),
      {
        type: 'turn_outcome',
        forkId: null,
        turnId: 'turn-1',
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [], yieldTarget: null }, requestId: null },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        cost: null,
        providerId: null,
        modelId: null,
      },
    ])
    expect(fork.processes.size).toBe(0)
  })

  it('turn_outcome leaves running processes intact', async () => {
    const fork = await makeDetachedProcessState([
      registeredEvent(),
      {
        type: 'turn_outcome',
        forkId: null,
        turnId: 'turn-1',
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [], yieldTarget: null }, requestId: null },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        cost: null,
        providerId: null,
        modelId: null,
      },
    ])
    expect(fork.processes.size).toBe(1)
    expect(fork.processes.get(12345)!.status).toBe('running')
  })

  it('agent_killed marks running processes of that fork as killed', async () => {
    const base = makeBaseLayer()
    const runtime = Layer.provideMerge(DetachedProcessProjection.Layer, base)

    const program = Effect.gen(function* () {
      const bus = yield* ProjectionBusTag<AppEvent>()
      const projection = yield* DetachedProcessProjection.Tag
      yield* bus.processEvent(registeredEvent({ forkId: 'fork-a' }) as any)
      yield* bus.processEvent({
        type: 'agent_killed',
        forkId: 'fork-a',
        parentForkId: null,
        agentId: 'agent-1',
        reason: 'test',
      } as any)
      return yield* projection.getFork('fork-a')
    })

    const fork = await Effect.runPromise(program.pipe(Effect.provide(runtime)) as any) as DetachedProcessState
    const proc = fork.processes.get(12345)
    expect(proc).toBeDefined()
    expect(proc!.status).toBe('killed')
    expect(proc!.exitCode).toBe(137)
  })

  it('worker_user_killed marks running processes of that fork as killed', async () => {
    const base = makeBaseLayer()
    const runtime = Layer.provideMerge(DetachedProcessProjection.Layer, base)

    const program = Effect.gen(function* () {
      const bus = yield* ProjectionBusTag<AppEvent>()
      const projection = yield* DetachedProcessProjection.Tag
      yield* bus.processEvent(registeredEvent({ forkId: 'fork-a' }) as any)
      yield* bus.processEvent({
        type: 'worker_user_killed',
        forkId: 'fork-a',
        parentForkId: null,
        agentId: 'agent-1',
        source: 'tab_close_confirm',
      } as any)
      return yield* projection.getFork('fork-a')
    })

    const fork = await Effect.runPromise(program.pipe(Effect.provide(runtime)) as any) as DetachedProcessState
    const proc = fork.processes.get(12345)
    expect(proc).toBeDefined()
    expect(proc!.status).toBe('killed')
    expect(proc!.exitCode).toBe(137)
  })

  it('fork isolation: registered events only affect their fork state', async () => {
    const base = makeBaseLayer()
    const runtime = Layer.provideMerge(DetachedProcessProjection.Layer, base)

    const program = Effect.gen(function* () {
      const bus = yield* ProjectionBusTag<AppEvent>()
      const projection = yield* DetachedProcessProjection.Tag
      yield* bus.processEvent(registeredEvent({ forkId: 'fork-a', pid: 111 }) as any)
      yield* bus.processEvent(registeredEvent({ forkId: 'fork-b', pid: 222 }) as any)
      const forkA = yield* projection.getFork('fork-a')
      const forkB = yield* projection.getFork('fork-b')
      return { forkA, forkB }
    })

    const { forkA, forkB } = await Effect.runPromise(program.pipe(Effect.provide(runtime)) as any) as { forkA: DetachedProcessState; forkB: DetachedProcessState }
    expect(forkA.processes.size).toBe(1)
    expect(forkA.processes.has(111)).toBe(true)
    expect(forkB.processes.size).toBe(1)
    expect(forkB.processes.has(222)).toBe(true)
  })
})

// ── B. TurnProjection shell_completed wake trigger ─────────────────────

describe('TurnProjection shell_completed wake trigger', () => {
  const idleEvents: AppEvent[] = [
    {
      type: 'agent_created',
      forkId: 'fork-a',
      parentForkId: null,
      agentId: 'agent-1',
      role: 'engineer',
      name: 'eng-1',
      taskId: 'task-1',
      mode: 'spawn',
      message: 'work',
      context: '',
    },
  ]

  it('when fork is idle, shell_completed enqueues a wake trigger', async () => {
    const fork = await makeTurnState(idleEvents, 'fork-a')
    expect(fork._tag).toBe('idle')
    expect((fork as TurnIdle).triggers.length).toBe(1)

    const next = await makeTurnState(
      [...idleEvents, completedEvent({ forkId: 'fork-a' })],
      'fork-a',
    )
    expect(next._tag).toBe('idle')
    expect((next as TurnIdle).triggers.length).toBe(2)
    expect((next as TurnIdle).triggers.some((t: any) => t._tag === 'wake')).toBe(true)
  })

  it('when fork is interrupting, shell_completed is ignored', async () => {
    const events: AppEvent[] = [
      ...idleEvents,
      {
        type: 'turn_started',
        forkId: 'fork-a',
        turnId: 'turn-1',
        chainId: 'chain-1',
      },
      {
        type: 'interrupt',
        forkId: 'fork-a',
      },
    ]
    const fork = await makeTurnState(events, 'fork-a')
    expect(fork._tag).toBe('interrupting')

    const next = await makeTurnState(
      [...events, completedEvent({ forkId: 'fork-a' })],
      'fork-a',
    )
    expect(next._tag).toBe('interrupting')
  })

  it('when fork is active, shell_completed is ignored', async () => {
    const events: AppEvent[] = [
      ...idleEvents,
      {
        type: 'turn_started',
        forkId: 'fork-a',
        turnId: 'turn-1',
        chainId: 'chain-1',
      },
    ]
    const fork = await makeTurnState(events, 'fork-a')
    expect(fork._tag).toBe('active')

    const next = await makeTurnState(
      [...events, completedEvent({ forkId: 'fork-a' })],
      'fork-a',
    )
    expect(next._tag).toBe('active')
  })
})

// ── C. WindowProjection shell_process_exited global handler ─────────────

describe('WindowProjection shell_process_exited global handler', () => {
  const sessionInit: AppEvent = {
    type: 'session_initialized',
    forkId: null,
    context: {
      cwd: '/tmp',
      scratchpadPath: '/tmp/scratch',
      platform: 'macos',
      shell: 'zsh',
      timezone: 'UTC',
      username: 'test',
      fullName: null,
      git: null,
      folderStructure: '',
      agentsFile: null,
      skills: null,
    },
    timestamp: ts(0),
  } as any

  const turnStart = (turnId: string, forkId: string | null = null): AppEvent => ({
    type: 'turn_started',
    forkId,
    turnId,
    chainId: 'chain-1',
    timestamp: ts(3),
  } as any)

  it('enqueues detached_process_exited timeline entry into owner fork', async () => {
    const fork = await makeWindowState([
      sessionInit,
      turnStart('turn-1', null),
      {
        type: 'turn_outcome',
        forkId: null,
        turnId: 'turn-1',
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [], yieldTarget: null }, requestId: null },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        cost: null,
        providerId: null,
        modelId: null,
      },
      registeredEvent(),
      exitedEvent(),
      turnStart('turn-2', null),
    ])

    const ctxMsgs = fork.messages.filter(m => m.type === 'context')
    expect(ctxMsgs.length).toBeGreaterThan(0)
    const ctx = ctxMsgs[ctxMsgs.length - 1] as Extract<ForkWindowState['messages'][number], { type: 'context' }>
    const entries = ctx.timeline.filter((e: any) => e.kind === 'detached_process_exited')
    expect(entries.length).toBe(1)
    expect(entries[0]).toMatchObject({
      kind: 'detached_process_exited',
      pid: 12345,
      command: 'sleep 30',
      exitCode: 0,
    })
  })

  it('also enqueues into root fork when owner !== root', async () => {
    const rootFork = await makeWindowState([
      sessionInit,
      {
        type: 'agent_created',
        forkId: 'fork-a',
        parentForkId: null,
        agentId: 'agent-1',
        role: 'engineer',
        name: 'eng-1',
        taskId: 'task-1',
        mode: 'spawn',
        message: 'work',
        context: '',
      },
      turnStart('turn-1', null),
      {
        type: 'turn_outcome',
        forkId: null,
        turnId: 'turn-1',
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [], yieldTarget: null }, requestId: null },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        cost: null,
        providerId: null,
        modelId: null,
      },
      registeredEvent({ forkId: 'fork-a' }),
      exitedEvent({ forkId: 'fork-a' }),
      turnStart('turn-2', null),
    ])

    const ctxMsgs = rootFork.messages.filter(m => m.type === 'context')
    expect(ctxMsgs.length).toBeGreaterThan(0)
    const ctx = ctxMsgs[ctxMsgs.length - 1] as Extract<ForkWindowState['messages'][number], { type: 'context' }>
    const entries = ctx.timeline.filter((e: any) => e.kind === 'detached_process_exited')
    expect(entries.length).toBe(1)
    expect(entries[0]).toMatchObject({
      kind: 'detached_process_exited',
      pid: 12345,
      command: 'sleep 30',
      exitCode: 0,
    })
  })

  it('falls back to empty paths if process not found', async () => {
    const fork = await makeWindowState([
      sessionInit,
      turnStart('turn-1', null),
      {
        type: 'turn_outcome',
        forkId: null,
        turnId: 'turn-1',
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: { _tag: 'Completed', completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [], yieldTarget: null }, requestId: null },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        cost: null,
        providerId: null,
        modelId: null,
      },
      // No registered event — process unknown
      exitedEvent(),
      turnStart('turn-2', null),
    ])

    const ctxMsgs = fork.messages.filter(m => m.type === 'context')
    expect(ctxMsgs.length).toBeGreaterThan(0)
    const ctx = ctxMsgs[ctxMsgs.length - 1] as Extract<ForkWindowState['messages'][number], { type: 'context' }>
    const entries = ctx.timeline.filter((e: any) => e.kind === 'detached_process_exited')
    expect(entries.length).toBe(1)
    expect(entries[0]).toMatchObject({
      kind: 'detached_process_exited',
      pid: 12345,
      stdoutPath: '',
      stderrPath: '',
    })
  })
})
