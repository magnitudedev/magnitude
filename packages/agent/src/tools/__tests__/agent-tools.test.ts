import { describe, expect, test } from 'bun:test'
import { Effect, Layer, Ref, Stream } from 'effect'
import { Fork, WorkerBusTag, type WorkerBusService } from '@magnitudedev/event-core'
import { noopToolContext } from '@magnitudedev/tools'
import { agentKillTool } from '../agent-tools'
import { AgentStateReaderTag } from '../fork'
import type { AgentStatusState } from '../../projections/agent-status'
import type { AppEvent } from '../../events'

const { ForkContext } = Fork

function makeAgentState(status: 'working' | 'idle', parentForkId: string | null): AgentStatusState {
  return {
    agents: new Map([
      ['agent-sub', {
        agentId: 'agent-sub',
        forkId: 'fork-sub',
        parentForkId,
        name: 'Subagent',
        role: 'worker',
        context: '',
        mode: 'spawn',
        taskId: 'task-1',
        message: null,
        status,
      }]
    ]),
    agentByForkId: new Map([['fork-sub', 'agent-sub']]),
  }
}

function makeLayer(state: AgentStatusState, eventsRef: Ref.Ref<AppEvent[]>) {
  return Layer.mergeAll(
    Layer.succeed(ForkContext, { forkId: 'parent-1' }),
    Layer.succeed(AgentStateReaderTag, {
      getAgentState: () => Effect.succeed(state),
      getAgent: (agentId: string) => Effect.succeed(state.agents.get(agentId)),
    }),
    Layer.succeed(WorkerBusTag<AppEvent>(), {
      publish: (event: AppEvent) => Ref.update(eventsRef, (events) => [...events, event]),
      subscribeToTypes: () => Stream.empty,
      stream: Stream.empty,
      subscribe: () => Effect.succeed(Stream.empty),
    } satisfies WorkerBusService<AppEvent>),
  )
}

function runPromise<A, E, R>(effect: Effect.Effect<A, E, R>) {
  return Effect.runPromise(effect as unknown as Effect.Effect<A, E, never>)
}

function runPromiseExit<A, E, R>(effect: Effect.Effect<A, E, R>) {
  return Effect.runPromiseExit(effect as unknown as Effect.Effect<A, E, never>)
}

describe('agent.kill tool validation matrix', () => {
  test('rejects unknown target', async () => {
    const eventsRef = await Effect.runPromise(Ref.make<AppEvent[]>([]))
    const state: AgentStatusState = { agents: new Map(), agentByForkId: new Map() }

    const exit = await Effect.runPromiseExit(
      agentKillTool.execute({ agentId: 'missing', reason: undefined }, noopToolContext).pipe(
        Effect.provide(makeLayer(state, eventsRef)),
      ) as unknown as Effect.Effect<{ readonly forkId: string; readonly agentId: string }, unknown, never>,
    )

    expect(exit._tag).toBe('Failure')
    expect((await Effect.runPromise(Ref.get(eventsRef))).length).toBe(0)
  })

  test('rejects non-child target', async () => {
    const eventsRef = await Effect.runPromise(Ref.make<AppEvent[]>([]))
    const exit = await Effect.runPromiseExit(
      agentKillTool.execute({ agentId: 'agent-sub', reason: undefined }, noopToolContext).pipe(
        Effect.provide(makeLayer(makeAgentState('working', 'other-parent'), eventsRef)),
      ) as unknown as Effect.Effect<{ readonly forkId: string; readonly agentId: string }, unknown, never>,
    )

    expect(exit._tag).toBe('Failure')
    expect((await Effect.runPromise(Ref.get(eventsRef))).length).toBe(0)
  })

  test('rejects idle target', async () => {
    const eventsRef = await Effect.runPromise(Ref.make<AppEvent[]>([]))
    const exit = await Effect.runPromiseExit(
      agentKillTool.execute({ agentId: 'agent-sub', reason: undefined }, noopToolContext).pipe(
        Effect.provide(makeLayer(makeAgentState('idle', 'parent-1'), eventsRef)),
      ) as unknown as Effect.Effect<{ readonly forkId: string; readonly agentId: string }, unknown, never>,
    )

    expect(exit._tag).toBe('Failure')
    expect((await Effect.runPromise(Ref.get(eventsRef))).length).toBe(0)
  })

  test('accepts idle target', async () => {
    const eventsRef = await Effect.runPromise(Ref.make<AppEvent[]>([]))
    const result = await Effect.runPromise(
      agentKillTool.execute({ agentId: 'agent-sub', reason: 'cleanup' }, noopToolContext).pipe(
        Effect.provide(makeLayer(makeAgentState('idle', 'parent-1'), eventsRef)),
      ) as unknown as Effect.Effect<{ readonly forkId: string; readonly agentId: string }, unknown, never>,
    )

    expect(result).toEqual({ agentId: 'agent-sub', forkId: 'fork-sub' })
    const events = await Effect.runPromise(Ref.get(eventsRef))
    expect(events.map((e) => e.type)).toEqual(['agent_killed'])
  })

  test('accepts working target', async () => {
    const eventsRef = await Effect.runPromise(Ref.make<AppEvent[]>([]))
    const result = await Effect.runPromise(
      agentKillTool.execute({ agentId: 'agent-sub', reason: undefined }, noopToolContext).pipe(
        Effect.provide(makeLayer(makeAgentState('working', 'parent-1'), eventsRef)),
      ) as unknown as Effect.Effect<{ readonly forkId: string; readonly agentId: string }, unknown, never>,
    )

    expect(result).toEqual({ agentId: 'agent-sub', forkId: 'fork-sub' })
    const events = await Effect.runPromise(Ref.get(eventsRef))
    expect(events.map((e) => e.type)).toEqual(['agent_killed'])
  })
})
