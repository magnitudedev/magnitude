import type { AppEvent } from '../events'
import type { ResponseBuilder } from './response-builder'
import type { MockTurnResponse } from './turn-script'

export type TurnFrame = MockTurnResponse | ResponseBuilder

export interface TurnsResult {
  readonly forks: Map<string, string>
  readonly turns: Extract<AppEvent, { type: 'turn_completed' }>[]
}

export interface TurnsBuilder {
  user(message: string): this
  lead(response: TurnFrame): this
  agent(agentId: string, response: TurnFrame): this
  agents(mapping: Record<string, TurnFrame>): this
  run(): Promise<TurnsResult>
}

type Step =
  | { kind: 'user'; message: string }
  | { kind: 'lead'; response: MockTurnResponse }
  | { kind: 'agent'; agentId: string; response: MockTurnResponse }
  | { kind: 'agents'; mapping: Record<string, MockTurnResponse> }

function isResponseBuilder(frame: TurnFrame): frame is ResponseBuilder {
  return 'yield' in frame && typeof frame.yield === 'function'
}

function toFrame(frame: TurnFrame): MockTurnResponse {
  return isResponseBuilder(frame) ? frame.yield() : frame
}

interface TurnsHarness {
  user(text: string): Promise<void>
  script: {
    next(frame: MockTurnResponse, forkId?: string | null): Promise<void>
    setResolver(
      resolver: ((input: { forkId: string | null; turnId: string }) => MockTurnResponse) | null,
    ): Promise<void>
  }
  wait: {
    event<T extends AppEvent['type']>(
      type: T,
      pred?: (e: Extract<AppEvent, { type: T }>) => boolean,
    ): Promise<Extract<AppEvent, { type: T }>>
    turnCompleted(forkId?: string | null): Promise<Extract<AppEvent, { type: 'turn_completed' }>>
    agentCreated(
      pred?: (e: Extract<AppEvent, { type: 'agent_created' }>) => boolean,
    ): Promise<Extract<AppEvent, { type: 'agent_created' }>>
  }
  events(): readonly AppEvent[]
  onEvent(cb: (e: AppEvent) => void): () => void
}

export function createTurnsBuilder(harness: TurnsHarness): TurnsBuilder {
  const steps: Step[] = []
  const forkByAgent = new Map<string, string>()

  const getOrWaitForkId = async (agentId: string): Promise<string> => {
    const known = forkByAgent.get(agentId)
    if (known) return known

    const events = harness.events()
    for (let i = events.length - 1; i >= 0; i -= 1) {
      const event = events[i]
      if (event.type === 'agent_created' && event.agentId === agentId) {
        forkByAgent.set(agentId, event.forkId)
        return event.forkId
      }
    }

    const agentCreated = await harness.wait.agentCreated((e) => e.agentId === agentId)
    forkByAgent.set(agentId, agentCreated.forkId)
    return agentCreated.forkId
  }

  const builder: TurnsBuilder = {
    user(message: string) {
      steps.push({ kind: 'user', message })
      return this
    },

    lead(response: TurnFrame) {
      steps.push({ kind: 'lead', response: toFrame(response) })
      return this
    },

    agent(agentId: string, response: TurnFrame) {
      steps.push({ kind: 'agent', agentId, response: toFrame(response) })
      return this
    },

    agents(mapping: Record<string, TurnFrame>) {
      const normalized: Record<string, MockTurnResponse> = {}
      for (const [agentId, frame] of Object.entries(mapping)) {
        normalized[agentId] = toFrame(frame)
      }
      steps.push({ kind: 'agents', mapping: normalized })
      return this
    },

    async run(): Promise<TurnsResult> {
      const pendingUserMessages: string[] = []
      const flushPendingUsers = async (): Promise<void> => {
        if (pendingUserMessages.length === 0) return
        const msgs = [...pendingUserMessages]
        pendingUserMessages.length = 0
        await Promise.all(msgs.map((m) => harness.user(m)))
      }

      const consumedTurnIds = new Set<string>()
      const waitForNextTurnCompleted = async (
        forkId: string | null,
      ): Promise<Extract<AppEvent, { type: 'turn_completed' }>> => {
        for (const event of harness.events()) {
          if (event.type !== 'turn_completed') continue
          if (event.forkId !== forkId) continue
          if (consumedTurnIds.has(event.turnId)) continue
          consumedTurnIds.add(event.turnId)
          return event
        }

        const next = await harness.wait.event(
          'turn_completed',
          (event) => event.forkId === forkId && !consumedTurnIds.has(event.turnId),
        )
        consumedTurnIds.add(next.turnId)
        return next
      }

      const rootQueue: MockTurnResponse[] = []
      const subagentQueues = new Map<string, MockTurnResponse[]>()

      for (const step of steps) {
        if (step.kind === 'lead') {
          rootQueue.push(step.response)
          continue
        }
        if (step.kind === 'agent') {
          const q = subagentQueues.get(step.agentId) ?? []
          q.push(step.response)
          subagentQueues.set(step.agentId, q)
          continue
        }
        if (step.kind === 'agents') {
          for (const [agentId, response] of Object.entries(step.mapping)) {
            const q = subagentQueues.get(agentId) ?? []
            q.push(response)
            subagentQueues.set(agentId, q)
          }
        }
      }

      const fallback: MockTurnResponse = { xml: '<yield/>' }

      const unsub = harness.onEvent((event: AppEvent) => {
        if (event.type === 'agent_created') {
          forkByAgent.set(event.agentId, event.forkId)
        }
      })

      try {
        await harness.script.setResolver(({ forkId }) => {
          if (forkId === null) {
            return rootQueue.shift() ?? fallback
          }

          const agentId = Array.from(forkByAgent.entries()).find(([, id]) => id === forkId)?.[0]
          if (!agentId) {
            return fallback
          }

          const q = subagentQueues.get(agentId)
          return q && q.length > 0 ? (q.shift() ?? fallback) : fallback
        })

        for (const step of steps) {
          if (step.kind === 'user') {
            pendingUserMessages.push(step.message)
            continue
          }

          if (step.kind === 'lead') {
            await flushPendingUsers()
            await waitForNextTurnCompleted(null)
            continue
          }

          if (step.kind === 'agent') {
            const forkId = await getOrWaitForkId(step.agentId)
            await flushPendingUsers()
            await waitForNextTurnCompleted(forkId)
            continue
          }

          const entries = Object.entries(step.mapping)
          const forkEntries = await Promise.all(
            entries.map(async ([agentId]) => [agentId, await getOrWaitForkId(agentId)] as const),
          )
          await flushPendingUsers()
          await Promise.all(
            forkEntries.map(([, forkId]) => waitForNextTurnCompleted(forkId)),
          )
        }

        await flushPendingUsers()
      } finally {
        await harness.script.setResolver(null)
        unsub()
      }

      const turns = harness
        .events()
        .filter((event): event is Extract<AppEvent, { type: 'turn_completed' }> => event.type === 'turn_completed')

      return {
        forks: new Map(forkByAgent),
        turns,
      }
    },
  }

  return builder
}

export function scenario(
  harness: TurnsHarness,
): TurnsBuilder & { run(userMessage?: string): Promise<TurnsResult> } {
  const builder = createTurnsBuilder(harness)
  const originalRun = builder.run.bind(builder)

  return Object.assign(builder, {
    run: async (userMessage?: string) => {
      if (userMessage) {
        builder.user(userMessage)
      }
      return originalRun()
    },
  })
}
