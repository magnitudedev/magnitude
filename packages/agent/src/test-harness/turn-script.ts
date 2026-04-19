import { Context, Effect, Layer, Ref } from 'effect'
import { YIELD_USER } from '@magnitudedev/xml-act'

export interface ScriptGate {
  wait(): Promise<void>
  release(): void
}

export function createScriptGate(): ScriptGate {
  let released = false
  const waiters = new Set<() => void>()

  return {
    wait(): Promise<void> {
      if (released) {
        return Promise.resolve()
      }

      return new Promise<void>((resolve) => {
        waiters.add(resolve)
      })
    },
    release(): void {
      if (released) {
        return
      }
      released = true
      for (const resolve of waiters) {
        resolve()
      }
      waiters.clear()
    },
  }
}

export interface MockTurnResponse {
  /** Complete XML response */
  readonly xml?: string
  /** Chunked XML response for streaming tests */
  readonly xmlChunks?: readonly string[]
  /** Optional usage override */
  readonly usage?: {
    readonly inputTokens?: number | null
    readonly outputTokens?: number | null
    readonly cacheReadTokens?: number | null
    readonly cacheWriteTokens?: number | null
  }
  /** Optional delay between chunks (for timing tests) */
  readonly delayMsBetweenChunks?: number
  /** Fault injection: stop stream before all chunks are sent */
  readonly terminateStreamEarly?: boolean
  /** Fault injection: throw error after Nth chunk */
  readonly failAfterChunk?: number
}

export interface MockTurnScriptInput {
  readonly forkId: string | null
  readonly turnId: string
}

export type MockTurnScriptResolver = (input: MockTurnScriptInput) => MockTurnResponse

export interface MockTurnScriptService {
  readonly enqueue: (frame: MockTurnResponse, forkId?: string | null) => Effect.Effect<void>
  readonly dequeue: (input: MockTurnScriptInput) => Effect.Effect<MockTurnResponse>
  readonly setResolver: (resolver: MockTurnScriptResolver | null) => Effect.Effect<void>
  readonly clear: () => Effect.Effect<void>
}

export class MockTurnScriptTag extends Context.Tag('MockTurnScript')<MockTurnScriptTag, MockTurnScriptService>() {}

interface MockTurnScriptState {
  readonly globalQueue: readonly MockTurnResponse[]
  readonly forkQueues: ReadonlyMap<string | null, readonly MockTurnResponse[]>
  readonly resolver: MockTurnScriptResolver | null
}

const defaultFrame: MockTurnResponse = {
  xml: `<message>ok</message>${YIELD_USER}`,
}

const initialState: MockTurnScriptState = {
  globalQueue: [],
  forkQueues: new Map(),
  resolver: null,
}

const makeMockTurnScript = Effect.gen(function* () {
  const stateRef = yield* Ref.make<MockTurnScriptState>(initialState)

  const enqueue: MockTurnScriptService['enqueue'] = (frame, forkId = null) =>
    Ref.update(stateRef, (state) => {
      if (forkId === null) {
        return { ...state, globalQueue: [...state.globalQueue, frame] }
      }

      const nextMap = new Map(state.forkQueues)
      const existing = nextMap.get(forkId) ?? []
      nextMap.set(forkId, [...existing, frame])
      return { ...state, forkQueues: nextMap }
    })

  const dequeue: MockTurnScriptService['dequeue'] = (input) => Effect.gen(function* () {
    const resolver = yield* Ref.get(stateRef).pipe(Effect.map((s) => s.resolver))
    if (resolver) {
      return resolver(input)
    }

    const result = yield* Ref.modify(stateRef, (state): [MockTurnResponse, MockTurnScriptState] => {
      const forkQueue = state.forkQueues.get(input.forkId) ?? []
      if (forkQueue.length > 0) {
        const [head, ...tail] = forkQueue
        const nextMap = new Map(state.forkQueues)
        if (tail.length === 0) nextMap.delete(input.forkId)
        else nextMap.set(input.forkId, tail)
        return [head, { ...state, forkQueues: nextMap }]
      }

      if (state.globalQueue.length > 0) {
        const [head, ...tail] = state.globalQueue
        return [head, { ...state, globalQueue: tail }]
      }

      return [defaultFrame, state]
    })

    return result
  })

  const setResolver: MockTurnScriptService['setResolver'] = (resolver) =>
    Ref.update(stateRef, (state) => ({ ...state, resolver }))

  const clear: MockTurnScriptService['clear'] = () =>
    Ref.set(stateRef, initialState)

  return {
    enqueue,
    dequeue,
    setResolver,
    clear,
  } satisfies MockTurnScriptService
})

export const MockTurnScriptLive = Layer.effect(
  MockTurnScriptTag,
  makeMockTurnScript,
)
