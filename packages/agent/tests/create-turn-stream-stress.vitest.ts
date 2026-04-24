import { describe, expect, it } from '@effect/vitest'
import { Cause, Effect, Exit, Ref, Scope, Stream } from 'effect'
import type { TurnEngineEvent, TurnEngineCrash } from '@magnitudedev/xml-act'
import type { CallUsage } from '@magnitudedev/providers'
import type { MessageDestination, TurnOutcome } from '../src/events'
import { createTurnStream } from '../src/execution/turn-stream'
import { TurnError } from '../src/execution/types'
import type { TurnEvent, TurnError as TurnErrorType, TurnEventSink, TurnStrategyResult } from '../src/execution/types'

type PublishedEvent =
  | { readonly type: 'message_start'; readonly id: string; readonly destination: MessageDestination }
  | { readonly type: 'message_chunk'; readonly id: string; readonly text: string }
  | { readonly type: 'message_end'; readonly id: string }
  | { readonly type: 'thinking_chunk'; readonly text: string }
  | { readonly type: 'thinking_end'; readonly about: string | null }
  | { readonly type: 'raw_response_chunk'; readonly text: string }
  | { readonly type: 'lens_start'; readonly name: string }
  | { readonly type: 'lens_chunk'; readonly text: string }
  | { readonly type: 'lens_end'; readonly name: string }
  | { readonly type: 'tool_event'; readonly toolCallId: string; readonly toolKey: string; readonly event: TurnEngineEvent }

const successfulTurnResult: TurnOutcome = {
  _tag: 'Completed',
  completion: { yieldTarget: 'user', feedback: [] },
}

const usage: CallUsage = {
  inputTokens: 1,
  outputTokens: 1,
  cacheReadTokens: 0,
  cacheWriteTokens: 0,
  inputCost: 0,
  outputCost: 0,
  totalCost: 0,
}

const finalTurnResult: TurnStrategyResult = {
  executeResult: { result: successfulTurnResult },
  usage,
}

const baseEvents: ReadonlyArray<TurnEvent> = [
  { _tag: 'RawResponseChunk', text: 'raw-1' },
  { _tag: 'ThinkingDelta', text: 'think-1' },
  { _tag: 'ThinkingEnd', about: 'analysis' },
  { _tag: 'MessageStart', id: 'm1', destination: { kind: 'user' } },
  { _tag: 'MessageChunk', id: 'm1', text: 'hello' },
  { _tag: 'MessageEnd', id: 'm1' },
  { _tag: 'LensStarted', name: 'turn' },
  { _tag: 'LensDelta', text: 'plan' },
  { _tag: 'LensEnded', name: 'turn' },
]

function toPublished(event: TurnEvent): PublishedEvent {
  switch (event._tag) {
    case 'MessageStart':
      return { type: 'message_start', id: event.id, destination: event.destination }
    case 'MessageChunk':
      return { type: 'message_chunk', id: event.id, text: event.text }
    case 'MessageEnd':
      return { type: 'message_end', id: event.id }
    case 'ThinkingDelta':
      return { type: 'thinking_chunk', text: event.text }
    case 'ThinkingEnd':
      return { type: 'thinking_end', about: event.about }
    case 'RawResponseChunk':
      return { type: 'raw_response_chunk', text: event.text }
    case 'LensStarted':
      return { type: 'lens_start', name: event.name }
    case 'LensDelta':
      return { type: 'lens_chunk', text: event.text }
    case 'LensEnded':
      return { type: 'lens_end', name: event.name }
    case 'ToolEvent':
      return { type: 'tool_event', toolCallId: event.toolCallId, toolKey: event.toolKey, event: event.event }
    case 'TurnResult':
      throw new Error('TurnResult is not published')
  }
}

function makeChunkEvents(count: number): ReadonlyArray<TurnEvent> {
  const events: Array<TurnEvent> = [{ _tag: 'MessageStart', id: 'm', destination: { kind: 'user' } }]
  for (let i = 0; i < count; i++) {
    events.push({ _tag: 'MessageChunk', id: 'm', text: `chunk-${i}` })
  }
  events.push({ _tag: 'MessageEnd', id: 'm' })
  return events
}

type ProducerPlan = {
  readonly events: ReadonlyArray<TurnEvent>
  readonly consumerDelayMs?: number
  readonly yieldsBetweenOffers?: number
  readonly yieldEvery?: number | null
  readonly sleepEvery?: { readonly every: number; readonly ms: number } | null
  readonly sleepBeforeTurnResultMs?: number
  readonly failAtIndex?: number | null
  readonly failMode?: 'typed' | 'defect'
}

function makeProducer(
  plan: ProducerPlan,
): (sink: TurnEventSink) => Effect.Effect<void, TurnEngineCrash | TurnErrorType, never> {
  return (sink: TurnEventSink) => Effect.gen(function* () {
    for (let i = 0; i < plan.events.length; i++) {
      if (plan.failAtIndex === i) {
        if (plan.failMode === 'typed') {
          return yield* Effect.fail(TurnError.StreamFailed({ message: `typed failure at ${i}` }))
        }
        return yield* Effect.die(new Error(`defect failure at ${i}`))
      }

      yield* sink.emit( plan.events[i])

      if (plan.yieldsBetweenOffers) {
        for (let j = 0; j < plan.yieldsBetweenOffers; j++) {
          yield* Effect.yieldNow()
        }
      }

      if (plan.yieldEvery && (i + 1) % plan.yieldEvery === 0) {
        yield* Effect.yieldNow()
      }

      if (plan.sleepEvery && (i + 1) % plan.sleepEvery.every === 0) {
        yield* Effect.sleep(`${plan.sleepEvery.ms} millis`)
      }
    }

    if (plan.sleepBeforeTurnResultMs) {
      yield* Effect.sleep(`${plan.sleepBeforeTurnResultMs} millis`)
    }

    if (plan.failAtIndex === plan.events.length) {
      if (plan.failMode === 'typed') {
        return yield* Effect.fail(TurnError.StreamFailed({ message: 'typed failure before TurnResult' }))
      }
      return yield* Effect.die(new Error('defect failure before TurnResult'))
    }

    yield* sink.emit( { _tag: 'TurnResult', value: finalTurnResult })
  })
}

function drainTurnStream<R>(
  turnStream: Stream.Stream<TurnEvent, TurnEngineCrash | TurnErrorType, R | Scope.Scope>,
  publish: (event: PublishedEvent) => Effect.Effect<void, never, never>,
): Effect.Effect<{ finalResult: TurnStrategyResult }, TurnEngineCrash | TurnErrorType, R> {
  return Effect.gen(function* () {
    let finalResult: TurnStrategyResult | null = null

    yield* Effect.scoped(turnStream.pipe(
      Stream.runForEach((event) => Effect.gen(function* () {
        if (event._tag === 'TurnResult') {
          finalResult = event.value
          return
        }
        yield* publish(toPublished(event))
      }))
    ))

    if (!finalResult) {
      return yield* Effect.die(new Error('Turn stream ended without TurnResult'))
    }

    return { finalResult }
  })
}

function runScenario(plan: ProducerPlan): Effect.Effect<{
  readonly exit: Exit.Exit<{ finalResult: TurnStrategyResult }, unknown>
  readonly published: ReadonlyArray<PublishedEvent>
}, never, never> {
  return Effect.scoped(Effect.gen(function* () {
    const publishedRef = yield* Ref.make<Array<PublishedEvent>>([])

    const publish = (event: PublishedEvent) =>
      Effect.gen(function* () {
        if (plan.consumerDelayMs && plan.consumerDelayMs > 0) {
          yield* Effect.sleep(`${plan.consumerDelayMs} millis`)
        }
        yield* Ref.update(publishedRef, (events) => [...events, event])
      })

    const exit = yield* drainTurnStream(
      createTurnStream(makeProducer(plan)),
      publish,
    ).pipe(Effect.exit)

    const published = yield* Ref.get(publishedRef)
    return { exit, published }
  }))
}

function expectSuccessWithAllEvents(
  result: { readonly exit: Exit.Exit<{ finalResult: TurnStrategyResult }, unknown>; readonly published: ReadonlyArray<PublishedEvent> },
  expectedEvents: ReadonlyArray<TurnEvent>,
) {
  expect(Exit.isSuccess(result.exit)).toBe(true)
  if (!Exit.isSuccess(result.exit)) return
  expect(result.exit.value.finalResult).toEqual(finalTurnResult)
  expect(result.published).toEqual(expectedEvents.map(toPublished))
}

describe('createTurnStream stress suite', () => {
  describe('green properties: delivery, ordering, lifecycle, and error propagation', () => {
    it.effect('delivers TurnResult for an empty turn', () =>
      Effect.gen(function* () {
        const result = yield* runScenario({ events: [] })
        expectSuccessWithAllEvents(result, [])
      }))

    it.effect('delivers one event before TurnResult', () =>
      Effect.gen(function* () {
        const events: ReadonlyArray<TurnEvent> = [{ _tag: 'RawResponseChunk', text: 'only' }]
        const result = yield* runScenario({ events })
        expectSuccessWithAllEvents(result, events)
      }))

    it.effect('delivers mixed event types in order', () =>
      Effect.gen(function* () {
        const result = yield* runScenario({ events: baseEvents })
        expectSuccessWithAllEvents(result, baseEvents)
      }))

    it.effect('preserves delivery when producer yields between all offers', () =>
      Effect.gen(function* () {
        const events = makeChunkEvents(100)
        const result = yield* runScenario({ events, yieldsBetweenOffers: 1 })
        expectSuccessWithAllEvents(result, events)
      }))

    it.effect('propagates typed producer failures after partial delivery', () =>
      Effect.gen(function* () {
        const events = makeChunkEvents(10)
        const result = yield* runScenario({
          events,
          failAtIndex: 5,
          failMode: 'typed',
        })

        expect(Exit.isFailure(result.exit)).toBe(true)
        if (!Exit.isFailure(result.exit)) return
        expect(Cause.failureOption(result.exit.cause)._tag).toBe('Some')
        expect(result.published).toEqual(events.slice(0, 5).map(toPublished))
      }))

    it.effect('propagates producer defects after partial delivery', () =>
      Effect.gen(function* () {
        const events = makeChunkEvents(10)
        const result = yield* runScenario({
          events,
          failAtIndex: 5,
          failMode: 'defect',
        })

        expect(Exit.isFailure(result.exit)).toBe(true)
        if (!Exit.isFailure(result.exit)) return
        expect(Cause.dieOption(result.exit.cause)._tag).toBe('Some')
        expect(result.published).toEqual(events.slice(0, 5).map(toPublished))
      }))

    it.effect('supports producer-side pacing without dropping events', () =>
      Effect.gen(function* () {
        const events = makeChunkEvents(10)
        const result = yield* runScenario({
          events,
          yieldEvery: 1,
        })

        expectSuccessWithAllEvents(result, events)
      }))
  })

  describe('red stress tests: known queue truncation bug', () => {
    const stressCases = [
      { name: '5000 events, slow consumer 0ms', eventCount: 5000, consumerDelayMs: 0, iterations: 3 },
      { name: '3000 events, slow consumer 0ms', eventCount: 3000, consumerDelayMs: 0, iterations: 3 },
      { name: '5000 events with sparse producer yields', eventCount: 5000, consumerDelayMs: 0, iterations: 2, yieldEvery: 1000 },
    ] as const

    for (const testCase of stressCases) {
      it.effect(`BUG: should deliver TurnResult under backlog pressure (${testCase.name})`, () =>
        Effect.gen(function* () {
          const events = makeChunkEvents(testCase.eventCount)

          for (let iteration = 0; iteration < testCase.iterations; iteration++) {
            const result = yield* runScenario({
              events,
              consumerDelayMs: testCase.consumerDelayMs,
              yieldEvery: 'yieldEvery' in testCase ? testCase.yieldEvery : null,
            })

            // Desired property. This should be green after the bug is fixed.
            expectSuccessWithAllEvents(result, events)
          }
        }), 5000)
    }

    it.effect('BUG: single-chunk producer should not lose TurnResult even with a large zero-delay backlog', () =>
      Effect.gen(function* () {
        const events = makeChunkEvents(3000)
        const result = yield* runScenario({
          events,
          consumerDelayMs: 0,
          yieldsBetweenOffers: 0,
          yieldEvery: null,
        })

        expectSuccessWithAllEvents(result, events)
      }), 5000)

    it.effect('BUG: partial producer yielding should still preserve TurnResult for a large backlog', () =>
      Effect.gen(function* () {
        const events = makeChunkEvents(5000)
        const result = yield* runScenario({
          events,
          consumerDelayMs: 0,
          yieldEvery: 1000,
        })

        expectSuccessWithAllEvents(result, events)
      }), 5000)
  })
})
