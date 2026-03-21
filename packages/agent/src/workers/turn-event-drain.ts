import { Effect, Scope, Stream } from 'effect'
import type { PublishFn } from '@magnitudedev/event-core'
import type { XmlRuntimeCrash } from '@magnitudedev/xml-act'

import type { AppEvent } from '../events'
import type { TurnError, TurnEvent, TurnStrategyResult } from '../execution/types'

/**
 * Drain a turn event stream, publishing each event to the bus.
 * Returns the final TurnStrategyResult.
 */
export function drainTurnEventStream<R>(
  turnStream: Stream.Stream<TurnEvent, XmlRuntimeCrash | TurnError, R | Scope.Scope>,
  forkId: string | null,
  turnId: string,
  publish: PublishFn<AppEvent>,
): Effect.Effect<{ finalResult: TurnStrategyResult }, XmlRuntimeCrash | TurnError, R> {
  return Effect.gen(function* () {
    let finalResult: TurnStrategyResult | null = null

    yield* Effect.scoped(turnStream.pipe(
      Stream.runForEach((event) => Effect.gen(function* () {
        switch (event._tag) {
          case 'MessageStart':
            yield* publish({ type: 'message_start', forkId, turnId, id: event.id, dest: event.dest })
            break
          case 'MessageChunk':
            yield* publish({ type: 'message_chunk', forkId, turnId, id: event.id, text: event.text })
            break
          case 'MessageEnd':
            yield* publish({ type: 'message_end', forkId, turnId, id: event.id })
            break
          case 'ThinkingDelta':
            yield* publish({ type: 'thinking_chunk', forkId, turnId, text: event.text })
            break
          case 'ThinkingEnd':
            yield* publish({ type: 'thinking_end', forkId, turnId, about: event.about })
            break
          case 'LensStarted':
            yield* publish({ type: 'lens_start', forkId, turnId, name: event.name })
            break
          case 'LensDelta':
            yield* publish({ type: 'lens_chunk', forkId, turnId, text: event.text })
            break
          case 'LensEnded':
            yield* publish({ type: 'lens_end', forkId, turnId, name: event.name })
            break
          case 'ToolEvent':
            yield* publish({ type: 'tool_event', forkId, turnId, toolCallId: event.toolCallId, toolKey: event.toolKey, event: event.event })
            break
          case 'TurnResult':
            finalResult = event.value
            break
        }
      }))
    ))

    if (!finalResult) {
      return yield* Effect.die(new Error('Turn stream ended without TurnResult'))
    }

    return { finalResult }
  })
}