import { Cause, Chunk, Effect, Stream, Queue, Scope, Option } from 'effect'
import type { TurnEngineCrash } from '@magnitudedev/xml-act'
import type { TurnEvent, TurnEventSink, TurnError } from './types'

type Envelope =
  | { readonly _tag: 'Event'; readonly event: TurnEvent }
  | { readonly _tag: 'Done' }
  | { readonly _tag: 'Failure'; readonly error: TurnEngineCrash | TurnError }
  | { readonly _tag: 'Defect'; readonly cause: Cause.Cause<never> }

/**
 * Create a turn event stream from an effectful producer.
 *
 * The producer receives a write-only sink for turn events.
 * Normal completion is signaled in-band through the envelope queue so
 * buffered events are never truncated by queue shutdown.
 *
 * The producer MUST emit a TurnResult event before returning successfully.
 * forkScoped still ensures the producer is interrupted when the stream scope closes.
 */
export function createTurnStream<R>(
  producer: (sink: TurnEventSink) => Effect.Effect<void, TurnEngineCrash | TurnError, R>
): Stream.Stream<TurnEvent, TurnEngineCrash | TurnError, R | Scope.Scope> {
  return Stream.unwrapScoped(
    Effect.gen(function* () {
      const queue = yield* Queue.unbounded<Envelope>()

      const sink: TurnEventSink = {
        emit: (event) => Queue.offer(queue, { _tag: 'Event', event }).pipe(Effect.asVoid),
      }

      yield* Effect.forkScoped(
        producer(sink).pipe(
          Effect.matchEffect({
            onSuccess: () => Queue.offer(queue, { _tag: 'Done' }).pipe(Effect.asVoid),
            onFailure: (error) => Queue.offer(queue, { _tag: 'Failure', error }).pipe(Effect.asVoid),
          }),
          Effect.catchAllCause((cause) =>
            Queue.offer(queue, { _tag: 'Defect', cause }).pipe(Effect.asVoid)
          ),
        )
      )

      return Stream.fromPull(
        Effect.succeed(
          Queue.take(queue).pipe(
            Effect.flatMap((item): Effect.Effect<Chunk.Chunk<TurnEvent>, Option.Option<TurnEngineCrash | TurnError>, never> => {
              switch (item._tag) {
                case 'Event':
                  return Effect.succeed(Chunk.of(item.event))
                case 'Done':
                  return Effect.fail(Option.none())
                case 'Failure':
                  return Effect.fail(Option.some(item.error))
                case 'Defect':
                  return Effect.failCause(item.cause)
              }
            })
          )
        )
      )
    })
  )
}
