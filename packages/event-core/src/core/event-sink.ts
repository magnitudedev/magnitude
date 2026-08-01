/**
 * EventSink
 *
 * In-memory accumulator for events awaiting durable persistence.
 * Scoped claims provide exclusive flush ownership without removing events
 * before their durable commit.
 *
 * Generic over E - the application's event union type.
 * Follows the same pattern as WorkerBus (GenericTag + layer factory).
 */

import { Effect, Ref, Context, Layer } from 'effect'
import type { Scope } from 'effect'
import type { BaseEvent, Timestamped } from './event-bus-core'

export interface PendingEventClaim<E extends BaseEvent = BaseEvent> {
  readonly events: ReadonlyArray<Timestamped<E>>
  readonly acknowledge: Effect.Effect<void>
}

export interface EventSinkService<E extends BaseEvent = BaseEvent> {
  append: (event: Timestamped<E>) => Effect.Effect<void>
  claimPending: () => Effect.Effect<PendingEventClaim<E>, never, Scope.Scope>
}

export const EventSinkTag = <E extends BaseEvent>() =>
  Context.GenericTag<EventSinkService<E>>('EventSink')

export function makeEventSinkLayer<E extends BaseEvent>(): Layer.Layer<EventSinkService<E>> {
  const Tag = EventSinkTag<E>()

  return Layer.scoped(Tag, Effect.gen(function* () {
    const pendingRef = yield* Ref.make<Timestamped<E>[]>([])
    const claimLock = yield* Effect.makeSemaphore(1)

    return {
      append: (event: Timestamped<E>) =>
        Ref.update(pendingRef, events => [...events, event]),

      claimPending: () =>
        Effect.acquireRelease(
          claimLock.take(1).pipe(
            Effect.zipRight(Ref.make<'active' | 'acknowledged' | 'closed'>('active'))
          ),
          claimState =>
            Ref.set(claimState, 'closed').pipe(
              Effect.zipRight(claimLock.release(1))
            ),
        ).pipe(
          Effect.flatMap(claimState => Effect.gen(function* () {
            const events = [...(yield* Ref.get(pendingRef))]

            const acknowledge = Effect.uninterruptible(
              Ref.modify(claimState, state =>
                state === 'active'
                  ? [true, 'acknowledged'] as const
                  : [false, state] as const
              ).pipe(
                Effect.flatMap(canAcknowledge => {
                  if (!canAcknowledge) {
                    return Effect.die(new Error('Pending event claim is no longer active'))
                  }

                  return Ref.modify(pendingRef, pending => {
                    const isClaimedPrefix =
                      events.length <= pending.length
                      && events.every((event, index) => pending[index] === event)

                    return isClaimedPrefix
                      ? [true, pending.slice(events.length)] as const
                      : [false, pending] as const
                  }).pipe(
                    Effect.flatMap(isClaimedPrefix =>
                      isClaimedPrefix
                        ? Effect.void
                        : Effect.die(new Error('Pending event claim no longer matches the queue prefix'))
                    )
                  )
                })
              )
            )

            return { events, acknowledge } satisfies PendingEventClaim<E>
          }))
        )
    }
  }))
}
