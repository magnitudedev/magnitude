import { Cause, Context, Data, Deferred, Effect, Layer, Queue } from 'effect'
import type { BaseEvent } from './event-bus-core'
import {
  ProjectionBusTag,
  isInsideProjectionBusTransaction,
  processAmbientChangeTransaction,
  type ProjectionBusService,
} from './projection-bus'
import type { AmbientDef } from '../ambient/define'

class UnregisteredAmbientDefect extends Data.TaggedError('UnregisteredAmbientDefect')<{
  readonly ambientName: string
}> {}

export interface AmbientService {
  register<T, R>(def: AmbientDef<T, R>): Effect.Effect<void, never, R>
  getValue<T, R>(def: AmbientDef<T, R>): T
  update<T, R>(def: AmbientDef<T, R>, value: T): Effect.Effect<void>
}

export const AmbientServiceTag = Context.GenericTag<AmbientService>('AmbientService')

export function makeAmbientServiceLayer<E extends BaseEvent>(): Layer.Layer<
  AmbientService,
  never,
  ProjectionBusService<E>
> {
  const BusTag = ProjectionBusTag<E>()

  return Layer.scoped(
    AmbientServiceTag,
    Effect.gen(function* () {
      const bus = yield* BusTag
      const values = new Map<AmbientDef<unknown, unknown>, unknown>()
      const updates = yield* Queue.unbounded<{
        readonly def: AmbientDef<unknown, unknown>
        readonly value: unknown
        readonly done: Deferred.Deferred<void, Cause.Cause<never>>
      }>()

      const processUpdate = (
        def: AmbientDef<unknown, unknown>,
        value: unknown,
      ): Effect.Effect<void> => Effect.gen(function* () {
        if (!values.has(def)) {
          return yield* Effect.die(
            new UnregisteredAmbientDefect({ ambientName: def.name })
          )
        }

        yield* processAmbientChangeTransaction(
          bus,
          def.name,
          value,
          Effect.sync(() => {
            values.set(def, value)
          }),
        )
      })

      yield* Effect.forkScoped(Effect.forever(
        Queue.take(updates).pipe(
          Effect.flatMap(({ def, value, done }) => processUpdate(def, value).pipe(
            Effect.matchCauseEffect({
              onFailure: (cause) => Deferred.fail(done, cause),
              onSuccess: () => Deferred.succeed(done, undefined),
            })
          ))
        )
      ))

      const service: AmbientService = {
        register<T, R>(def: AmbientDef<T, R>) {
          if (values.has(def)) {
            return Effect.void
          }

          return Effect.gen(function* () {
            const initial =
              Effect.isEffect(def.initial)
                ? yield* def.initial
                : def.initial

            values.set(def, initial)
          })
        },

        getValue<T, R>(def: AmbientDef<T, R>): T {
          if (!values.has(def)) {
            throw new UnregisteredAmbientDefect({ ambientName: def.name })
          }
          return values.get(def) as T
        },

        update<T, R>(def: AmbientDef<T, R>, value: T) {
          return Effect.gen(function* () {
            if (yield* isInsideProjectionBusTransaction(bus)) {
              return yield* processUpdate(
                def as AmbientDef<unknown, unknown>,
                value,
              )
            }

            const done = yield* Deferred.make<void, Cause.Cause<never>>()
            yield* Queue.offer(updates, {
              def: def as AmbientDef<unknown, unknown>,
              value,
              done,
            })
            yield* Deferred.await(done).pipe(
              Effect.catchAll((cause) => Effect.failCause(cause))
            )
          })
        },
      }
      return service
    })
  )
}
