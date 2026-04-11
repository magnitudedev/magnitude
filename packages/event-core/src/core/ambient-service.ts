import { Context, Data, Effect, Layer } from 'effect'
import type { BaseEvent } from './event-bus-core'
import { ProjectionBusTag, type ProjectionBusService } from './projection-bus'
import type { AmbientDef, AmbientSnapshot } from '../ambient/define'

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

  return Layer.effect(
    AmbientServiceTag,
    Effect.gen(function* () {
      const bus = yield* BusTag
      const snapshots = new Map<AmbientDef<unknown, unknown>, AmbientSnapshot<unknown>>()

      return {
        register<T, R>(def: AmbientDef<T, R>) {
          if (snapshots.has(def)) {
            return Effect.void
          }

          return Effect.gen(function* () {
            const initial =
              Effect.isEffect(def.initial)
                ? yield* def.initial
                : def.initial

            snapshots.set(def, {
              value: initial,
              version: 0
            })
          })
        },

        getValue<T, R>(def: AmbientDef<T, R>): T {
          return snapshots.get(def)!.value as T
        },

        update<T, R>(def: AmbientDef<T, R>, value: T) {
          const current = snapshots.get(def)
          if (!current) {
            return Effect.die(new UnregisteredAmbientDefect({ ambientName: def.name }))
          }

          const nextSnapshot: AmbientSnapshot<T> = {
            value,
            version: current.version + 1
          }

          snapshots.set(def, nextSnapshot)
          return bus.processAmbientChange(def.name, value)
        }
      } satisfies AmbientService
    })
  )
}
