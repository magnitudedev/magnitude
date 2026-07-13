import { Context, Effect, Layer, PubSub, Ref, Stream } from "effect"
import { AcnRpcCommandActivity } from "@magnitudedev/protocol"

export interface AcnActivityState {
  readonly lastCommandAt: number
  readonly lastActivityAt: number
}

export interface AcnActivityTrackerApi {
  readonly markCommand: (operation: string) => Effect.Effect<void>
  readonly touch: (reason: string) => Effect.Effect<void>
  readonly current: Effect.Effect<AcnActivityState>
  readonly changes: Stream.Stream<void>
}

export class AcnActivityTracker extends Context.Tag("AcnActivityTracker")<
  AcnActivityTracker,
  AcnActivityTrackerApi
>() {}

interface ActivityState {
  readonly lastCommandAt: number
  readonly lastActivityAt: number
}

const now = () => Date.now()

export const AcnActivityTrackerLive: Layer.Layer<AcnActivityTracker> =
  Layer.effect(
    AcnActivityTracker,
    Effect.gen(function* () {
      const startedAt = now()
      const state = yield* Ref.make<ActivityState>({
        lastCommandAt: 0,
        lastActivityAt: startedAt,
      })
      const changes = yield* PubSub.unbounded<void>()

      const publishChange = PubSub.publish(changes, undefined).pipe(Effect.asVoid)

      const touch = (_reason: string) =>
        Ref.update(state, (current) => ({
          ...current,
          lastActivityAt: now(),
        })).pipe(Effect.zipRight(publishChange))

      const markCommand = (operation: string) =>
        Ref.update(state, (current) => {
          const timestamp = now()
          return {
            ...current,
            lastCommandAt: timestamp,
            lastActivityAt: timestamp,
          }
        }).pipe(
          Effect.zipRight(publishChange),
          Effect.annotateLogs({ operation }),
        )

      return {
        markCommand,
        touch,
        current: Ref.get(state),
        changes: Stream.fromPubSub(changes).pipe(Stream.map(() => undefined)),
      }
    }),
  )

export const AcnRpcCommandActivityLive: Layer.Layer<
  AcnRpcCommandActivity,
  never,
  AcnActivityTracker
> = Layer.effect(
  AcnRpcCommandActivity,
  Effect.map(AcnActivityTracker, (activity) =>
    ({ rpc, next }) =>
      activity.markCommand(rpc._tag).pipe(
        Effect.zipRight(next),
        Effect.tap(() => activity.touch("command-completed")),
      ),
  ),
)
