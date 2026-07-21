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
  readonly hasActiveWork: Effect.Effect<boolean>
  readonly withActiveWork: <A, E, R>(label: string, effect: Effect.Effect<A, E, R>) => Effect.Effect<A, E, R>
  readonly acquireActiveWork: (label: string) => Effect.Effect<Effect.Effect<void>>
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
      const activeWork = yield* Ref.make(0)

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

      const acquire = (label: string) => Ref.update(activeWork, (count) => count + 1).pipe(
        Effect.zipRight(touch(`active-work-started:${label}`)),
      )
      const release = (label: string) => Ref.update(activeWork, (count) => Math.max(0, count - 1)).pipe(
        Effect.zipRight(touch(`active-work-finished:${label}`)),
      )

      return {
        markCommand,
        touch,
        current: Ref.get(state),
        changes: Stream.fromPubSub(changes).pipe(Stream.map(() => undefined)),
        hasActiveWork: Ref.get(activeWork).pipe(Effect.map((count) => count > 0)),
        withActiveWork: (label, effect) => Effect.acquireUseRelease(
          acquire(label),
          () => effect,
          () => release(label),
        ),
        acquireActiveWork: (label) => acquire(label).pipe(Effect.as(release(label))),
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
