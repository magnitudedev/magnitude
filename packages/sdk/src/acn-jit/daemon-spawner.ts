import { Context, Effect, Option, Schema, Stream } from "effect"
import { DaemonError, DaemonSpawnFailed } from "./errors"
import { AcnLifecycleObservationSchema } from "./lifecycle"

export const DaemonSpawnEventSchema = Schema.Union(
  Schema.TaggedStruct("Observation", {
    observation: AcnLifecycleObservationSchema,
  }),
  Schema.TaggedStruct("Ready", {
    url: Schema.String.pipe(Schema.minLength(1)),
  }),
)
export type DaemonSpawnEvent = typeof DaemonSpawnEventSchema.Type

/**
 * The environment-specific boundary for daemon discovery and startup.
 *
 * Startup is a stream because observations occur before the final endpoint.
 * Local implementations produce it directly; remote implementations transport
 * the same schema. Failures remain in the typed stream error channel.
 */
export interface DaemonSpawner {
  readonly discover: () => Effect.Effect<
    Option.Option<string>,
    DaemonError,
    never
  >
  readonly spawn: (
    command: Option.Option<ReadonlyArray<string>>,
  ) => Stream.Stream<DaemonSpawnEvent, DaemonError>
}

export class DaemonSpawnerTag extends Context.Tag("DaemonSpawner")<
  DaemonSpawnerTag,
  DaemonSpawner
>() {}

interface SpawnReduction {
  readonly ready: Option.Option<string>
}

export const runDaemonSpawn = (
  events: Stream.Stream<DaemonSpawnEvent, DaemonError>,
): Effect.Effect<string, DaemonError> =>
  events.pipe(
    Stream.runFoldEffect(
      { ready: Option.none<string>() } satisfies SpawnReduction,
      (state, event) => {
        if (event._tag === "Observation") {
          if (Option.isSome(state.ready)) {
            return new DaemonSpawnFailed({
              reason: "Daemon spawn emitted an observation after readiness",
            })
          }
          return Effect.succeed(state)
        }
        if (Option.isSome(state.ready)) {
          return new DaemonSpawnFailed({
            reason: "Daemon spawn emitted readiness more than once",
          })
        }
        return Effect.succeed({ ready: Option.some(event.url) })
      },
    ),
    Effect.flatMap(({ ready }) =>
      Option.match(ready, {
        onNone: () =>
          new DaemonSpawnFailed({
            reason: "Daemon spawn ended without a ready endpoint",
          }),
        onSome: Effect.succeed,
      }),
    ),
  )
