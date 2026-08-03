import {
  AcnEndpointSchema,
  type AcnEndpoint,
} from "@magnitudedev/acn-protocol"
import { Context, Effect, Option, Schema, Stream } from "effect"
import { DaemonSpawnFailed, type DaemonError } from "./errors"
import { AcnLifecycleObservationSchema } from "./lifecycle"

export const DaemonLaunchEventSchema = Schema.Union(
  Schema.TaggedStruct("Observation", {
    observation: AcnLifecycleObservationSchema,
  }),
  Schema.TaggedStruct("Ready", {
    endpoint: AcnEndpointSchema,
  }),
)
export type DaemonLaunchEvent = typeof DaemonLaunchEventSchema.Type

/** Mutation boundary for launching a Magnitude daemon. */
export interface DaemonLauncher {
  readonly launch: (
    command: Option.Option<ReadonlyArray<string>>,
  ) => Stream.Stream<DaemonLaunchEvent, DaemonError>
}

export const DaemonLauncher = Context.GenericTag<DaemonLauncher>(
  "@magnitudedev/sdk/DaemonLauncher",
)

interface LaunchReduction {
  readonly ready: Option.Option<AcnEndpoint>
}

export const runDaemonLaunch = (
  events: Stream.Stream<DaemonLaunchEvent, DaemonError>,
): Effect.Effect<AcnEndpoint, DaemonError> =>
  events.pipe(
    Stream.runFoldEffect(
      { ready: Option.none<AcnEndpoint>() } satisfies LaunchReduction,
      (state, event) => {
        if (event._tag === "Observation") {
          if (Option.isSome(state.ready)) {
            return new DaemonSpawnFailed({
              reason: "Daemon launch emitted an observation after readiness",
            })
          }
          return Effect.succeed(state)
        }
        if (Option.isSome(state.ready)) {
          return new DaemonSpawnFailed({
            reason: "Daemon launch emitted readiness more than once",
          })
        }
        return Effect.succeed({ ready: Option.some(event.endpoint) })
      },
    ),
    Effect.flatMap(({ ready }) =>
      Option.match(ready, {
        onNone: () => new DaemonSpawnFailed({
          reason: "Daemon launch ended without readiness",
        }),
        onSome: Effect.succeed,
      }),
    ),
  )
