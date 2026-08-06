import {
  AcnIdentitySchema,
  AcnInstanceSchema,
  type AcnInstance,
} from "@magnitudedev/acn-protocol";
import { Context, Effect, Option, Schema, Stream } from "effect";
import type { DaemonError } from "./errors";
import { DaemonSpawnFailed } from "./errors";
import { AcnLifecycleObservationSchema } from "./lifecycle";

export const AcnLaunchRequestSchema = Schema.Struct({
  identity: AcnIdentitySchema,
  replace: Schema.optionalWith(AcnInstanceSchema, {
    as: "Option",
    exact: true,
  }),
  command: Schema.optionalWith(Schema.Array(Schema.String), {
    as: "Option",
    exact: true,
  }),
});
export type AcnLaunchRequest = typeof AcnLaunchRequestSchema.Type;

export const AcnLaunchEventSchema = Schema.Union(
  Schema.TaggedStruct("Observation", {
    observation: AcnLifecycleObservationSchema,
  }),
  Schema.TaggedStruct("Ready", {
    instance: AcnInstanceSchema,
  })
);
export type AcnLaunchEvent = typeof AcnLaunchEventSchema.Type;

export interface AcnProcessManager {
  readonly observeCurrent: Effect.Effect<
    Option.Option<AcnInstance>,
    DaemonError
  >;
  readonly launch: (
    request: AcnLaunchRequest
  ) => Stream.Stream<AcnLaunchEvent, DaemonError>;
  readonly terminate: (
    instance: AcnInstance
  ) => Effect.Effect<void, DaemonError>;
}

export const AcnProcessManager = Context.GenericTag<AcnProcessManager>(
  "@magnitudedev/sdk/AcnProcessManager"
);

export const runAcnLaunch = (
  events: Stream.Stream<AcnLaunchEvent, DaemonError>
): Effect.Effect<AcnInstance, DaemonError> =>
  events.pipe(
    Stream.runFoldEffect(Option.none<AcnInstance>(), (ready, event) => {
      if (event._tag === "Observation") {
        return Option.isSome(ready)
          ? Effect.fail(
              new DaemonSpawnFailed({
                reason: "ACN launch emitted progress after readiness",
              })
            )
          : Effect.succeed(ready);
      }
      return Option.isSome(ready)
        ? Effect.fail(
            new DaemonSpawnFailed({
              reason: "ACN launch emitted readiness more than once",
            })
          )
        : Effect.succeed(Option.some(event.instance));
    }),
    Effect.flatMap(
      Option.match({
        onNone: () =>
          Effect.fail(
            new DaemonSpawnFailed({
              reason: "ACN launch ended without readiness",
            })
          ),
        onSome: Effect.succeed,
      })
    )
  );
