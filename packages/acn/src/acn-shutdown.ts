import { Context, Deferred, Effect, Layer, Option } from "effect";

export type AcnShutdownReason =
  | "idle"
  | "upgrade"
  | "ownership-lost"
  | "icn-exited"
  | "signal"
  | "fatal";

export interface AcnShutdownRequest {
  readonly reason: AcnShutdownReason;
  readonly detail?: string;
}

export interface AcnShutdownApi {
  /** First request wins; subsequent requests observe the same shutdown. */
  readonly request: (request: AcnShutdownRequest) => Effect.Effect<boolean>;
  readonly await: Effect.Effect<AcnShutdownRequest>;
  readonly current: Effect.Effect<Option.Option<AcnShutdownRequest>>;
}

export class AcnShutdown extends Context.Tag("AcnShutdown")<
  AcnShutdown,
  AcnShutdownApi
>() {}

export const AcnShutdownLive: Layer.Layer<AcnShutdown> = Layer.effect(
  AcnShutdown,
  Effect.gen(function* () {
    const requested = yield* Deferred.make<AcnShutdownRequest>();

    return {
      request: (request) => Deferred.succeed(requested, request),
      await: Deferred.await(requested),
      current: Deferred.poll(requested).pipe(
        Effect.flatMap(
          Option.match({
            onNone: () => Effect.succeed(Option.none<AcnShutdownRequest>()),
            onSome: Effect.map(Option.some),
          }),
        ),
      ),
    };
  })
);
