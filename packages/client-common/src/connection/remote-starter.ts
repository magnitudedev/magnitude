import { HttpClient } from "@effect/platform";
import {
  MagnitudeServiceStarter,
  ServiceStartErrorSchema,
  ServiceStartFailed,
  ServiceStartProgressSchema,
} from "@magnitudedev/sdk";
import { Effect, Schema, Stream } from "effect";

/** Startup-only host bridge. Application operations use the daemon's normal RPC endpoint. */
export const RemoteServiceStartMessageSchema = Schema.Union(
  ServiceStartProgressSchema,
  Schema.TaggedStruct("Failed", { error: ServiceStartErrorSchema })
);
export type RemoteServiceStartMessage =
  typeof RemoteServiceStartMessageSchema.Type;

export const makeRemoteServiceStarter = (
  origin: string
): Effect.Effect<MagnitudeServiceStarter, never, HttpClient.HttpClient> =>
  Effect.gen(function* () {
    const http = yield* HttpClient.HttpClient;
    return {
      start: Stream.unwrap(
        http.post(`${origin}/service/start`).pipe(
          Effect.flatMap((response) =>
            response.status === 200
              ? Effect.succeed(response.stream)
              : Effect.fail(
                  new ServiceStartFailed({
                    message: `Service starter returned HTTP ${response.status}`,
                  })
                )
          )
        )
      ).pipe(
        Stream.decodeText(),
        Stream.splitLines,
        Stream.filter((line) => line.length > 0),
        Stream.mapEffect(
          Schema.decodeUnknown(
            Schema.parseJson(RemoteServiceStartMessageSchema)
          )
        ),
        Stream.mapError(
          (error) => new ServiceStartFailed({ message: String(error) })
        ),
        Stream.mapEffect((message) =>
          message._tag === "Failed"
            ? Effect.fail(message.error)
            : Effect.succeed(message)
        )
      ),
    };
  });
