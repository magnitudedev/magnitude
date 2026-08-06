import * as HttpClient from "@effect/platform/HttpClient";
import * as HttpClientRequest from "@effect/platform/HttpClientRequest";
import * as HttpClientResponse from "@effect/platform/HttpClientResponse";
import {
  AcnInstanceSchema,
  type AcnInstance,
} from "@magnitudedev/acn-protocol";
import { Effect, Option, Schema, Stream } from "effect";
import {
  AcnLaunchEventSchema,
  AcnLaunchRequestSchema,
  AcnProcessManager,
  type AcnLaunchEvent,
} from "./acn-process-manager";
import {
  DaemonDiscoveryFailed,
  DaemonError,
  DaemonSpawnFailed,
  type DaemonError as DaemonErrorType,
} from "./errors";

export const RemoteAcnCurrentResponseSchema = Schema.Struct({
  instance: Schema.NullOr(AcnInstanceSchema),
});

export const RemoteAcnLaunchMessageSchema = Schema.Union(
  AcnLaunchEventSchema,
  Schema.TaggedStruct("Failed", {
    error: DaemonError,
  })
);
export type RemoteAcnLaunchMessage = typeof RemoteAcnLaunchMessageSchema.Type;

export const RemoteAcnTerminateRequestSchema = Schema.Struct({
  instance: AcnInstanceSchema,
});

export const RemoteAcnErrorResponseSchema = Schema.Struct({
  error: DaemonError,
});

const extractDaemonError = (
  response: HttpClientResponse.HttpClientResponse
): Effect.Effect<DaemonErrorType> =>
  response.json.pipe(
    Effect.catchAll(() => Effect.succeed(null)),
    Effect.flatMap((json) =>
      Schema.decodeUnknown(RemoteAcnErrorResponseSchema)(json).pipe(
        Effect.map((body) => body.error),
        Effect.catchAll(() => Effect.succeed(new DaemonSpawnFailed({
          reason: `Invalid ACN manager error response (HTTP ${response.status})`,
        })))
      )
    )
  );

const instanceRoute = (
  proxyUrl: string,
  instance: AcnInstance
): AcnInstance => ({
  ...instance,
  url: `${proxyUrl}/acn/${encodeURIComponent(instance.id)}`,
});

export const makeRemoteAcnProcessManager = (
  proxyUrl: string
): Effect.Effect<AcnProcessManager, never, HttpClient.HttpClient> =>
  Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient;

    const observeCurrent: AcnProcessManager["observeCurrent"] = client
      .execute(HttpClientRequest.get(`${proxyUrl}/acn/current`))
      .pipe(
        Effect.timeout("2 seconds"),
        Effect.mapError(
          (cause) =>
            new DaemonDiscoveryFailed({
              reason: `ACN observation request failed: ${String(cause)}`,
            })
        ),
        Effect.flatMap((response) =>
          response.status >= 200 && response.status < 300
            ? response.json.pipe(
                Effect.flatMap(
                  Schema.decodeUnknown(RemoteAcnCurrentResponseSchema)
                ),
                Effect.mapError(
                  (cause) =>
                    new DaemonDiscoveryFailed({
                      reason: `Invalid ACN observation response: ${String(
                        cause
                      )}`,
                    })
                )
              )
            : extractDaemonError(response).pipe(Effect.flatMap(Effect.fail))
        ),
        Effect.map(({ instance }) =>
          Option.fromNullable(instance).pipe(
            Option.map((value) => instanceRoute(proxyUrl, value))
          )
        )
      );

    const launch: AcnProcessManager["launch"] = (request) =>
      Stream.unwrap(
        Effect.gen(function* () {
          const body = yield* Schema.encode(AcnLaunchRequestSchema)(
            request
          ).pipe(
            Effect.mapError(
              (cause) =>
                new DaemonSpawnFailed({
                  reason: `Failed to encode ACN launch request: ${String(
                    cause
                  )}`,
                })
            )
          );
          const httpRequest = yield* HttpClientRequest.post(
            `${proxyUrl}/acn/launch`
          ).pipe(
            HttpClientRequest.bodyJson(body),
            Effect.mapError(
              (cause) =>
                new DaemonSpawnFailed({
                  reason: `Failed to build ACN launch request: ${String(
                    cause
                  )}`,
                })
            )
          );
          const response = yield* client.execute(httpRequest).pipe(
            Effect.mapError(
              (cause) =>
                new DaemonSpawnFailed({
                  reason: `ACN launch request failed: ${String(cause)}`,
                })
            )
          );
          if (response.status < 200 || response.status >= 300) {
            return yield* extractDaemonError(response).pipe(Effect.flatMap(Effect.fail));
          }
          return response.stream.pipe(
            Stream.mapError(
              (cause) =>
                new DaemonSpawnFailed({
                  reason: `ACN launch stream failed: ${String(cause)}`,
                })
            ),
            Stream.decodeText(),
            Stream.splitLines,
            Stream.filter((line) => line.length > 0),
            Stream.mapEffect((line) =>
              Schema.decodeUnknown(
                Schema.parseJson(RemoteAcnLaunchMessageSchema)
              )(line).pipe(
                Effect.mapError(
                  (cause) =>
                    new DaemonSpawnFailed({
                      reason: `Invalid ACN launch response: ${String(cause)}`,
                    })
                )
              )
            ),
            Stream.mapEffect((message) =>
              message._tag === "Failed"
                ? Effect.fail(message.error)
                : Effect.succeed(message)
            ),
            Stream.map(
              (event): AcnLaunchEvent =>
                event._tag === "Ready"
                  ? {
                      ...event,
                      instance: instanceRoute(proxyUrl, event.instance),
                    }
                  : event
            )
          );
        })
      );

    const terminate: AcnProcessManager["terminate"] = (instance) =>
      Schema.encode(RemoteAcnTerminateRequestSchema)({ instance }).pipe(
        Effect.flatMap((body) =>
          HttpClientRequest.post(`${proxyUrl}/acn/terminate`).pipe(
            HttpClientRequest.bodyJson(body)
          )
        ),
        Effect.flatMap(client.execute),
        Effect.mapError(
          (cause) =>
            new DaemonSpawnFailed({
              reason: `ACN termination request failed: ${String(cause)}`,
            })
        ),
        Effect.flatMap((response) =>
          response.status >= 200 && response.status < 300
            ? Effect.void
            : extractDaemonError(response).pipe(Effect.flatMap(Effect.fail))
        )
      );

    return AcnProcessManager.of({ observeCurrent, launch, terminate });
  });
