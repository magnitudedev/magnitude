import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { Effect, Option, Schema, Stream } from "effect"
import type { DaemonDiscovery } from "./daemon-discovery"
import { DaemonStatusSchema } from "./daemon-discovery"
import type { DaemonLauncher } from "./daemon-launcher"
import { DaemonLaunchEventSchema } from "./daemon-launcher"
import { DaemonDiscoveryFailed, DaemonSpawnFailed } from "./errors"

export const RemoteDaemonCurrentResponseSchema = Schema.Struct({
  daemon: Schema.Union(DaemonStatusSchema, Schema.Null),
})
export type RemoteDaemonCurrentResponse =
  typeof RemoteDaemonCurrentResponseSchema.Type

export const RemoteDaemonLaunchRequestSchema = Schema.Struct({
  command: Schema.optionalWith(Schema.Array(Schema.String), {
    as: "Option",
    exact: true,
  }),
})
export type RemoteDaemonLaunchRequest =
  typeof RemoteDaemonLaunchRequestSchema.Type

export const RemoteDaemonLaunchMessageSchema = Schema.Union(
  DaemonLaunchEventSchema,
  Schema.TaggedStruct("Failed", {
    message: Schema.String.pipe(Schema.minLength(1)),
  }),
)
export type RemoteDaemonLaunchMessage =
  typeof RemoteDaemonLaunchMessageSchema.Type

export const RemoteDaemonErrorResponseSchema = Schema.Struct({
  error: Schema.String,
})

const extractErrorMessage = (
  response: HttpClientResponse.HttpClientResponse,
): Effect.Effect<string> =>
  response.json.pipe(
    Effect.catchAll(() => Effect.succeed(null)),
    Effect.flatMap((json) =>
      Schema.decodeUnknown(RemoteDaemonErrorResponseSchema)(json).pipe(
        Effect.map((body) => body.error),
        Effect.catchAll(() => Effect.succeed(`HTTP ${response.status}`)),
      ),
    ),
  )

export const makeRemoteDaemonDiscovery = (
  proxyUrl: string,
): Effect.Effect<DaemonDiscovery, never, HttpClient.HttpClient> =>
  Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    return {
      current: () =>
        client.execute(HttpClientRequest.get(`${proxyUrl}/current`)).pipe(
          Effect.timeout("2 seconds"),
          Effect.mapError((cause) =>
            new DaemonDiscoveryFailed({
              reason: `Daemon discovery request failed: ${String(cause)}`,
            }),
          ),
          Effect.flatMap((response) =>
            response.status >= 200 && response.status < 300
              ? response.json.pipe(
                  Effect.mapError((cause) =>
                    new DaemonDiscoveryFailed({
                      reason: `Failed to read daemon discovery response: ${String(cause)}`,
                    }),
                  ),
                  Effect.flatMap((body) =>
                    Schema.decodeUnknown(RemoteDaemonCurrentResponseSchema)(body).pipe(
                      Effect.mapError((cause) =>
                        new DaemonDiscoveryFailed({
                          reason: `Invalid daemon discovery response: ${String(cause)}`,
                        }),
                      ),
                    ),
                  ),
                )
              : extractErrorMessage(response).pipe(
                  Effect.flatMap((reason) =>
                    Effect.fail(new DaemonDiscoveryFailed({ reason })),
                  ),
                ),
          ),
          Effect.map(({ daemon }) =>
            Option.fromNullable(daemon).pipe(
              Option.map((status) => ({ ...status, url: proxyUrl })),
            ),
          ),
        ),
    }
  })

export const makeRemoteDaemonLauncher = (
  proxyUrl: string,
): Effect.Effect<DaemonLauncher, never, HttpClient.HttpClient> =>
  Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    return {
      launch: (command) =>
        Stream.unwrap(Effect.gen(function* () {
          const body = yield* Schema.encode(RemoteDaemonLaunchRequestSchema)({
            command,
          }).pipe(
            Effect.mapError((cause) => new DaemonSpawnFailed({
              reason: `Failed to encode daemon launch request: ${String(cause)}`,
            })),
          )
          const request = yield* HttpClientRequest.post(`${proxyUrl}/launch`).pipe(
            HttpClientRequest.bodyJson(body),
            Effect.mapError((cause) => new DaemonSpawnFailed({
              reason: `Failed to build daemon launch request: ${String(cause)}`,
            })),
          )
          const response = yield* client.execute(request).pipe(
            Effect.mapError((cause) => new DaemonSpawnFailed({
              reason: `Daemon launch request failed: ${String(cause)}`,
            })),
          )
          if (response.status < 200 || response.status >= 300) {
            const reason = yield* extractErrorMessage(response)
            return yield* new DaemonSpawnFailed({ reason })
          }
          return response.stream.pipe(
            Stream.mapError((cause) => new DaemonSpawnFailed({
              reason: `Daemon launch stream failed: ${String(cause)}`,
            })),
            Stream.decodeText(),
            Stream.splitLines,
            Stream.filter((line) => line.length > 0),
            Stream.mapEffect((line) =>
              Schema.decodeUnknown(
                Schema.parseJson(RemoteDaemonLaunchMessageSchema),
              )(line).pipe(
                Effect.mapError(
                  (cause) =>
                    new DaemonSpawnFailed({
                      reason: `Invalid daemon launch response: ${String(cause)}`,
                    }),
                ),
              ),
            ),
            Stream.mapEffect((message) =>
              message._tag === "Failed"
                ? Effect.fail(new DaemonSpawnFailed({ reason: message.message }))
                : Effect.succeed(message),
            ),
            Stream.map((event) =>
              event._tag === "Ready"
                ? { ...event, endpoint: { ...event.endpoint, url: proxyUrl } }
                : event,
            ),
          )
        })),
    }
  })
