import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import { Effect, Schema, Stream } from "effect"
import { AcnReady, type AcnInstance } from "@magnitudedev/acn-protocol"
import {
  AcnEnsureRequestSchema,
  AcnInstanceManager,
  RemoteAcnEnsureMessageSchema,
  type AcnEnsureEvent,
} from "./acn-instance-manager"
import { AcnAdministrationFailed, AcnEnsuranceError, AcnEnsuranceFailed } from "./errors"

export const RemoteAcnErrorResponseSchema = Schema.Struct({ error: AcnEnsuranceError })

const failure = (reason: string) => new AcnEnsuranceFailed({ reason })
type ReadyInstance = AcnInstance<AcnReady>

const extractError = (response: { readonly status: number; readonly json: Effect.Effect<unknown, unknown> }) =>
  response.json.pipe(
    Effect.flatMap(Schema.decodeUnknown(RemoteAcnErrorResponseSchema)),
    Effect.map((body) => body.error),
    Effect.catchAll(() => Effect.succeed(failure(`Invalid ACN ensure error response (HTTP ${response.status})`))),
  )

const proxyRoute = (proxyUrl: string, instance: ReadyInstance): ReadyInstance => ({
  ...instance,
  url: `${proxyUrl}/acn/${encodeURIComponent(instance.id)}`,
})

export const makeRemoteAcnInstanceManager = (
  proxyUrl: string,
): Effect.Effect<AcnInstanceManager, never, HttpClient.HttpClient> => Effect.gen(function* () {
  const client = yield* HttpClient.HttpClient
  const ensure: AcnInstanceManager["ensure"] = (request) => Stream.unwrap(Effect.gen(function* () {
    const body = yield* Schema.encode(AcnEnsureRequestSchema)(request).pipe(
      Effect.mapError((error) => failure(`Failed to encode ACN ensure request: ${String(error)}`)),
    )
    const httpRequest = yield* HttpClientRequest.post(`${proxyUrl}/acn/ensure`).pipe(
      HttpClientRequest.bodyJson(body),
      Effect.mapError((error) => failure(`Failed to build ACN ensure request: ${String(error)}`)),
    )
    const response = yield* client.execute(httpRequest).pipe(
      Effect.mapError((error) => failure(`ACN ensure request failed: ${String(error)}`)),
    )
    if (response.status < 200 || response.status >= 300) {
      return yield* extractError(response).pipe(Effect.flatMap(Effect.fail))
    }
    return response.stream.pipe(
      Stream.mapError((error) => failure(`ACN ensure stream failed: ${String(error)}`)),
      Stream.decodeText(),
      Stream.splitLines,
      Stream.filter((line) => line.length > 0),
      Stream.mapEffect((line) => Schema.decodeUnknown(
        Schema.parseJson(RemoteAcnEnsureMessageSchema),
      )(line).pipe(
        Effect.mapError((error) => failure(`Invalid ACN ensure response: ${String(error)}`)),
      )),
      Stream.mapEffect((message) => message._tag === "Failed"
        ? Effect.fail(message.error)
        : Effect.succeed(message)),
      Stream.map((event): AcnEnsureEvent => event._tag === "Ready"
        ? { ...event, instance: proxyRoute(proxyUrl, event.instance) }
        : event),
    )
  }))
  const stop = Effect.fail(new AcnAdministrationFailed({
    reason: "Remote ACN administration is not exposed by this host",
  }))
  return AcnInstanceManager.of({ ensure, stop })
})
