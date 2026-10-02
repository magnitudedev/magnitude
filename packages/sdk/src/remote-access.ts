import { HttpBody, HttpClient, HttpClientRequest } from "@effect/platform"
import { Effect, Schema } from "effect"
import {
  REMOTE_ACCESS_SESSION_PATH,
  REMOTE_ACCESS_SIGN_OUT_PATH,
  RemoteAccessStatus,
  SignInFailure,
} from "@magnitudedev/acn-protocol"
import { InvalidServiceResponse, ServiceUnavailable } from "./connection-errors"

const unavailable = (origin: string) => () => new ServiceUnavailable({ origin, message: "Magnitude service is unavailable" })
const invalid = (origin: string) => () => new InvalidServiceResponse({ origin, message: "Invalid Magnitude sign-in response" })

/**
 * Browser sign-in to a Magnitude service on another device. The session is an HttpOnly cookie, so
 * these requests only work from a page the service itself served.
 */
export const readRemoteAccess = (origin: string) => Effect.gen(function* () {
  const http = yield* HttpClient.HttpClient
  const response = yield* http.get(`${origin}${REMOTE_ACCESS_SESSION_PATH}`).pipe(Effect.mapError(unavailable(origin)))
  if (response.status !== 200) return yield* invalid(origin)()
  return yield* response.json.pipe(Effect.flatMap(Schema.decodeUnknown(RemoteAccessStatus)), Effect.mapError(invalid(origin)))
})

export const signInRemotely = (origin: string, key: string) => Effect.gen(function* () {
  const http = yield* HttpClient.HttpClient
  const response = yield* http.execute(HttpClientRequest.post(`${origin}${REMOTE_ACCESS_SESSION_PATH}`, {
    body: HttpBody.unsafeJson({ key }),
  })).pipe(Effect.mapError(unavailable(origin)))
  if (response.status === 204) return
  const failure = yield* response.json.pipe(Effect.flatMap(Schema.decodeUnknown(SignInFailure)), Effect.mapError(invalid(origin)))
  return yield* failure
})

export const signOutRemotely = (origin: string) => Effect.gen(function* () {
  const http = yield* HttpClient.HttpClient
  const response = yield* http.execute(HttpClientRequest.post(`${origin}${REMOTE_ACCESS_SIGN_OUT_PATH}`)).pipe(Effect.mapError(unavailable(origin)))
  if (response.status !== 204) return yield* invalid(origin)()
})
