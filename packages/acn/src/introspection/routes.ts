import { HttpServerResponse } from "@effect/platform"
import * as HttpLayerRouter from "@effect/platform/HttpLayerRouter"
import * as HttpRouter from "@effect/platform/HttpRouter"
import type { HttpServerRequest } from "@effect/platform/HttpServerRequest"
import { Effect, Layer, Stream } from "effect"
import { SessionError } from "@magnitudedev/protocol"
import { sessionErrorMessage } from "../session-errors"
import { AcnIntrospector } from "./service"

const encoder = new TextEncoder()

const json = (body: unknown, status = 200) =>
  HttpServerResponse.json(body, { status }).pipe(Effect.orDie)

const sessionErrorStatus = (error: SessionError): number => {
  switch (error._tag) {
    case "SessionNotFound":
      return 404
    case "InvalidSessionPath":
    case "SessionAlreadyExists":
    case "SessionStartFailed":
    case "SessionOperationFailed":
      return 400
  }
  return 500
}

const sessionErrorJson = (error: SessionError) =>
  json({
    error: error._tag,
    message: sessionErrorMessage(error),
  }, sessionErrorStatus(error))

const forkIdFromRequest = (request: HttpServerRequest): string | null => {
  const url = new URL(request.url, "http://localhost")
  const raw = url.searchParams.get("forkId")
  return raw && raw.length > 0 ? raw : null
}

const sessionIdParam = Effect.gen(function* () {
  const params = yield* HttpRouter.params
  return params.sessionId ? decodeURIComponent(params.sessionId) : ""
})

const currentSessionResponse = (request: HttpServerRequest) =>
  Effect.gen(function* () {
    const introspector = yield* AcnIntrospector
    const sessionId = yield* sessionIdParam
    const introspection = yield* introspector.currentSession(sessionId, forkIdFromRequest(request))
    return yield* json(introspection)
  }).pipe(
    Effect.catchAll(sessionErrorJson),
  )

const sessionChangesResponse = (request: HttpServerRequest) =>
  Effect.gen(function* () {
    const introspector = yield* AcnIntrospector
    const sessionId = yield* sessionIdParam
    const forkId = forkIdFromRequest(request)

    return HttpServerResponse.stream(
      introspector.sessionChanges(sessionId, forkId).pipe(
        Stream.map((payload) => encoder.encode(`data: ${JSON.stringify(payload)}\n\n`)),
        Stream.catchAll((error) =>
          Stream.succeed(
            encoder.encode(`event: error\ndata: ${JSON.stringify({
              error: error._tag,
              message: sessionErrorMessage(error),
            })}\n\n`),
          )
        ),
      ),
      {
        contentType: "text/event-stream",
        headers: {
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
        },
      },
    )
  })

const AcnIntrospectionRoutesLive = Layer.mergeAll(
  HttpLayerRouter.add("GET", "/dev/introspection", Effect.gen(function* () {
    const introspector = yield* AcnIntrospector
    return yield* json(yield* introspector.currentOverview)
  })),
  HttpLayerRouter.add("GET", "/dev/sessions", Effect.gen(function* () {
    const introspector = yield* AcnIntrospector
    const overview = yield* introspector.currentOverview
    return yield* json({ sessions: overview.sessions, activity: overview.activity, timestamp: overview.timestamp })
  })),
  HttpLayerRouter.add("GET", "/dev/sessions/:sessionId", currentSessionResponse),
  HttpLayerRouter.add("GET", "/dev/sessions/:sessionId/stream", sessionChangesResponse),
)

const AcnIntrospectionRoutesDisabled: Layer.Layer<never, never, never> = Layer.empty

export function AcnIntrospectionRoutes(enabled: true): typeof AcnIntrospectionRoutesLive
export function AcnIntrospectionRoutes(enabled: false): typeof AcnIntrospectionRoutesDisabled
export function AcnIntrospectionRoutes(
  enabled: boolean,
): typeof AcnIntrospectionRoutesLive | typeof AcnIntrospectionRoutesDisabled
export function AcnIntrospectionRoutes(enabled: boolean) {
  return enabled ? AcnIntrospectionRoutesLive : AcnIntrospectionRoutesDisabled
}
