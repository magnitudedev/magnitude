import { BunHttpServer, BunFileSystem, BunPath, BunCommandExecutor } from "@effect/platform-bun"
import { HttpServerResponse } from "@effect/platform"
import * as HttpLayerRouter from "@effect/platform/HttpLayerRouter"
import * as HttpServerRequest from "@effect/platform/HttpServerRequest"
import { RpcSerialization, RpcServer } from "@effect/rpc"
import { Effect, Layer, Runtime } from "effect"
import {
  StorageLive,
  GlobalStorageLive,
  ProjectStorageLiveFromCwd,
  VersionLive,
} from "@magnitudedev/storage"
import { MagnitudeRpcs } from "@magnitudedev/protocol"
import { HandlersLive } from "./handlers"
import { DaemonLifecycleLive, defaultDataDir } from "./daemon-lifecycle"
import { AgentFactoryLive } from "./agent-factory"
import { AgentRuntimeLive } from "./agent-runtime"
import { AccountLive } from "./account"
import { SharedProviderClientLive, SharedProviderClientRefLive } from "./shared-client"
import { ActiveSessionStatusesLive } from "./active-session-statuses"
import { AcnActivityTrackerLive, AcnRpcCommandActivityLive } from "./activity-tracker"
import { DisplayViewStreamsLive } from "./display-view-streams"
import { AcnDisplayViewIntrospectorLive, AcnIntrospectorLive, AcnIntrospectionRoutes } from "./introspection"
import { SessionCommandsLive } from "./session-commands"
import { SessionDestroyerLive } from "./session-destroyer"
import { SessionDraftsLive } from "./session-drafts"
import { SessionLifecycleLive } from "./session-lifecycle"
import { SessionRuntimeOptionsStoreLive } from "./session-runtime-options"
import { SessionStoreLive } from "./session-store"
import { ACN_VERSION } from "./version"
import { TracingLayer } from "./tracing"
import { makeHealthResponse } from "./identity"

export interface AcnServerOptions {
  readonly register?: boolean
  readonly debug?: boolean
}

const CORS_ALLOWED_HEADERS = "Content-Type, Content-Length, traceparent, tracestate, baggage, b3, x-b3-traceid, x-b3-spanid, x-b3-parentspanid, x-b3-sampled, x-b3-flags"
const LOCAL_HTTP_ORIGIN = /^https?:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/

function isAllowedCorsOrigin(origin: string): boolean {
  return LOCAL_HTTP_ORIGIN.test(origin) || origin === "file://" || origin === "null"
}

function corsHeadersFor(request: HttpServerRequest.HttpServerRequest): Record<string, string> | null {
  const origin = request.headers.origin
  if (!origin || !isAllowedCorsOrigin(origin)) return null

  return {
    "access-control-allow-origin": origin,
    "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": CORS_ALLOWED_HEADERS,
    "access-control-max-age": "86400",
    "vary": "Origin",
  }
}

function withCors(
  response: HttpServerResponse.HttpServerResponse,
  request: HttpServerRequest.HttpServerRequest,
) {
  const headers = corsHeadersFor(request)
  return headers ? HttpServerResponse.setHeaders(response, headers) : response
}

const disallowedCorsResponse = HttpServerResponse.empty({ status: 403 })

// CORS middleware — browser clients are limited to local web origins and
// Electron's packaged renderer origin. Non-browser clients normally omit
// Origin and do not need CORS headers.
const CorsMiddleware = HttpLayerRouter.use((router) =>
  router.addGlobalMiddleware((effect) =>
    Effect.gen(function* () {
      const request = yield* HttpServerRequest.HttpServerRequest
      const response = yield* effect
      return withCors(response, request)
    }),
  ),
)

// OPTIONS preflight handler — catches all OPTIONS requests.
const OptionsRoute = HttpLayerRouter.add("OPTIONS", "*", (request) => {
  const headers = corsHeadersFor(request)
  if (!headers) return Effect.succeed(disallowedCorsResponse)
  return Effect.succeed(HttpServerResponse.setHeaders(HttpServerResponse.empty({ status: 204 }), headers))
})

// Health route
const HealthRoute = HttpLayerRouter.add("GET", "/health", HttpServerResponse.json(makeHealthResponse(ACN_VERSION)))

// RPC route
const RpcRoute = RpcServer.layerHttpRouter({
  group: MagnitudeRpcs,
  path: "/rpc",
  protocol: "http",
}).pipe(
  Layer.provide(RpcSerialization.layerNdjson),
)

// Combine routes
const AllDebugRoutes = Layer.mergeAll(CorsMiddleware, OptionsRoute, HealthRoute, RpcRoute, AcnIntrospectionRoutes(true))
const AllBaseRoutes = Layer.mergeAll(CorsMiddleware, OptionsRoute, HealthRoute, RpcRoute, AcnIntrospectionRoutes(false))

function AllRoutes(debug: true): typeof AllDebugRoutes
function AllRoutes(debug: false): typeof AllBaseRoutes
function AllRoutes(debug: boolean): typeof AllDebugRoutes | typeof AllBaseRoutes
function AllRoutes(debug: boolean) {
  return debug ? AllDebugRoutes : AllBaseRoutes
}

const AcnProcessHandlersLive = Layer.scopedDiscard(
  Effect.gen(function* () {
    const runtime = yield* Effect.runtime<never>()

    const uncaughtExceptionHandler = (error: Error) => {
      Runtime.runPromise(
        runtime,
        Effect.gen(function* () {
          yield* Effect.logError("Uncaught exception in ACN process").pipe(
            Effect.annotateLogs({ error: error.stack ?? String(error) })
          )
          return yield* Effect.sync(() => process.exit(1))
        })
      ).catch(() => process.exit(1))
    }

    const unhandledRejectionHandler = (reason: unknown) => {
      Runtime.runPromise(
        runtime,
        Effect.gen(function* () {
          const message = reason instanceof Error ? reason.stack ?? String(reason) : String(reason)
          yield* Effect.logError("Unhandled promise rejection in ACN process").pipe(
            Effect.annotateLogs({ reason: message })
          )
          return yield* Effect.sync(() => process.exit(1))
        })
      ).catch(() => process.exit(1))
    }

    process.on("uncaughtException", uncaughtExceptionHandler)
    process.on("unhandledRejection", unhandledRejectionHandler)

    yield* Effect.addFinalizer(() =>
      Effect.sync(() => {
        process.off("uncaughtException", uncaughtExceptionHandler)
        process.off("unhandledRejection", unhandledRejectionHandler)
      })
    )
  })
)

const makeAcnServicesBase = (debug: boolean) => {
  const storageBase = Layer.mergeAll(
    GlobalStorageLive,
    VersionLive(ACN_VERSION),
    ProjectStorageLiveFromCwd(process.cwd()),
    BunFileSystem.layer,
    BunPath.layer,
  )

  const storageLayer = StorageLive.pipe(
    Layer.provide(storageBase)
  )

  const services = Layer.mergeAll(
    SessionStoreLive,
    SessionRuntimeOptionsStoreLive,
    AgentFactoryLive({ debug, version: ACN_VERSION }),
  ).pipe(
    Layer.provideMerge(storageLayer)
  )

  const withRuntime = Layer.provideMerge(AgentRuntimeLive, services)
  // SharedProviderClientLive needs both MagnitudeStorage (from services) and
  // GlobalStorage (from storageBase). Merge GlobalStorageLive alongside so
  // the service is in context.
  const withGlobalStorage = Layer.mergeAll(withRuntime, GlobalStorageLive)
  const withSharedClientRef = Layer.provideMerge(SharedProviderClientRefLive, withGlobalStorage)
  const withSharedClient = Layer.provideMerge(SharedProviderClientLive, withSharedClientRef)
  const withDrafts = Layer.provideMerge(SessionDraftsLive, withSharedClient)
  const withDestroyer = Layer.provideMerge(SessionDestroyerLive, withDrafts)
  return Layer.provideMerge(AcnActivityTrackerLive, withDestroyer)
}

const AcnBaseServicesLayer = (debug: boolean) => {
  const withActivity = makeAcnServicesBase(debug)
  const withCommandTracking = Layer.provideMerge(AcnRpcCommandActivityLive, withActivity)
  const withCommands = Layer.provideMerge(SessionCommandsLive, withCommandTracking)
  const withLifecycle = Layer.provideMerge(SessionLifecycleLive, withCommands)
  const withAccount = Layer.provideMerge(AccountLive, withLifecycle)
  const withActiveSessionStatuses = Layer.provideMerge(ActiveSessionStatusesLive, withAccount)
  const withStreams = Layer.provideMerge(DisplayViewStreamsLive, withActiveSessionStatuses)
  return withStreams
}

const AcnDebugServicesLayer = (debug: boolean) => {
  const withActivity = makeAcnServicesBase(debug)
  const withDisplayIntrospection = Layer.provideMerge(AcnDisplayViewIntrospectorLive, withActivity)
  const withIntrospection = Layer.provideMerge(AcnIntrospectorLive, withDisplayIntrospection)
  const withCommandTracking = Layer.provideMerge(AcnRpcCommandActivityLive, withIntrospection)
  const withCommands = Layer.provideMerge(SessionCommandsLive, withCommandTracking)
  const withLifecycle = Layer.provideMerge(SessionLifecycleLive, withCommands)
  const withAccount = Layer.provideMerge(AccountLive, withLifecycle)
  const withActiveSessionStatuses = Layer.provideMerge(ActiveSessionStatusesLive, withAccount)
  const withStreams = Layer.provideMerge(DisplayViewStreamsLive, withActiveSessionStatuses)
  return withStreams
}

const AcnServerBaseLayer = (options: AcnServerOptions) =>
  HttpLayerRouter.serve(AllRoutes(false)).pipe(
    // HandlersLive consumes the ACN services directly.
    Layer.provide(HandlersLive),
    // DaemonLifecycle needs runtime/activity + HttpServer + FileSystem.
    Layer.provide(
      DaemonLifecycleLive({
        version: ACN_VERSION,
        register: options.register ?? false,
        debug: false,
        idleTimeoutMinutes: 30,
        checkIntervalSeconds: 60,
        dataDir: defaultDataDir(),
      })
    ),
    Layer.provide(AcnProcessHandlersLive),
    Layer.provide(AcnBaseServicesLayer(false)),
    // CommandExecutor (used by ops.ts) requires FileSystem, so provide it before BunFileSystem
    Layer.provide(BunCommandExecutor.layer),
    // FileSystem (shared by Daemon + CommandExecutor)
    Layer.provide(BunFileSystem.layer),
    Layer.provide(BunPath.layer),
    // HttpServer (shared by Daemon + HttpLayerRouter.serve)
    Layer.provide(BunHttpServer.layer({ port: 0, hostname: "127.0.0.1", idleTimeout: 255 })),
    // OTLP tracing — only active when MAGNITUDE_OTEL_ENDPOINT or
    // MAGNITUDE_OTEL=1 is set. Exports all RPC + HTTP spans to motel (or
    // any OTLP-compatible collector). Zero overhead when disabled.
    Layer.provide(TracingLayer)
  )

const AcnServerDebugLayer = (options: AcnServerOptions) =>
  HttpLayerRouter.serve(AllRoutes(true)).pipe(
    // HandlersLive consumes the ACN services directly.
    Layer.provide(HandlersLive),
    // DaemonLifecycle needs runtime/activity + HttpServer + FileSystem.
    Layer.provide(
      DaemonLifecycleLive({
        version: ACN_VERSION,
        register: options.register ?? false,
        debug: true,
        idleTimeoutMinutes: 30,
        checkIntervalSeconds: 60,
        dataDir: defaultDataDir(),
      })
    ),
    Layer.provide(AcnProcessHandlersLive),
    Layer.provide(AcnDebugServicesLayer(true)),
    // CommandExecutor (used by ops.ts) requires FileSystem, so provide it before BunFileSystem
    Layer.provide(BunCommandExecutor.layer),
    // FileSystem (shared by Daemon + CommandExecutor)
    Layer.provide(BunFileSystem.layer),
    Layer.provide(BunPath.layer),
    // HttpServer (shared by Daemon + HttpLayerRouter.serve)
    Layer.provide(BunHttpServer.layer({ port: 0, hostname: "127.0.0.1", idleTimeout: 255 })),
    // OTLP tracing — only active when MAGNITUDE_OTEL_ENDPOINT or
    // MAGNITUDE_OTEL=1 is set. Exports all RPC + HTTP spans to motel (or
    // any OTLP-compatible collector). Zero overhead when disabled.
    Layer.provide(TracingLayer)
  )

export const AcnServerLayer = (options: AcnServerOptions = {}): Layer.Layer<never, never, never> =>
  options.debug ? AcnServerDebugLayer(options) : AcnServerBaseLayer(options)
