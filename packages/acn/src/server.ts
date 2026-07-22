import { BunHttpServer, BunFileSystem, BunPath, BunCommandExecutor } from "@effect/platform-bun"
import { FetchHttpClient, HttpServerResponse } from "@effect/platform"
import * as HttpLayerRouter from "@effect/platform/HttpLayerRouter"
import * as HttpServerRequest from "@effect/platform/HttpServerRequest"
import { RpcSerialization, RpcServer } from "@effect/rpc"
import { Context, Effect, Exit, Layer, Runtime, Scope } from "effect"
import {
  StorageLive,
  GlobalStorage,
  MagnitudeStorage,
  makeGlobalStorage,
  ProjectStorageLiveFromCwd,
  VersionLive,
} from "@magnitudedev/storage"
import { MagnitudeRpcs } from "@magnitudedev/protocol"
import { IcnProcess } from "@magnitudedev/icn"
import { HandlersLive } from "./handlers"
import { DaemonLifecycleLive, defaultDataDir } from "./daemon-lifecycle"
import { AgentFactoryLive } from "./agent-factory"
import { AgentRuntimeLive } from "./agent-runtime"
import { ProviderModelCatalogLive } from "./provider-model-catalog"
import { ProviderCredentialsLive } from "./provider-credentials"
import { ModelSlotCoordinatorLive } from "./model-slot-coordinator"
import { MagnitudeCloudUsageLive } from "./magnitude-cloud-usage"
import { ProviderClientRegistryLive, SharedProviderClientLive } from "./shared-client"
import { ActiveSessionStatusesLive } from "./active-session-statuses"
import {
  AcnActivityTracker,
  AcnActivityTrackerLive,
  AcnRpcDemandLive,
} from "./activity-tracker"
import { DisplayViewStreamsLive } from "./display-view-streams"
import {
  AcnDisplayViewIntrospectorLive,
  AcnIntrospectorLive,
  AcnIntrospectionRoutes,
} from "./introspection"
import { SessionCommandsLive } from "./session-commands"
import { SessionDraftsLive } from "./session-drafts"
import { SessionLifecycleLive } from "./session-lifecycle"
import { SessionRuntimeOptionsStoreLive } from "./session-runtime-options"
import { makeModelConfigurationLayer } from "./model-configuration"
import { makeAcnIcn } from "./icn"
import { LocalModelInventoryLive } from "./local-model-inventory"
import { LocalInferenceHardwareLive } from "./local-inference-hardware"
import { OnboardingLive } from "./onboarding"
import { SessionStoreLive } from "./session-store"
import { ACN_VERSION } from "./version"
import { TracingLayer } from "./tracing"
import { ACN_OWNER_ID, ACN_SHUTDOWN_TOKEN, makeHealthResponse } from "./identity"
import { MirroredStateChangesLive } from "./mirrored-state"
import { AcnShutdown, AcnShutdownLive } from "./acn-shutdown"
import { acquireAcnMachineOwnership } from "./machine-ownership"
import { AcnSubscriptions, AcnSubscriptionsLive } from "./acn-subscriptions"
import { acnSubscriptionProtocolLayer } from "./acn-subscription-protocol"

export interface AcnServerOptions {
  readonly register?: boolean
  readonly debug?: boolean
  readonly dataDir?: string
}

const CORS_ALLOWED_HEADERS =
  "Content-Type, Content-Length, traceparent, tracestate, baggage, b3, x-b3-traceid, x-b3-spanid, x-b3-parentspanid, x-b3-sampled, x-b3-flags"
const LOCAL_HTTP_ORIGIN = /^https?:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/

function isAllowedCorsOrigin(origin: string): boolean {
  return LOCAL_HTTP_ORIGIN.test(origin) || origin === "file://" || origin === "null"
}

function corsHeadersFor(
  request: HttpServerRequest.HttpServerRequest,
): Record<string, string> | null {
  const origin = request.headers.origin
  if (!origin || !isAllowedCorsOrigin(origin)) return null

  return {
    "access-control-allow-origin": origin,
    "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": CORS_ALLOWED_HEADERS,
    "access-control-max-age": "86400",
    vary: "Origin",
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
  return Effect.succeed(
    HttpServerResponse.setHeaders(HttpServerResponse.empty({ status: 204 }), headers),
  )
})

// Health route
const HealthRoute = HttpLayerRouter.add(
  "GET",
  "/health",
  HttpServerResponse.json(makeHealthResponse(ACN_VERSION)),
)

const ShutdownRoute = HttpLayerRouter.add("POST", "/shutdown", (request) =>
  Effect.gen(function* () {
    if (request.headers.authorization !== `Bearer ${ACN_SHUTDOWN_TOKEN}`) {
      return HttpServerResponse.empty({ status: 401 })
    }
    yield* (yield* AcnShutdown).request({ reason: "upgrade" })
    return HttpServerResponse.empty({ status: 202 })
  }),
)

// RPC route
const RpcHttpProtocol = RpcServer.layerProtocolHttpRouter({
  path: "/rpc",
}).pipe(Layer.provide(RpcSerialization.layerNdjson))

const RpcRoute = RpcServer.layer(MagnitudeRpcs).pipe(
  Layer.provide(acnSubscriptionProtocolLayer(RpcHttpProtocol)),
)

// Combine routes
const AllDebugRoutes = Layer.mergeAll(
  CorsMiddleware,
  OptionsRoute,
  HealthRoute,
  ShutdownRoute,
  RpcRoute,
  AcnIntrospectionRoutes(true),
)
const AllBaseRoutes = Layer.mergeAll(
  CorsMiddleware,
  OptionsRoute,
  HealthRoute,
  ShutdownRoute,
  RpcRoute,
  AcnIntrospectionRoutes(false),
)

const AcnProcessHandlersLive = Layer.scopedDiscard(
  Effect.gen(function* () {
    const shutdown = yield* AcnShutdown
    const runtime = yield* Effect.runtime<never>()

    const uncaughtExceptionHandler = (error: Error) => {
      Runtime.runPromise(
        runtime,
        Effect.gen(function* () {
          yield* Effect.logError("Uncaught exception in ACN process").pipe(
            Effect.annotateLogs({ error: error.stack ?? String(error) }),
          )
          yield* shutdown.request({
            reason: "fatal",
            detail: error.stack ?? String(error),
          })
        }),
      ).catch(() => undefined)
    }

    const unhandledRejectionHandler = (reason: unknown) => {
      Runtime.runPromise(
        runtime,
        Effect.gen(function* () {
          const message = reason instanceof Error ? reason.stack ?? String(reason) : String(reason)
          yield* Effect.logError("Unhandled promise rejection in ACN process").pipe(
            Effect.annotateLogs({ reason: message }),
          )
          yield* shutdown.request({
            reason: "fatal",
            detail: message,
          })
        }),
      ).catch(() => undefined)
    }

    const requestSignalShutdown = (signal: NodeJS.Signals) => {
      Runtime.runPromise(
        runtime,
        shutdown.request({ reason: "signal", detail: signal }),
      ).catch(() => undefined)
    }
    const sigintHandler = () => requestSignalShutdown("SIGINT")
    const sigtermHandler = () => requestSignalShutdown("SIGTERM")

    process.on("uncaughtException", uncaughtExceptionHandler)
    process.on("unhandledRejection", unhandledRejectionHandler)
    process.on("SIGINT", sigintHandler)
    process.on("SIGTERM", sigtermHandler)

    yield* Effect.addFinalizer(() =>
      Effect.sync(() => {
        process.off("uncaughtException", uncaughtExceptionHandler)
        process.off("unhandledRejection", unhandledRejectionHandler)
        process.off("SIGINT", sigintHandler)
        process.off("SIGTERM", sigtermHandler)
      }),
    )
  }),
)

const makeAcnServicesBase = (debug: boolean, dataDir: string) => {
  const storageBase = Layer.mergeAll(
    VersionLive(ACN_VERSION),
    ProjectStorageLiveFromCwd(process.cwd()),
  )

  const storageLayer = StorageLive.pipe(Layer.provide(storageBase))

  const storageServices = Layer.mergeAll(SessionStoreLive, SessionRuntimeOptionsStoreLive).pipe(
    Layer.provideMerge(storageLayer),
  )

  const withActivity = Layer.provideMerge(AcnActivityTrackerLive("30 minutes", false), storageServices)
  const withSubscriptions = Layer.provideMerge(AcnSubscriptionsLive, withActivity)
  const withMirroredStateChanges = Layer.provideMerge(MirroredStateChangesLive, withSubscriptions)
  const localServices = addLocalInferenceServices(withMirroredStateChanges, dataDir)
  const withSharedClient = Layer.provideMerge(SharedProviderClientLive, localServices)
  const withCatalog = Layer.provideMerge(ProviderModelCatalogLive, withSharedClient)
  const withCredentials = Layer.provideMerge(ProviderCredentialsLive, withCatalog)
  const withCloudUsage = Layer.provideMerge(MagnitudeCloudUsageLive, withCredentials)
  const withModelSlots = Layer.provideMerge(ModelSlotCoordinatorLive, withCloudUsage)
  const withFactory = Layer.provideMerge(
    AgentFactoryLive({ debug, version: ACN_VERSION }),
    withModelSlots,
  )
  const withRuntime = Layer.provideMerge(AgentRuntimeLive, withFactory)
  const withDrafts = Layer.provideMerge(SessionDraftsLive, withRuntime)
  return withDrafts
}

const addLocalInferenceServices = <A, E, R>(
  base: Layer.Layer<A, E, R>,
  dataDir: string,
) => {
  const withIcn = Layer.provideMerge(makeAcnIcn(dataDir), base)
  const withConfiguration = Layer.provideMerge(makeModelConfigurationLayer(), withIcn)
  const withHardware = Layer.provideMerge(LocalInferenceHardwareLive, withConfiguration)
  const withInventory = Layer.provideMerge(LocalModelInventoryLive, withHardware)
  const withOnboarding = Layer.provideMerge(OnboardingLive, withInventory)
  const withProviderClients = Layer.provideMerge(ProviderClientRegistryLive, withOnboarding)
  return withProviderClients
}

const addCommonAcnServices = <A, E, R>(services: Layer.Layer<A, E, R>) => {
  const withDemand = Layer.provideMerge(AcnRpcDemandLive, services)
  const withCommands = Layer.provideMerge(SessionCommandsLive, withDemand)
  const withLifecycle = Layer.provideMerge(SessionLifecycleLive, withCommands)
  const withActiveSessionStatuses = Layer.provideMerge(ActiveSessionStatusesLive, withLifecycle)
  const withStreams = Layer.provideMerge(DisplayViewStreamsLive, withActiveSessionStatuses)
  return withStreams
}

const AcnBaseServicesLayer = (dataDir: string) =>
  addCommonAcnServices(makeAcnServicesBase(false, dataDir))

const AcnDebugServicesLayer = (dataDir: string) => {
  const withActivity = makeAcnServicesBase(true, dataDir)
  const withDisplayIntrospection = Layer.provideMerge(AcnDisplayViewIntrospectorLive, withActivity)
  return addCommonAcnServices(Layer.provideMerge(AcnIntrospectorLive, withDisplayIntrospection))
}

const makeAcnApplication = <A, E, R>(
  routes: Layer.Layer<A, E, R>,
  options: AcnServerOptions,
  debug: boolean,
) =>
  HttpLayerRouter.serve(routes).pipe(
    // HandlersLive consumes the ACN services directly.
    Layer.provide(HandlersLive),
    // DaemonLifecycle needs runtime/activity + HttpServer + FileSystem.
    Layer.provide(
      DaemonLifecycleLive({
        version: ACN_VERSION,
        register: options.register ?? false,
        debug,
        dataDir: options.dataDir ?? defaultDataDir(),
      }),
    ),
    Layer.provide(AcnProcessHandlersLive),
  )

const makeAcnServerLayer = (options: AcnServerOptions, debug: boolean) => {
  const dataDir = options.dataDir ?? defaultDataDir()
  const application = debug
    ? makeAcnApplication(AllDebugRoutes, options, true).pipe(
        Layer.provideMerge(AcnDebugServicesLayer(dataDir)),
      )
    : makeAcnApplication(AllBaseRoutes, options, false).pipe(
        Layer.provideMerge(AcnBaseServicesLayer(dataDir)),
      )

  return application.pipe(
    Layer.provideMerge(AcnShutdownLive),
    Layer.provide(
      Layer.succeed(
        GlobalStorage,
        GlobalStorage.of(makeGlobalStorage({ root: dataDir })),
      ),
    ),
    // CommandExecutor (used by ops.ts) requires FileSystem, so provide it before BunFileSystem
    Layer.provide(BunCommandExecutor.layer),
    // FileSystem (shared by Daemon + CommandExecutor)
    Layer.provide(BunFileSystem.layer),
    Layer.provide(BunPath.layer),
    Layer.provide(FetchHttpClient.layer),
    // HttpServer (shared by Daemon + HttpLayerRouter.serve)
    Layer.provide(BunHttpServer.layer({ port: 0, hostname: "127.0.0.1", idleTimeout: 255 })),
    // OTLP tracing — only active when MAGNITUDE_OTEL_ENDPOINT or
    // MAGNITUDE_OTEL=1 is set. Exports all RPC + HTTP spans to motel (or
    // any OTLP-compatible collector). Zero overhead when disabled.
    Layer.provide(TracingLayer),
  )
}

const AcnServerLayer = (options: AcnServerOptions = {}) =>
  makeAcnServerLayer(options, options.debug === true)

/**
 * Runs one ACN generation until its shutdown coordinator is requested. Scope
 * closure then stops HTTP, disposes sessions, and reaps the private ICN.
 */
export const launchAcnServer = (options: AcnServerOptions = {}) =>
  Effect.scoped(
    Effect.gen(function* () {
      yield* acquireAcnMachineOwnership({
        dataDir: options.dataDir ?? defaultDataDir(),
        id: ACN_OWNER_ID,
        version: ACN_VERSION,
      })
      const applicationScope = yield* Scope.make()
      yield* Effect.addFinalizer(() => Scope.close(applicationScope, Exit.void))
      const services = yield* Layer.buildWithScope(AcnServerLayer(options), applicationScope)
      const shutdown = Context.get(services, AcnShutdown)
      const subscriptions = Context.get(services, AcnSubscriptions)
      const icn = Context.get(services, IcnProcess)
      const activity = Context.get(services, AcnActivityTracker)
      yield* activity.ready
      const request = yield* shutdown.await
      yield* Effect.logInfo("ACN shutdown requested").pipe(
        Effect.annotateLogs({
          reason: request.reason,
          detail: request.detail ?? null,
        }),
      )
      // Linearize shutdown against RPC admission before any application
      // finalizer begins. Existing exact leases remain releasable while HTTP
      // and session scopes drain.
      yield* activity.gate.closeAdmission
      yield* subscriptions.terminate
      // Close consumers before the owned ICN. Layer finalizer ordering stops
      // observation fibers and HTTP admission before the ICN finalizer sends
      // its termination signal, so graceful drain cannot be prolonged by new
      // internal requests.
      yield* Scope.close(applicationScope, Exit.void)
      yield* icn.shutdownResult.pipe(Effect.orDie)
    }),
  )
