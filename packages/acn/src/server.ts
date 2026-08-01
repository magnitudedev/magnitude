import {
  BunHttpServer,
  BunFileSystem,
  BunPath,
  BunCommandExecutor,
} from "@effect/platform-bun"
import { FetchHttpClient, HttpServerResponse } from "@effect/platform"
import * as HttpLayerRouter from "@effect/platform/HttpLayerRouter"
import * as HttpServer from "@effect/platform/HttpServer"
import * as HttpServerRequest from "@effect/platform/HttpServerRequest"
import { RpcSerialization, RpcServer } from "@effect/rpc"
import {
  Cause,
  Context,
  Effect,
  Exit,
  Layer,
  Option,
  Runtime,
  Schema,
  Scope,
} from "effect"
import {
  StorageLive,
  GlobalStorage,
  MagnitudeStorage,
  makeGlobalStorage,
  ProjectStorageLiveFromCwd,
  VersionLive,
} from "@magnitudedev/storage"
import {
  AcnHealthResponseSchema,
  MagnitudeRpcs,
} from "@magnitudedev/acn-protocol"
import { IcnProcess, makeIcnProvider } from "@magnitudedev/icn"
import { HandlersLive } from "./handlers"
import { DaemonLifecycleLive, defaultDataDir } from "./daemon-lifecycle"
import { AgentFactoryLive } from "./agent-factory"
import { AgentRuntimeLive } from "./agent-runtime"
import { ProviderModelCatalogLive } from "./provider-model-catalog"
import { ProviderCredentialsLive } from "./provider-credentials"
import { ModelSlotControllerLive } from "./model-slot-controller"
import { MagnitudeCloudUsageLive } from "./magnitude-cloud-usage"
import {
  ProviderClientRegistryLive,
  SharedProviderClientLive,
} from "./shared-client"
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
  AcnIntrospector,
  installAcnIntrospectionRoutes,
  type AcnIntrospectorApi,
} from "./introspection"
import { SessionCommandsLive } from "./session-commands"
import { SessionDraftsLive } from "./session-drafts"
import { SessionLifecycleLive } from "./session-lifecycle"
import { SessionRuntimeOptionsStoreLive } from "./session-runtime-options"
import { makeModelConfigurationLayer } from "./model-configuration"
import { makeAcnIcn } from "./icn"
import { LocalModelAutoSetupLive } from "./local-model-auto-setup"
import { LocalModelEvaluationsLive } from "./local-model-evaluations"
import { LocalModelPackagesLive } from "./local-model-packages"
import { makeLocalModelRecommendationsLive } from "./local-model-recommendations"
import { LocalModelsLive } from "./local-models"
import { LocalProviderOfferingsLive } from "./local-provider-offerings"
import { LocalProviderOfferingProjectionLive } from "./local-provider-offering-projection"
import { LocalProviderResolverLive } from "./local-provider-resolver"
import { LocalInferenceHardwareLive } from "./local-inference-hardware"
import { OnboardingLive } from "./onboarding"
import { SessionStoreLive } from "./session-store"
import { ACN_VERSION } from "./version"
import { TracingLayer } from "./tracing"
import {
  ACN_OWNER_ID,
  makeHealthResponse,
} from "./identity"
import { MirroredStateChangesLive } from "./mirrored-state"
import { AcnShutdown, AcnShutdownLive } from "./acn-shutdown"
import { acquireAcnMachineOwnership } from "./machine-ownership"
import {
  readRegistrationOwnership,
  registrationIsOwnedBy,
  registrationPath,
} from "./daemon-registration"
import { AcnSubscriptions, AcnSubscriptionsLive } from "./acn-subscriptions"
import { makeAcnSubscriptionProtocol } from "./acn-subscription-protocol"
import { AcnStartupState } from "./startup-state"
import { makeAcnStartupState } from "./startup-state"

export interface AcnServerOptions {
  readonly register?: boolean
  readonly debug?: boolean
  readonly dataDir?: string
}

const CORS_ALLOWED_HEADERS =
  "Content-Type, Content-Length, traceparent, tracestate, baggage, b3, x-b3-traceid, x-b3-spanid, x-b3-parentspanid, x-b3-sampled, x-b3-flags"
const LOCAL_HTTP_ORIGIN =
  /^https?:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/

function isAllowedCorsOrigin(origin: string): boolean {
  return (
    LOCAL_HTTP_ORIGIN.test(origin) || origin === "file://" || origin === "null"
  )
}

function corsHeadersFor(
  request: HttpServerRequest.HttpServerRequest
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
  request: HttpServerRequest.HttpServerRequest
) {
  const headers = corsHeadersFor(request)
  return headers ? HttpServerResponse.setHeaders(response, headers) : response
}

const disallowedCorsResponse = HttpServerResponse.empty({ status: 403 })
const encodeHealthResponse = Schema.encode(AcnHealthResponseSchema)

// OPTIONS preflight handler — catches all OPTIONS requests.
const OptionsRouteHandler = (request: HttpServerRequest.HttpServerRequest) => {
  const headers = corsHeadersFor(request)
  if (!headers) return Effect.succeed(disallowedCorsResponse)
  return Effect.succeed(
    HttpServerResponse.setHeaders(
      HttpServerResponse.empty({ status: 204 }),
      headers
    )
  )
}

const AcnProcessHandlersLive = Layer.scopedDiscard(
  Effect.gen(function* () {
    const shutdown = yield* AcnShutdown
    const runtime = yield* Effect.runtime<never>()

    const uncaughtExceptionHandler = (error: Error) => {
      Runtime.runPromise(
        runtime,
        Effect.gen(function* () {
          yield* Effect.logError("Uncaught exception in ACN process").pipe(
            Effect.annotateLogs({ error: error.stack ?? String(error) })
          )
          yield* shutdown.request({
            reason: "fatal",
            detail: error.stack ?? String(error),
          })
        })
      ).catch(() => undefined)
    }

    const unhandledRejectionHandler = (reason: unknown) => {
      Runtime.runPromise(
        runtime,
        Effect.gen(function* () {
          const message =
            reason instanceof Error
              ? reason.stack ?? String(reason)
              : String(reason)
          yield* Effect.logError(
            "Unhandled promise rejection in ACN process"
          ).pipe(Effect.annotateLogs({ reason: message }))
          yield* shutdown.request({
            reason: "fatal",
            detail: message,
          })
        })
      ).catch(() => undefined)
    }

    const requestSignalShutdown = (signal: NodeJS.Signals) => {
      Runtime.runPromise(
        runtime,
        shutdown.request({ reason: "signal", detail: signal })
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
      })
    )
  })
)

const makeAcnServicesBase = (debug: boolean, dataDir: string) => {
  const storageBase = Layer.mergeAll(
    VersionLive(ACN_VERSION),
    ProjectStorageLiveFromCwd(process.cwd())
  )

  const storageLayer = StorageLive.pipe(Layer.provide(storageBase))

  const storageServices = Layer.mergeAll(
    SessionStoreLive,
    SessionRuntimeOptionsStoreLive
  ).pipe(Layer.provideMerge(storageLayer))

  const withActivity = Layer.provideMerge(
    AcnActivityTrackerLive("30 minutes", false),
    storageServices
  )
  const withSubscriptions = Layer.provideMerge(
    AcnSubscriptionsLive,
    withActivity
  )
  const withMirroredStateChanges = Layer.provideMerge(
    MirroredStateChangesLive,
    withSubscriptions
  )
  const localServices = addLocalInferenceServices(
    withMirroredStateChanges,
    dataDir
  )
  const withSharedClient = Layer.provideMerge(
    SharedProviderClientLive,
    localServices
  )
  const withCatalog = Layer.provideMerge(
    ProviderModelCatalogLive,
    withSharedClient
  )
  const withCredentials = Layer.provideMerge(
    ProviderCredentialsLive,
    withCatalog
  )
  const withCloudUsage = Layer.provideMerge(
    MagnitudeCloudUsageLive,
    withCredentials
  )
  const withModelSlots = Layer.provideMerge(
    ModelSlotControllerLive,
    withCloudUsage
  )
  const withFactory = Layer.provideMerge(
    AgentFactoryLive({ debug, version: ACN_VERSION }),
    withModelSlots
  )
  const withRuntime = Layer.provideMerge(AgentRuntimeLive, withFactory)
  const withDrafts = Layer.provideMerge(SessionDraftsLive, withRuntime)
  return withDrafts
}

const addLocalInferenceServices = <A, E, R>(
  base: Layer.Layer<A, E, R>,
  dataDir: string
) => {
  const withIcn = Layer.provideMerge(makeAcnIcn(dataDir), base)
  const withConfiguration = Layer.provideMerge(
    makeModelConfigurationLayer(),
    withIcn
  )
  const withHardware = Layer.provideMerge(
    LocalInferenceHardwareLive,
    withConfiguration
  )
  const withPackages = Layer.provideMerge(LocalModelPackagesLive, withHardware)
  const withEvaluations = Layer.provideMerge(
    LocalModelEvaluationsLive,
    withPackages
  )
  const withOfferings = Layer.provideMerge(
    LocalProviderOfferingsLive,
    withEvaluations
  )
  const withOfferingProjection = Layer.provideMerge(
    LocalProviderOfferingProjectionLive,
    withOfferings
  )
  const withRecommendations = Layer.provideMerge(
    makeLocalModelRecommendationsLive(),
    withOfferingProjection
  )
  const withAutoSetup = Layer.provideMerge(
    LocalModelAutoSetupLive,
    withRecommendations
  )
  const withLocalModels = Layer.provideMerge(LocalModelsLive, withAutoSetup)
  const withOnboarding = Layer.provideMerge(OnboardingLive, withLocalModels)
  const withResolver = Layer.provideMerge(
    LocalProviderResolverLive,
    withOnboarding
  )
  const withIcnProvider = Layer.provideMerge(makeIcnProvider(), withResolver)
  const withProviderClients = Layer.provideMerge(
    ProviderClientRegistryLive,
    withIcnProvider
  )
  return withProviderClients
}

const addCommonAcnServices = <A, E, R>(services: Layer.Layer<A, E, R>) => {
  const withDemand = Layer.provideMerge(AcnRpcDemandLive, services)
  const withCommands = Layer.provideMerge(SessionCommandsLive, withDemand)
  const withLifecycle = Layer.provideMerge(SessionLifecycleLive, withCommands)
  const withActiveSessionStatuses = Layer.provideMerge(
    ActiveSessionStatusesLive,
    withLifecycle
  )
  const withStreams = Layer.provideMerge(
    DisplayViewStreamsLive,
    withActiveSessionStatuses
  )
  return withStreams
}

const AcnBaseServicesLayer = (dataDir: string) =>
  addCommonAcnServices(makeAcnServicesBase(false, dataDir))

const AcnDebugServicesLayer = (dataDir: string) => {
  const withActivity = makeAcnServicesBase(true, dataDir)
  const withDisplayIntrospection = Layer.provideMerge(
    AcnDisplayViewIntrospectorLive,
    withActivity
  )
  return addCommonAcnServices(
    Layer.provideMerge(AcnIntrospectorLive, withDisplayIntrospection)
  )
}

const makeAcnInfrastructure = (
  options: AcnServerOptions,
  startup: AcnStartupState
) => {
  const dataDir = options.dataDir ?? defaultDataDir()
  return Layer.mergeAll(
    Layer.succeed(AcnStartupState, startup),
    AcnShutdownLive,
    Layer.succeed(
      GlobalStorage,
      GlobalStorage.of(makeGlobalStorage({ root: dataDir }))
    ),
    BunFileSystem.layer,
    BunCommandExecutor.layer.pipe(Layer.provide(BunFileSystem.layer)),
    BunPath.layer,
    FetchHttpClient.layer,
    BunHttpServer.layer({ port: 0, hostname: "127.0.0.1", idleTimeout: 255 }),
    HttpLayerRouter.layer,
    RpcSerialization.layerNdjson,
    TracingLayer
  )
}

/**
 * Runs one ACN generation until its shutdown coordinator is requested. Scope
 * closure then stops HTTP, disposes sessions, and reaps the private ICN.
 */
export const launchAcnServer = (options: AcnServerOptions = {}) =>
  Effect.scoped(
    Effect.gen(function* () {
      const dataDir = options.dataDir ?? defaultDataDir()
      const debug = options.debug === true
      const startup = yield* makeAcnStartupState()
      yield* startup.starting("WaitingForOwnership", Option.none())
      const applicationScope = yield* Scope.make()
      yield* Effect.addFinalizer(() => Scope.close(applicationScope, Exit.void))

      const infrastructure = yield* Layer.buildWithScope(
        makeAcnInfrastructure(options, startup),
        applicationScope
      )
      const router = Context.get(infrastructure, HttpLayerRouter.HttpRouter)
      const server = Context.get(infrastructure, HttpServer.HttpServer)
      const shutdown = Context.get(infrastructure, AcnShutdown)

      yield* router.addGlobalMiddleware((responseEffect) =>
        Effect.gen(function* () {
          const request = yield* HttpServerRequest.HttpServerRequest
          return withCors(yield* responseEffect, request)
        })
      )
      yield* router.add("OPTIONS", "*", OptionsRouteHandler)
      yield* router.add(
        "GET",
        "/health",
        startup.get.pipe(
          Effect.map((state) => makeHealthResponse(ACN_VERSION, state)),
          Effect.flatMap(encodeHealthResponse),
          Effect.flatMap(HttpServerResponse.json),
          Effect.orDie
        )
      )
      yield* router.add("POST", "/rpc", startup.rpc)
      yield* server
        .serve(router.asHttpEffect())
        .pipe(Effect.provide(infrastructure))
      yield* Layer.buildWithScope(
        Layer.merge(
          DaemonLifecycleLive({
            version: ACN_VERSION,
            register: options.register ?? false,
            debug,
            dataDir,
          }),
          AcnProcessHandlersLive
        ),
        applicationScope
      ).pipe(Effect.provide(infrastructure))

      const ownership = yield* Effect.raceFirst(
        acquireAcnMachineOwnership({
          dataDir,
          id: ACN_OWNER_ID,
          version: ACN_VERSION,
        }).pipe(Effect.as(true)),
        shutdown.await.pipe(Effect.as(false))
      ).pipe(Effect.provide(infrastructure))
      if (!ownership) {
        const request = yield* shutdown.await
        yield* Effect.logInfo(
          "ACN retired before acquiring active runtime ownership"
        ).pipe(
          Effect.annotateLogs({
            reason: request.reason,
            detail: request.detail ?? null,
          })
        )
        return
      }
      if (options.register ?? false) {
        const registration = yield* readRegistrationOwnership(
          registrationPath(dataDir)
        ).pipe(Effect.provide(infrastructure))
        if (!registrationIsOwnedBy(registration, ACN_OWNER_ID)) {
          yield* shutdown.request({ reason: "ownership-lost" })
          return
        }
      }
      const pendingShutdown = yield* shutdown.current
      if (Option.isSome(pendingShutdown)) return
      yield* startup.starting("Resolving", Option.none())

      const application = Effect.gen(function* () {
        const builtServices = yield* debug
          ? Layer.buildWithScope(
              AcnDebugServicesLayer(dataDir),
              applicationScope
            ).pipe(
              Effect.provide(infrastructure),
              Effect.map((context) => ({
                context,
                introspector: Option.some(
                  Context.get(context, AcnIntrospector)
                ),
              }))
            )
          : Layer.buildWithScope(
              AcnBaseServicesLayer(dataDir),
              applicationScope
            ).pipe(
              Effect.provide(infrastructure),
              Effect.map((context) => ({
                context,
                introspector: Option.none<AcnIntrospectorApi>(),
              }))
            )
        const acnServices = builtServices.context
        const serviceContext = Context.merge(infrastructure, acnServices)
        const handlers = yield* Layer.buildWithScope(
          HandlersLive,
          applicationScope
        ).pipe(Effect.provide(serviceContext))
        const applicationContext = Context.merge(serviceContext, handlers)

        const rpcRouter = yield* HttpLayerRouter.make
        const rawProtocol = yield* RpcServer.makeProtocolHttpRouter({
          path: "/rpc",
        }).pipe(
          Effect.provideService(HttpLayerRouter.HttpRouter, rpcRouter),
          Effect.provide(infrastructure)
        )
        const protocol = yield* makeAcnSubscriptionProtocol(rawProtocol).pipe(
          Effect.provide(applicationContext)
        )
        yield* RpcServer.make(MagnitudeRpcs).pipe(
          Effect.provideService(RpcServer.Protocol, protocol),
          Effect.provide(applicationContext),
          Effect.forkIn(applicationScope)
        )
        if (Option.isSome(builtServices.introspector)) {
          yield* installAcnIntrospectionRoutes(
            router,
            builtServices.introspector.value
          )
        }

        yield* startup.ready(rpcRouter.asHttpEffect().pipe(Effect.orDie))
        return {
          subscriptions: Context.get(applicationContext, AcnSubscriptions),
          icn: Context.get(applicationContext, IcnProcess),
          activity: Context.get(applicationContext, AcnActivityTracker),
        }
      })

      const { subscriptions, icn, activity } = yield* application.pipe(
        Effect.tapErrorCause((cause) =>
          startup
            .failed("Magnitude could not prepare local inference", true)
            .pipe(
              Effect.zipRight(
                Effect.logError("ACN application startup failed").pipe(
                  Effect.annotateLogs({ cause: Cause.pretty(cause) })
                )
              )
            )
        )
      )
      yield* activity.ready
      const request = yield* shutdown.await
      yield* Effect.logInfo("ACN shutdown requested").pipe(
        Effect.annotateLogs({
          reason: request.reason,
          detail: request.detail ?? null,
        })
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
    })
  )
