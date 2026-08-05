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
  Fiber,
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
import { LocalModelAssessmentsLive } from "./local-model-assessments"
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
import { acquireAcnMachineOwnership } from "./machine-ownership"
import {
  readRegistrationOwnership,
  removeExactInstance,
  registrationIsOwnedBy,
  registrationPath,
  writeInstanceAtomic,
} from "./daemon-registration"
import { AcnSubscriptions, AcnSubscriptionsLive } from "./acn-subscriptions"
import { makeAcnSubscriptionProtocol } from "./acn-subscription-protocol"
import {
  AcnServiceLifecycle,
  makeAcnServiceLifecycle,
  type AcnServiceLifecycleApi,
} from "./service-lifecycle"
import { removePeerAcns } from "./peer-acn"
import { currentProcessStartIdentity } from "./process-identity"

export interface AcnServerOptions {
  readonly register?: boolean
  readonly debug?: boolean
  readonly dataDir?: string
}

const CORS_ALLOWED_HEADERS =
  "Content-Type, Content-Length, traceparent, tracestate, baggage, b3, x-b3-traceid, x-b3-spanid, x-b3-parentspanid, x-b3-sampled, x-b3-flags"
const LOCAL_HTTP_ORIGIN =
  /^https?:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/

const closeScopeBounded = (scope: Scope.CloseableScope) =>
  Effect.gen(function* () {
    const closing = yield* Scope.close(scope, Exit.void).pipe(Effect.forkDaemon)
    yield* Effect.raceFirst(
      Fiber.await(closing),
      Effect.sleep("1 second"),
    ).pipe(Effect.asVoid)
  })

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
    const lifecycle = yield* AcnServiceLifecycle
    const runtime = yield* Effect.runtime<never>()

    const uncaughtExceptionHandler = (error: Error) => {
      Runtime.runPromise(
        runtime,
        Effect.gen(function* () {
          yield* Effect.logError("Uncaught exception in ACN process").pipe(
            Effect.annotateLogs({ error: error.stack ?? String(error) })
          )
          yield* lifecycle.beginStopping({
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
          yield* lifecycle.beginStopping({
            reason: "fatal",
            detail: message,
          })
        })
      ).catch(() => undefined)
    }

    const requestSignalShutdown = (signal: NodeJS.Signals) => {
      Runtime.runPromise(
        runtime,
        lifecycle.beginStopping({ reason: "signal", detail: signal })
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
    AcnActivityTrackerLive,
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
  const withAssessments = Layer.provideMerge(
    LocalModelAssessmentsLive,
    withPackages
  )
  const withOfferings = Layer.provideMerge(
    LocalProviderOfferingsLive,
    withAssessments
  )
  const withOfferingProjection = Layer.provideMerge(
    LocalProviderOfferingProjectionLive,
    withOfferings
  )
  const withRecommendations = Layer.provideMerge(
    makeLocalModelRecommendationsLive(),
    withOfferingProjection
  )
  const withLocalModels = Layer.provideMerge(LocalModelsLive, withRecommendations)
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
  lifecycle: AcnServiceLifecycleApi,
) => {
  const dataDir = options.dataDir ?? defaultDataDir()
  return Layer.mergeAll(
    Layer.succeed(AcnServiceLifecycle, lifecycle),
    Layer.succeed(
      GlobalStorage,
      GlobalStorage.of(makeGlobalStorage({ root: dataDir }))
    ),
    BunFileSystem.layer,
    BunCommandExecutor.layer.pipe(Layer.provide(BunFileSystem.layer)),
    BunPath.layer,
    FetchHttpClient.layer,
    // Finite unary RPCs may legitimately run for the full duration of model
    // download or loading. Bun counts an in-flight handler that has not yet
    // emitted response bytes as idle, so any non-zero server timeout would
    // turn operation duration into a connection reset.
    BunHttpServer.layer({ port: 0, hostname: "127.0.0.1", idleTimeout: 0 }),
    HttpLayerRouter.layer,
    RpcSerialization.layerNdjson,
    TracingLayer
  )
}

/**
 * Runs one ACN process until its lifecycle enters Stopping. Scope
 * closure then stops HTTP, disposes sessions, and reaps the private ICN.
 */
export const launchAcnServer = (options: AcnServerOptions = {}) =>
  Effect.scoped(
    Effect.gen(function* () {
      const dataDir = options.dataDir ?? defaultDataDir()
      const debug = options.debug === true
      const processStartIdentity = yield* currentProcessStartIdentity.pipe(
        Effect.provide(
          BunCommandExecutor.layer.pipe(Layer.provide(BunFileSystem.layer)),
        ),
        Effect.orDie,
      )
      const instanceRecord = {
        id: ACN_OWNER_ID,
        version: ACN_VERSION,
        url: Option.none<string>(),
        pid: process.pid,
        processStartIdentity,
      }
      const publishedInstancePath = yield* writeInstanceAtomic(
        dataDir,
        instanceRecord,
      ).pipe(
        Effect.provide(BunFileSystem.layer),
        Effect.orDie,
      )
      const instance = {
        path: publishedInstancePath,
        record: instanceRecord,
      }
      yield* Effect.addFinalizer(() =>
        removeExactInstance(instance).pipe(
          Effect.provide(BunFileSystem.layer),
          Effect.ignore,
        ),
      )
      const lifecycle = yield* makeAcnServiceLifecycle()
      const applicationScope = yield* Scope.make()
      const closeApplicationScope = yield* Effect.cached(
        closeScopeBounded(applicationScope),
      )
      yield* Effect.addFinalizer(() => closeApplicationScope)

      const infrastructure = yield* Layer.buildWithScope(
        makeAcnInfrastructure(options, lifecycle),
        applicationScope
      )
      const router = Context.get(infrastructure, HttpLayerRouter.HttpRouter)
      const server = Context.get(infrastructure, HttpServer.HttpServer)

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
        lifecycle.state.pipe(
          Effect.map((state) => makeHealthResponse(ACN_VERSION, state)),
          Effect.flatMap(encodeHealthResponse),
          Effect.flatMap(HttpServerResponse.json),
          Effect.orDie
        )
      )
      yield* router.add("POST", "/rpc", lifecycle.dispatchRpc)
      yield* router.add(
        "POST",
        "/shutdown",
        Effect.gen(function* () {
          const request = yield* HttpServerRequest.HttpServerRequest
          if (request.headers["x-magnitude-acn-id"] !== ACN_OWNER_ID) {
            return HttpServerResponse.empty({ status: 409 })
          }
          yield* lifecycle.beginStopping({ reason: "peer-request" })
          return HttpServerResponse.empty({ status: 202 })
        }),
      )
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
            instance,
          }),
          AcnProcessHandlersLive
        ),
        applicationScope
      ).pipe(Effect.provide(infrastructure))

      if (options.register ?? false) {
        yield* removePeerAcns(dataDir, ACN_OWNER_ID).pipe(
          Effect.provide(infrastructure),
          Effect.tapError((error) =>
            lifecycle.beginStopping({
              reason: "startup-failed",
              detail: `Unable to remove ACN peer ${error.id}: ${error.reason}`,
            }),
          ),
        )
      }

      const ownership = yield* Effect.raceFirst(
        acquireAcnMachineOwnership({
          dataDir,
          id: ACN_OWNER_ID,
          version: ACN_VERSION,
        }).pipe(
          Effect.timeoutOption("10 seconds"),
          Effect.flatMap(
            Option.match({
              onSome: () => Effect.succeed(true),
              onNone: () => lifecycle.beginStopping({
                reason: "startup-failed",
                detail: "Timed out acquiring active ACN ownership",
              }).pipe(Effect.as(false)),
            }),
          ),
        ),
        lifecycle.awaitStopping.pipe(Effect.as(false))
      ).pipe(Effect.provide(infrastructure))
      if (!ownership) {
        const request = yield* lifecycle.awaitStopping
        yield* Effect.logInfo(
          "ACN retired before acquiring active runtime ownership"
        ).pipe(
          Effect.annotateLogs({
            reason: request.reason,
            detail: Option.getOrNull(request.safeDetail),
          })
        )
        return
      }
      if (options.register ?? false) {
        const registration = yield* readRegistrationOwnership(
          registrationPath(dataDir)
        ).pipe(Effect.provide(infrastructure))
        if (!registrationIsOwnedBy(registration, ACN_OWNER_ID)) {
          yield* lifecycle.beginStopping({ reason: "ownership-lost" })
          return
        }
      }
      const currentLifecycle = yield* lifecycle.state
      if (currentLifecycle._tag === "Stopping") return
      yield* lifecycle.reportStarting("Resolving", Option.none())

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

        yield* lifecycle.becomeReady(rpcRouter.asHttpEffect().pipe(Effect.orDie))
        return {
          subscriptions: Context.get(applicationContext, AcnSubscriptions),
          icn: Context.get(applicationContext, IcnProcess),
        }
      })

      const { subscriptions, icn } = yield* application.pipe(
        Effect.timeout("5 minutes"),
        Effect.tapErrorCause((cause) =>
          lifecycle
            .beginStopping({
              reason: "startup-failed",
              detail: "Magnitude could not prepare local inference",
            })
            .pipe(
              Effect.zipRight(
                Effect.logError("ACN application startup failed").pipe(
                  Effect.annotateLogs({ cause: Cause.pretty(cause) })
                )
              )
            )
        )
      )
      const request = yield* lifecycle.awaitStopping
      yield* Effect.logInfo("ACN shutdown requested").pipe(
        Effect.annotateLogs({
          reason: request.reason,
          detail: Option.getOrNull(request.safeDetail),
        })
      )
      yield* subscriptions.terminate
      // Close consumers before the owned ICN. Layer finalizer ordering stops
      // observation fibers and HTTP admission before the ICN finalizer sends
      // its termination signal, so graceful drain cannot be prolonged by new
      // internal requests.
      yield* closeApplicationScope
      yield* icn.shutdown.pipe(Effect.orDie)
    })
  )
