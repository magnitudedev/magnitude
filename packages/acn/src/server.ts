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
  Data,
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
import {
  AcnProcessRevisionSchema,
  applyAcnProcessCommand,
  currentProcessStartIdentity,
  readAcnProcessState,
  type ExactAcnCandidate,
} from "@magnitudedev/acn-protocol/process-state"
import { IcnProcess, makeIcnProvider } from "@magnitudedev/icn"
import { HandlersLive } from "./handlers"
import { defaultDataDir } from "./data-dir"
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
  ACN_INSTANCE_ID,
  makeHealthResponse,
} from "./identity"
import { MirroredStateChangesLive } from "./mirrored-state"
import { AcnSubscriptions, AcnSubscriptionsLive } from "./acn-subscriptions"
import { makeAcnSubscriptionProtocol } from "./acn-subscription-protocol"
import {
  AcnServiceLifecycle,
  makeAcnServiceLifecycle,
  type AcnServiceLifecycleApi,
} from "./service-lifecycle"
import { ClientLeaseManagerLive } from "./client-lease-manager"
import { ModelResidencyPolicyLive } from "./model-residency-policy"

export interface AcnServerOptions {
  readonly register?: boolean
  readonly debug?: boolean
  readonly dataDir?: string
  readonly changeRevision?: Option.Option<number>
}

class AcnBootstrapRejected extends Data.TaggedError("AcnBootstrapRejected")<{
  readonly reason: string
}> {}

const awaitBootstrapRelease = Effect.async<void, AcnBootstrapRejected>((resume) => {
  const onData = (chunk: Buffer) => {
    cleanup()
    chunk.includes(0x31)
      ? resume(Effect.void)
      : resume(Effect.fail(new AcnBootstrapRejected({ reason: "invalid bootstrap release" })))
  }
  const onEnd = () => {
    cleanup()
    resume(Effect.fail(new AcnBootstrapRejected({ reason: "bootstrap owner exited before state transfer" })))
  }
  const onError = (error: Error) => {
    cleanup()
    resume(Effect.fail(new AcnBootstrapRejected({ reason: error.message })))
  }
  const cleanup = () => {
    process.stdin.off("data", onData)
    process.stdin.off("end", onEnd)
    process.stdin.off("error", onError)
  }
  process.stdin.once("data", onData)
  process.stdin.once("end", onEnd)
  process.stdin.once("error", onError)
  process.stdin.resume()
  return Effect.sync(cleanup)
})

const acnServerUrl = (address: HttpServer.Address): string => {
  if (address._tag === "UnixAddress") {
    throw new TypeError("Unix sockets are not supported for ACN coordination")
  }
  const hostname = address.hostname === "0.0.0.0" ? "127.0.0.1" : address.hostname
  return `http://${hostname}:${address.port}`
}

const CORS_ALLOWED_HEADERS =
  "Content-Type, Content-Length, x-magnitude-acn-id, traceparent, tracestate, baggage, b3, x-b3-traceid, x-b3-spanid, x-b3-parentspanid, x-b3-sampled, x-b3-flags"
const LOCAL_HTTP_ORIGIN =
  /^https?:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/

const closeApplication = (scope: Scope.CloseableScope) =>
  Effect.gen(function* () {
    const closing = yield* Scope.close(scope, Exit.void).pipe(Effect.forkDaemon)
    yield* Effect.raceFirst(
      Fiber.await(closing),
      Effect.sleep("5 seconds"),
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
  const withResidencyPolicy = Layer.provideMerge(
    ModelResidencyPolicyLive,
    withIcn,
  )
  const withConfiguration = Layer.provideMerge(
    makeModelConfigurationLayer(),
    withResidencyPolicy
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
  const withClientLeases = Layer.provideMerge(ClientLeaseManagerLive, withDemand)
  const withCommands = Layer.provideMerge(SessionCommandsLive, withClientLeases)
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
      const candidate: Option.Option<ExactAcnCandidate> = options.register === true
        ? yield* Effect.gen(function* () {
            yield* awaitBootstrapRelease
            const rawChangeRevision = Option.getOrUndefined(options.changeRevision ?? Option.none())
            if (
              rawChangeRevision === undefined ||
              !Number.isSafeInteger(rawChangeRevision) ||
              rawChangeRevision <= 0
            ) {
              return yield* new AcnBootstrapRejected({ reason: "registered ACN is missing a valid change revision" })
            }
            const changeRevision = AcnProcessRevisionSchema.make(rawChangeRevision)
            const state = yield* readAcnProcessState(dataDir)
            if (
              Option.isNone(state) ||
              state.value.mode._tag !== "Changing" ||
              state.value.mode.changeRevision !== changeRevision ||
              state.value.mode.owner._tag !== "Candidate"
            ) return yield* new AcnBootstrapRejected({ reason: "process state does not name this candidate change" })
            const expected = state.value.mode.owner.candidate
            if (
              expected.pid !== process.pid ||
              expected.processStartIdentity !== processStartIdentity ||
              expected.identity !== ACN_VERSION
            ) return yield* new AcnBootstrapRejected({ reason: "process state does not name this exact candidate" })
            return Option.some(expected)
          }).pipe(
            Effect.provide(BunFileSystem.layer),
            Effect.orDie,
          )
        : Option.none()
      const lifecycle = yield* makeAcnServiceLifecycle()
      const applicationScope = yield* Scope.make()
      const closeApplicationScope = yield* Effect.cached(
        closeApplication(applicationScope),
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
      yield* router.add(
        "POST",
        "/rpc",
        Effect.gen(function* () {
          const request = yield* HttpServerRequest.HttpServerRequest
          return request.headers["x-magnitude-acn-id"] === ACN_INSTANCE_ID
            ? yield* lifecycle.dispatchRpc
            : HttpServerResponse.empty({ status: 409 })
        }),
      )
      yield* router.add(
        "POST",
        "/shutdown",
        Effect.gen(function* () {
          const request = yield* HttpServerRequest.HttpServerRequest
          if (request.headers["x-magnitude-acn-id"] !== ACN_INSTANCE_ID) {
            return HttpServerResponse.empty({ status: 409 })
          }
          yield* lifecycle.beginStopping({ reason: "replacement" })
          return HttpServerResponse.empty({ status: 202 })
        }),
      )
      yield* server
        .serve(router.asHttpEffect())
        .pipe(Effect.provide(infrastructure))
      yield* Layer.buildWithScope(
        AcnProcessHandlersLive,
        applicationScope
      ).pipe(Effect.provide(infrastructure))

      if (Option.isSome(candidate)) {
        const state = yield* readAcnProcessState(dataDir).pipe(Effect.provide(infrastructure))
        const admitted = yield* applyAcnProcessCommand({
          dataDirectory: dataDir,
          expectedRevision: Option.map(state, (value) => value.revision),
          command: {
            _tag: "CandidateAdmitted",
            candidate: candidate.value,
            id: ACN_INSTANCE_ID,
            url: acnServerUrl(server.address),
          },
        }).pipe(Effect.provide(infrastructure), Effect.either)
        if (admitted._tag === "Left") {
          yield* lifecycle.beginStopping({
            reason: "ownership-lost",
            detail: String(admitted.left),
          })
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
