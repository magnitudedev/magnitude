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
  Context,
  Data,
  Deferred,
  Duration,
  Effect,
  Exit,
  Layer,
  Option,
  Ref,
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
  AcnBoundary,
  AcnRpc,
} from "@magnitudedev/acn-protocol"
import {
  ProcessGroupController,
  makeAcnOwnerStore,
  type AcnOwnerStoreError,
  type AcnOwnerStore,
  type ExactProcess,
} from "@magnitudedev/acn-protocol/coordination"
import { BunSqliteDriverLayer } from "@magnitudedev/acn-protocol/coordination/bun"
import { ProcessGroupControllerLive } from "@magnitudedev/acn-protocol/coordination/exact-process"
import { IcnProcess, makeIcnProvider } from "@magnitudedev/icn"
import { AcnBoundaryLive } from "./boundary/acn"
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
import { ModelSelectionLive } from "./model-selection"
import { LocalModelConfigurationCoordinatorLive } from "./local-model-configuration-coordinator"
import { makeAcnIcn } from "./icn"
import { LocalModelAssessmentsLive } from "./local-model-assessments"
import { LocalModelAssessorLive } from "./local-model-assessor"
import { LocalModelConfigurationResolverLive } from "./local-model-configuration-resolver"
import { LocalModelPackagesLive } from "./local-model-packages"
import { makeLocalModelRecommendationsLive } from "./local-model-recommendations"
import { LocalModelsLive } from "./local-models"
import { LocalProviderOfferingsLive } from "./local-provider-offerings"
import { installAcnOwnershipMonitor } from "./ownership-monitor"
import { LocalProviderResolverLive } from "./local-provider-resolver"
import { LocalInferenceHardwareLive } from "./local-inference-hardware"
import { OnboardingLive } from "./onboarding"
import { CustomEndpointsLive } from "./custom-endpoints"
import { CustomEndpointReconcilerLive } from "./custom-endpoint-reconciler"
import { FileMentionSearcherLive } from "./file-mention-searcher"
import { FileSystemManagerLive } from "./file-system-manager"
import { GitInspectorLive } from "./git-inspector"
import { ProjectFileManagerLive } from "./project-file-manager"
import { ProjectInspectorLive } from "./project-inspector"
import { ProjectManagerLive } from "./project-manager"
import { ProjectStoreLive } from "./project-store"
import { SessionInspectorLive } from "./session-inspector"
import { ACN_REVISION, ACN_VERSION } from "./version"
import { TracingLayer } from "./tracing"
import {
  ACN_INSTANCE_ID,
  makeHealthResponse,
} from "./identity"
import { AcnChangesLive, AcnStorageChangesLive } from "./changes"
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
  readonly parentBound?: boolean
  readonly debug?: boolean
  readonly dataDir?: string
  readonly port?: number
}

export const ACN_PUBLIC_PORT = 10_100

class AcnBootstrapRejected extends Data.TaggedError("AcnBootstrapRejected")<{
  readonly reason: string
}> {}

class InferenceProxyFailed extends Data.TaggedError("InferenceProxyFailed")<{
  readonly cause: unknown
}> {}

class AcnRestartRequired extends Data.TaggedError("AcnRestartRequired")<{
  readonly reason: string
}> {}

type ParentBindingState = "Pending" | "Admitted" | "Lost"

const makeParentBinding = (
  enabled: boolean,
): Effect.Effect<{
  readonly admit: <A, E>(
    effect: Effect.Effect<A, E>,
    admitted: (value: A) => boolean,
  ) => Effect.Effect<A, E | AcnBootstrapRejected>
}, never, Scope.Scope> => Effect.gen(function* () {
  if (!enabled) return { admit: (effect) => effect }
  const state = yield* Ref.make<ParentBindingState>("Pending")
  const lock = yield* Effect.makeSemaphore(1)
  const lost = yield* Deferred.make<void>()
  const runtime = yield* Effect.runtime<never>()
  const reportLoss = () => Runtime.runSync(runtime, Deferred.succeed(lost, undefined))
  const onEnd = () => {
    reportLoss()
  }
  const onError = () => {
    reportLoss()
  }
  process.stdin.once("end", onEnd)
  process.stdin.once("error", onError)
  process.stdin.resume()
  yield* Effect.addFinalizer(() => Effect.sync(() => {
    process.stdin.off("end", onEnd)
    process.stdin.off("error", onError)
  }))
  yield* Deferred.await(lost).pipe(
    Effect.flatMap(() => lock.withPermits(1)(Ref.update(state, (current) =>
      current === "Pending" ? "Lost" : current))),
    Effect.forkScoped,
  )
  return {
    admit: (effect, isAdmitted) => lock.withPermits(1)(Effect.uninterruptibleMask((restore) =>
      Effect.gen(function* () {
        if ((yield* Ref.get(state)) === "Lost" || Option.isSome(yield* Deferred.poll(lost))) {
          yield* Ref.set(state, "Lost")
          return yield* new AcnBootstrapRejected({
            reason: "ACN spawning parent exited before admission",
          })
        }
        const value = yield* restore(Effect.raceFirst(
          effect,
          Deferred.await(lost).pipe(Effect.flatMap(() => Effect.fail(new AcnBootstrapRejected({
            reason: "ACN spawning parent exited before admission",
          })))),
        ))
        if (Option.isSome(yield* Deferred.poll(lost))) {
          yield* Ref.set(state, "Lost")
          return yield* new AcnBootstrapRejected({
            reason: "ACN spawning parent exited before admission",
          })
        }
        if (isAdmitted(value)) yield* Ref.set(state, "Admitted")
        return value
      }),
    )),
  }
})

const acnServerUrl = (address: HttpServer.Address): string => {
  if (address._tag === "UnixAddress") {
    throw new TypeError("Unix sockets are not supported for ACN coordination")
  }
  const hostname = address.hostname === "0.0.0.0" ? "127.0.0.1" : address.hostname
  return `http://${hostname}:${address.port}`
}

const CORS_ALLOWED_HEADERS =
  "Accept, Authorization, Content-Type, Content-Length, Magnitude-Include-Progress, x-magnitude-acn-id, traceparent, tracestate, baggage, b3, x-b3-traceid, x-b3-spanid, x-b3-parentspanid, x-b3-sampled, x-b3-flags"
const LOCAL_HTTP_ORIGIN =
  /^https?:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/
const LOCAL_HTTP_HOST = /^(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/

const closeApplication = (scope: Scope.CloseableScope) =>
  Scope.close(scope, Exit.void).pipe(
    Effect.disconnect,
    Effect.timeoutOption(Duration.seconds(5)),
    Effect.asVoid,
  )

const boundedShutdownStep = (
  effect: Effect.Effect<unknown, unknown>,
  timeout: Duration.DurationInput = Duration.seconds(5),
) => effect.pipe(
  Effect.disconnect,
  Effect.timeoutOption(timeout),
  Effect.asVoid,
)

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
    "access-control-allow-methods": "GET, POST, PUT, DELETE, OPTIONS",
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
    const processEvents = process as unknown as NodeJS.EventEmitter

    processEvents.on("uncaughtException", uncaughtExceptionHandler)
    processEvents.on("unhandledRejection", unhandledRejectionHandler)
    processEvents.on("SIGINT", sigintHandler)
    processEvents.on("SIGTERM", sigtermHandler)

    yield* Effect.addFinalizer(() =>
      Effect.sync(() => {
        processEvents.removeListener("uncaughtException", uncaughtExceptionHandler)
        processEvents.removeListener("unhandledRejection", unhandledRejectionHandler)
        processEvents.removeListener("SIGINT", sigintHandler)
        processEvents.removeListener("SIGTERM", sigtermHandler)
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

  // The two durable read authorities plus host observation services. Platform
  // requirements (FileSystem, Path, CommandExecutor) flow from infrastructure.
  const domainCore = Layer.mergeAll(
    FileSystemManagerLive,
    GitInspectorLive,
    ProjectStoreLive,
    SessionInspectorLive,
  ).pipe(Layer.provideMerge(storageLayer))

  const storageServices = Layer.mergeAll(
    SessionRuntimeOptionsStoreLive
  ).pipe(Layer.provideMerge(domainCore))

  const withSubscriptions = Layer.provideMerge(
    AcnSubscriptionsLive,
    storageServices
  )
  // The change registry serves `StreamChanges`; storage change streams are
  // forwarded into it here, versioned snapshots publish their own pokes.
  const withChanges = Layer.provideMerge(
    AcnStorageChangesLive,
    Layer.provideMerge(AcnChangesLive, withSubscriptions),
  )
  const localServices = addLocalInferenceServices(
    withChanges,
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
  const withCustomEndpointReconciliation = Layer.provideMerge(
    CustomEndpointReconcilerLive,
    withModelSlots,
  )
  const withFactory = Layer.provideMerge(
    AgentFactoryLive({ debug, version: ACN_VERSION }),
    withCustomEndpointReconciliation
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
  const withSelection = Layer.provideMerge(
    ModelSelectionLive,
    withResidencyPolicy
  )
  const withConfigurationCoordinator = Layer.provideMerge(
    LocalModelConfigurationCoordinatorLive,
    withSelection,
  )
  const withCustomEndpoints = Layer.provideMerge(
    CustomEndpointsLive,
    withConfigurationCoordinator,
  )
  const withHardware = Layer.provideMerge(
    LocalInferenceHardwareLive,
    withCustomEndpoints
  )
  const withPackages = Layer.provideMerge(LocalModelPackagesLive, withHardware)
  const withAssessments = Layer.provideMerge(
    LocalModelAssessmentsLive,
    withPackages
  )
  const withAssessor = Layer.provideMerge(
    LocalModelAssessorLive,
    withAssessments,
  )
  const withConfigurationResolver = Layer.provideMerge(
    LocalModelConfigurationResolverLive,
    withAssessor,
  )
  const withOfferings = Layer.provideMerge(
    LocalProviderOfferingsLive,
    withConfigurationResolver
  )
  const withRecommendations = Layer.provideMerge(
    makeLocalModelRecommendationsLive(),
    withOfferings
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
  const withClientLeases = Layer.provideMerge(ClientLeaseManagerLive, services)
  const withMentionSearcher = Layer.provideMerge(FileMentionSearcherLive, withClientLeases)
  const withCommands = Layer.provideMerge(SessionCommandsLive, withMentionSearcher)
  const withLifecycle = Layer.provideMerge(SessionLifecycleLive, withCommands)
  const withProjectManager = Layer.provideMerge(ProjectManagerLive, withLifecycle)
  const withProjectInspector = Layer.provideMerge(ProjectInspectorLive, withProjectManager)
  const withProjectFiles = Layer.provideMerge(ProjectFileManagerLive, withProjectInspector)
  const withActiveSessionStatuses = Layer.provideMerge(
    ActiveSessionStatusesLive,
    withProjectFiles
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
  const services = makeAcnServicesBase(true, dataDir)
  const withDisplayIntrospection = Layer.provideMerge(
    AcnDisplayViewIntrospectorLive,
    services
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
    BunHttpServer.layer({
      // Candidate coordination endpoints must remain independently bindable so
      // concurrent candidates can reach atomic owner admission. The admitted
      // process opens the stable public inference listener separately.
      port: options.port ?? 0,
      hostname: "127.0.0.1",
      idleTimeout: 0,
    }),
    HttpLayerRouter.layer,
    RpcSerialization.layerNdjson,
    TracingLayer
  )
}

const makePublicInferenceInfrastructure = (lifecycle: AcnServiceLifecycleApi) =>
  Layer.mergeAll(
    Layer.succeed(AcnServiceLifecycle, lifecycle),
    BunHttpServer.layer({
      port: ACN_PUBLIC_PORT,
      hostname: "127.0.0.1",
      idleTimeout: 0,
    }),
    HttpLayerRouter.layer,
    TracingLayer,
  )

/**
 * Runs one ACN process until its lifecycle enters Stopping. Scope
 * closure then stops HTTP, disposes sessions, and reaps the private ICN.
 */
const rejectCoordinationFailure = <A>(
  effect: Effect.Effect<A, AcnOwnerStoreError | AcnBootstrapRejected>,
): Effect.Effect<A, AcnBootstrapRejected> => effect.pipe(
  Effect.mapError((error) => error instanceof AcnBootstrapRejected
    ? error
    : new AcnBootstrapRejected({ reason: `${error._tag}: ${error.message}` })),
)

const HOP_BY_HOP_HEADERS = [
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
] as const

interface InferenceProxyTarget {
  readonly origin: URL
  readonly clientOptions: {
    readonly headers?: Readonly<Record<string, string>>
  }
}

export const proxyInferenceWebRequest = async (
  source: Request,
  icn: InferenceProxyTarget,
  fetchTarget: typeof fetch = fetch,
  signal: AbortSignal = source.signal,
): Promise<Response> => {
    const incoming = new URL(source.url)
    const targetPath = incoming.pathname.slice("/inference".length) || "/"
    const target = new URL(`${targetPath}${incoming.search}`, icn.origin)
    const headers = new Headers(source.headers)
    headers.delete("host")
    headers.delete("authorization")
    headers.delete("x-magnitude-acn-id")
    for (const header of HOP_BY_HOP_HEADERS) headers.delete(header)
    const privateAuthorization = icn.clientOptions.headers?.authorization
    if (privateAuthorization !== undefined) {
      headers.set("authorization", privateAuthorization)
    }

    const response = await fetchTarget(target, {
      method: source.method,
      headers,
      body: source.body,
      signal,
      redirect: "manual",
    })
    const responseHeaders = new Headers(response.headers)
    for (const header of HOP_BY_HOP_HEADERS) responseHeaders.delete(header)

    if (targetPath === "/openapi.json" && response.ok) {
      try {
        const document = await response.json() as Record<string, unknown>
        // The representation changes when the public server base is injected;
        // representation metadata from the private response is no longer valid.
        responseHeaders.delete("content-length")
        responseHeaders.delete("content-encoding")
        responseHeaders.delete("content-md5")
        responseHeaders.delete("etag")
        return Response.json({
          ...document,
          servers: [{ url: "/inference" }],
        }, { status: response.status, headers: responseHeaders })
      } catch {
        return new Response("Invalid ICN OpenAPI document", { status: 502 })
      }
    }
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    })
}

const makeInferenceProxy = (icn: Context.Tag.Service<typeof IcnProcess>) =>
  (request: HttpServerRequest.HttpServerRequest) => Effect.gen(function* () {
    // The wildcard proxy route is also the most specific OPTIONS route. Handle
    // browser preflight locally instead of forwarding it to an ICN operation.
    if (request.method === "OPTIONS") return yield* OptionsRouteHandler(request)
    const source = request.source
    if (!(source instanceof Request)) {
      return HttpServerResponse.text("Unsupported request transport", { status: 500 })
    }
    const response = yield* Effect.tryPromise({
      try: (signal) => proxyInferenceWebRequest(source, icn, fetch, signal),
      catch: (cause) => new InferenceProxyFailed({ cause }),
    }).pipe(Effect.option)
    return Option.isNone(response)
      ? HttpServerResponse.text("Local inference service unavailable", { status: 502 })
      : HttpServerResponse.fromWeb(response.value)
  })

const predecessorAbsent = (
  owner: Option.Option<{ readonly pid: number; readonly processStartIdentity: ExactProcess["processStartIdentity"] }>,
): Effect.Effect<boolean, AcnBootstrapRejected, ProcessGroupController> => Option.match(owner, {
  onNone: () => Effect.succeed(true),
  onSome: (process) => ProcessGroupController.pipe(
    Effect.flatMap((processes) => processes.observe({ leader: process })),
    Effect.map((observed) => observed._tag === "ProcessGroupAbsent"),
    Effect.mapError((error) => new AcnBootstrapRejected({ reason: error.message })),
  ),
})

export const launchAcnServer = (options: AcnServerOptions = {}) =>
  Effect.scoped(Effect.gen(function* () {
    const dataDir = options.dataDir ?? defaultDataDir()
    const debug = options.debug === true
    const parentBinding = yield* makeParentBinding(options.parentBound === true)

    const ownerStore = yield* makeAcnOwnerStore(dataDir).pipe(
      Effect.provide(Layer.merge(BunFileSystem.layer, BunPath.layer)),
    )

    const currentProcess = yield* ProcessGroupController.pipe(
      Effect.flatMap((processes) => processes.currentProcess),
      Effect.mapError((error) => new AcnBootstrapRejected({ reason: error.message })),
    )

    const lifecycle = yield* makeAcnServiceLifecycle()
    const applicationScope = yield* Scope.make()
    const closeApplicationScope = yield* Effect.cached(closeApplication(applicationScope))
    yield* Effect.addFinalizer(() => closeApplicationScope)
    const infrastructure = yield* Layer.buildWithScope(
      makeAcnInfrastructure(options, lifecycle),
      applicationScope,
    )
    const router = Context.get(infrastructure, HttpLayerRouter.HttpRouter)
    const server = Context.get(infrastructure, HttpServer.HttpServer)
    const address = server.address
    if (address._tag === "UnixAddress") {
      return yield* new AcnBootstrapRejected({ reason: "ACN requires a loopback TCP endpoint" })
    }

    yield* router.addGlobalMiddleware((responseEffect) => Effect.gen(function* () {
      const request = yield* HttpServerRequest.HttpServerRequest
      const host = request.headers.host
      if (host === undefined || !LOCAL_HTTP_HOST.test(host)) {
        return HttpServerResponse.text("Invalid Host header", { status: 421 })
      }
      return withCors(yield* responseEffect, request)
    }))
    yield* router.add("OPTIONS", "/*", OptionsRouteHandler)
    yield* router.add("GET", "/health", lifecycle.state.pipe(
      Effect.flatMap((state) => encodeHealthResponse(makeHealthResponse(ACN_VERSION, state)).pipe(
        Effect.flatMap((body) => HttpServerResponse.json(body, {
          status: state._tag === "Ready" ? 200 : 503,
        })),
      )),
      Effect.orDie,
    ))
    yield* router.add("POST", "/rpc", Effect.gen(function* () {
      const request = yield* HttpServerRequest.HttpServerRequest
      return request.headers["x-magnitude-acn-id"] === ACN_INSTANCE_ID
        ? yield* lifecycle.dispatchRpc
        : HttpServerResponse.empty({ status: 409 })
    }))
    yield* router.add("POST", "/shutdown", lifecycle.beginStopping({ reason: "administrative" }).pipe(
      Effect.as(HttpServerResponse.empty({ status: 202 })),
    ))
    yield* server.serve(router.asHttpEffect()).pipe(Effect.provide(infrastructure))

    const expectedOwner = yield* rejectCoordinationFailure(ownerStore.current)
    if (!(yield* predecessorAbsent(Option.map(expectedOwner, (owner) => ({
      pid: owner.pid,
      processStartIdentity: owner.processStartIdentity,
    }))))) return
    const admittedOwner = { ...currentProcess, port: address.port }
    const admission = yield* parentBinding.admit(
      ownerStore.replaceOwner(
        expectedOwner,
        admittedOwner,
      ),
      (result) => result._tag === "Replaced",
    ).pipe(rejectCoordinationFailure)
    if (admission._tag !== "Replaced") return

    yield* installAcnOwnershipMonitor(ownerStore, admittedOwner, lifecycle).pipe(
      Effect.provideService(Scope.Scope, applicationScope),
    )

    yield* Layer.buildWithScope(AcnProcessHandlersLive, applicationScope).pipe(
      Effect.provide(infrastructure),
    )

    yield* lifecycle.reportStarting("Resolving", Option.none())
    const application = Effect.gen(function* () {
      const builtServices = yield* debug
        ? Layer.buildWithScope(AcnDebugServicesLayer(dataDir), applicationScope).pipe(
            Effect.provide(infrastructure),
            Effect.map((context) => ({
              context,
              introspector: Option.some(Context.get(context, AcnIntrospector)),
            })),
          )
        : Layer.buildWithScope(AcnBaseServicesLayer(dataDir), applicationScope).pipe(
            Effect.provide(infrastructure),
            Effect.map((context) => ({
              context,
              introspector: Option.none<AcnIntrospectorApi>(),
            })),
          )
      const serviceContext = Context.merge(infrastructure, builtServices.context)
      const handlers = yield* Layer.buildWithScope(AcnBoundaryLive, applicationScope).pipe(
        Effect.provide(serviceContext),
      )
      const applicationContext = Context.merge(serviceContext, handlers)
      const rpcRouter = yield* HttpLayerRouter.make
      const rawProtocol = yield* RpcServer.makeProtocolHttpRouter({ path: "/rpc" }).pipe(
        Effect.provideService(HttpLayerRouter.HttpRouter, rpcRouter),
        Effect.provide(infrastructure),
      )
      const protocol = yield* makeAcnSubscriptionProtocol(rawProtocol).pipe(
        Effect.provide(applicationContext),
      )
      yield* AcnRpc.makeRpcServer(AcnBoundary).pipe(
        Effect.provideService(RpcServer.Protocol, protocol),
        Effect.provide(applicationContext),
        Effect.forkIn(applicationScope),
      )
      if (Option.isSome(builtServices.introspector)) {
        yield* installAcnIntrospectionRoutes(router, builtServices.introspector.value)
      }
      const icn = Context.get(applicationContext, IcnProcess)
      const publicInfrastructure = yield* Layer.buildWithScope(
        makePublicInferenceInfrastructure(lifecycle),
        applicationScope,
      )
      const publicRouter = Context.get(publicInfrastructure, HttpLayerRouter.HttpRouter)
      const publicServer = Context.get(publicInfrastructure, HttpServer.HttpServer)
      yield* publicRouter.addGlobalMiddleware((responseEffect) => Effect.gen(function* () {
        const request = yield* HttpServerRequest.HttpServerRequest
        const host = request.headers.host
        if (host === undefined || !LOCAL_HTTP_HOST.test(host)) {
          return HttpServerResponse.text("Invalid Host header", { status: 421 })
        }
        return withCors(yield* responseEffect, request)
      }))
      yield* publicRouter.add("OPTIONS", "/*", OptionsRouteHandler)
      yield* publicRouter.add("GET", "/health", lifecycle.state.pipe(
        Effect.flatMap((state) => encodeHealthResponse(makeHealthResponse(ACN_VERSION, state)).pipe(
          Effect.flatMap((body) => HttpServerResponse.json(body, {
            status: state._tag === "Ready" ? 200 : 503,
          })),
        )),
        Effect.orDie,
      ))
      yield* publicRouter.prefixed("/inference").add("*", "/*", makeInferenceProxy(icn))
      yield* publicServer.serve(publicRouter.asHttpEffect()).pipe(Effect.provide(publicInfrastructure))
      yield* lifecycle.becomeReady(rpcRouter.asHttpEffect().pipe(Effect.orDie))
      return {
        subscriptions: Context.get(applicationContext, AcnSubscriptions),
        icn,
      }
    })

    const startup = application.pipe(
      Effect.timeout(Duration.minutes(5)),
      Effect.tapError((error) => lifecycle.beginStopping({
        reason: "startup-failed",
        detail: "Magnitude could not prepare local inference",
      }).pipe(Effect.zipRight(Effect.logError("ACN application startup failed", error)))),
    )
    const started = yield* Effect.raceFirst(
      startup.pipe(Effect.disconnect, Effect.map(Option.some)),
      lifecycle.awaitStopping.pipe(Effect.map(() => Option.none())),
    )
    if (Option.isNone(started)) {
      yield* closeApplicationScope
      return
    }
    const { subscriptions, icn } = started.value
    const request = yield* lifecycle.awaitStopping
    yield* Effect.logInfo("ACN shutdown requested").pipe(Effect.annotateLogs({
      reason: request.reason,
      detail: Option.getOrNull(request.safeDetail),
    }))
    yield* boundedShutdownStep(subscriptions.terminate)
    yield* closeApplicationScope
    yield* boundedShutdownStep(icn.shutdown, Duration.seconds(2))
    if (request.reason === "fatal"
      || request.reason === "icn-exited"
      || request.reason === "startup-failed") {
      return yield* new AcnRestartRequired({ reason: request.reason })
    }
  })).pipe(
    Effect.provideService(ProcessGroupController, ProcessGroupControllerLive),
    Effect.provide(BunSqliteDriverLayer),
  )
