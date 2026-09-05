import {
  BunHttpServer,
  BunFileSystem,
  BunPath,
  BunCommandExecutor,
} from "@effect/platform-bun"
import { FetchHttpClient, HttpServerResponse, Socket as PlatformSocket } from "@effect/platform"
import * as HttpLayerRouter from "@effect/platform/HttpLayerRouter"
import * as HttpServer from "@effect/platform/HttpServer"
import * as HttpServerRequest from "@effect/platform/HttpServerRequest"
import { RpcSerialization, RpcServer } from "@effect/rpc"
import {
  Context,
  Cause,
  Data,
  Deferred,
  Duration,
  Effect,
  Exit,
  Fiber,
  Layer,
  Option,
  Queue,
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
  MagnitudeHealthResponseSchema,
  AcnRpcGroup, } from "@magnitudedev/acn-protocol"
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
import { makeAcnIcn } from "./icn"
import { LocalModelSourcesLive } from "./local-model-sources"
import { LocalModelsLive } from "./local-models"
import { LocalModelRemovalsLive } from "./local-model-removals"
import { ModelCatalogLive } from "./model-catalog"
import { ModelCommandsLive } from "./model-commands"
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
import {
  type InferenceProxyTarget,
  makeAnthropicGateway,
  makeCodexGateway,
  codexWebSocketTarget,
  proxyLocalAnthropicInferenceRequest,
  proxyOpenAiInferenceRequest,
} from "./inference-gateway"

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
  readonly reason: "fatal" | "icn-exited" | "startup-failed"
  readonly message: string
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
  "Accept, Authorization, Content-Type, Content-Length, Magnitude-Include-Progress, anthropic-version, anthropic-beta, x-api-key, x-magnitude-acn-id, traceparent, tracestate, baggage, b3, x-b3-traceid, x-b3-spanid, x-b3-parentspanid, x-b3-sampled, x-b3-flags"
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
    "access-control-expose-headers": "request-id, x-request-id",
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
const encodeHealthResponse = Schema.encode(MagnitudeHealthResponseSchema)

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
  const withModelCatalog = Layer.provideMerge(ModelCatalogLive, withCatalog)
  const withCredentials = Layer.provideMerge(
    ProviderCredentialsLive,
    withModelCatalog
  )
  const withCloudUsage = Layer.provideMerge(
    MagnitudeCloudUsageLive,
    withCredentials
  )
  const withModelSlots = Layer.provideMerge(
    ModelSlotControllerLive,
    withCloudUsage
  )
  const withModelCommands = Layer.provideMerge(ModelCommandsLive, withModelSlots)
  const withCustomEndpointReconciliation = Layer.provideMerge(
    CustomEndpointReconcilerLive,
    withModelCommands,
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
  const withModelRemovals = Layer.provideMerge(LocalModelRemovalsLive, withIcn)
  const withSelection = Layer.provideMerge(
    ModelSelectionLive,
    withModelRemovals
  )
  const withCustomEndpoints = Layer.provideMerge(
    CustomEndpointsLive,
    withSelection,
  )
  const withHardware = Layer.provideMerge(
    LocalInferenceHardwareLive,
    withCustomEndpoints
  )
  const withCatalogAdapter = Layer.provideMerge(LocalModelSourcesLive, withHardware)
  const withLocalModels = Layer.provideMerge(LocalModelsLive, withCatalogAdapter)
  const withOfferings = Layer.provideMerge(LocalProviderOfferingsLive, withLocalModels)
  const withOnboarding = Layer.provideMerge(OnboardingLive, withOfferings)
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
  const withMentionSearcher = Layer.provideMerge(FileMentionSearcherLive, services)
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
    BunHttpServer.layer({
      // Candidate coordination endpoints must remain independently bindable so
      // concurrent candidates can reach atomic owner admission. The admitted
      // process opens the stable public application listener separately.
      port: options.port ?? 0,
      hostname: "127.0.0.1",
      idleTimeout: 0,
    }),
    HttpLayerRouter.layer,
    RpcSerialization.layerNdjson,
    TracingLayer
  )
}

const makePublicInfrastructure = (lifecycle: AcnServiceLifecycleApi) =>
  Layer.mergeAll(
    Layer.succeed(AcnServiceLifecycle, lifecycle),
    BunHttpServer.layer({
      port: ACN_PUBLIC_PORT,
      hostname: "127.0.0.1",
      // Unary RPCs can run throughout model acquisition/loading before emitting
      // any response bytes. Operation duration must not become an idle timeout.
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

export const proxyInferenceWebRequest = async (
  source: Request,
  icn: InferenceProxyTarget,
  fetchTarget: typeof fetch = fetch,
  signal: AbortSignal = source.signal,
): Promise<Response> => {
  return proxyOpenAiInferenceRequest(source, icn, fetchTarget, signal)
}

const makeCodexWebSocketProxy = (
  request: HttpServerRequest.HttpServerRequest,
  source: Request,
  icn: InferenceProxyTarget,
) => Effect.scoped(Effect.gen(function* () {
  const incoming = yield* request.upgrade
  type ClientEvent =
    | { readonly _tag: "Message"; readonly message: string | Uint8Array }
    | { readonly _tag: "Closed" }
  const messages = yield* Queue.unbounded<ClientEvent>()
  yield* incoming.runRaw((message) => Queue.offer(messages, { _tag: "Message", message })).pipe(
    Effect.onExit(() => Queue.offer(messages, { _tag: "Closed" })),
    Effect.forkScoped,
  )
  const incomingWriter = yield* incoming.writer
  const BunWebSocket = WebSocket as unknown as new (
    url: string | URL,
    options: { readonly headers: Readonly<Record<string, string>> },
  ) => WebSocket
  let active: {
    readonly key: string
    readonly scope: Scope.CloseableScope
    readonly writer: (
      chunk: Uint8Array | string | PlatformSocket.CloseEvent,
    ) => Effect.Effect<void, PlatformSocket.SocketError>
  } | undefined
  while (true) {
    const event = yield* Queue.take(messages)
    if (event._tag === "Closed") break
    const target = codexWebSocketTarget(event.message, source.headers, icn)
    if (target._tag === "Invalid") {
      yield* incomingWriter(new PlatformSocket.CloseEvent(1008, target.message))
      break
    }
    const key = `${target.route}:${target.url.href}`
    if (active?.key !== key) {
      if (active !== undefined) yield* Scope.close(active.scope, Exit.void)
      const outgoingScope = yield* Scope.make()
      yield* Effect.addFinalizer(() => Scope.close(outgoingScope, Exit.void))
      const outgoing = yield* PlatformSocket.fromWebSocket(Effect.acquireRelease(
        Effect.sync(() => new BunWebSocket(target.url, {
          headers: Object.fromEntries(target.headers.entries()),
        })),
        (socket) => Effect.sync(() => socket.close(1000)),
      )).pipe(Scope.extend(outgoingScope))
      const writer = yield* outgoing.writer
      yield* outgoing.runRaw((message) => incomingWriter(message)).pipe(
        Effect.onExit((exit) => Exit.isInterrupted(exit)
          ? Effect.void
          : incomingWriter(new PlatformSocket.CloseEvent(
            1011,
            "Upstream WebSocket closed",
          )).pipe(Effect.ignore)),
        Effect.forkIn(outgoingScope),
      )
      active = { key, scope: outgoingScope, writer }
    }
    const sent = yield* active.writer(target.firstMessage).pipe(Effect.either)
    if (sent._tag === "Left") {
      yield* incomingWriter(new PlatformSocket.CloseEvent(
        1011,
        "Unable to write to upstream WebSocket",
      )).pipe(Effect.ignore)
      break
    }
  }
  return HttpServerResponse.empty()
})).pipe(
  Effect.catchAllCause((cause) => Effect.logDebug(
    "Codex WebSocket proxy closed",
    Cause.pretty(cause),
  ).pipe(Effect.as(HttpServerResponse.empty()))),
)

const makeInferenceProxy = (
  icn: InferenceProxyTarget,
  protocol: "openai" | "anthropic" | "codex" | "claude-code",
) => {
  const anthropicGateway = protocol === "claude-code"
    ? makeAnthropicGateway(icn)
    : undefined
  const codexGateway = protocol === "codex" ? makeCodexGateway(icn) : undefined
  return (request: HttpServerRequest.HttpServerRequest) => Effect.gen(function* () {
    // The wildcard proxy route is also the most specific OPTIONS route. Handle
    // browser preflight locally instead of forwarding it to an ICN operation.
    if (request.method === "OPTIONS") return yield* OptionsRouteHandler(request)
    const source = request.source
    if (!(source instanceof Request)) {
      return HttpServerResponse.text("Unsupported request transport", { status: 500 })
    }
    if (protocol === "codex" && source.headers.get("upgrade")?.toLowerCase() === "websocket") {
      return yield* makeCodexWebSocketProxy(request, source, icn)
    }
    const response = anthropicGateway !== undefined
      ? yield* anthropicGateway.route(source).pipe(Effect.either)
      : codexGateway !== undefined
        ? yield* codexGateway.route(source).pipe(Effect.either)
        : yield* Effect.tryPromise({
          try: (signal) => protocol === "openai"
            ? proxyInferenceWebRequest(source, icn, fetch, signal)
            : proxyLocalAnthropicInferenceRequest(source, icn, fetch, signal),
          catch: (cause) => new InferenceProxyFailed({ cause }),
        }).pipe(Effect.either)
    if (response._tag === "Right") {
      return HttpServerResponse.fromWeb(response.right)
    }
    yield* Effect.logError("Inference gateway failed", response.left)
    const requestId = `req_acn_gateway_${Date.now()}`
    const body = protocol === "anthropic" || protocol === "claude-code"
      ? {
          type: "error",
          error: {
            type: "api_error",
            message: "Local inference gateway unavailable",
          },
          request_id: requestId,
        }
      : {
          error: {
            message: "Local inference gateway unavailable",
            type: "server_error",
            param: null,
            code: "gateway_unavailable",
          },
        }
    return yield* HttpServerResponse.json(body, {
      status: 502,
      headers: { "request-id": requestId },
    }).pipe(Effect.orDie)
  })
}

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

const installAcnHealthRoutes = (
  router: HttpLayerRouter.HttpRouter,
  lifecycle: AcnServiceLifecycleApi,
) => Effect.gen(function* () {
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
})

export const installAcnControlRoutes = (
  router: HttpLayerRouter.HttpRouter,
  lifecycle: AcnServiceLifecycleApi,
) => Effect.gen(function* () {
  yield* installAcnHealthRoutes(router, lifecycle)
  yield* router.add("POST", "/shutdown", lifecycle.beginStopping({ reason: "administrative" }).pipe(
    Effect.as(HttpServerResponse.empty({ status: 202 })),
  ))
})

export const installAcnPublicRoutes = (
  router: HttpLayerRouter.HttpRouter,
  lifecycle: AcnServiceLifecycleApi,
  icn: InferenceProxyTarget,
) => Effect.gen(function* () {
  yield* installAcnHealthRoutes(router, lifecycle)
  yield* router.add("POST", "/rpc", Effect.gen(function* () {
    const request = yield* HttpServerRequest.HttpServerRequest
    return request.headers["x-magnitude-acn-id"] === ACN_INSTANCE_ID
      ? yield* lifecycle.dispatchRpc
      : HttpServerResponse.empty({ status: 409 })
  }))
  yield* router.prefixed("/inference/v1/proxies/codex").add(
    "*", "/*", makeInferenceProxy(icn, "codex"),
  )
  yield* router.prefixed("/inference/v1").add(
    "*", "/*", makeInferenceProxy(icn, "openai"),
  )
  yield* router.prefixed("/inference/anthropic/proxies/claude-code").add(
    "*", "/*", makeInferenceProxy(icn, "claude-code"),
  )
  yield* router.prefixed("/inference/anthropic").add(
    "*", "/*", makeInferenceProxy(icn, "anthropic"),
  )
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

    yield* installAcnControlRoutes(router, lifecycle)
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
      yield* RpcServer.make(AcnRpcGroup).pipe(
        Effect.provideService(RpcServer.Protocol, protocol),
        Effect.provide(applicationContext),
        Effect.forkIn(applicationScope),
      )
      if (Option.isSome(builtServices.introspector)) {
        yield* installAcnIntrospectionRoutes(router, builtServices.introspector.value)
      }
      const icn = Context.get(applicationContext, IcnProcess)
      const publicInfrastructure = yield* Layer.buildWithScope(
        makePublicInfrastructure(lifecycle),
        applicationScope,
      )
      const publicRouter = Context.get(publicInfrastructure, HttpLayerRouter.HttpRouter)
      const publicServer = Context.get(publicInfrastructure, HttpServer.HttpServer)
      yield* installAcnPublicRoutes(publicRouter, lifecycle, icn)
      yield* publicServer.serve(publicRouter.asHttpEffect()).pipe(Effect.provide(publicInfrastructure))
      yield* lifecycle.becomeReady(rpcRouter.asHttpEffect().pipe(Effect.orDie))
      return {
        subscriptions: Context.get(applicationContext, AcnSubscriptions),
        icn,
      }
    })

    const startup = application.pipe(
      Effect.timeout(Duration.minutes(5)),
      Effect.tapErrorCause((cause) => lifecycle.beginStopping({
        reason: "startup-failed",
        detail: Cause.pretty(cause),
      }).pipe(Effect.zipRight(Effect.logError("ACN application startup failed", cause)))),
    )
    const started = yield* Effect.raceFirst(
      startup.pipe(Effect.disconnect, Effect.map(Option.some)),
      lifecycle.awaitStopping.pipe(Effect.map(() => Option.none())),
    )
    const request = yield* lifecycle.awaitStopping
    if (Option.isNone(started)) {
      yield* closeApplicationScope
    } else {
      const { subscriptions, icn } = started.value
      yield* Effect.logInfo("ACN shutdown requested").pipe(Effect.annotateLogs({
        reason: request.reason,
        detail: Option.getOrNull(request.safeDetail),
      }))
      yield* boundedShutdownStep(subscriptions.terminate)
      yield* closeApplicationScope
      yield* boundedShutdownStep(icn.shutdown, Duration.seconds(2))
    }
    if (request.reason === "fatal"
      || request.reason === "icn-exited"
      || request.reason === "startup-failed") {
      return yield* new AcnRestartRequired({
        reason: request.reason,
        message: Option.getOrElse(request.safeDetail, () => `ACN stopped because ${request.reason}`),
      })
    }
  })).pipe(
    Effect.provideService(ProcessGroupController, ProcessGroupControllerLive),
    Effect.provide(BunSqliteDriverLayer),
  )
