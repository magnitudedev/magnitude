import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import {
  ClientIdSchema,
  AcnRpc,
  AcnBoundary,
  AcnReady,
  AcnRpcRecoveryPolicyTag,
  type AcnInstance,
  type AcnIdentity,
  type AcnTarget,
  type ClientId,
  type ClientLeaseMutationResult,
} from "@magnitudedev/acn-protocol"
import { RpcClient, RpcClientError, RpcSerialization } from "@effect/rpc"
import {
  Cause,
  Context,
  Data,
  Deferred,
  Duration,
  Effect,
  Either,
  Exit,
  Fiber,
  Layer,
  Option,
  Ref,
  Schedule,
  Scope,
  Stream,
  SubscriptionRef,
} from "effect"
import { isInterruptedExit, recoveringProtocolLayer as jitRecoveringProtocolLayer } from "../jit-rpc"
import { SDK_ACN_TARGET } from "../version"
import {
  ACN_ENSURE_TIMEOUT,
  AcnInstanceManager,
  runAcnEnsure,
} from "./acn-instance-manager"
import { acnSubscriptionProtocol } from "./acn-subscription-protocol"
import { type AcnEnsuranceError, AcnEnsuranceFailed } from "./errors"
import { makeAcnLifecycle, type AcnLifecycle, type AcnLifecycleState } from "./lifecycle"

type ReadyInstance = AcnInstance<AcnReady>

const makeAcnRpcClient = () => AcnRpc.makeRpcClient(AcnBoundary)
const recoveryPolicy = (tag: string) => {
  const operation = AcnRpc.operation(AcnBoundary, tag)
  if (operation === undefined) throw new TypeError(`Unknown ACN operation ${tag}`)
  const policy = Context.getOption(operation.annotations, AcnRpcRecoveryPolicyTag)
  if (Option.isNone(policy)) throw new TypeError(`Finite ACN operation ${tag} has no recovery policy`)
  return policy.value
}

const CLIENT_LEASE_RENEWAL_INTERVAL = Duration.seconds(15)
const CLIENT_LEASE_ESTABLISH_TIMEOUT = Duration.seconds(5)
const CLIENT_LEASE_ESTABLISH_RETRY_DELAY = Duration.millis(250)
const CLIENT_LEASE_RELEASE_TIMEOUT = Duration.seconds(2)

type ReleaseClientLeaseThrough = (client: ClientLeaseRpcClient) => Effect.Effect<
  ClientLeaseMutationResult,
  RpcClientError.RpcClientError | Cause.TimeoutException
>
/** The transport-internal RPC client; lease and close observation run through it. */
type AcnClient = Effect.Effect.Success<ReturnType<typeof makeAcnRpcClient>>
type ClientLeaseRpcClient = Pick<AcnClient, "RenewClientLease" | "ReleaseClientLease">

export interface AcnClientLeaseOwner {
  readonly clientId: ClientId
  readonly establishThrough: (
    client: ClientLeaseRpcClient,
  ) => Effect.Effect<void, RpcClientError.RpcClientError>
  readonly stop: Effect.Effect<void>
  readonly releaseThrough: ReleaseClientLeaseThrough
}

export const makeAcnClientLeaseOwner = (
  clientId: ClientId,
): Effect.Effect<AcnClientLeaseOwner, never, Scope.Scope> =>
  Effect.gen(function* () {
    const released = yield* Ref.make(Option.none<ClientLeaseMutationResult>())
    const releaseLock = yield* Effect.makeSemaphore(1)
    const started = yield* Deferred.make<void>()
    const currentClient = yield* Ref.make(Option.none<ClientLeaseRpcClient>())
    const renew = Ref.get(currentClient).pipe(Effect.flatMap(Option.match({
      onNone: () => Effect.void,
      onSome: (client) => client.RenewClientLease({ clientId }).pipe(
      Effect.tapError((error) => Effect.logWarning("Failed to renew ACN client lease").pipe(
        Effect.annotateLogs({ clientId, error: String(error) }),
      )),
      Effect.ignore,
      ),
    })))
    const heartbeat = yield* Deferred.await(started).pipe(
      Effect.zipRight(Effect.sleep(CLIENT_LEASE_RENEWAL_INTERVAL)),
      Effect.zipRight(renew.pipe(Effect.repeat(Schedule.spaced(CLIENT_LEASE_RENEWAL_INTERVAL)))),
      Effect.forkScoped,
    )
    const establishThrough: AcnClientLeaseOwner["establishThrough"] = (client) =>
      client.RenewClientLease({ clientId }).pipe(
        Effect.tap(() => Ref.set(currentClient, Option.some(client))),
        Effect.tap(() => Deferred.succeed(started, undefined)),
        Effect.asVoid,
      )
    const stop = Fiber.interrupt(heartbeat)
    const releaseThrough: ReleaseClientLeaseThrough = (releaseClient) =>
      releaseLock.withPermits(1)(Ref.get(released).pipe(
        Effect.flatMap(Option.match({
          onSome: Effect.succeed,
          onNone: () => stop.pipe(
            Effect.zipRight(releaseClient.ReleaseClientLease({ clientId }).pipe(
              Effect.timeout(CLIENT_LEASE_RELEASE_TIMEOUT),
            )),
            Effect.tap((result) => Ref.set(released, Option.some(result))),
          ),
        })),
      ))
    yield* Effect.addFinalizer(() => stop)
    return { clientId, establishThrough, stop, releaseThrough }
  })

export interface AcnStartup {
  readonly state: AcnLifecycle
  readonly prepare: Effect.Effect<AcnLifecycleState>
  readonly retry: Effect.Effect<void, AcnEnsuranceError>
}

export interface AcnClientCloseReport {
  readonly connectedClientCount: number
}
export type AcnClientCloseResult = Option.Option<AcnClientCloseReport>

export interface AcnJitRuntime {
  readonly identity: Effect.Effect<AcnIdentity>
  readonly identityChanges: Stream.Stream<AcnIdentity>
  readonly protocolLayer: Layer.Layer<RpcClient.Protocol, never, HttpClient.HttpClient>
  readonly close: Effect.Effect<AcnClientCloseResult>
  readonly startup: AcnStartup
}

interface AcnAssociation {
  readonly target: AcnTarget
  readonly selected: Option.Option<ReadyInstance>
}

class AcnRuntimeClosed extends Data.TaggedError("AcnRuntimeClosed") {}
type SelectionError = AcnEnsuranceError | AcnRuntimeClosed
const runtimeClosed = () => new AcnRuntimeClosed()

const sameReadyOccurrence = (left: ReadyInstance, right: ReadyInstance): boolean =>
  left.id === right.id &&
  left.pid === right.pid &&
  left.processStartIdentity === right.processStartIdentity

const resultOption = <A, E, R>(effect: Effect.Effect<A, E, R>): Effect.Effect<Option.Option<A>, never, R> =>
  effect.pipe(
    Effect.exit,
    Effect.map((exit) => Exit.isSuccess(exit) ? Option.some(exit.value) : Option.none()),
  )

const { RpcClientError: TransportError } = RpcClientError
const unavailableError = (cause: SelectionError): RpcClientError.RpcClientError =>
  new TransportError({
    reason: "Unknown",
    message: cause._tag === "AcnRuntimeClosed"
      ? "ACN client runtime is closed"
      : `ACN unavailable: ${cause._tag}${"reason" in cause ? `: ${String(cause.reason)}` : ""}`,
    cause,
  })

export const makeAcnJitRuntime = (): Effect.Effect<
  AcnJitRuntime,
  never,
  AcnInstanceManager | HttpClient.HttpClient | Scope.Scope
> => Effect.gen(function* () {
  const manager = yield* AcnInstanceManager
  const httpClient = yield* HttpClient.HttpClient
  const selectionScope = yield* Scope.make()
  yield* Effect.addFinalizer(() => Scope.close(selectionScope, Exit.void))
  const lifecycle = yield* makeAcnLifecycle()
  const association = yield* SubscriptionRef.make<AcnAssociation>({
    target: SDK_ACN_TARGET,
    selected: Option.none(),
  })
  const admission = yield* Effect.makeSemaphore(1)
  const activeSelection = yield* Ref.make(
    Option.none<Deferred.Deferred<ReadyInstance, SelectionError>>(),
  )
  const open = yield* Ref.make(true)
  yield* Effect.addFinalizer(() => Ref.set(open, false))
  const clientId = ClientIdSchema.make(globalThis.crypto.randomUUID())
  const owner = yield* makeAcnClientLeaseOwner(clientId)

  const exactClient = (instance: ReadyInstance) => Layer.buildWithScope(
    RpcClient.layerProtocolHttp({
      url: `${instance.url}/rpc`,
      transformClient: HttpClient.mapRequest(
        HttpClientRequest.setHeader("x-magnitude-acn-id", instance.id),
      ),
    }).pipe(
      Layer.provide(RpcSerialization.layerNdjson),
      Layer.provide(Layer.succeed(HttpClient.HttpClient, httpClient)),
    ),
    selectionScope,
  ).pipe(
    Effect.flatMap((context) => makeAcnRpcClient().pipe(
      Effect.provide(context),
      Effect.provideService(Scope.Scope, selectionScope),
    )),
  )

  const finishFailedSelection = (
    deferred: Deferred.Deferred<ReadyInstance, SelectionError>,
    cause: Cause.Cause<AcnEnsuranceError>,
  ) => admission.withPermits(1)(Effect.gen(function* () {
    const current = yield* Ref.get(activeSelection)
    if (Option.isNone(current) || current.value !== deferred) return
    yield* Ref.set(activeSelection, Option.none())
    if (!(yield* Ref.get(open))) {
      yield* Deferred.fail(deferred, runtimeClosed())
      return
    }
    const failure = Option.getOrUndefined(Cause.failureOption(cause))
    if (failure !== undefined) yield* lifecycle.fail(failure)
    yield* Deferred.failCause(deferred, cause)
  })).pipe(Effect.uninterruptible)

  const finishReadySelection = (
    deferred: Deferred.Deferred<ReadyInstance, SelectionError>,
    ready: ReadyInstance,
    client: ClientLeaseRpcClient,
  ): Effect.Effect<boolean> => admission.withPermits(1)(
    Effect.uninterruptibleMask((restore) => Effect.gen(function* () {
      const current = yield* Ref.get(activeSelection)
      if (Option.isNone(current) || current.value !== deferred) return true
      if (!(yield* Ref.get(open))) {
        yield* Ref.set(activeSelection, Option.none())
        yield* Deferred.fail(deferred, runtimeClosed())
        return true
      }
      const established = yield* restore(owner.establishThrough(client).pipe(
        Effect.timeout(CLIENT_LEASE_ESTABLISH_TIMEOUT),
        Effect.either,
      ))
      if (Either.isLeft(established)) return false
      const previous = yield* SubscriptionRef.get(association)
      const target = ready.revision > previous.target.revision
        ? { revision: ready.revision, identity: ready.identity }
        : previous.target
      yield* Ref.set(activeSelection, Option.none())
      yield* SubscriptionRef.set(association, { target, selected: Option.some(ready) })
      yield* lifecycle.ready
      yield* Deferred.succeed(deferred, ready)
      return true
    })),
  )

  const launchSelection = (
    deferred: Deferred.Deferred<ReadyInstance, SelectionError>,
    target: AcnTarget,
  ): Effect.Effect<void> => Effect.suspend(() => runAcnEnsure(manager.ensure({ target }).pipe(
    Stream.tap((event) => event._tag === "Observation"
      ? lifecycle.report(event.observation)
      : Effect.void),
  )).pipe(
    Effect.exit,
    Effect.flatMap((exit) => {
      if (Exit.isFailure(exit)) return finishFailedSelection(deferred, exit.cause)
      return exactClient(exit.value).pipe(
        Effect.flatMap((client) => finishReadySelection(deferred, exit.value, client)),
        Effect.flatMap((finished) => finished
          ? Effect.void
          : Effect.sleep(CLIENT_LEASE_ESTABLISH_RETRY_DELAY).pipe(
            Effect.zipRight(launchSelection(deferred, target)),
          )),
      )
    }),
  ))

  const admitSelectionUnlocked: Effect.Effect<
    Effect.Effect<ReadyInstance, SelectionError>
  > = Effect.gen(function* () {
    if (!(yield* Ref.get(open))) return yield* Effect.succeed(Effect.fail(runtimeClosed()))
    const selected = (yield* SubscriptionRef.get(association)).selected
    if (Option.isSome(selected)) return yield* Effect.succeed(
      Effect.succeed(selected.value) as Effect.Effect<ReadyInstance, SelectionError>,
    )
    const active = yield* Ref.get(activeSelection)
    if (Option.isSome(active)) return yield* Effect.succeed(Deferred.await(active.value))
    const deferred = yield* Deferred.make<ReadyInstance, SelectionError>()
    const target = (yield* SubscriptionRef.get(association)).target
    yield* Ref.set(activeSelection, Option.some(deferred))
    const selection = launchSelection(deferred, target).pipe(
      Effect.timeoutFail({
        duration: ACN_ENSURE_TIMEOUT,
        onTimeout: () => new AcnEnsuranceFailed({
          reason: "ACN client selection did not converge within its absolute deadline",
        }),
      }),
      Effect.catchAll((error) => finishFailedSelection(deferred, Cause.fail(error))),
    )
    yield* Effect.forkIn(selection, selectionScope)
    return yield* Effect.succeed(Deferred.await(deferred))
  })

  const endpoint: Effect.Effect<ReadyInstance, SelectionError> = Effect.flatten(
    admission.withPermits(1)(admitSelectionUnlocked),
  )

  const recover = (failed: ReadyInstance): Effect.Effect<ReadyInstance, SelectionError> =>
    Effect.flatten(admission.withPermits(1)(Effect.gen(function* () {
      if (!(yield* Ref.get(open))) return yield* Effect.succeed(Effect.fail(runtimeClosed()))
      const current = yield* SubscriptionRef.get(association)
      if (Option.isSome(current.selected) && !sameReadyOccurrence(current.selected.value, failed)) {
        return yield* Effect.succeed(
          Effect.succeed(current.selected.value) as Effect.Effect<ReadyInstance, SelectionError>,
        )
      }
      if (Option.isSome(current.selected) && sameReadyOccurrence(current.selected.value, failed)) {
        yield* SubscriptionRef.set(association, { ...current, selected: Option.none() })
      }
      return yield* admitSelectionUnlocked
    })))

  const recoveringProtocolLayer = jitRecoveringProtocolLayer({
    endpoint,
    recover,
    rpcPath: "/rpc",
    streamProtocol: acnSubscriptionProtocol,
    isEndpointRetirementExit: isInterruptedExit,
    classifyInfraError: unavailableError,
    recoveryPolicy,
  })

  yield* lifecycle.report({ _tag: "Starting", phase: "PreparingAcn" })
  yield* Effect.forkIn(endpoint.pipe(Effect.ignore), selectionScope)

  const prepare = lifecycle.get.pipe(
    Effect.flatMap((state) => state._tag === "Checking"
      ? lifecycle.changes.pipe(
        Stream.filter((next) => next._tag !== "Checking"),
        Stream.runHead,
        Effect.flatMap(Option.match({
          onNone: () => Effect.dieMessage("ACN lifecycle ended before startup became visible"),
          onSome: Effect.succeed,
        })),
      )
      : Effect.succeed(state)),
  )

  const retry = lifecycle.report({ _tag: "Starting", phase: "PreparingAcn" }).pipe(
    Effect.zipRight(endpoint),
    Effect.mapError((error) => error._tag === "AcnRuntimeClosed"
      ? new AcnEnsuranceFailed({ reason: "ACN client runtime is closed" })
      : error),
    Effect.asVoid,
  )

  const closeResult = yield* Ref.make(Option.none<AcnClientCloseResult>())
  const closeLock = yield* Effect.makeSemaphore(1)
  const close: AcnJitRuntime["close"] = closeLock.withPermits(1)(Ref.get(closeResult).pipe(
    Effect.flatMap(Option.match({
      onSome: Effect.succeed,
      onNone: () => Effect.gen(function* () {
        yield* admission.withPermits(1)(Ref.set(open, false))
        yield* Scope.close(selectionScope, Exit.void)
        yield* owner.stop
        const selected = (yield* SubscriptionRef.get(association)).selected
        if (Option.isNone(selected)) {
          const result = Option.none<AcnClientCloseReport>()
          yield* Ref.set(closeResult, Option.some(result))
          return result
        }
        const result = yield* Effect.scoped(Effect.gen(function* () {
          const closeClient = yield* makeAcnRpcClient()
          const release = yield* resultOption(owner.releaseThrough(closeClient))
          return Option.map(release, ({ connectedClientCount }) => ({ connectedClientCount }))
        }).pipe(
          Effect.provide(jitRecoveringProtocolLayer({
            endpoint: Effect.succeed(selected.value),
            recover: () => Effect.fail(runtimeClosed()),
            rpcPath: "/rpc",
            streamProtocol: acnSubscriptionProtocol,
            isEndpointRetirementExit: isInterruptedExit,
            classifyInfraError: unavailableError,
            recoveryPolicy,
          }).pipe(Layer.provide(Layer.succeed(HttpClient.HttpClient, httpClient))),
        )))
        yield* Ref.set(closeResult, Option.some(result))
        return result
      }),
    })),
  ))

  return {
    identity: SubscriptionRef.get(association).pipe(Effect.map((current) => current.target.identity)),
    identityChanges: association.changes.pipe(
      Stream.map((current) => current.target.identity),
      Stream.changes,
    ),
    startup: { state: lifecycle, prepare, retry },
    close,
    protocolLayer: recoveringProtocolLayer,
  }
})
