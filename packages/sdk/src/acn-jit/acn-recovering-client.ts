import * as HttpClient from "@effect/platform/HttpClient"
import {
  ClientIdSchema,
  MagnitudeRpcs,
  type AcnIdentity,
  type ClientId,
  type ClientLeaseMutationResult,
  type ModelSlotsState,
} from "@magnitudedev/acn-protocol"
import { compareAcnIdentities } from "@magnitudedev/acn-protocol/acn-identity"
import { RpcClient, RpcClientError } from "@effect/rpc"
import {
  Cause,
  Data,
  Deferred,
  Duration,
  Effect,
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
import { SDK_VERSION } from "../version"
import type { AcnClient } from "../protocol"
import { AcnEnsurer, runAcnEnsure, type ReadyAcn } from "./acn-ensurer"
import { acnSubscriptionProtocol } from "./acn-subscription-protocol"
import { type AcnEnsuranceError, AcnEnsuranceFailed } from "./errors"
import { makeAcnLifecycle, type AcnLifecycle, type AcnLifecycleState } from "./lifecycle"

const CLIENT_LEASE_RENEWAL_INTERVAL = Duration.seconds(15)
const CLIENT_LEASE_RELEASE_TIMEOUT = Duration.seconds(2)
const CLIENT_CLOSE_OBSERVATION_TIMEOUT = Duration.seconds(2)

type ReleaseClientLeaseThrough = (client: ClientLeaseRpcClient) => Effect.Effect<
  ClientLeaseMutationResult,
  RpcClientError.RpcClientError | Cause.TimeoutException
>
type ClientLeaseRpcClient = Pick<AcnClient, "RenewClientLease" | "ReleaseClientLease">

export interface AcnClientLeaseOwner {
  readonly clientId: ClientId
  readonly start: Effect.Effect<void>
  readonly stop: Effect.Effect<void>
  readonly releaseThrough: ReleaseClientLeaseThrough
}

export const makeAcnClientLeaseOwner = (
  clientId: ClientId,
  client: ClientLeaseRpcClient,
): Effect.Effect<AcnClientLeaseOwner, never, Scope.Scope> =>
  Effect.gen(function* () {
    const released = yield* Ref.make(Option.none<ClientLeaseMutationResult>())
    const releaseLock = yield* Effect.makeSemaphore(1)
    const started = yield* Deferred.make<void>()
    const renew = client.RenewClientLease({ clientId }).pipe(
      Effect.tapError((error) => Effect.logWarning("Failed to renew ACN client lease").pipe(
        Effect.annotateLogs({ clientId, error: String(error) }),
      )),
      Effect.ignore,
    )
    const heartbeat = yield* Deferred.await(started).pipe(
      Effect.zipRight(renew.pipe(Effect.repeat(Schedule.spaced(CLIENT_LEASE_RENEWAL_INTERVAL)))),
      Effect.forkScoped,
    )
    const start = Deferred.succeed(started, undefined).pipe(Effect.asVoid)
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
    return { clientId, start, stop, releaseThrough }
  })

export interface AcnStartup {
  readonly state: AcnLifecycle
  readonly prepare: Effect.Effect<AcnLifecycleState>
  readonly retry: Effect.Effect<void, AcnEnsuranceError>
}

export interface AcnClientCloseReport {
  readonly modelSlots: ModelSlotsState
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
  readonly identity: AcnIdentity
  readonly selected: Option.Option<ReadyAcn>
}

class AcnRuntimeClosed extends Data.TaggedError("AcnRuntimeClosed") {}
type SelectionError = AcnEnsuranceError | AcnRuntimeClosed
const runtimeClosed = () => new AcnRuntimeClosed()

const sameReadyOccurrence = (left: ReadyAcn, right: ReadyAcn): boolean =>
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
  AcnEnsurer | HttpClient.HttpClient | Scope.Scope
> => Effect.gen(function* () {
  const ensurer = yield* AcnEnsurer
  const httpClient = yield* HttpClient.HttpClient
  const runtimeScope = yield* Scope.Scope
  const selectionScope = yield* Scope.make()
  yield* Effect.addFinalizer(() => Scope.close(selectionScope, Exit.void))
  const lifecycle = yield* makeAcnLifecycle()
  const association = yield* SubscriptionRef.make<AcnAssociation>({
    identity: SDK_VERSION,
    selected: Option.none(),
  })
  const admission = yield* Effect.makeSemaphore(1)
  const activeSelection = yield* Ref.make(
    Option.none<Deferred.Deferred<ReadyAcn, SelectionError>>(),
  )
  const open = yield* Ref.make(true)
  yield* Effect.addFinalizer(() => Ref.set(open, false))
  const clientId = ClientIdSchema.make(globalThis.crypto.randomUUID())

  // The lease client depends on the recovering protocol built below. No
  // selection is admitted until this is replaced with the inert owner's start.
  let installLease: Effect.Effect<void> = Effect.dieMessage("lease installer was not initialized")

  const finishSelection = (
    deferred: Deferred.Deferred<ReadyAcn, SelectionError>,
    exit: Exit.Exit<ReadyAcn, AcnEnsuranceError>,
  ) => admission.withPermits(1)(Effect.gen(function* () {
    const current = yield* Ref.get(activeSelection)
    if (Option.isNone(current) || current.value !== deferred) return
    yield* Ref.set(activeSelection, Option.none())
    if (!(yield* Ref.get(open))) {
      yield* Deferred.fail(deferred, runtimeClosed())
      return
    }
    if (Exit.isFailure(exit)) {
      const failure = Option.getOrUndefined(Cause.failureOption(exit.cause))
      if (failure !== undefined) yield* lifecycle.fail(failure)
      yield* Deferred.done(deferred, exit)
      return
    }
    const ready = exit.value
    const previous = yield* SubscriptionRef.get(association)
    const identity = compareAcnIdentities(ready.identity, previous.identity) > 0
      ? ready.identity
      : previous.identity
    yield* SubscriptionRef.set(association, { identity, selected: Option.some(ready) })
    yield* lifecycle.ready
    yield* installLease
    yield* Deferred.succeed(deferred, ready)
  })).pipe(Effect.uninterruptible)

  const launchSelection = (
    deferred: Deferred.Deferred<ReadyAcn, SelectionError>,
    identity: AcnIdentity,
  ) => Effect.uninterruptibleMask((restore) =>
    restore(runAcnEnsure(ensurer.ensure({ minimumIdentity: identity }).pipe(
      Stream.tap((event) => event._tag === "Observation"
        ? lifecycle.report(event.observation)
        : Effect.void),
    ))).pipe(
      Effect.exit,
      Effect.flatMap((exit) => finishSelection(deferred, exit)),
    ),
  )

  const admitSelectionUnlocked: Effect.Effect<
    Effect.Effect<ReadyAcn, SelectionError>
  > = Effect.gen(function* () {
    if (!(yield* Ref.get(open))) return yield* Effect.succeed(Effect.fail(runtimeClosed()))
    const selected = (yield* SubscriptionRef.get(association)).selected
    if (Option.isSome(selected)) return yield* Effect.succeed(
      Effect.succeed(selected.value) as Effect.Effect<ReadyAcn, SelectionError>,
    )
    const active = yield* Ref.get(activeSelection)
    if (Option.isSome(active)) return yield* Effect.succeed(Deferred.await(active.value))
    const deferred = yield* Deferred.make<ReadyAcn, SelectionError>()
    const identity = (yield* SubscriptionRef.get(association)).identity
    yield* Ref.set(activeSelection, Option.some(deferred))
    yield* Effect.forkIn(launchSelection(deferred, identity), selectionScope)
    return yield* Effect.succeed(Deferred.await(deferred))
  })

  const endpoint: Effect.Effect<ReadyAcn, SelectionError> = Effect.flatten(
    admission.withPermits(1)(admitSelectionUnlocked),
  )

  const recover = (failed: ReadyAcn): Effect.Effect<ReadyAcn, SelectionError> =>
    Effect.flatten(admission.withPermits(1)(Effect.gen(function* () {
      if (!(yield* Ref.get(open))) return yield* Effect.succeed(Effect.fail(runtimeClosed()))
      const current = yield* SubscriptionRef.get(association)
      if (Option.isSome(current.selected) && !sameReadyOccurrence(current.selected.value, failed)) {
        return yield* Effect.succeed(
          Effect.succeed(current.selected.value) as Effect.Effect<ReadyAcn, SelectionError>,
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
  })

  const leaseProtocolContext = yield* Layer.build(recoveringProtocolLayer)
  const leaseClient = yield* RpcClient.make(MagnitudeRpcs).pipe(Effect.provide(leaseProtocolContext))
  const owner = yield* makeAcnClientLeaseOwner(clientId, leaseClient)
  installLease = owner.start

  yield* lifecycle.report({ _tag: "Starting", phase: "Discovering" })
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

  const retry = lifecycle.report({ _tag: "Starting", phase: "Discovering" }).pipe(
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
        const closeProtocolContext = yield* Layer.buildWithScope(
          jitRecoveringProtocolLayer({
            endpoint: Effect.succeed(selected.value),
            recover: () => Effect.fail(runtimeClosed()),
            rpcPath: "/rpc",
            streamProtocol: acnSubscriptionProtocol,
            isEndpointRetirementExit: isInterruptedExit,
            classifyInfraError: unavailableError,
          }).pipe(Layer.provide(Layer.succeed(HttpClient.HttpClient, httpClient))),
          runtimeScope,
        )
        const closeClient = yield* RpcClient.make(MagnitudeRpcs).pipe(
          Effect.provide(closeProtocolContext),
          Effect.provideService(Scope.Scope, runtimeScope),
        )
        const modelSlots = yield* closeClient.GetModelSlots({}).pipe(
          Effect.map((result) => result.state),
          Effect.timeout(CLIENT_CLOSE_OBSERVATION_TIMEOUT),
          resultOption,
        )
        const release = yield* resultOption(owner.releaseThrough(closeClient))
        const result = Option.all({ modelSlots, release }).pipe(
          Option.map(({ modelSlots, release }) => ({
            modelSlots,
            connectedClientCount: release.connectedClientCount,
          })),
        )
        yield* Ref.set(closeResult, Option.some(result))
        return result
      }),
    })),
  ))

  return {
    identity: SubscriptionRef.get(association).pipe(Effect.map((current) => current.identity)),
    identityChanges: association.changes.pipe(
      Stream.map((current) => current.identity),
      Stream.changes,
    ),
    startup: { state: lifecycle, prepare, retry },
    close,
    protocolLayer: recoveringProtocolLayer,
  }
})
