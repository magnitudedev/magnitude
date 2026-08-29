import * as HttpClient from "@effect/platform/HttpClient"
import {
  AcnRpc,
  AcnBoundary,
  AcnReady,
  AcnRpcRecoveryPolicyTag,
  type AcnInstance,
  type AcnIdentity,
  type AcnTarget,
} from "@magnitudedev/acn-protocol"
import { RpcClient, RpcClientError } from "@effect/rpc"
import {
  Cause,
  Context,
  Data,
  Deferred,
  Effect,
  Exit,
  Layer,
  Option,
  Ref,
  Schema,
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
import {
  AcnLifecycleStateSchema,
  makeAcnLifecycle,
  type AcnLifecycle,
  type AcnLifecycleOwner,
  type AcnLifecycleState,
} from "./lifecycle"

type ReadyInstance = AcnInstance<AcnReady>

const recoveryPolicy = (tag: string) => {
  const operation = AcnRpc.operation(AcnBoundary, tag)
  if (operation === undefined) throw new TypeError(`Unknown ACN operation ${tag}`)
  const policy = Context.getOption(operation.annotations, AcnRpcRecoveryPolicyTag)
  if (Option.isNone(policy)) throw new TypeError(`Finite ACN operation ${tag} has no recovery policy`)
  return policy.value
}

export interface AcnStartup {
  readonly state: AcnLifecycle
  readonly prepare: Effect.Effect<AcnLifecycleState>
  readonly awaitReady: Effect.Effect<void, AcnEnsuranceError>
  readonly retry: Effect.Effect<void, AcnEnsuranceError>
  readonly recovery: AcnRecovery
}

export class AcnRecoveryInactive extends Schema.TaggedClass<AcnRecoveryInactive>()(
  "Inactive",
  {},
) {}
export class AcnRecovering extends Schema.TaggedClass<AcnRecovering>()(
  "Recovering",
  {
    occurrence: Schema.Number.pipe(Schema.int(), Schema.positive()),
    lifecycle: AcnLifecycleStateSchema,
  },
) {}
export class AcnRecovered extends Schema.TaggedClass<AcnRecovered>()(
  "Recovered",
  { occurrence: Schema.Number.pipe(Schema.int(), Schema.positive()) },
) {}

export type AcnRecoveryState =
  | AcnRecoveryInactive
  | AcnRecovering
  | AcnRecovered

export interface AcnRecovery {
  readonly get: Effect.Effect<AcnRecoveryState>
  readonly changes: Stream.Stream<AcnRecoveryState>
}

export interface AcnConnection {
  readonly identity: Effect.Effect<AcnIdentity>
  readonly identityChanges: Stream.Stream<AcnIdentity>
  readonly protocolLayer: Layer.Layer<RpcClient.Protocol, never, HttpClient.HttpClient>
  readonly close: Effect.Effect<void>
  readonly startup: AcnStartup
}

interface SelectionPresentation {
  readonly lifecycle: AcnLifecycleOwner
  readonly recoveryOccurrence: Option.Option<number>
}

interface AcnAssociation {
  readonly target: AcnTarget
  readonly selected: Option.Option<ReadyInstance>
}

class AcnConnectionClosed extends Data.TaggedError("AcnConnectionClosed") {}
type SelectionError = AcnEnsuranceError | AcnConnectionClosed
const connectionClosed = () => new AcnConnectionClosed()

const sameReadyOccurrence = (left: ReadyInstance, right: ReadyInstance): boolean =>
  left.id === right.id &&
  left.pid === right.pid &&
  left.processStartIdentity === right.processStartIdentity

const { RpcClientError: TransportError } = RpcClientError
const unavailableError = (cause: SelectionError): RpcClientError.RpcClientError =>
  new TransportError({
    reason: "Unknown",
    message: cause._tag === "AcnConnectionClosed"
      ? "Magnitude service client is closed"
      : `Magnitude service unavailable: ${cause._tag}${"reason" in cause ? `: ${String(cause.reason)}` : ""}`,
    cause,
  })

export const makeAcnConnection = (): Effect.Effect<
  AcnConnection,
  never,
  AcnInstanceManager | Scope.Scope
> => Effect.gen(function* () {
  const manager = yield* AcnInstanceManager
  const selectionScope = yield* Scope.make()
  yield* Effect.addFinalizer(() => Scope.close(selectionScope, Exit.void))
  const lifecycle = yield* makeAcnLifecycle()
  const recoveryState = yield* SubscriptionRef.make<AcnRecoveryState>(new AcnRecoveryInactive({}))
  const hasSelected = yield* Ref.make(false)
  const recoveryOccurrence = yield* Ref.make(0)
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
  const finishFailedSelection = (
    deferred: Deferred.Deferred<ReadyInstance, SelectionError>,
    cause: Cause.Cause<AcnEnsuranceError>,
    presentation: SelectionPresentation,
  ) => admission.withPermits(1)(Effect.gen(function* () {
    const current = yield* Ref.get(activeSelection)
    if (Option.isNone(current) || current.value !== deferred) return
    yield* Ref.set(activeSelection, Option.none())
    if (!(yield* Ref.get(open))) {
      yield* Deferred.fail(deferred, connectionClosed())
      return
    }
    const failure = Option.getOrUndefined(Cause.failureOption(cause))
    if (failure !== undefined) {
      yield* presentation.lifecycle.fail(failure)
      if (Option.isSome(presentation.recoveryOccurrence)) {
        yield* SubscriptionRef.set(recoveryState, new AcnRecovering({
          occurrence: presentation.recoveryOccurrence.value,
          lifecycle: yield* presentation.lifecycle.get,
        }))
      }
    }
    yield* Deferred.failCause(deferred, cause)
  })).pipe(Effect.uninterruptible)

  const finishReadySelection = (
    deferred: Deferred.Deferred<ReadyInstance, SelectionError>,
    ready: ReadyInstance,
    presentation: SelectionPresentation,
  ): Effect.Effect<void> => admission.withPermits(1)(
    Effect.gen(function* () {
      const current = yield* Ref.get(activeSelection)
      if (Option.isNone(current) || current.value !== deferred) return
      if (!(yield* Ref.get(open))) {
        yield* Ref.set(activeSelection, Option.none())
        yield* Deferred.fail(deferred, connectionClosed())
        return
      }
      const previous = yield* SubscriptionRef.get(association)
      const target = ready.revision > previous.target.revision
        ? { revision: ready.revision, identity: ready.identity }
        : previous.target
      yield* Ref.set(activeSelection, Option.none())
      yield* SubscriptionRef.set(association, { target, selected: Option.some(ready) })
      yield* presentation.lifecycle.ready
      yield* Ref.set(hasSelected, true)
      if (Option.isSome(presentation.recoveryOccurrence)) {
        yield* SubscriptionRef.set(
          recoveryState,
          new AcnRecovered({ occurrence: presentation.recoveryOccurrence.value }),
        )
      }
      yield* Deferred.succeed(deferred, ready)
    }).pipe(Effect.uninterruptible),
  )

  const launchSelection = (
    deferred: Deferred.Deferred<ReadyInstance, SelectionError>,
    target: AcnTarget,
    presentation: SelectionPresentation,
  ): Effect.Effect<void> => Effect.suspend(() => runAcnEnsure(manager.ensure({ target }).pipe(
    Stream.tap((event) => event._tag === "Observation"
      ? presentation.lifecycle.report(event.observation).pipe(
          Effect.zipRight(Option.match(presentation.recoveryOccurrence, {
            onNone: () => Effect.void,
            onSome: (occurrence) => presentation.lifecycle.get.pipe(
              Effect.flatMap((current) => SubscriptionRef.set(recoveryState, new AcnRecovering({
                occurrence,
                lifecycle: current,
              }))),
            ),
          })),
        )
      : Effect.void),
  )).pipe(
    Effect.exit,
    Effect.flatMap((exit) => {
      if (Exit.isFailure(exit)) return finishFailedSelection(deferred, exit.cause, presentation)
      return finishReadySelection(deferred, exit.value, presentation)
    }),
  ))

  const admitSelectionUnlocked: Effect.Effect<
    Effect.Effect<ReadyInstance, SelectionError>
  > = Effect.gen(function* () {
    if (!(yield* Ref.get(open))) return yield* Effect.succeed(Effect.fail(connectionClosed()))
    const selected = (yield* SubscriptionRef.get(association)).selected
    if (Option.isSome(selected)) return yield* Effect.succeed(
      Effect.succeed(selected.value) as Effect.Effect<ReadyInstance, SelectionError>,
    )
    const active = yield* Ref.get(activeSelection)
    if (Option.isSome(active)) return yield* Effect.succeed(Deferred.await(active.value))
    const deferred = yield* Deferred.make<ReadyInstance, SelectionError>()
    const target = (yield* SubscriptionRef.get(association)).target
    const recovering = yield* Ref.get(hasSelected)
    const selectionLifecycle = recovering ? yield* makeAcnLifecycle() : lifecycle
    const occurrence = recovering
      ? Option.some(yield* Ref.updateAndGet(recoveryOccurrence, (value) => value + 1))
      : Option.none<number>()
    const presentation: SelectionPresentation = {
      lifecycle: selectionLifecycle,
      recoveryOccurrence: occurrence,
    }
    if (Option.isSome(occurrence)) {
      yield* SubscriptionRef.set(recoveryState, new AcnRecovering({
        occurrence: occurrence.value,
        lifecycle: yield* selectionLifecycle.get,
      }))
    }
    yield* Ref.set(activeSelection, Option.some(deferred))
    const selection = launchSelection(deferred, target, presentation).pipe(
      Effect.timeoutFail({
        duration: ACN_ENSURE_TIMEOUT,
        onTimeout: () => new AcnEnsuranceFailed({
          reason: "ACN client selection did not converge within its absolute deadline",
        }),
      }),
      Effect.catchAll((error) => finishFailedSelection(deferred, Cause.fail(error), presentation)),
    )
    yield* Effect.forkIn(selection, selectionScope)
    return yield* Effect.succeed(Deferred.await(deferred))
  })

  const endpoint: Effect.Effect<ReadyInstance, SelectionError> = Effect.flatten(
    admission.withPermits(1)(admitSelectionUnlocked),
  )

  const recover = (failed: ReadyInstance): Effect.Effect<ReadyInstance, SelectionError> =>
    Effect.flatten(admission.withPermits(1)(Effect.gen(function* () {
      if (!(yield* Ref.get(open))) return yield* Effect.succeed(Effect.fail(connectionClosed()))
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

  const retry = endpoint.pipe(
    Effect.mapError((error) => error._tag === "AcnConnectionClosed"
      ? new AcnEnsuranceFailed({ reason: "Magnitude service connection is closed" })
      : error),
    Effect.asVoid,
  )

  const close = yield* Effect.cached(
    admission.withPermits(1)(Ref.set(open, false)).pipe(
      Effect.zipRight(Scope.close(selectionScope, Exit.void)),
    ),
  )

  return {
    identity: SubscriptionRef.get(association).pipe(Effect.map((current) => current.target.identity)),
    identityChanges: association.changes.pipe(
      Stream.map((current) => current.target.identity),
      Stream.changes,
    ),
    startup: {
      state: lifecycle,
      prepare,
      awaitReady: endpoint.pipe(
        Effect.mapError((error) => error._tag === "AcnConnectionClosed"
          ? new AcnEnsuranceFailed({ reason: "Magnitude service connection is closed" })
          : error),
        Effect.asVoid,
      ),
      retry,
      recovery: {
        get: SubscriptionRef.get(recoveryState),
        changes: recoveryState.changes,
      },
    },
    close,
    protocolLayer: recoveringProtocolLayer,
  }
})
