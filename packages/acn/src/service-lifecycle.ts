import {
  AcnServiceLifecycleFsm,
  AcnStarting,
  type AcnHealthState,
  type AcnStartupActivity,
  type AcnStartupProgress,
  type AcnStopping,
  type AcnStoppingReason,
} from "@magnitudedev/acn-protocol"
import { HttpServerRequest, HttpServerResponse } from "@effect/platform"
import {
  Clock,
  Context,
  Deferred,
  Duration,
  Effect,
  Fiber,
  Layer,
  Option,
  Ref,
  type Scope,
} from "effect"

export type AcnRpcApplication = Effect.Effect<
  HttpServerResponse.HttpServerResponse,
  never,
  HttpServerRequest.HttpServerRequest | Scope.Scope
>

export interface AcnStopRequest {
  readonly reason: AcnStoppingReason
  readonly detail?: string
}

interface AcnRuntimeState {
  readonly lifecycle: AcnHealthState
  readonly rpc: Option.Option<AcnRpcApplication>
  readonly clientsPresent: boolean
  readonly idleDeadline: Option.Option<bigint>
  readonly idleRevision: number
}

export interface AcnServiceLifecycleApi {
  readonly state: Effect.Effect<AcnHealthState>
  readonly dispatchRpc: AcnRpcApplication
  readonly reportStarting: (
    activity: AcnStartupActivity,
    progress: Option.Option<AcnStartupProgress>,
  ) => Effect.Effect<void>
  readonly becomeReady: (rpc: AcnRpcApplication) => Effect.Effect<void>
  readonly setClientPresence: (present: boolean) => Effect.Effect<boolean>
  readonly beginStopping: (request: AcnStopRequest) => Effect.Effect<boolean>
  readonly awaitStopping: Effect.Effect<AcnStopping>
}

export class AcnServiceLifecycle extends Context.Tag("AcnServiceLifecycle")<
  AcnServiceLifecycle,
  AcnServiceLifecycleApi
>() {}

const unavailable = (state: AcnHealthState) =>
  HttpServerResponse.text(
    state._tag === "Stopping" ? "Magnitude is stopping" : "Magnitude is starting",
    {
      status: 503,
      headers: { "retry-after": "1" },
    },
  )

const DEFAULT_ACN_IDLE_TIMEOUT = Duration.minutes(30)

export const makeAcnServiceLifecycle = (
  idleTimeout: Duration.DurationInput = DEFAULT_ACN_IDLE_TIMEOUT,
): Effect.Effect<AcnServiceLifecycleApi, never, Scope.Scope> =>
  Effect.gen(function* () {
    const timeout = Duration.decode(idleTimeout)
    const timeoutNanos = yield* Option.match(Duration.toNanos(timeout), {
      onNone: () => Effect.dieMessage("ACN idle timeout must be finite"),
      onSome: Effect.succeed,
    })
    if (timeoutNanos <= 0n) {
      return yield* Effect.dieMessage("ACN idle timeout must be positive")
    }

    const runtime = yield* Ref.make<AcnRuntimeState>({
      lifecycle: new AcnStarting({
        activity: "WaitingForOwnership",
        progress: Option.none(),
      }),
      rpc: Option.none(),
      clientsPresent: false,
      idleDeadline: Option.none(),
      idleRevision: 0,
    })
    const transitionLock = yield* Effect.makeSemaphore(1)
    const stopping = yield* Deferred.make<AcnStopping>()
    const scope = yield* Effect.scope
    const idleFiber = yield* Ref.make<Option.Option<Fiber.RuntimeFiber<void, never>>>(Option.none())

    const replaceState = (next: AcnRuntimeState) => Ref.set(runtime, next)

    const cancelIdle = Ref.getAndSet(idleFiber, Option.none()).pipe(
      Effect.flatMap(Option.match({
        onNone: () => Effect.void,
        onSome: Fiber.interruptFork,
      })),
    )

    let armIdle = (_deadline: bigint, _revision: number): Effect.Effect<void> =>
      Effect.dieMessage("ACN idle timer initialized without retirement transition")

    const commitStopping = (
      current: AcnRuntimeState,
      request: AcnStopRequest,
    ) => Effect.gen(function* () {
      if (current.lifecycle._tag === "Stopping") return false
      const next = AcnServiceLifecycleFsm.transition(
        current.lifecycle,
        "Stopping",
        {
          reason: request.reason,
          safeDetail: Option.fromNullable(request.detail).pipe(
            Option.filter((detail) => detail.length > 0),
          ),
        },
      )
      yield* replaceState({
        lifecycle: next,
        rpc: Option.none(),
        clientsPresent: current.clientsPresent,
        idleDeadline: Option.none(),
        idleRevision: current.idleRevision + 1,
      })
      yield* Deferred.succeed(stopping, next)
      return true
    })

    const beginStopping: AcnServiceLifecycleApi["beginStopping"] = (request) =>
      transitionLock.withPermits(1)(
        Ref.get(runtime).pipe(
          Effect.flatMap((current) => commitStopping(current, request)),
          Effect.uninterruptible,
        ),
      )

    const reportStarting: AcnServiceLifecycleApi["reportStarting"] =
      (activity, progress) =>
        transitionLock.withPermits(1)(
          Effect.gen(function* () {
            const current = yield* Ref.get(runtime)
            if (current.lifecycle._tag === "Stopping") return
            if (current.lifecycle._tag !== "Starting") {
              return yield* Effect.dieMessage(
                "ACN startup activity was reported after readiness",
              )
            }
            yield* replaceState({
              lifecycle: AcnServiceLifecycleFsm.hold(current.lifecycle, {
                activity,
                progress,
              }),
              rpc: current.rpc,
              clientsPresent: current.clientsPresent,
              idleDeadline: current.idleDeadline,
              idleRevision: current.idleRevision,
            })
          }),
        )

    const becomeReady: AcnServiceLifecycleApi["becomeReady"] = (rpc) =>
      transitionLock.withPermits(1)(
        Effect.gen(function* () {
          const current = yield* Ref.get(runtime)
          if (current.lifecycle._tag === "Stopping") return
          if (current.lifecycle._tag !== "Starting") {
            return yield* Effect.dieMessage("ACN became ready more than once")
          }
          const now = yield* Clock.currentTimeNanos
          const deadline = now + timeoutNanos
          const revision = current.idleRevision + 1
          yield* replaceState({
            lifecycle: AcnServiceLifecycleFsm.transition(
              current.lifecycle,
              "Ready",
              {},
            ),
            rpc: Option.some(rpc),
            clientsPresent: current.clientsPresent,
            idleDeadline: current.clientsPresent
              ? Option.none()
              : Option.some(deadline),
            idleRevision: revision,
          })
          if (!current.clientsPresent) yield* armIdle(deadline, revision)
        }).pipe(Effect.uninterruptible),
      )

    const setClientPresence: AcnServiceLifecycleApi["setClientPresence"] = (present) =>
      transitionLock.withPermits(1)(
        Effect.gen(function* () {
          const current = yield* Ref.get(runtime)
          if (current.lifecycle._tag === "Stopping") return false
          if (current.clientsPresent === present) return true
          const now = yield* Clock.currentTimeNanos
          const deadline = now + timeoutNanos
          const revision = current.idleRevision + 1
          yield* replaceState({
            lifecycle: current.lifecycle,
            rpc: current.rpc,
            clientsPresent: present,
            idleDeadline: current.lifecycle._tag === "Ready" && !present
              ? Option.some(deadline)
              : Option.none(),
            idleRevision: revision,
          })
          if (present || current.lifecycle._tag !== "Ready") {
            yield* cancelIdle
          } else {
            yield* armIdle(deadline, revision)
          }
          return true
        }).pipe(Effect.uninterruptible),
      )

    armIdle = (deadline, revision) =>
      Effect.gen(function* () {
        yield* cancelIdle
        const now = yield* Clock.currentTimeNanos
        const remaining = deadline > now
          ? Duration.nanos(deadline - now)
          : Duration.zero
        const fiber = yield* Effect.sleep(remaining).pipe(
          Effect.zipRight(
            transitionLock.withPermits(1)(
              Effect.gen(function* () {
                const current = yield* Ref.get(runtime)
                if (
                  current.lifecycle._tag !== "Ready" ||
                  current.clientsPresent ||
                  current.idleRevision !== revision ||
                  Option.isNone(current.idleDeadline) ||
                  current.idleDeadline.value !== deadline ||
                  (yield* Clock.currentTimeNanos) < deadline
                ) {
                  return
                }
                yield* commitStopping(current, { reason: "idle" })
              }).pipe(Effect.uninterruptible),
            ),
          ),
          Effect.interruptible,
          Effect.forkIn(scope),
        )
        yield* Ref.set(idleFiber, Option.some(fiber))
      })

    const state = Ref.get(runtime).pipe(
      Effect.map((current) => current.lifecycle),
    )

    return AcnServiceLifecycle.of({
      state,
      dispatchRpc: Ref.get(runtime).pipe(
        Effect.flatMap((current) =>
          current.lifecycle._tag === "Ready" && Option.isSome(current.rpc)
            ? current.rpc.value
            : Effect.succeed(unavailable(current.lifecycle)),
        ),
      ),
      reportStarting,
      becomeReady,
      setClientPresence,
      beginStopping,
      awaitStopping: Deferred.await(stopping),
    })
  })

export const AcnServiceLifecycleLive = (
  idleTimeout: Duration.DurationInput = DEFAULT_ACN_IDLE_TIMEOUT,
): Layer.Layer<AcnServiceLifecycle> =>
  Layer.scoped(AcnServiceLifecycle, makeAcnServiceLifecycle(idleTimeout))
