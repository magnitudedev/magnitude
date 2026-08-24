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
import { Context, Deferred, Effect, Layer, Option, Ref, type Scope } from "effect"

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

export const makeAcnServiceLifecycle = (): Effect.Effect<AcnServiceLifecycleApi, never, Scope.Scope> =>
  Effect.gen(function* () {
    const runtime = yield* Ref.make<AcnRuntimeState>({
      lifecycle: new AcnStarting({
        activity: "WaitingForOwnership",
        progress: Option.none(),
      }),
      rpc: Option.none(),
      clientsPresent: false,
    })
    const transitionLock = yield* Effect.makeSemaphore(1)
    const stopping = yield* Deferred.make<AcnStopping>()
    const replaceState = (next: AcnRuntimeState) => Ref.set(runtime, next)

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
          yield* replaceState({
            lifecycle: AcnServiceLifecycleFsm.transition(
              current.lifecycle,
              "Ready",
              {},
            ),
            rpc: Option.some(rpc),
            clientsPresent: current.clientsPresent,
          })
        }).pipe(Effect.uninterruptible),
      )

    const setClientPresence: AcnServiceLifecycleApi["setClientPresence"] = (present) =>
      transitionLock.withPermits(1)(
        Effect.gen(function* () {
          const current = yield* Ref.get(runtime)
          if (current.lifecycle._tag === "Stopping") return false
          if (current.clientsPresent === present) return true
          yield* replaceState({
            lifecycle: current.lifecycle,
            rpc: current.rpc,
            clientsPresent: present,
          })
          return true
        }).pipe(Effect.uninterruptible),
      )

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

export const AcnServiceLifecycleLive: Layer.Layer<AcnServiceLifecycle> =
  Layer.scoped(AcnServiceLifecycle, makeAcnServiceLifecycle())
