import {
  AcnServiceLifecycleFsm,
  AcnStarting,
  summarizeInferenceHardware,
  type AcnHealthState,
  type AcnStartupActivity,
  type AcnStartupProgress,
  type AcnStartupRetry,
  type AcnStopping,
  type AcnStoppingReason,
} from "@magnitudedev/acn-protocol"
import type { HardwareSnapshot, ModelInstancesSnapshot } from "@magnitudedev/icn-protocol/schemas"
import { HttpServerRequest, HttpServerResponse } from "@effect/platform"
import { Context, Deferred, Effect, Layer, Option, SubscriptionRef, Stream, type Scope } from "effect"

export type AcnRpcApplication = Effect.Effect<
  HttpServerResponse.HttpServerResponse,
  never,
  HttpServerRequest.HttpServerRequest | Scope.Scope
>

export interface AcnStopRequest {
  readonly reason: AcnStoppingReason
  readonly detail?: string
  readonly permanent?: boolean
}

interface AcnRuntimeState {
  readonly lifecycle: AcnHealthState
  readonly rpc: Option.Option<AcnRpcApplication>
}

export interface AcnServiceLifecycleApi {
  readonly changes: Stream.Stream<AcnHealthState>
  readonly state: Effect.Effect<AcnHealthState>
  readonly dispatchRpc: AcnRpcApplication
  readonly reportStarting: (
    activity: AcnStartupActivity,
    progress: Option.Option<AcnStartupProgress>,
  ) => Effect.Effect<void>
  /** Keeps the current activity and progress while a failed startup step waits to run again. */
  readonly reportRetrying: (retry: AcnStartupRetry) => Effect.Effect<void>
  readonly becomeReady: (
    rpc: AcnRpcApplication,
    machine: Option.Option<{ readonly hardware: HardwareSnapshot; readonly instances: ModelInstancesSnapshot }>,
  ) => Effect.Effect<void>
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
    const runtime = yield* SubscriptionRef.make<AcnRuntimeState>({
      lifecycle: new AcnStarting({
        activity: "WaitingForOwnership",
        progress: Option.none(),
        retry: Option.none(),
      }),
      rpc: Option.none(),
    })
    const transitionLock = yield* Effect.makeSemaphore(1)
    const stopping = yield* Deferred.make<AcnStopping>()
    const replaceState = (next: AcnRuntimeState) => SubscriptionRef.set(runtime, next)

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
          permanent: request.permanent ? Option.some(true) : Option.none(),
        },
      )
      yield* replaceState({
        lifecycle: next,
        rpc: Option.none(),
      })
      yield* Deferred.succeed(stopping, next)
      return true
    })

    const beginStopping: AcnServiceLifecycleApi["beginStopping"] = (request) =>
      transitionLock.withPermits(1)(
        SubscriptionRef.get(runtime).pipe(
          Effect.flatMap((current) => commitStopping(current, request)),
          Effect.uninterruptible,
        ),
      )

    const holdStarting = (
      updates: Partial<Pick<AcnStarting, "activity" | "progress" | "retry">>,
    ) =>
      transitionLock.withPermits(1)(
        Effect.gen(function* () {
          const current = yield* SubscriptionRef.get(runtime)
          if (current.lifecycle._tag === "Stopping") return
          if (current.lifecycle._tag !== "Starting") {
            return yield* Effect.dieMessage(
              "ACN startup activity was reported after readiness",
            )
          }
          yield* replaceState({
            lifecycle: AcnServiceLifecycleFsm.hold(current.lifecycle, updates),
            rpc: current.rpc,
          })
        }),
      )

    const reportStarting: AcnServiceLifecycleApi["reportStarting"] =
      (activity, progress) => holdStarting({ activity, progress, retry: Option.none() })

    const reportRetrying: AcnServiceLifecycleApi["reportRetrying"] =
      (retry) => holdStarting({ retry: Option.some(retry) })

    const becomeReady: AcnServiceLifecycleApi["becomeReady"] = (rpc, machine) =>
      transitionLock.withPermits(1)(
        Effect.gen(function* () {
          const current = yield* SubscriptionRef.get(runtime)
          if (current.lifecycle._tag === "Stopping") return
          if (current.lifecycle._tag !== "Starting") {
            return yield* Effect.dieMessage("ACN became ready more than once")
          }
          // A snapshot whose placements do not resolve breaks ICN's contract, as in the hardware projection.
          const summary = yield* Option.match(machine, {
            onNone: () => Effect.succeed(Option.none()),
            onSome: ({ hardware, instances }) => summarizeInferenceHardware(hardware, instances).pipe(
              Effect.map(Option.some),
              Effect.orDie,
            ),
          })
          yield* replaceState({
            lifecycle: AcnServiceLifecycleFsm.transition(
              current.lifecycle,
              "Ready",
              { hardware: summary },
            ),
            rpc: Option.some(rpc),
          })
        }).pipe(Effect.uninterruptible),
      )

    const state = SubscriptionRef.get(runtime).pipe(
      Effect.map((current) => current.lifecycle),
    )

    return AcnServiceLifecycle.of({
      state,
      changes: runtime.changes.pipe(Stream.map(value => value.lifecycle)),
      dispatchRpc: SubscriptionRef.get(runtime).pipe(
        Effect.flatMap((current) =>
          current.lifecycle._tag === "Ready" && Option.isSome(current.rpc)
            ? current.rpc.value
            : Effect.succeed(unavailable(current.lifecycle)),
        ),
      ),
      reportStarting,
      reportRetrying,
      becomeReady,
      beginStopping,
      awaitStopping: Deferred.await(stopping),
    })
  })

export const AcnServiceLifecycleLive: Layer.Layer<AcnServiceLifecycle> =
  Layer.scoped(AcnServiceLifecycle, makeAcnServiceLifecycle())
