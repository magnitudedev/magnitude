import {
  AcnReady,
  AcnServiceLifecycleFsm,
  AcnStarting,
  type AcnHealthState,
  type AcnStartupActivity,
  type AcnStartupProgress,
  type AcnStopping,
  type AcnStoppingReason,
} from "@magnitudedev/acn-protocol";
import { HttpServerRequest, HttpServerResponse } from "@effect/platform";
import {
  Context,
  Deferred,
  Duration,
  Effect,
  Layer,
  Option,
  Ref,
  type Scope,
} from "effect";
import {
  makeResourceUseGate,
  type ResourceRetired,
  type ResourceUseGateSnapshot,
} from "./resource-use-gate";

export type AcnRpcApplication = Effect.Effect<
  HttpServerResponse.HttpServerResponse,
  never,
  HttpServerRequest.HttpServerRequest | Scope.Scope
>;

export interface AcnStopRequest {
  readonly reason: AcnStoppingReason;
  readonly detail?: string;
}

interface AcnRuntimeState {
  readonly lifecycle: AcnHealthState;
  readonly rpc: Option.Option<AcnRpcApplication>;
}

export interface AcnServiceLifecycleApi {
  readonly state: Effect.Effect<AcnHealthState>;
  readonly dispatchRpc: AcnRpcApplication;
  readonly reportStarting: (
    activity: AcnStartupActivity,
    progress: Option.Option<AcnStartupProgress>,
  ) => Effect.Effect<void>;
  readonly becomeReady: (rpc: AcnRpcApplication) => Effect.Effect<void>;
  readonly beginStopping: (request: AcnStopRequest) => Effect.Effect<boolean>;
  readonly awaitStopping: Effect.Effect<AcnStopping>;
  readonly acquireActivity: (
    label: string,
  ) => Effect.Effect<Effect.Effect<void>, ResourceRetired>;
  readonly joinActivityIfBusy: (
    label: string,
  ) => Effect.Effect<Option.Option<Effect.Effect<void>>, ResourceRetired>;
  readonly withActivity: <A, E, R>(
    label: string,
    effect: Effect.Effect<A, E, R>,
  ) => Effect.Effect<A, E | ResourceRetired, R>;
  readonly activity: Effect.Effect<ResourceUseGateSnapshot>;
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
  );

const DEFAULT_ACN_IDLE_TIMEOUT = Duration.minutes(30);

export const makeAcnServiceLifecycle = (
  idleTimeout: Duration.DurationInput = DEFAULT_ACN_IDLE_TIMEOUT,
): Effect.Effect<AcnServiceLifecycleApi, never, Scope.Scope> =>
  Effect.gen(function* () {
    const runtime = yield* Ref.make<AcnRuntimeState>({
      lifecycle: new AcnStarting({
        activity: "WaitingForOwnership",
        progress: Option.none(),
      }),
      rpc: Option.none(),
    });
    const transitionLock = yield* Effect.makeSemaphore(1);
    const stopping = yield* Deferred.make<AcnStopping>();

    let beginStopping: AcnServiceLifecycleApi["beginStopping"] = () =>
      Effect.dieMessage("ACN lifecycle initialized without stopping transition");

    const gate = yield* makeResourceUseGate({
      resource: "acn",
      generation: 1,
      idleTimeout,
      retire: () =>
        Effect.suspend(() => beginStopping({ reason: "idle" })).pipe(Effect.as(true)),
    });
    const releaseBootstrap = yield* gate.acquire("acn-startup").pipe(Effect.orDie);

    beginStopping = (request) =>
      transitionLock.withPermits(1)(
        Effect.gen(function* () {
          const current = yield* Ref.get(runtime);
          if (current.lifecycle._tag === "Stopping") return false;
          yield* gate.closeAdmission;
          const next = AcnServiceLifecycleFsm.transition(
            current.lifecycle,
            "Stopping",
            {
              reason: request.reason,
              safeDetail: Option.fromNullable(request.detail).pipe(
                Option.filter((detail) => detail.length > 0),
              ),
            },
          );
          yield* Ref.set(runtime, {
            lifecycle: next,
            rpc: Option.none(),
          });
          yield* Deferred.succeed(stopping, next);
          return true;
        }).pipe(Effect.uninterruptible),
      );

    const reportStarting: AcnServiceLifecycleApi["reportStarting"] =
      (activity, progress) =>
        transitionLock.withPermits(1)(
          Effect.gen(function* () {
            const current = yield* Ref.get(runtime);
            if (current.lifecycle._tag === "Stopping") return;
            if (current.lifecycle._tag !== "Starting") {
              return yield* Effect.dieMessage(
                "ACN startup activity was reported after readiness",
              );
            }
            yield* Ref.set(runtime, {
              ...current,
              lifecycle: AcnServiceLifecycleFsm.hold(current.lifecycle, {
                activity,
                progress,
              }),
            });
          }),
        );

    const becomeReady: AcnServiceLifecycleApi["becomeReady"] = (rpc) =>
      transitionLock.withPermits(1)(
        Effect.gen(function* () {
          const current = yield* Ref.get(runtime);
          if (current.lifecycle._tag === "Stopping") return;
          if (current.lifecycle._tag !== "Starting") {
            return yield* Effect.dieMessage("ACN became ready more than once");
          }
          yield* releaseBootstrap;
          yield* Ref.set(runtime, {
            lifecycle: AcnServiceLifecycleFsm.transition(
              current.lifecycle,
              "Ready",
              {},
            ),
            rpc: Option.some(rpc),
          });
        }),
      );

    const state = Ref.get(runtime).pipe(
      Effect.map((current) => current.lifecycle),
    );

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
      beginStopping,
      awaitStopping: Deferred.await(stopping),
      acquireActivity: gate.acquire,
      joinActivityIfBusy: gate.joinIfBusy,
      withActivity: gate.withUse,
      activity: gate.snapshot,
    });
  });

export const AcnServiceLifecycleLive = (
  idleTimeout: Duration.DurationInput = DEFAULT_ACN_IDLE_TIMEOUT,
): Layer.Layer<AcnServiceLifecycle> =>
  Layer.scoped(AcnServiceLifecycle, makeAcnServiceLifecycle(idleTimeout));
