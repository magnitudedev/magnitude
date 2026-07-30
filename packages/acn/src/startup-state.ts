import {
  type AcnHealthState,
  type AcnInstallationPlan,
  type AcnInstallationPhase,
  type AcnStartupActivity,
  type AcnStartupProgress,
} from "@magnitudedev/acn-protocol";
import { HttpServerRequest, HttpServerResponse } from "@effect/platform";
import {
  Context,
  Effect,
  Layer,
  Option,
  SubscriptionRef,
  type Scope,
} from "effect";

export type AcnRpcApplication = Effect.Effect<
  HttpServerResponse.HttpServerResponse,
  never,
  HttpServerRequest.HttpServerRequest | Scope.Scope
>;

export interface AcnStartupState {
  readonly get: Effect.Effect<AcnHealthState>;
  readonly rpc: AcnRpcApplication;
  readonly starting: (
    activity: AcnStartupActivity,
    progress: Option.Option<AcnStartupProgress>
  ) => Effect.Effect<void>;
  readonly installing: (
    phase: AcnInstallationPhase,
    plan: AcnInstallationPlan,
    progress: Option.Option<AcnStartupProgress>
  ) => Effect.Effect<void>;
  readonly ready: (rpc: AcnRpcApplication) => Effect.Effect<void>;
  readonly failed: (message: string, retryable: boolean) => Effect.Effect<void>;
}

export const AcnStartupState = Context.GenericTag<AcnStartupState>(
  "@magnitudedev/acn/AcnStartupState"
);

export const makeAcnStartupState = (): Effect.Effect<AcnStartupState> =>
  Effect.gen(function* () {
    const state = yield* SubscriptionRef.make<{
      readonly health: AcnHealthState;
      readonly rpc: Option.Option<AcnRpcApplication>;
    }>({
      health: {
        _tag: "Starting",
        activity: "Resolving",
        progress: Option.none(),
      },
      rpc: Option.none(),
    });
    return {
      get: SubscriptionRef.get(state).pipe(
        Effect.map((current) => current.health)
      ),
      rpc: SubscriptionRef.get(state).pipe(
        Effect.flatMap((current) =>
          Option.match(current.rpc, {
            onNone: () =>
              Effect.succeed(
                HttpServerResponse.text("Magnitude is starting", {
                  status: 503,
                  headers: { "retry-after": "1" },
                })
              ),
            onSome: (rpc) => rpc,
          })
        )
      ),
      starting: (activity, progress) =>
        SubscriptionRef.set(state, {
          health: {
            _tag: "Starting",
            activity,
            progress,
          },
          rpc: Option.none(),
        }),
      installing: (phase, plan, progress) =>
        SubscriptionRef.set(state, {
          health: {
            _tag: "Installing",
            phase,
            plan,
            progress,
          },
          rpc: Option.none(),
        }),
      ready: (rpc) =>
        SubscriptionRef.set(state, {
          health: { _tag: "Ready" },
          rpc: Option.some(rpc),
        }),
      failed: (message, retryable) =>
        SubscriptionRef.set(state, {
          health: {
            _tag: "Failed",
            message,
            retryable,
          },
          rpc: Option.none(),
        }),
    };
  });

export const AcnStartupStateLive: Layer.Layer<AcnStartupState> = Layer.effect(
  AcnStartupState,
  makeAcnStartupState()
);
