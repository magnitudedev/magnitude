import * as HttpClient from "@effect/platform/HttpClient";
import type { AcnIdentity, AcnInstance } from "@magnitudedev/acn-protocol";
import {
  canUseAcnIdentity,
  compareAcnIdentities,
} from "@magnitudedev/acn-protocol/acn-identity";
import { RpcClient, RpcClientError } from "@effect/rpc";
import { Effect, Fiber, Layer, Option, Stream, SubscriptionRef } from "effect";
import {
  isInterruptedExit,
  recoveringProtocolLayer as jitRecoveringProtocolLayer,
} from "../jit-rpc";
import { AcnProcessManager, runAcnLaunch } from "./acn-process-manager";
import type { DaemonError } from "./errors";
import { SDK_VERSION } from "../version";
import { acnSubscriptionProtocol } from "./acn-subscription-protocol";
import {
  makeAcnLifecycle,
  acnLifecycleObservationFromHealthState,
  acnStartupProgressKey,
  type AcnLifecycle,
  type AcnLifecycleState,
} from "./lifecycle";

export interface AcnJitRuntimeOptions {
  readonly launchCommand: Option.Option<ReadonlyArray<string>>;
}

export interface AcnStartup {
  readonly state: AcnLifecycle;
  readonly prepare: Effect.Effect<AcnLifecycleState>;
  readonly retry: Effect.Effect<void, DaemonError>;
}

export interface AcnJitRuntime {
  readonly identity: Effect.Effect<AcnIdentity>;
  readonly identityChanges: Stream.Stream<AcnIdentity>;
  readonly protocolLayer: Layer.Layer<
    RpcClient.Protocol,
    never,
    HttpClient.HttpClient
  >;
  readonly startup: AcnStartup;
}

interface AcnAssociation {
  readonly identity: AcnIdentity;
  readonly selected: Option.Option<AcnInstance>;
}

const { RpcClientError: TransportError } = RpcClientError;

const unavailableError = (cause: DaemonError): RpcClientError.RpcClientError =>
  new TransportError({
    reason: "Unknown",
    message: `ACN unavailable: ${cause._tag}`,
    cause,
  });

export const makeAcnJitRuntime = (
  options: AcnJitRuntimeOptions = { launchCommand: Option.none() }
): Effect.Effect<AcnJitRuntime, never, AcnProcessManager> =>
  Effect.gen(function* () {
    const processManager = yield* AcnProcessManager;
    const lifecycle = yield* makeAcnLifecycle();
    const association = yield* SubscriptionRef.make<AcnAssociation>({
      identity: SDK_VERSION,
      selected: Option.none(),
    });
    const selectionLock = yield* Effect.makeSemaphore(1);

    const reportInstance = (instance: AcnInstance) =>
      Option.match(acnLifecycleObservationFromHealthState(instance.lifecycle), {
        onNone: () => Effect.void,
        onSome: lifecycle.report,
      });

    const reconcile = (instance: AcnInstance) =>
      Effect.gen(function* () {
        const current = yield* SubscriptionRef.get(association);
        if (!canUseAcnIdentity(current.identity, instance.identity)) {
          return Option.none<AcnInstance>();
        }
        const identity =
          compareAcnIdentities(instance.identity, current.identity) > 0
            ? instance.identity
            : current.identity;
        const selected =
          instance.lifecycle._tag === "Ready"
            ? Option.some(instance)
            : Option.none<AcnInstance>();
        yield* SubscriptionRef.set(association, { identity, selected });
        if (Option.isSome(selected)) yield* lifecycle.ready;
        else yield* reportInstance(instance);
        return Option.some(instance);
      });

    const launchCurrent = (
      replace: Option.Option<AcnInstance> = Option.none()
    ) =>
      Effect.suspend(() =>
        SubscriptionRef.get(association).pipe(
          Effect.flatMap((current) =>
            runAcnLaunch(
              processManager
                .launch({
                  identity: current.identity,
                  replace,
                  command: options.launchCommand,
                })
                .pipe(
                  Stream.tap((event) =>
                    event._tag === "Observation"
                      ? lifecycle.report(event.observation)
                      : Effect.void
                  )
                )
            )
          ),
          Effect.flatMap((instance) =>
            reconcile(instance).pipe(Effect.as(instance))
          )
        )
      );

    const observeStarting = (
      initial: AcnInstance
    ): Effect.Effect<AcnInstance, DaemonError> =>
      Effect.gen(function* () {
        let observed = initial;
        let key = acnStartupProgressKey(initial.lifecycle);
        let deadline = Date.now() + 30_000;
        while (true) {
          if (observed.lifecycle._tag === "Ready") {
            yield* reconcile(observed);
            return observed;
          }
          if (observed.lifecycle._tag === "Stopping") {
            return yield* launchCurrent(Option.some(observed));
          }
          yield* reportInstance(observed);
          if (Date.now() >= deadline) {
            return yield* launchCurrent(Option.some(observed));
          }
          yield* Effect.sleep("250 millis");
          const current = yield* processManager.observeCurrent;
          if (Option.isNone(current)) return yield* launchCurrent();
          if (current.value.id !== observed.id) return yield* ensureUnlocked;
          observed = current.value;
          const nextKey = acnStartupProgressKey(observed.lifecycle);
          if (nextKey !== key) {
            key = nextKey;
            deadline = Date.now() + 30_000;
          }
        }
      });

    const publicationGrace = Effect.gen(function* () {
      const deadline = Date.now() + 2_000;
      while (Date.now() < deadline) {
        yield* Effect.sleep("100 millis");
        const current = yield* processManager.observeCurrent;
        if (Option.isSome(current)) return yield* ensureObserved(current.value);
      }
      return yield* launchCurrent();
    });

    const ensureObserved = (
      instance: AcnInstance
    ): Effect.Effect<AcnInstance, DaemonError> =>
      Effect.gen(function* () {
        const accepted = yield* reconcile(instance);
        if (Option.isNone(accepted)) return yield* launchCurrent();
        if (instance.lifecycle._tag === "Ready") return instance;
        if (instance.lifecycle._tag === "Starting")
          return yield* observeStarting(instance);
        return yield* launchCurrent(Option.some(instance));
      });

    const ensureUnlocked: Effect.Effect<AcnInstance, DaemonError> =
      Effect.suspend(() =>
        processManager.observeCurrent.pipe(
          Effect.flatMap(
            Option.match({
              onNone: () => publicationGrace,
              onSome: ensureObserved,
            })
          )
        )
      );

    const ensure = selectionLock
      .withPermits(1)(ensureUnlocked)
      .pipe(Effect.tapError(lifecycle.fail));

    const recover = (failed: AcnInstance) =>
      selectionLock
        .withPermits(1)(
          Effect.gen(function* () {
            const current = yield* SubscriptionRef.get(association);
            const replacement = Option.filter(
              current.selected,
              (instance) => instance.id !== failed.id
            );
            if (Option.isSome(replacement)) {
              return replacement.value;
            }
            yield* SubscriptionRef.set(association, {
              ...current,
              selected: Option.none(),
            });
            yield* lifecycle.report({ _tag: "Starting", phase: "Discovering" });
            return yield* ensureUnlocked;
          })
        )
        .pipe(Effect.tapError(lifecycle.fail));

    const prepare = Effect.gen(function* () {
      const initial = yield* lifecycle.get;
      if (initial._tag !== "Checking") return initial;
      const firstVisible = yield* lifecycle.changes.pipe(
        Stream.filter((state) => state._tag !== "Checking"),
        Stream.runHead,
        Effect.fork
      );
      yield* ensure.pipe(Effect.ignore, Effect.forkDaemon);
      const observed = yield* Fiber.join(firstVisible);
      return yield* Option.match(observed, {
        onNone: () =>
          Effect.dieMessage(
            "ACN lifecycle ended before publishing a visible startup state"
          ),
        onSome: Effect.succeed,
      });
    });

    const retry = lifecycle
      .report({
        _tag: "Starting" as const,
        phase: "Discovering" as const,
      })
      .pipe(Effect.zipRight(ensure), Effect.asVoid);

    return {
      identity: SubscriptionRef.get(association).pipe(
        Effect.map((current) => current.identity)
      ),
      identityChanges: association.changes.pipe(
        Stream.map((current) => current.identity),
        Stream.changes
      ),
      startup: {
        state: lifecycle,
        prepare,
        retry,
      },
      protocolLayer: jitRecoveringProtocolLayer({
        endpoint: ensure,
        recover,
        rpcPath: "/rpc",
        streamProtocol: acnSubscriptionProtocol,
        isEndpointRetirementExit: isInterruptedExit,
        classifyInfraError: unavailableError,
      }),
    };
  });
