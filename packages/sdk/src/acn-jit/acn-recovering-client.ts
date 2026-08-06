import * as HttpClient from "@effect/platform/HttpClient";
import {
  ClientIdSchema,
  MagnitudeRpcs,
  type AcnIdentity,
  type AcnInstance,
  type ClientId,
  type ClientLeaseMutationResult,
  type ModelSlotsState,
} from "@magnitudedev/acn-protocol";
import {
  canUseAcnIdentity,
  compareAcnIdentities,
} from "@magnitudedev/acn-protocol/acn-identity";
import { RpcClient, RpcClientError } from "@effect/rpc";
import {
  Cause,
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
} from "effect";
import {
  isInterruptedExit,
  recoveringProtocolLayer as jitRecoveringProtocolLayer,
} from "../jit-rpc";
import { AcnProcessManager, runAcnLaunch } from "./acn-process-manager";
import { DaemonDiscoveryFailed, type DaemonError } from "./errors";
import { SDK_VERSION } from "../version";
import { acnSubscriptionProtocol } from "./acn-subscription-protocol";
import {
  makeAcnLifecycle,
  acnLifecycleObservationFromHealthState,
  acnStartupProgressKey,
  type AcnLifecycle,
  type AcnLifecycleState,
} from "./lifecycle";
import type { AcnClient } from "../protocol";

const CLIENT_LEASE_RENEWAL_INTERVAL = Duration.seconds(15);
const CLIENT_LEASE_RELEASE_TIMEOUT = Duration.seconds(2);
const CLIENT_CLOSE_OBSERVATION_TIMEOUT = Duration.seconds(2);

type ReleaseClientLeaseThrough = (client: ClientLeaseRpcClient) => Effect.Effect<
  ClientLeaseMutationResult,
  RpcClientError.RpcClientError | Cause.TimeoutException
>;

type ClientLeaseRpcClient = Pick<
  AcnClient,
  "RenewClientLease" | "ReleaseClientLease"
>;

export interface AcnClientLeaseOwner {
  readonly clientId: ClientId;
  readonly stop: Effect.Effect<void>;
  readonly releaseThrough: ReleaseClientLeaseThrough;
}

/** Owns one client's scoped heartbeat and graceful release capability. */
export const makeAcnClientLeaseOwner = (
  clientId: ClientId,
  client: ClientLeaseRpcClient
): Effect.Effect<AcnClientLeaseOwner, never, Scope.Scope> =>
  Effect.gen(function* () {
    const released = yield* Ref.make(Option.none<ClientLeaseMutationResult>());
    const releaseLock = yield* Effect.makeSemaphore(1);

    const renew = client.RenewClientLease({ clientId }).pipe(
      Effect.tapError((error) =>
        Effect.logWarning("Failed to renew ACN client lease").pipe(
          Effect.annotateLogs({ clientId, error: String(error) })
        )
      ),
      Effect.ignore
    );
    const heartbeat = yield* renew.pipe(
      Effect.repeat(Schedule.spaced(CLIENT_LEASE_RENEWAL_INTERVAL)),
      Effect.forkScoped
    );

    const stop = Fiber.interrupt(heartbeat);
    const releaseThrough: ReleaseClientLeaseThrough = (releaseClient) =>
      releaseLock.withPermits(1)(
        Ref.get(released).pipe(
          Effect.flatMap(
            Option.match({
              onSome: Effect.succeed,
              onNone: () =>
                stop.pipe(
                  Effect.zipRight(
                    releaseClient
                      .ReleaseClientLease({ clientId })
                      .pipe(Effect.timeout(CLIENT_LEASE_RELEASE_TIMEOUT))
                  ),
                  Effect.tap((result) => Ref.set(released, Option.some(result)))
                ),
            })
          )
        )
      );

    yield* Effect.addFinalizer(() => stop);
    return { clientId, stop, releaseThrough };
  });

export interface AcnJitRuntimeOptions {
  readonly launchCommand: Option.Option<ReadonlyArray<string>>;
}

export interface AcnStartup {
  readonly state: AcnLifecycle;
  readonly prepare: Effect.Effect<AcnLifecycleState>;
  readonly retry: Effect.Effect<void, DaemonError>;
}

export interface AcnClientCloseReport {
  readonly modelSlots: ModelSlotsState;
  /** The authoritative connected count after this client's lease was removed. */
  readonly connectedClientCount: number;
}

export type AcnClientCloseResult = Option.Option<AcnClientCloseReport>;

export interface AcnJitRuntime {
  readonly identity: Effect.Effect<AcnIdentity>;
  readonly identityChanges: Stream.Stream<AcnIdentity>;
  readonly protocolLayer: Layer.Layer<
    RpcClient.Protocol,
    never,
    HttpClient.HttpClient
  >;
  /** Closes this interactive client lifetime and returns complete exit observation when available. */
  readonly close: Effect.Effect<AcnClientCloseResult>;
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

const runtimeClosedError = () =>
  new DaemonDiscoveryFailed({ reason: "ACN client runtime is closed" });

export const makeAcnJitRuntime = (
  options: AcnJitRuntimeOptions = { launchCommand: Option.none() }
): Effect.Effect<
  AcnJitRuntime,
  never,
  AcnProcessManager | HttpClient.HttpClient | Scope.Scope
> =>
  Effect.gen(function* () {
    const processManager = yield* AcnProcessManager;
    const httpClient = yield* HttpClient.HttpClient;
    const runtimeScope = yield* Scope.Scope;
    const lifecycle = yield* makeAcnLifecycle();
    const association = yield* SubscriptionRef.make<AcnAssociation>({
      identity: SDK_VERSION,
      selected: Option.none(),
    });
    const selectionLock = yield* Effect.makeSemaphore(1);
    const open = yield* Ref.make(true);
    const clientId = yield* Effect.sync(() =>
      ClientIdSchema.make(globalThis.crypto.randomUUID())
    );

    const reportInstance = (instance: AcnInstance) =>
      Option.match(acnLifecycleObservationFromHealthState(instance.lifecycle), {
        onNone: () => Effect.void,
        onSome: lifecycle.report,
      });

    const requireOpen = Ref.get(open).pipe(
      Effect.flatMap((isOpen) =>
        isOpen ? Effect.void : Effect.fail(runtimeClosedError())
      )
    );

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
      Effect.gen(function* () {
        yield* requireOpen;
        const current = yield* SubscriptionRef.get(association);
        yield* requireOpen;
        const instance = yield* runAcnLaunch(
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
        );
        yield* reconcile(instance);
        return instance;
      });

    const observeStarting = (
      initial: AcnInstance
    ): Effect.Effect<AcnInstance, DaemonError> =>
      Effect.gen(function* () {
        let observed = initial;
        let key = acnStartupProgressKey(initial.lifecycle);
        let deadline = Date.now() + 30_000;
        while (true) {
          yield* requireOpen;
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
        yield* requireOpen;
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
        requireOpen.pipe(
          Effect.zipRight(processManager.observeCurrent),
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
            yield* requireOpen;
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

    const retry = requireOpen.pipe(
      Effect.zipRight(
        lifecycle.report({
          _tag: "Starting" as const,
          phase: "Discovering" as const,
        })
      ),
      Effect.zipRight(ensure),
      Effect.asVoid
    );

    const recoveringProtocolLayer = jitRecoveringProtocolLayer({
      endpoint: ensure,
      recover,
      rpcPath: "/rpc",
      streamProtocol: acnSubscriptionProtocol,
      isEndpointRetirementExit: isInterruptedExit,
      classifyInfraError: unavailableError,
    });

    // RpcClient.Protocol is single-consumer: each RpcClient.make owns its run
    // loop. The lease client therefore gets a dedicated protocol instance.
    // Application protocol instances still share this runtime's endpoint
    // selection and recovery authority through the closures above.
    const leaseProtocolContext = yield* Layer.build(recoveringProtocolLayer);
    const leaseClient = yield* RpcClient.make(MagnitudeRpcs).pipe(
      Effect.provide(leaseProtocolContext)
    );
    const leaseOwner = yield* makeAcnClientLeaseOwner(clientId, leaseClient);
    yield* Effect.addFinalizer(() => Ref.set(open, false));
    const closeResult = yield* Ref.make(Option.none<AcnClientCloseResult>());
    const closeLock = yield* Effect.makeSemaphore(1);

    const close: AcnJitRuntime["close"] = closeLock.withPermits(1)(
      Ref.get(closeResult).pipe(
        Effect.flatMap(
          Option.match({
            onSome: Effect.succeed,
            onNone: () =>
              Effect.gen(function* () {
                yield* Ref.set(open, false);
                yield* leaseOwner.stop;
                const selected = (yield* SubscriptionRef.get(association)).selected;
                if (Option.isNone(selected)) {
                  const result = Option.none<AcnClientCloseReport>();
                  yield* Ref.set(closeResult, Option.some(result));
                  return result;
                }

                const closeProtocolContext = yield* Layer.buildWithScope(
                  jitRecoveringProtocolLayer({
                    endpoint: Effect.succeed(selected.value),
                    recover: () => Effect.fail(runtimeClosedError()),
                    rpcPath: "/rpc",
                    streamProtocol: acnSubscriptionProtocol,
                    isEndpointRetirementExit: isInterruptedExit,
                    classifyInfraError: unavailableError,
                  }).pipe(
                    Layer.provide(
                      Layer.succeed(HttpClient.HttpClient, httpClient)
                    )
                  ),
                  runtimeScope
                );
                const closeClient = yield* RpcClient.make(MagnitudeRpcs).pipe(
                  Effect.provide(closeProtocolContext),
                  Effect.provideService(Scope.Scope, runtimeScope)
                );
                const modelSlots = yield* closeClient.GetModelSlots({}).pipe(
                  Effect.map((result) => result.state),
                  Effect.timeout(CLIENT_CLOSE_OBSERVATION_TIMEOUT),
                  Effect.exit,
                  Effect.map(
                    Exit.match({
                      onFailure: () => Option.none<ModelSlotsState>(),
                      onSuccess: Option.some,
                    })
                  )
                );
                const release = yield* leaseOwner.releaseThrough(closeClient).pipe(
                  Effect.exit,
                  Effect.map(
                    Exit.match({
                      onFailure: () => Option.none<ClientLeaseMutationResult>(),
                      onSuccess: Option.some,
                    })
                  )
                );
                const result = Option.all({ modelSlots, release }).pipe(
                  Option.map(({ modelSlots, release }) => ({
                    modelSlots,
                    connectedClientCount: release.connectedClientCount,
                  }))
                );
                yield* Ref.set(closeResult, Option.some(result));
                return result;
              }),
          })
        )
      )
    );

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
      close,
      protocolLayer: recoveringProtocolLayer,
    };
  });
