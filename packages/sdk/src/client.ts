import { HttpClient } from "@effect/platform";
import { RpcClient, RpcClientError, RpcSerialization } from "@effect/rpc";
import {
  MagnitudeHealthResponseSchema,
  AcnIdentitySchema,
  AcnInstanceIdSchema,
  AcnRpcGroup,
  AcnRpcRecoveryPolicyTag,
  MagnitudeRpcs,
  MAGNITUDE_RPC_VERSION,
  namespaceClient,
  ServiceStartProgressSchema,
  serviceProgressFromHealth,
  type TreeClient,
} from "@magnitudedev/acn-protocol";
import { FSM } from "@magnitudedev/utils";
import {
  Cause,
  Context,
  Deferred,
  Duration,
  Effect,
  Exit,
  Layer,
  Option,
  Schema,
  Scope,
  Stream,
  SubscriptionRef,
} from "effect";
import { MagnitudeServiceStarter } from "./service-starter";
import {
  ConnectionClosed,
  ConnectionErrorSchema,
  InvalidServiceResponse,
  ProtocolMismatch,
  ServiceUnavailable,
  type ConnectionError,
} from "./connection-errors";
import { acnSubscriptionProtocol } from "./acn-jit/acn-subscription-protocol";
import { isInterruptedExit, makeRecoveringProtocol } from "./jit-rpc";
import { MAGNITUDE_SERVICE_ORIGIN } from "@magnitudedev/acn-protocol";

export const ServiceInfoSchema = Schema.Struct({
  id: AcnInstanceIdSchema,
  version: AcnIdentitySchema,
  rpcVersion: Schema.Number.pipe(Schema.int(), Schema.positive()),
});
export type ServiceInfo = typeof ServiceInfoSchema.Type;
class Idle extends Schema.TaggedClass<Idle>()("Idle", {}) {}
class Connecting extends Schema.TaggedClass<Connecting>()("Connecting", {
  reason: Schema.Literal("initial", "recovery", "retry"),
  activity: Schema.optionalWith(ServiceStartProgressSchema, {
    as: "Option",
    exact: true,
  }),
}) {}
class Ready extends Schema.TaggedClass<Ready>()("Ready", {
  service: ServiceInfoSchema,
}) {}
class Failed extends Schema.TaggedClass<Failed>()("Failed", {
  error: ConnectionErrorSchema,
}) {}
class Closed extends Schema.TaggedClass<Closed>()("Closed", {}) {}
const States = { Idle, Connecting, Ready, Failed, Closed };
export const ConnectionStateSchema = Schema.Union(...Object.values(States));
export type ConnectionState = typeof ConnectionStateSchema.Type;
const ConnectionFsm = FSM.defineFSM(States, {
  Idle: ["Connecting", "Closed"],
  Connecting: ["Connecting", "Ready", "Failed", "Closed"],
  Ready: ["Connecting", "Closed"],
  Failed: ["Connecting", "Closed"],
  Closed: [],
} as const);

export interface MagnitudeConnection {
  readonly state: Effect.Effect<ConnectionState>;
  readonly changes: Stream.Stream<ConnectionState>;
  readonly connect: Effect.Effect<void, ConnectionError>;
}
export type MagnitudeClientError =
  | RpcClientError.RpcClientError
  | ConnectionError;
type Operations = TreeClient<typeof MagnitudeRpcs, MagnitudeClientError>;
export interface MagnitudeClient extends Operations {
  readonly connection: Operations["connection"] & MagnitudeConnection;
}
export const MagnitudeClient = Object.assign(
  Context.GenericTag<MagnitudeClient>("@magnitudedev/sdk/MagnitudeClient"),
  { layer: clientLayer }
);

export interface ClientOptions {
  readonly origin?: string;
  readonly connectTimeout?: Duration.DurationInput;
}
export function clientLayer(
  options: ClientOptions & { readonly autoStart: false }
): Layer.Layer<MagnitudeClient, never, HttpClient.HttpClient>;
export function clientLayer(
  options?: ClientOptions & { readonly autoStart?: true }
): Layer.Layer<
  MagnitudeClient,
  never,
  HttpClient.HttpClient | MagnitudeServiceStarter
>;
export function clientLayer(
  options: ClientOptions & { readonly autoStart?: boolean } = {}
): Layer.Layer<
  MagnitudeClient,
  never,
  HttpClient.HttpClient | MagnitudeServiceStarter
> {
  return Layer.scoped(MagnitudeClient, makeClient(options));
}

const makeClient = (
  options: ClientOptions & { readonly autoStart?: boolean }
) =>
  Effect.gen(function* () {
    const http = yield* HttpClient.HttpClient;
    const starter =
      options.autoStart === false
        ? Option.none()
        : Option.some(yield* MagnitudeServiceStarter);
    const origin = (options.origin ?? MAGNITUDE_SERVICE_ORIGIN).replace(
      /\/$/,
      ""
    );
    const scope = yield* Scope.make();
    const state = yield* SubscriptionRef.make<ConnectionState>(new Idle({}));
    const admission = yield* Effect.makeSemaphore(1);
    // These are scoped single-flight mechanics; public status is read-only.
    let active: Deferred.Deferred<ServiceInfo, ConnectionError> | undefined;
    let closed = false;
    let connectedBefore = false;

    yield* Effect.addFinalizer(() =>
      admission
        .withPermits(1)(
          Effect.gen(function* () {
            closed = true;
            const current = yield* SubscriptionRef.get(state);
            if (current._tag !== "Closed")
              yield* SubscriptionRef.set(
                state,
                ConnectionFsm.transition(current, "Closed", {})
              );
            if (active !== undefined)
              yield* Deferred.fail(active, new ConnectionClosed({}));
          })
        )
        .pipe(Effect.zipRight(Scope.close(scope, Exit.void)))
    );

    const unavailable = (message: string) =>
      new ServiceUnavailable({ origin, message });
    const probe = http.get(`${origin}/health`).pipe(
      Effect.timeoutFail({
        duration: "2 seconds",
        onTimeout: () => unavailable("Magnitude service health timed out"),
      }),
      Effect.mapError((error) =>
        error instanceof ServiceUnavailable
          ? error
          : unavailable("Magnitude service is unavailable")
      ),
      Effect.flatMap((response) =>
        response.status !== 200 && response.status !== 503
          ? Effect.fail(
              new InvalidServiceResponse({
                origin,
                message: `Health returned HTTP ${response.status}`,
              })
            )
          : response.json.pipe(
              Effect.flatMap(
                Schema.decodeUnknown(MagnitudeHealthResponseSchema)
              ),
              Effect.mapError(
                () =>
                  new InvalidServiceResponse({
                    origin,
                    message: "Invalid Magnitude health response",
                  })
              )
            )
      )
    );
    const report = (activity: typeof ServiceStartProgressSchema.Type) =>
      admission.withPermits(1)(
        Effect.gen(function* () {
          const current = yield* SubscriptionRef.get(state);
          if (!closed && current._tag === "Connecting")
            yield* SubscriptionRef.set(
              state,
              ConnectionFsm.transition(current, "Connecting", {
                reason: current.reason,
                activity: Option.some(activity),
              })
            );
        })
      );
    const validate = (
      health: typeof MagnitudeHealthResponseSchema.Type
    ): Effect.Effect<ServiceInfo, ConnectionError> =>
      health.rpcVersion !== MAGNITUDE_RPC_VERSION
        ? Effect.fail(
            new ProtocolMismatch({
              expected: MAGNITUDE_RPC_VERSION,
              actual: health.rpcVersion,
              daemonVersion: health.version,
            })
          )
        : Effect.succeed({
            id: health.id,
            version: health.version,
            rpcVersion: health.rpcVersion,
          });

    const acquire = Effect.gen(function* () {
      let started = false;
      while (true) {
        const observed = yield* Effect.either(probe);
        if (observed._tag === "Left") {
          if (observed.left._tag !== "ServiceUnavailable")
            return yield* observed.left;
          if (!started && Option.isSome(starter)) {
            started = true;
            yield* report({ _tag: "Starting", phase: "PreparingAcn" });
            yield* Effect.scoped(
              Effect.gen(function* () {
                // Observe health progress while a shell command is waiting for startup.
                yield* probe.pipe(
                  Effect.flatMap((health) =>
                    Option.match(serviceProgressFromHealth(health.state), {
                      onNone: () => Effect.void,
                      onSome: report,
                    })
                  ),
                  Effect.ignore,
                  Effect.zipRight(Effect.sleep("250 millis")),
                  Effect.forever,
                  Effect.forkScoped
                );
                yield* starter.value.start.pipe(Stream.runForEach(report));
              })
            );
            continue;
          }
          if (!started) return yield* observed.left;
        } else {
          const health = observed.right;
          const service = yield* validate(health);
          if (health.state._tag === "Ready") return service;
          yield* Option.match(serviceProgressFromHealth(health.state), {
            onNone: () => Effect.void,
            onSome: report,
          });
        }
        yield* Effect.sleep("250 millis");
      }
    }).pipe(
      Effect.timeoutFail({
        duration: options.connectTimeout ?? "10 minutes",
        onTimeout: () =>
          unavailable(
            "Magnitude service did not become ready before the connection deadline"
          ),
      })
    );

    const select = (
      failedId?: string
    ): Effect.Effect<ServiceInfo, ConnectionError> =>
      Effect.flatten(
        admission
          .withPermits(1)(
            Effect.gen(function* () {
              if (closed) return Effect.fail(new ConnectionClosed({}));
              const current = yield* SubscriptionRef.get(state);
              if (
                current._tag === "Ready" &&
                (failedId === undefined || current.service.id !== failedId)
              )
                return Effect.succeed(current.service);
              if (active !== undefined) return Deferred.await(active);
              const pending = yield* Deferred.make<
                ServiceInfo,
                ConnectionError
              >();
              active = pending;
              const reason = connectedBefore
                ? "recovery"
                : current._tag === "Failed"
                ? "retry"
                : "initial";
              if (current._tag === "Closed")
                return Effect.fail(new ConnectionClosed({}));
              yield* SubscriptionRef.set(
                state,
                ConnectionFsm.transition(current, "Connecting", {
                  reason,
                  activity: Option.none(),
                })
              );
              yield* acquire.pipe(
                Effect.exit,
                Effect.flatMap((result) =>
                  admission.withPermits(1)(
                    Effect.gen(function* () {
                      if (closed || active !== pending) return;
                      active = undefined;
                      const before = yield* SubscriptionRef.get(state);
                      if (before._tag !== "Connecting") return;
                      if (Exit.isSuccess(result)) {
                        connectedBefore = true;
                        yield* SubscriptionRef.set(
                          state,
                          ConnectionFsm.transition(before, "Ready", {
                            service: result.value,
                          })
                        );
                      } else {
                        const failure = Option.getOrElse(
                          Cause.failureOption(result.cause),
                          () =>
                            unavailable(
                              "Magnitude connection attempt was interrupted"
                            )
                        );
                        yield* SubscriptionRef.set(
                          state,
                          ConnectionFsm.transition(before, "Failed", {
                            error: failure,
                          })
                        );
                      }
                      yield* Deferred.done(pending, result);
                    })
                  )
                ),
                Effect.interruptible,
                Effect.forkIn(scope)
              );
              return Deferred.await(pending);
            })
          )
          .pipe(Effect.uninterruptible)
      );

    const protocol = yield* makeRecoveringProtocol({
      endpoint: select().pipe(
        Effect.map((service) => ({ id: service.id, url: origin }))
      ),
      recover: (failed) =>
        select(failed.id).pipe(
          Effect.map((service) => ({ id: service.id, url: origin }))
        ),
      rpcPath: "/rpc",
      streamProtocol: acnSubscriptionProtocol,
      isEndpointRetirementExit: isInterruptedExit,
      recoveryPolicy: (tag) => {
        const rpc = AcnRpcGroup.requests.get(tag);
        if (rpc === undefined) throw new TypeError(`Unknown RPC: ${tag}`);
        return Option.getOrThrow(
          Context.getOption(rpc.annotations, AcnRpcRecoveryPolicyTag)
        );
      },
      classifyInfraError: (error) =>
        new RpcClientError.RpcClientError({
          reason: "Unknown",
          message: error._tag,
          cause: error,
        }),
    }).pipe(Effect.provide(RpcSerialization.layerNdjson));
    const raw = yield* RpcClient.make(AcnRpcGroup, { flatten: true }).pipe(
      Effect.provideService(RpcClient.Protocol, protocol)
    );
    // Expose connection errors directly instead of requiring callers to inspect a transport cause.
    const unwrap = <E>(
      error: E | RpcClientError.RpcClientError
    ): E | MagnitudeClientError =>
      error instanceof RpcClientError.RpcClientError &&
      Schema.is(ConnectionErrorSchema)(error.cause)
        ? error.cause
        : error;
    const mapped = namespaceClient<typeof MagnitudeRpcs, RpcClientError.RpcClientError, MagnitudeClientError>(MagnitudeRpcs, raw, unwrap);
    return MagnitudeClient.of({
      ...mapped,
      connection: {
        ...mapped.connection,
        state: SubscriptionRef.get(state),
        changes: state.changes,
        connect: select().pipe(Effect.asVoid),
      },
    });
  });
