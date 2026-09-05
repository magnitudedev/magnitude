import {
  MagnitudeClient,
  type ConnectionError,
  type ConnectionState,
  type AcnIdentity,
} from "@magnitudedev/sdk";
import {
  Clock,
  Context,
  Equal,
  Effect,
  Exit,
  Layer,
  Option,
  Schema,
  Scope,
  Stream,
  SubscriptionRef,
} from "effect";
import {
  ServiceLifecycleStateSchema,
  ServiceReady,
  renderLifecycle,
  lifecycleIsTimeDependent,
  type ServiceLifecycle,
  type ServiceLifecycleState,
} from "./lifecycle";

import {
  initialPresentation,
  reducePresentation,
  type PresentationModel,
} from "./presentation";

export interface ServiceStartup {
  readonly state: ServiceLifecycle;
  readonly prepare: Effect.Effect<ServiceLifecycleState>;
  readonly awaitReady: Effect.Effect<void, ConnectionError>;
  readonly retry: Effect.Effect<void, ConnectionError>;
  readonly recovery: ServiceRecovery;
}
export class ServiceRecoveryInactive extends Schema.TaggedClass<ServiceRecoveryInactive>()(
  "Inactive",
  {}
) {}
export class ServiceRecovering extends Schema.TaggedClass<ServiceRecovering>()(
  "Recovering",
  {
    occurrence: Schema.Number.pipe(Schema.int(), Schema.positive()),
    lifecycle: ServiceLifecycleStateSchema,
  }
) {}
export class ServiceRecovered extends Schema.TaggedClass<ServiceRecovered>()(
  "Recovered",
  {
    occurrence: Schema.Number.pipe(Schema.int(), Schema.positive()),
  }
) {}
export type ServiceRecoveryState =
  | ServiceRecoveryInactive
  | ServiceRecovering
  | ServiceRecovered;
export interface ServiceRecovery {
  readonly get: Effect.Effect<ServiceRecoveryState>;
  readonly changes: Stream.Stream<ServiceRecoveryState>;
}
export interface FirstPartyConnection {
  readonly client: MagnitudeClient;
  readonly identity: Effect.Effect<AcnIdentity, ConnectionError>;
  readonly identityChanges: Stream.Stream<AcnIdentity>;
  readonly close: Effect.Effect<void>;
  readonly startup: ServiceStartup;
}

/** Presentation only. Connection selection, startup, and recovery belong to the SDK. */
export const makeFirstPartyConnection = <R>(
  clientLayer: Layer.Layer<MagnitudeClient, never, R>
): Effect.Effect<FirstPartyConnection, never, R | Scope.Scope> =>
  Effect.gen(function* () {
    const scope = yield* Scope.make();
    const close = yield* Effect.cached(Scope.close(scope, Exit.void));
    yield* Effect.addFinalizer(() => close);
    const client = Context.get(
      yield* Layer.buildWithScope(clientLayer, scope),
      MagnitudeClient
    );
    const presentation = yield* SubscriptionRef.make<PresentationModel>(
      initialPresentation
    );
    yield* client.connection.changes.pipe(
      Stream.scanEffect(initialPresentation, (previous: PresentationModel, state: ConnectionState) =>
        Effect.map(Clock.currentTimeMillis, (now) =>
          reducePresentation(previous, state, now)
        )
      ),
      Stream.runForEach((model) => SubscriptionRef.set(presentation, model)),
      Effect.forkIn(scope)
    );
    const render = (model: PresentationModel, now: number) => ({
      startup:
        model._tag === "Bootstrapping"
          ? renderLifecycle(model.lifecycle, now)
          : new ServiceReady({}),
      recovery:
        model._tag === "Recovering"
          ? new ServiceRecovering({
              occurrence: model.occurrence,
              lifecycle: renderLifecycle(model.lifecycle, now),
            })
          : model._tag === "Online" && model.occurrence > 0
          ? new ServiceRecovered({ occurrence: model.occurrence })
          : new ServiceRecoveryInactive({}),
    });
    const get = SubscriptionRef.get(presentation).pipe(
      Effect.flatMap((model) =>
        Effect.map(Clock.currentTimeMillis, (now) => render(model, now))
      )
    );
    const changes = presentation.changes.pipe(
      Stream.flatMap(
        (model) => {
          const snapshot = Effect.map(Clock.currentTimeMillis, (now) =>
            render(model, now)
          );
          const initial = Stream.fromEffect(snapshot);
          return model._tag !== "Online" &&
            lifecycleIsTimeDependent(model.lifecycle)
            ? Stream.merge(
                initial,
                Stream.tick("50 millis").pipe(Stream.mapEffect(() => snapshot))
              )
            : initial;
        },
        { switch: true }
      )
    );
    const initial = {
      get: Effect.map(get, (value) => value.startup),
      changes: changes.pipe(
        Stream.map((value) => value.startup),
        Stream.changesWith((left, right) => Equal.equals(left, right))
      ),
    };
    const identities = client.connection.changes.pipe(
      Stream.filterMap((state) =>
        state._tag === "Ready"
          ? Option.some(state.service.version)
          : Option.none()
      ),
      Stream.changes
    );
    return {
      client,
      identity: client.connection.connect.pipe(
        Effect.zipRight(identities.pipe(Stream.runHead)),
        Effect.map(Option.getOrThrow)
      ),
      identityChanges: identities,
      close,
      startup: {
        state: initial,
        prepare: client.connection.connect.pipe(
          Effect.ignore,
          Effect.forkIn(scope),
          Effect.zipRight(initial.get)
        ),
        awaitReady: client.connection.connect,
        retry: client.connection.connect,
        recovery: {
          get: Effect.map(get, (value) => value.recovery),
          changes: changes.pipe(
            Stream.map((value) => value.recovery),
            Stream.changesWith((left, right) => Equal.equals(left, right))
          ),
        },
      },
    };
  });
