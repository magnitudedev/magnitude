import { Effect, Queue, Scope } from "effect"

export const EngineRuntimeObservationTypeId: unique symbol = Symbol.for(
  "@magnitudedev/event-engine/Runtime/Observation",
)

export interface ObservableEngineRuntime {
  readonly [EngineRuntimeObservationTypeId]: {
    /** One pending item at most, after each settlement the subscriber has not caught up with. */
    readonly subscribe: Effect.Effect<Queue.Dequeue<void>, never, Scope.Scope>
    /** Runs an effect over one settled state, whose reads of the engine read directly. */
    readonly settled: <A, E, R>(effect: Effect.Effect<A, E, R>) => Effect.Effect<A, E, R>
  }
}

export const RuntimeObservation = {
  subscribe: (runtime: ObservableEngineRuntime): Effect.Effect<Queue.Dequeue<void>, never, Scope.Scope> =>
    runtime[EngineRuntimeObservationTypeId].subscribe,
  settled: <A, E, R>(runtime: ObservableEngineRuntime, effect: Effect.Effect<A, E, R>): Effect.Effect<A, E, R> =>
    runtime[EngineRuntimeObservationTypeId].settled(effect),
}
