import { Context, Effect, Equal, Layer, Stream } from "effect"
import type { EngineClosed, EngineRuntime } from "./engine"
import type { ProjectionReader } from "./projection"
import { type ObservableEngineRuntime, RuntimeObservation } from "./runtime-internal"
import type { Emit } from "./worker"

type EffectFunction = (...args: readonly any[]) => Effect.Effect<any, any, any>
type QueryFunction = (...args: readonly any[]) => Effect.Effect<any, any, never>
type EffectSuccessOf<F extends EffectFunction> = ReturnType<F> extends Effect.Effect<infer A, any, any> ? A : never
type EffectErrorOf<F extends EffectFunction> = ReturnType<F> extends Effect.Effect<any, infer E, any> ? E : never

/**
 * A transaction's body holds the acceptance lock: `read` reads directly in it, and `emit`,
 * `interrupt` and `transact` die as a defect in it instead of waiting on themselves.
 */
export interface MutationContext<Event, StoreError = never> {
  readonly emit: Emit<Event, StoreError | EngineClosed>
  readonly read: ProjectionReader
  readonly interrupt: EngineRuntime<Event, StoreError>["interrupt"]
  /** The runtime's transaction: reads settled state and decides under the acceptance lock. */
  readonly transact: EngineRuntime<Event, StoreError>["transact"]
}

export interface QueryContext {
  readonly read: ProjectionReader
}

export interface Query<Fn extends QueryFunction> {
  readonly get: Fn
  readonly changes: QueryChanges<Fn>
}

type QueryChanges<Fn extends QueryFunction> = (
  ...args: Parameters<Fn>
) => Stream.Stream<EffectSuccessOf<Fn>, EffectErrorOf<Fn>>

type HostedQueries<Queries extends Readonly<Record<string, QueryFunction>>> = {
  readonly [Name in keyof Queries]: Query<Queries[Name]>
}

export interface FacadeService<
  Mutations extends Readonly<Record<string, EffectFunction>>,
  Queries extends Readonly<Record<string, QueryFunction>>,
> {
  readonly mutations: Mutations
  readonly queries: HostedQueries<Queries>
}

export interface FacadeHostOptions<Event, StoreError = never> {
  readonly runtime: EngineRuntime<Event, StoreError>
  /** How mutations emit: the runtime's `accept`, unless given. */
  readonly emit?: Emit<Event, StoreError | EngineClosed>
}

export interface FacadeDefinition<
  Event,
  Mutations extends Readonly<Record<string, EffectFunction>>,
  Queries extends Readonly<Record<string, QueryFunction>>,
  StoreError = never,
> {
  readonly name: string
  readonly Tag: Context.Tag<FacadeService<Mutations, Queries>, FacadeService<Mutations, Queries>>
  readonly make: (options: FacadeHostOptions<Event, StoreError>) => FacadeService<Mutations, Queries>
  readonly host: (options: FacadeHostOptions<Event, StoreError>) => Layer.Layer<FacadeService<Mutations, Queries>>
}

export interface FacadeConfig<
  Event,
  Mutations extends Readonly<Record<string, EffectFunction>>,
  Queries extends Readonly<Record<string, QueryFunction>>,
  StoreError = never,
> {
  readonly name: string
  readonly mutations: (context: MutationContext<Event, StoreError>) => Mutations
  readonly queries: (context: QueryContext) => Queries
}

// A hosted query reads one settled state. Its changes read on subscribing, after the latest
// settlement each time it catches up, and once more after the engine has closed, so the final state
// is never missed; consecutive equal values are emitted once.
const makeQuery = <Fn extends QueryFunction>(
  runtime: ObservableEngineRuntime,
  query: Fn,
): Query<Fn> => {
  const get = ((...args: Parameters<Fn>) => RuntimeObservation.settled(runtime, query(...args))) as Fn
  const changes = ((...args: Parameters<Fn>) => Stream.unwrapScoped(
    Effect.map(RuntimeObservation.subscribe(runtime), (occurrences) =>
      Stream.succeed(undefined).pipe(
        Stream.concat(Stream.fromQueue(occurrences)),
        Stream.concat(Stream.succeed(undefined)),
        Stream.mapEffect(() => get(...args)),
      )),
  ).pipe(Stream.changesWith(Equal.equals))) as QueryChanges<Fn>

  return { get, changes }
}

export const makeFacadeBuilder = <Event, StoreError = never>() => ({
  define: <
    const Mutations extends Readonly<Record<string, EffectFunction>>,
    const Queries extends Readonly<Record<string, QueryFunction>>,
  >(config: FacadeConfig<Event, Mutations, Queries, StoreError>): FacadeDefinition<Event, Mutations, Queries, StoreError> => {
    const Tag = Context.GenericTag<FacadeService<Mutations, Queries>>(
      `@magnitudedev/event-engine/facade/${config.name}`,
    )

    const make = ({ runtime, emit = runtime.accept }: FacadeHostOptions<Event, StoreError>): FacadeService<Mutations, Queries> => {
      const mutations = config.mutations({
        emit,
        read: runtime.read,
        interrupt: runtime.interrupt,
        transact: runtime.transact,
      })
      const definitions = config.queries({ read: runtime.read })
      const queries = Object.fromEntries(
        Object.entries(definitions).map(([name, query]) => [name, makeQuery(runtime, query)]),
      ) as HostedQueries<Queries>

      return { mutations, queries }
    }

    return {
      name: config.name,
      Tag,
      make,
      host: (options) => Layer.succeed(Tag, make(options)),
    }
  },
})
