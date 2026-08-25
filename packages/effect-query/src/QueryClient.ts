import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
import * as Cause from "effect/Cause"
import * as Chunk from "effect/Chunk"
import * as Data from "effect/Data"
import * as Effect from "effect/Effect"
import * as HashSet from "effect/HashSet"
import * as Layer from "effect/Layer"
import * as Option from "effect/Option"
import * as Stream from "effect/Stream"
import * as SubscriptionRef from "effect/SubscriptionRef"
import type { Query, QueryAtom, State as QueryState } from "./Query.js"
import {
  type ErasedQueryEntry,
  makeQueryCache,
  mutationMatches,
  QueryCacheTypeId,
  QueryClient,
  type QueryCache,
  type QueryFetch,
  queryEntry,
  queryMatches,
  subscriptionMatches,
  type QueryClientEvent,
  type QueryFilter,
  type QueryMetadata
} from "./internal.js"
import type { AnyMutationState, MutationFilter, SubscriptionFilter } from "./Model.js"

export type { QueryClientEvent, QueryFilter, QueryMetadata, SubscriptionFilter }
export { QueryClient }

export interface QueryBatchFailure {
  readonly name: string
  readonly keyHash: number
  readonly cause: Cause.Cause<unknown>
}

export class QueryBatchError extends Data.TaggedError("QueryBatchError")<{
  readonly failures: ReadonlyArray<QueryBatchFailure>
}> {}

export interface Service {
  /** The client's caches; package-internal. */
  readonly [QueryCacheTypeId]: QueryCache
  readonly resolve: <Input, Data, Error, Requirements>(
    query: Query<Input, Data, Error, Requirements>,
    input: Input
  ) => QueryAtom<Input, Data, Error, Requirements>
  readonly fetch: <Input, Data, Error, Requirements>(
    query: QueryAtom<Input, Data, Error, Requirements>
  ) => Effect.Effect<Data, Error>
  readonly ensure: <Input, Data, Error, Requirements>(
    query: QueryAtom<Input, Data, Error, Requirements>
  ) => Effect.Effect<Data, Error>
  readonly prefetch: <Input, Data, Error, Requirements>(
    query: QueryAtom<Input, Data, Error, Requirements>
  ) => Effect.Effect<void>
  readonly invalidate: (
    filter?: QueryFilter,
    options?: { readonly refetch?: boolean }
  ) => Effect.Effect<void>
  readonly refetch: (filter?: QueryFilter) => Effect.Effect<void, QueryBatchError>
  readonly cancel: (filter?: QueryFilter) => Effect.Effect<void>
  readonly remove: (filter?: QueryFilter) => Effect.Effect<void>
  readonly getState: <Input, Data, Error, Requirements>(
    query: QueryAtom<Input, Data, Error, Requirements>
  ) => Effect.Effect<Option.Option<QueryState<Data, Error>>>
  readonly setData: <Input, Data, Error, Requirements>(
    query: QueryAtom<Input, Data, Error, Requirements>,
    update: (current: Option.Option<Data>) => Data
  ) => Effect.Effect<void>
  readonly isFetching: (filter?: QueryFilter) => Atom.Atom<number>
  readonly isMutating: (filter?: MutationFilter) => Atom.Atom<number>
  readonly mutationState: (filter?: MutationFilter) => Atom.Atom<ReadonlyArray<AnyMutationState>>
  /** Reopens every matching mounted subscription now (`Atom.Reset`). */
  readonly reconnect: (filter?: SubscriptionFilter) => Effect.Effect<void>
  readonly events: Stream.Stream<QueryClientEvent>
}

/** Await the exact fetch ticket while the query remains in this cache. */
const awaitFetch = <Data>(
  cache: QueryCache,
  entry: ErasedQueryEntry,
  fetch: QueryFetch<Data>
): Effect.Effect<Data, unknown> => Effect.raceFirst(
  fetch.await,
  cache.awaitRemoval(entry).pipe(Effect.zipRight(Effect.interrupt))
)

const makeService = (
  registry: AtomRegistry.Registry,
  cache: QueryCache,
  resolve: Service["resolve"]
): Service => {
  const queriesAtom = Atom.subscriptionRef(cache.queries)
  const mutationsAtom = Atom.subscriptionRef(cache.mutations)

  const entries = (filter?: QueryFilter): Effect.Effect<ReadonlyArray<ErasedQueryEntry>> =>
    Effect.map(SubscriptionRef.get(cache.queries), (current) =>
      [...current].filter((entry) => queryMatches(registry, entry, filter)))

  /** Make the entry current in the cache (restoring it if removed) and materialize its atoms. */
  const entryFor = <Input, Data, Error, Requirements>(
    query: QueryAtom<Input, Data, Error, Requirements>
  ): Effect.Effect<readonly [ReturnType<typeof queryEntry<Data>>, boolean]> => Effect.gen(function*() {
    const entry = queryEntry(query)
    const existed = HashSet.has(yield* SubscriptionRef.get(cache.queries), entry)
    yield* cache.restoreQuery(entry)
    registry.get(query)
    return [entry, existed] as const
  })

  const fetchQuery = <Input, Data, Error, Requirements>(
    query: QueryAtom<Input, Data, Error, Requirements>
  ): Effect.Effect<Data, Error> => Effect.gen(function*() {
    const [entry, existed] = yield* entryFor(query)
    const state = entry.state(registry)
    if (state.fetchStatus !== "paused" && entry.hasData(registry) && !state.isStale) {
      return Option.getOrThrow(entry.getData(registry))
    }
    const fetch = !existed && state.fetchStatus !== "paused"
      ? entry.join(registry)
      : entry.start(registry)
    return yield* awaitFetch(cache, entry, fetch) as Effect.Effect<Data, Error>
  })

  const ensureQuery = <Input, Data, Error, Requirements>(
    query: QueryAtom<Input, Data, Error, Requirements>
  ): Effect.Effect<Data, Error> => Effect.gen(function*() {
    const [entry, existed] = yield* entryFor(query)
    if (entry.hasData(registry)) {
      if (existed && entry.state(registry).isStale && entry.state(registry).fetchStatus !== "fetching") {
        entry.start(registry)
      }
      return Option.getOrThrow(entry.getData(registry))
    }
    const fetch = !existed && entry.state(registry).fetchStatus !== "paused"
      ? entry.join(registry)
      : entry.start(registry)
    return yield* awaitFetch(cache, entry, fetch) as Effect.Effect<Data, Error>
  })

  return {
    [QueryCacheTypeId]: cache,
    resolve,
    fetch: fetchQuery,
    ensure: ensureQuery,
    prefetch: (query) => Effect.ignore(fetchQuery(query)),
    invalidate: (filter, options) => Effect.gen(function*() {
      for (const entry of yield* entries(filter)) {
        entry.invalidate(registry)
        yield* cache.emit({ _tag: "QueryInvalidated", name: entry.name, keyHash: entry.keyHash })
        if (options?.refetch !== false) entry.start(registry, { cancelRefetch: true })
      }
    }),
    refetch: (filter) => Effect.gen(function*() {
      const started = (yield* entries(filter)).map((entry) => ({
        entry,
        fetch: entry.start(registry, { cancelRefetch: true })
      }))
      const exits = yield* Effect.forEach(
        started,
        ({ entry, fetch }) => Effect.exit(awaitFetch(cache, entry, fetch)),
        { concurrency: "unbounded" }
      )
      const failures: Array<QueryBatchFailure> = exits.flatMap((exit, index) => exit._tag === "Failure"
        ? [{ name: started[index]!.entry.name, keyHash: started[index]!.entry.keyHash, cause: exit.cause }]
        : [])
      if (failures.length > 0) return yield* new QueryBatchError({ failures })
    }),
    cancel: (filter) => Effect.gen(function*() {
      for (const entry of yield* entries(filter)) yield* entry.cancel(registry)
    }),
    remove: (filter) => Effect.gen(function*() {
      for (const entry of yield* entries(filter)) {
        yield* cache.removeQuery(entry)
        yield* entry.cancel(registry)
      }
    }),
    getState: (query) => Effect.map(SubscriptionRef.get(cache.queries), (current) =>
      HashSet.has(current, queryEntry(query))
        ? Option.some(registry.get(query))
        : Option.none()),
    setData: (query, update) => Effect.gen(function*() {
      const [entry] = yield* entryFor(query)
      yield* entry.setData(registry, update)
    }),
    isFetching: (filter) => Atom.readable((get) => {
      const matching = [...get(queriesAtom)].filter((entry) => queryMatches(registry, entry, filter))
      for (const entry of matching) get(entry.stateAtom)
      return matching.filter((entry) => entry.state(registry).fetchStatus === "fetching").length
    }),
    isMutating: (filter) => Atom.readable((get) =>
      Chunk.toReadonlyArray(get(mutationsAtom))
        .filter((state) => mutationMatches(state, filter))
        .filter((state) => state.result.waiting || state.result._tag === "Initial")
        .length),
    mutationState: (filter) => Atom.readable((get) =>
      Chunk.toReadonlyArray(get(mutationsAtom)).filter((state) => mutationMatches(state, filter))),
    reconnect: (filter) => Effect.gen(function*() {
      for (const entry of yield* SubscriptionRef.get(cache.subscriptions)) {
        if (subscriptionMatches(entry, filter)) entry.reconnect(registry)
      }
    }),
    events: cache.events
  }
}

/** Internal layer constructor used by a connection-scoped Client. Owns the client's caches. */
export const makeLayer = (
  resolve: Service["resolve"]
): Layer.Layer<QueryClient, never, AtomRegistry.AtomRegistry> =>
  Layer.scoped(QueryClient, Effect.gen(function*() {
    const registry = yield* AtomRegistry.AtomRegistry
    const cache = yield* makeQueryCache
    return makeService(registry, cache, resolve)
  }))

export const fetch = <Input, Data, Error, Requirements>(
  query: Query<Input, Data, Error, Requirements>,
  input: Input
): Effect.Effect<Data, Error, QueryClient> =>
  Effect.flatMap(QueryClient, (client) => client.fetch(client.resolve(query, input)))

export const ensure = <Input, Data, Error, Requirements>(
  query: Query<Input, Data, Error, Requirements>,
  input: Input
): Effect.Effect<Data, Error, QueryClient> =>
  Effect.flatMap(QueryClient, (client) => client.ensure(client.resolve(query, input)))

export const prefetch = <Input, Data, Error, Requirements>(
  query: Query<Input, Data, Error, Requirements>,
  input: Input
): Effect.Effect<void, never, QueryClient> =>
  Effect.flatMap(QueryClient, (client) => client.prefetch(client.resolve(query, input)))

export const invalidate = (
  filter?: QueryFilter,
  options?: { readonly refetch?: boolean }
): Effect.Effect<void, never, QueryClient> => Effect.flatMap(QueryClient, (client) => client.invalidate(filter, options))

export const refetch = (filter?: QueryFilter): Effect.Effect<void, QueryBatchError, QueryClient> =>
  Effect.flatMap(QueryClient, (client) => client.refetch(filter))

export const cancel = (filter?: QueryFilter): Effect.Effect<void, never, QueryClient> =>
  Effect.flatMap(QueryClient, (client) => client.cancel(filter))

export const remove = (filter?: QueryFilter): Effect.Effect<void, never, QueryClient> =>
  Effect.flatMap(QueryClient, (client) => client.remove(filter))

export const getState = <Input, Data, Error, Requirements>(
  query: Query<Input, Data, Error, Requirements>,
  input: Input
): Effect.Effect<Option.Option<QueryState<Data, Error>>, never, QueryClient> =>
  Effect.flatMap(QueryClient, (client) => client.getState(client.resolve(query, input)))

export const setData = <Input, Data, Error, Requirements>(
  query: Query<Input, Data, Error, Requirements>,
  input: Input,
  update: (current: Option.Option<Data>) => Data
): Effect.Effect<void, never, QueryClient> =>
  Effect.flatMap(QueryClient, (client) => client.setData(client.resolve(query, input), update))

export const isFetching = (filter?: QueryFilter): Effect.Effect<Atom.Atom<number>, never, QueryClient> =>
  Effect.map(QueryClient, (client) => client.isFetching(filter))

export const isMutating = (filter?: MutationFilter): Effect.Effect<Atom.Atom<number>, never, QueryClient> =>
  Effect.map(QueryClient, (client) => client.isMutating(filter))

export const mutationState = (
  filter?: MutationFilter
): Effect.Effect<Atom.Atom<ReadonlyArray<AnyMutationState>>, never, QueryClient> =>
  Effect.map(QueryClient, (client) => client.mutationState(filter))

export const reconnect = (filter?: SubscriptionFilter): Effect.Effect<void, never, QueryClient> =>
  Effect.flatMap(QueryClient, (client) => client.reconnect(filter))
