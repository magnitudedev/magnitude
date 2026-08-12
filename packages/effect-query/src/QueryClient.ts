import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
import * as Cause from "effect/Cause"
import * as Context from "effect/Context"
import * as Data from "effect/Data"
import * as Effect from "effect/Effect"
import * as Exit from "effect/Exit"
import * as FiberId from "effect/FiberId"
import * as Layer from "effect/Layer"
import * as Option from "effect/Option"
import type * as Stream from "effect/Stream"
import type { Query, QueryAtom, State as QueryState } from "./Query.js"
import {
  type ErasedQueryEntry,
  getClientCore,
  mutationMatches,
  queryEntry,
  queryMatches,
  registerEntry,
  type QueryClientEvent,
  type QueryFilter,
  type QueryMetadata
} from "./internal.js"
import type { AnyMutationState, MutationFilter } from "./Model.js"

export type { QueryClientEvent, QueryFilter, QueryMetadata }

export interface QueryBatchFailure {
  readonly name: string
  readonly keyHash: number
  readonly cause: Cause.Cause<unknown>
}

export class QueryBatchError extends Data.TaggedError("QueryBatchError")<{
  readonly failures: ReadonlyArray<QueryBatchFailure>
}> {}

export interface Service {
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
  readonly events: Stream.Stream<QueryClientEvent>
}

export class QueryClient extends Context.Tag("@magnitudedev/effect-query/QueryClient")<QueryClient, Service>() {}

function awaitQuery<Input, Data, Error, Requirements>(
  registry: AtomRegistry.Registry,
  query: QueryAtom<Input, Data, Error, Requirements>
): Effect.Effect<Data, Error> {
  return Effect.async((resume) => {
    const core = getClientCore(registry)
    let unsubscribe: (() => void) | undefined
    let completed = false
    const inspect = (state: QueryState<Data, Error>) => {
      if (completed) return
      if (core.removed.has(queryEntry(query))) {
        completed = true
        resume(Exit.interrupt(FiberId.none))
        unsubscribe?.()
        return
      }
      if (state.result._tag === "Failure" && !state.result.waiting) {
        completed = true
        resume(Exit.failCause(state.result.cause))
        unsubscribe?.()
        return
      }
      if (state.result._tag === "Success" && !state.result.waiting) {
        completed = true
        resume(Exit.succeed(state.result.value))
        unsubscribe?.()
      }
    }
    inspect(registry.get(query))
    if (!completed) unsubscribe = registry.subscribe(query, inspect)
    return Effect.sync(() => unsubscribe?.())
  })
}

const awaitErased = (
  registry: AtomRegistry.Registry,
  entry: ErasedQueryEntry
): Effect.Effect<void, unknown> => Effect.async((resume) => {
  let unsubscribe: (() => void) | undefined
  let completed = false
  const inspect = (state: ReturnType<ErasedQueryEntry["state"]>) => {
    if (completed || state.fetchStatus !== "idle") return
    completed = true
    const failure = entry.failureCause(registry)
    resume(Option.isSome(failure) ? Exit.failCause(failure.value) : Exit.void)
    unsubscribe?.()
  }
  inspect(entry.state(registry))
  if (!completed) unsubscribe = registry.subscribe(entry.stateAtom, inspect)
  return Effect.sync(() => unsubscribe?.())
})

const makeService = (
  registry: AtomRegistry.Registry,
  resolve: Service["resolve"]
): Service => {
  const core = getClientCore(registry)
  const entries = (filter?: QueryFilter) => [...core.entries].filter((entry) => queryMatches(registry, entry, filter))
  const entryFor = <Input, Data, Error, Requirements>(
    query: QueryAtom<Input, Data, Error, Requirements>
  ): readonly [ReturnType<typeof queryEntry<Data>>, boolean] => {
    const entry = queryEntry(query)
    const existed = core.entries.has(entry)
    registry.get(query)
    if (!core.entries.has(entry)) {
      core.removed.delete(entry)
      registerEntry(registry, entry)
    }
    return [entry, existed]
  }
  const fetchQuery = <Input, Data, Error, Requirements>(
    query: QueryAtom<Input, Data, Error, Requirements>
  ): Effect.Effect<Data, Error> => Effect.suspend(() => {
    const [entry, existed] = entryFor(query)
    const state = entry.state(registry)
    if (existed && state.fetchStatus !== "paused" && entry.hasData(registry) && !state.isStale) {
      return Effect.succeed(Option.getOrThrow(entry.getData(registry)))
    }
    if ((existed || state.fetchStatus === "paused") && state.fetchStatus !== "fetching") entry.start(registry)
    return awaitQuery(registry, query)
  })

  return {
    resolve,
    fetch: fetchQuery,
    ensure: (query) => Effect.suspend(() => {
      const [entry] = entryFor(query)
      if (entry.hasData(registry)) {
        if (entry.state(registry).isStale && entry.state(registry).fetchStatus !== "fetching") entry.start(registry)
        return Effect.succeed(Option.getOrThrow(entry.getData(registry)))
      }
      if (entry.state(registry).fetchStatus !== "fetching") entry.start(registry)
      return awaitQuery(registry, query)
    }),
    prefetch: (query) => Effect.ignore(fetchQuery(query)),
    invalidate: (filter, options) => Effect.sync(() => {
      for (const entry of entries(filter)) {
        entry.invalidate(registry)
        core.emit({ _tag: "QueryInvalidated", name: entry.name, keyHash: entry.keyHash })
        if (options?.refetch !== false) entry.start(registry, { cancelRefetch: true })
      }
      core.touch()
    }),
    refetch: (filter) => Effect.gen(function*() {
      const failures: Array<QueryBatchFailure> = []
      for (const entry of entries(filter)) {
        entry.start(registry, { cancelRefetch: true })
        const state = yield* Effect.exit(awaitErased(registry, entry))
        if (state._tag === "Failure") failures.push({ name: entry.name, keyHash: entry.keyHash, cause: state.cause })
      }
      if (failures.length > 0) return yield* new QueryBatchError({ failures })
    }),
    cancel: (filter) => Effect.sync(() => {
      for (const entry of entries(filter)) entry.cancel(registry)
      core.touch()
    }),
    remove: (filter) => Effect.sync(() => {
      for (const entry of entries(filter)) entry.remove(registry)
      core.touch()
    }),
    getState: (query) => Effect.sync(() => {
      const entry = queryEntry(query)
      return core.entries.has(entry)
        ? Option.some(registry.get(query))
        : Option.none()
    }),
    setData: (query, update) => Effect.sync(() => {
      const [entry] = entryFor(query)
      entry.setData(registry, update)
      core.touch()
    }),
    isFetching: (filter) => Atom.readable((get) => {
      get(core.revision)
      const matching = entries(filter)
      for (const entry of matching) get(entry.stateAtom)
      return matching.filter((entry) => entry.state(registry).fetchStatus === "fetching").length
    }),
    isMutating: (filter) => Atom.readable((get) => {
      get(core.revision)
      return core.mutationStates.filter((state) => mutationMatches(state, filter)).filter((state) =>
        state.result.waiting || state.result._tag === "Initial"
      ).length
    }),
    mutationState: (filter) => Atom.readable((get) => {
      get(core.revision)
      return core.mutationStates.filter((state) => mutationMatches(state, filter))
    }),
    events: core.events
  }
}

/** Internal layer constructor used by a connection-scoped Client. */
export const makeLayer = (
  resolve: Service["resolve"]
): Layer.Layer<QueryClient, never, AtomRegistry.AtomRegistry> =>
  Layer.effect(QueryClient, Effect.map(AtomRegistry.AtomRegistry, (registry) => makeService(registry, resolve)))

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
