import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import * as Cause from "effect/Cause"
import * as Chunk from "effect/Chunk"
import * as Clock from "effect/Clock"
import * as Context from "effect/Context"
import * as Effect from "effect/Effect"
import * as Equal from "effect/Equal"
import * as Fiber from "effect/Fiber"
import * as HashMap from "effect/HashMap"
import * as HashSet from "effect/HashSet"
import * as Option from "effect/Option"
import * as PubSub from "effect/PubSub"
import * as Ref from "effect/Ref"
import type * as Scope from "effect/Scope"
import * as Stream from "effect/Stream"
import * as SubscriptionRef from "effect/SubscriptionRef"
import * as SynchronizedRef from "effect/SynchronizedRef"
import {
  type AnyMutationState,
  type MutationDefinition,
  type MutationState,
  type MutationStateId,
  type MutationScope,
  type MutationFilter,
  type QueryClientEvent,
  type QueryDefinition,
  type QueryEntryState,
  type QueryFilter,
  type QueryKey,
  type SubscriptionDefinition,
  type SubscriptionEntryState,
  type SubscriptionFilter
} from "./Model.js"
import type { Service as QueryClientService } from "./QueryClient.js"

export const QueryEntryTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/QueryEntry")
export const MutationInternalTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/MutationInternal")
export const SubscriptionEntryTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/SubscriptionEntry")
export const QueryCacheTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/QueryCache")

/** The connection's query client. Declared here so materialized atoms can reach their owning cache. */
export class QueryClient extends Context.Tag("@magnitudedev/effect-query/QueryClient")<
  QueryClient,
  QueryClientService
>() {}

export interface ErasedSubscriptionEntry {
  readonly stateAtom: Atom.Atom<SubscriptionEntryState>
  readonly definition: SubscriptionDefinition
  readonly name: string
  readonly key: QueryKey
  readonly keyHash: number
  readonly reconnect: (registry: AtomRegistry.Registry) => void
  readonly close: (registry: AtomRegistry.Registry) => void
}

export interface SubscriptionEntry<Event> extends ErasedSubscriptionEntry {
  /** Events of the shared stream; keeps the subscription mounted while consumed. */
  readonly events: (registry: AtomRegistry.Registry) => Stream.Stream<Event>
}

export interface SubscriptionEntryCarrier<Event> {
  readonly [SubscriptionEntryTypeId]: SubscriptionEntry<Event>
}

export const subscriptionEntry = <Event>(carrier: SubscriptionEntryCarrier<Event>): SubscriptionEntry<Event> =>
  carrier[SubscriptionEntryTypeId]

export interface ErasedQueryEntry {
  readonly stateAtom: Atom.Atom<QueryEntryState>
  readonly definition: QueryDefinition
  readonly name: string
  readonly key: QueryKey
  readonly keyHash: number
  readonly state: (registry: AtomRegistry.Registry) => QueryEntryState
  readonly failureCause: (registry: AtomRegistry.Registry) => Option.Option<Cause.Cause<unknown>>
  readonly start: (
    registry: AtomRegistry.Registry,
    options?: { readonly cancelRefetch?: boolean }
  ) => void
  readonly cancel: (registry: AtomRegistry.Registry) => void
  readonly invalidate: (registry: AtomRegistry.Registry) => void
}

export interface QueryEntry<Data> extends ErasedQueryEntry {
  readonly setData: (
    registry: AtomRegistry.Registry,
    update: (current: Option.Option<Data>) => Data
  ) => Effect.Effect<void>
  readonly hasData: (registry: AtomRegistry.Registry) => boolean
  readonly getData: (registry: AtomRegistry.Registry) => Option.Option<Data>
}

export interface QueryEntryCarrier<Data> {
  readonly [QueryEntryTypeId]: QueryEntry<Data>
}

export const queryEntry = <Data>(carrier: QueryEntryCarrier<Data>): QueryEntry<Data> =>
  carrier[QueryEntryTypeId]

export interface MutationController<Input, Output, Error> {
  /** Submit one invocation and return its id; the invocation runs in the owning cache's scope. */
  readonly submit: (registry: AtomRegistry.Registry, input: Input) => MutationStateId
  /** Resolve when the invocation's history entry is terminal, or fail if the runtime never built. */
  readonly await: (registry: AtomRegistry.Registry, id: MutationStateId) => Effect.Effect<Output, Error>
}

export interface MutationControllerCarrier<Input, Output, Error> {
  readonly [MutationInternalTypeId]: MutationController<Input, Output, Error>
}

export const mutationController = <Input, Output, Error>(
  carrier: MutationControllerCarrier<Input, Output, Error>
): MutationController<Input, Output, Error> =>
  carrier[MutationInternalTypeId]

/**
 * The QueryClient's caches: retained query and subscription entries, mutation
 * history, in-flight invocations, scope semaphores, and the client event bus.
 * Built in the QueryClient layer's scope; every operation is a pure in-memory
 * Effect that never touches the atom graph.
 */
export interface QueryCache {
  readonly queries: SubscriptionRef.SubscriptionRef<HashSet.HashSet<ErasedQueryEntry>>
  readonly subscriptions: SubscriptionRef.SubscriptionRef<HashSet.HashSet<ErasedSubscriptionEntry>>
  readonly mutations: SubscriptionRef.SubscriptionRef<Chunk.Chunk<AnyMutationState>>
  readonly events: Stream.Stream<QueryClientEvent>
  readonly emit: (event: QueryClientEvent) => Effect.Effect<void>
  readonly registerQuery: (entry: ErasedQueryEntry) => Effect.Effect<void>
  readonly unregisterQuery: (entry: ErasedQueryEntry) => Effect.Effect<void>
  /** Tombstone the entry and drop it from the cache; it stays out until restored. */
  readonly removeQuery: (entry: ErasedQueryEntry) => Effect.Effect<void>
  /** Clear the tombstone and re-add the entry. */
  readonly restoreQuery: (entry: ErasedQueryEntry) => Effect.Effect<void>
  /** Resolves once the entry has been removed. */
  readonly awaitRemoval: (entry: ErasedQueryEntry) => Effect.Effect<void>
  readonly registerSubscription: (entry: ErasedSubscriptionEntry) => Effect.Effect<void>
  readonly unregisterSubscription: (entry: ErasedSubscriptionEntry) => Effect.Effect<void>
  readonly addMutation: (state: AnyMutationState) => Effect.Effect<void>
  readonly settleMutation: (
    id: MutationStateId,
    result: AtomResult.Result<unknown, unknown>
  ) => Effect.Effect<void>
  readonly forgetMutation: (id: MutationStateId) => Effect.Effect<void>
  /** Resolves with the invocation's terminal state. */
  readonly awaitMutation: (id: MutationStateId) => Effect.Effect<AnyMutationState>
  /** Run one invocation as a fiber of the cache scope. */
  readonly runInvocation: (
    id: MutationStateId,
    mutation: MutationDefinition,
    effect: Effect.Effect<unknown, unknown>
  ) => Effect.Effect<void>
  readonly interruptInvocations: (mutation: MutationDefinition) => Effect.Effect<void>
  readonly scopeSemaphore: (scope: MutationScope) => Effect.Effect<Effect.Semaphore>
}

const isTerminal = (result: AtomResult.Result<unknown, unknown>): boolean =>
  result._tag !== "Initial" && !result.waiting

interface Invocation {
  readonly mutation: MutationDefinition
  readonly fiber: Fiber.RuntimeFiber<unknown, unknown>
}

export const makeQueryCache: Effect.Effect<QueryCache, never, Scope.Scope> = Effect.gen(function*() {
  const scope = yield* Effect.scope
  const queries = yield* SubscriptionRef.make(HashSet.empty<ErasedQueryEntry>())
  const subscriptions = yield* SubscriptionRef.make(HashSet.empty<ErasedSubscriptionEntry>())
  const mutations = yield* SubscriptionRef.make(Chunk.empty<AnyMutationState>())
  const tombstones = yield* SubscriptionRef.make(HashSet.empty<ErasedQueryEntry>())
  const queryDefinitions = yield* Ref.make(HashMap.empty<string, QueryDefinition>())
  const subscriptionDefinitions = yield* Ref.make(HashMap.empty<string, SubscriptionDefinition>())
  const invocations = yield* Ref.make(HashMap.empty<MutationStateId, Invocation>())
  const scopes = yield* SynchronizedRef.make(HashMap.empty<MutationScope, Effect.Semaphore>())
  const pubsub = yield* Effect.acquireRelease(PubSub.unbounded<QueryClientEvent>(), PubSub.shutdown)

  const emit = (event: QueryClientEvent): Effect.Effect<void> => PubSub.publish(pubsub, event).pipe(Effect.asVoid)

  const registerQuery = (entry: ErasedQueryEntry): Effect.Effect<void> => Effect.gen(function*() {
    if (HashSet.has(yield* SubscriptionRef.get(tombstones), entry)) return
    const conflicting = HashMap.get(yield* Ref.get(queryDefinitions), entry.name)
    if (Option.isSome(conflicting) && conflicting.value !== entry.definition) {
      return yield* Effect.die(new Error(`Duplicate query definition name: ${entry.name}`))
    }
    yield* Ref.update(queryDefinitions, HashMap.set(entry.name, entry.definition))
    const added = yield* SubscriptionRef.modify(queries, (current) =>
      HashSet.has(current, entry) ? [false, current] : [true, HashSet.add(current, entry)])
    if (added) yield* emit({ _tag: "QueryCreated", name: entry.name, keyHash: entry.keyHash })
  })

  const unregisterQuery = (entry: ErasedQueryEntry): Effect.Effect<void> => Effect.gen(function*() {
    const remaining = yield* SubscriptionRef.modify(queries, (current) => {
      if (!HashSet.has(current, entry)) return [Option.none<HashSet.HashSet<ErasedQueryEntry>>(), current]
      const next = HashSet.remove(current, entry)
      return [Option.some(next), next]
    })
    if (Option.isNone(remaining)) return
    if (!HashSet.some(remaining.value, (candidate) => candidate.definition === entry.definition)) {
      yield* Ref.update(queryDefinitions, HashMap.remove(entry.name))
    }
    yield* emit({ _tag: "QueryRemoved", name: entry.name, keyHash: entry.keyHash })
  })

  const removeQuery = (entry: ErasedQueryEntry): Effect.Effect<void> =>
    SubscriptionRef.update(tombstones, HashSet.add(entry)).pipe(
      Effect.zipRight(SubscriptionRef.update(queries, HashSet.remove(entry)))
    )

  const restoreQuery = (entry: ErasedQueryEntry): Effect.Effect<void> =>
    SubscriptionRef.update(tombstones, HashSet.remove(entry)).pipe(Effect.zipRight(registerQuery(entry)))

  const awaitRemoval = (entry: ErasedQueryEntry): Effect.Effect<void> =>
    tombstones.changes.pipe(
      Stream.filter((current) => HashSet.has(current, entry)),
      Stream.runHead,
      Effect.asVoid
    )

  const registerSubscription = (entry: ErasedSubscriptionEntry): Effect.Effect<void> => Effect.gen(function*() {
    const conflicting = HashMap.get(yield* Ref.get(subscriptionDefinitions), entry.name)
    if (Option.isSome(conflicting) && conflicting.value !== entry.definition) {
      return yield* Effect.die(new Error(`Duplicate subscription definition name: ${entry.name}`))
    }
    yield* Ref.update(subscriptionDefinitions, HashMap.set(entry.name, entry.definition))
    yield* SubscriptionRef.update(subscriptions, HashSet.add(entry))
  })

  const unregisterSubscription = (entry: ErasedSubscriptionEntry): Effect.Effect<void> => Effect.gen(function*() {
    const remaining = yield* SubscriptionRef.modify(subscriptions, (current) => {
      const next = HashSet.remove(current, entry)
      return [next, next]
    })
    if (!HashSet.some(remaining, (candidate) => candidate.definition === entry.definition)) {
      yield* Ref.update(subscriptionDefinitions, HashMap.remove(entry.name))
    }
  })

  const addMutation = (state: AnyMutationState): Effect.Effect<void> =>
    SubscriptionRef.update(mutations, Chunk.append(state))

  const settleMutation = (
    id: MutationStateId,
    result: AtomResult.Result<unknown, unknown>
  ): Effect.Effect<void> => Effect.gen(function*() {
    const now = yield* Clock.currentTimeMillis
    const terminal = isTerminal(result)
    const previous = yield* SubscriptionRef.modify(mutations, (current) => {
      const index = Chunk.findFirstIndex(current, (state) => state.id === id)
      if (Option.isNone(index)) return [Option.none<AnyMutationState>(), current]
      const state = Chunk.unsafeGet(current, index.value)
      const next: AnyMutationState = {
        ...state,
        result,
        settledAt: terminal ? Option.some(now) : Option.none()
      }
      return [Option.some(state), Chunk.replace(current, index.value, next)]
    })
    if (Option.isSome(previous) && terminal) {
      yield* emit({
        _tag: "MutationSettled",
        name: previous.value.mutation.name,
        id,
        success: result._tag === "Success"
      })
    }
  })

  const forgetMutation = (id: MutationStateId): Effect.Effect<void> =>
    SubscriptionRef.update(mutations, Chunk.filter((state) => state.id !== id))

  const awaitMutation = (id: MutationStateId): Effect.Effect<AnyMutationState> =>
    mutations.changes.pipe(
      Stream.filterMap(Chunk.findFirst((state) => state.id === id && isTerminal(state.result))),
      Stream.runHead,
      Effect.flatMap(Option.match({
        onNone: () => Effect.dieMessage(`Mutation ${id}: history closed before the invocation settled`),
        onSome: Effect.succeed
      }))
    )

  const runInvocation = (
    id: MutationStateId,
    mutation: MutationDefinition,
    effect: Effect.Effect<unknown, unknown>
  ): Effect.Effect<void> => Effect.gen(function*() {
    const fiber = yield* Effect.forkIn(
      effect.pipe(Effect.ensuring(Ref.update(invocations, HashMap.remove(id)))),
      scope
    )
    yield* Ref.update(invocations, HashMap.set(id, { mutation, fiber }))
  })

  const interruptInvocations = (mutation: MutationDefinition): Effect.Effect<void> =>
    Ref.get(invocations).pipe(
      Effect.flatMap((current) => Effect.forEach(
        HashMap.values(current),
        (invocation) => invocation.mutation === mutation ? Fiber.interruptFork(invocation.fiber) : Effect.void,
        { discard: true }
      ))
    )

  const scopeSemaphore = (key: MutationScope): Effect.Effect<Effect.Semaphore> =>
    SynchronizedRef.modifyEffect(scopes, (current) => Option.match(HashMap.get(current, key), {
      onSome: (semaphore) => Effect.succeed([semaphore, current] as const),
      onNone: () => Effect.map(Effect.makeSemaphore(1), (semaphore) =>
        [semaphore, HashMap.set(current, key, semaphore)] as const)
    }))

  return {
    queries,
    subscriptions,
    mutations,
    events: Stream.fromPubSub(pubsub),
    emit,
    registerQuery,
    unregisterQuery,
    removeQuery,
    restoreQuery,
    awaitRemoval,
    registerSubscription,
    unregisterSubscription,
    addMutation,
    settleMutation,
    forgetMutation,
    awaitMutation,
    runInvocation,
    interruptInvocations,
    scopeSemaphore
  }
})

export const subscriptionMatches = (
  entry: ErasedSubscriptionEntry,
  filter: SubscriptionFilter | undefined
): boolean => {
  if (filter === undefined) return true
  if (filter.definition !== undefined && filter.definition !== entry.definition) return false
  if (filter.name !== undefined && filter.name !== entry.name) return false
  if (filter.key !== undefined && !Equal.equals(filter.key, entry.key)) return false
  return true
}

const keyMatches = (filterKey: QueryKey, entryKey: QueryKey, exact: boolean): boolean =>
  !exact && Array.isArray(filterKey) && Array.isArray(entryKey)
    ? filterKey.length <= entryKey.length
      && filterKey.every((part, index) => Equal.equals(part, entryKey[index]))
    : Equal.equals(filterKey, entryKey)

export const queryMatches = (
  registry: AtomRegistry.Registry,
  entry: ErasedQueryEntry,
  filter: QueryFilter | undefined
): boolean => {
  if (filter === undefined) return true
  if (filter.definition !== undefined && filter.definition !== entry.definition) return false
  if (filter.name !== undefined && filter.name !== entry.name) return false
  if (filter.key !== undefined && !keyMatches(filter.key, entry.key, filter.exact !== false)) return false
  const state = entry.state(registry)
  if (filter.stale !== undefined && filter.stale !== state.isStale) return false
  if (filter.fetchStatus !== undefined && filter.fetchStatus !== state.fetchStatus) return false
  return filter.predicate?.({
    definition: entry.definition,
    name: entry.name,
    key: entry.key,
    state
  }) ?? true
}

export const mutationStatus = (state: AnyMutationState): "pending" | "success" | "error" =>
  state.result.waiting || state.result._tag === "Initial"
    ? "pending"
    : state.result._tag === "Success" ? "success" : "error"

export const mutationMatches = (state: AnyMutationState, filter: MutationFilter | undefined): boolean => {
  if (filter === undefined) return true
  if (filter.mutation !== undefined && filter.mutation !== state.mutation) return false
  if (filter.scope !== undefined && !Option.contains(state.scope, filter.scope)) return false
  if (filter.status !== undefined && filter.status !== mutationStatus(state)) return false
  return filter.predicate?.(state) ?? true
}

export type {
  AnyMutationState,
  MutationState,
  MutationStateId,
  MutationFilter,
  QueryClientEvent,
  QueryDefinition,
  QueryEntryState,
  QueryFilter,
  QueryKey,
  QueryMetadata,
  SubscriptionDefinition,
  SubscriptionEntryState,
  SubscriptionFilter,
  SubscriptionStatus
} from "./Model.js"
