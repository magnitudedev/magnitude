import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import * as Cause from "effect/Cause"
import * as Effect from "effect/Effect"
import * as Equal from "effect/Equal"
import * as Option from "effect/Option"
import * as PubSub from "effect/PubSub"
import * as Stream from "effect/Stream"
import {
  type AnyMutationState,
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

export const QueryEntryTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/QueryEntry")
export const MutationInternalTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/MutationInternal")
export const SubscriptionEntryTypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/SubscriptionEntry")

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
  readonly remove: (registry: AtomRegistry.Registry) => void
}

export interface QueryEntry<Data> extends ErasedQueryEntry {
  readonly setData: (
    registry: AtomRegistry.Registry,
    update: (current: Option.Option<Data>) => Data
  ) => void
  readonly hasData: (registry: AtomRegistry.Registry) => boolean
  readonly getData: (registry: AtomRegistry.Registry) => Option.Option<Data>
}

export interface QueryEntryCarrier<Data> {
  readonly [QueryEntryTypeId]: QueryEntry<Data>
}

export const queryEntry = <Data>(carrier: QueryEntryCarrier<Data>): QueryEntry<Data> =>
  carrier[QueryEntryTypeId]

export interface MutationInvocation<Output, Error> {
  readonly id: MutationStateId
  readonly await: Effect.Effect<Output, Error>
}

export interface MutationController<Input, Output, Error> {
  readonly invoke: (
    registry: AtomRegistry.Registry,
    input: Input
  ) => MutationInvocation<Output, Error>
}

export interface MutationControllerCarrier<Input, Output, Error> {
  readonly [MutationInternalTypeId]: MutationController<Input, Output, Error>
}

export const mutationController = <Input, Output, Error>(
  carrier: MutationControllerCarrier<Input, Output, Error>
): MutationController<Input, Output, Error> =>
  carrier[MutationInternalTypeId]

export interface ClientCore {
  readonly registry: AtomRegistry.Registry
  readonly entries: Set<ErasedQueryEntry>
  readonly definitions: Map<string, QueryDefinition>
  readonly subscriptions: Set<ErasedSubscriptionEntry>
  readonly subscriptionDefinitions: Map<string, SubscriptionDefinition>
  readonly removed: Set<ErasedQueryEntry>
  readonly mutationStates: Array<AnyMutationState>
  readonly mutationScopes: Map<MutationScope, Effect.Semaphore>
  readonly revision: Atom.Writable<number>
  readonly events: Stream.Stream<QueryClientEvent>
  readonly emit: (event: QueryClientEvent) => void
  readonly touch: () => void
}

const clients = new WeakMap<AtomRegistry.Registry, ClientCore>()

export const getClientCore = (registry: AtomRegistry.Registry): ClientCore => {
  const existing = clients.get(registry)
  if (existing !== undefined) return existing

  const pubsub = Effect.runSync(PubSub.unbounded<QueryClientEvent>())
  const revision = Atom.keepAlive(Atom.make(0))
  // Entries register while an atom is being evaluated (possibly during a React
  // render); the registry-wide revision therefore advances after the current
  // evaluation, coalescing concurrent touches into one write.
  let touchPending = false
  const core: ClientCore = {
    registry,
    entries: new Set(),
    definitions: new Map(),
    subscriptions: new Set(),
    subscriptionDefinitions: new Map(),
    removed: new Set(),
    mutationStates: [],
    mutationScopes: new Map(),
    revision,
    events: Stream.fromPubSub(pubsub),
    emit: (event) => {
      Effect.runSync(PubSub.publish(pubsub, event))
    },
    touch: () => {
      if (touchPending) return
      touchPending = true
      queueMicrotask(() => {
        touchPending = false
        // Nothing observes the revision until something read it; a disposed
        // registry has no nodes at all.
        if (registry.getNodes().has(revision)) registry.update(revision, (value) => value + 1)
      })
    }
  }
  clients.set(registry, core)
  return core
}

export const registerEntry = (registry: AtomRegistry.Registry, entry: ErasedQueryEntry): (() => void) => {
  const core = getClientCore(registry)
  if (core.removed.has(entry)) return () => {}
  const conflicting = core.definitions.get(entry.name)
  if (conflicting !== undefined && conflicting !== entry.definition) {
    throw new Error(`Duplicate query definition name: ${entry.name}`)
  }
  core.definitions.set(entry.name, entry.definition)
  if (!core.entries.has(entry)) {
    core.entries.add(entry)
    core.emit({ _tag: "QueryCreated", name: entry.name, keyHash: entry.keyHash })
    core.touch()
  }
  return () => {
    queueMicrotask(() => {
      if (registry.getNodes().has(entry.stateAtom) || !core.entries.delete(entry)) return
      if (![...core.entries].some((candidate) => candidate.definition === entry.definition)) {
        core.definitions.delete(entry.name)
      }
      core.emit({ _tag: "QueryRemoved", name: entry.name, keyHash: entry.keyHash })
    })
  }
}

export const registerSubscriptionEntry = (
  registry: AtomRegistry.Registry,
  entry: ErasedSubscriptionEntry
): (() => void) => {
  const core = getClientCore(registry)
  const conflicting = core.subscriptionDefinitions.get(entry.name)
  if (conflicting !== undefined && conflicting !== entry.definition) {
    throw new Error(`Duplicate subscription definition name: ${entry.name}`)
  }
  core.subscriptionDefinitions.set(entry.name, entry.definition)
  if (!core.subscriptions.has(entry)) {
    core.subscriptions.add(entry)
    core.touch()
  }
  return () => {
    queueMicrotask(() => {
      if (registry.getNodes().has(entry.stateAtom) || !core.subscriptions.delete(entry)) return
      if (![...core.subscriptions].some((candidate) => candidate.definition === entry.definition)) {
        core.subscriptionDefinitions.delete(entry.name)
      }
    })
  }
}

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

export const addMutationState = <Input, Output, Error>(
  core: ClientCore,
  state: MutationState<Input, Output, Error>
): void => {
  core.mutationStates.push(state)
  core.touch()
}

export const settleMutationState = <Output, Error>(
  core: ClientCore,
  id: MutationStateId,
  result: AtomResult.Result<Output, Error>
): void => {
  const index = core.mutationStates.findIndex((state) => state.id === id)
  if (index < 0) return
  const previous = core.mutationStates[index]
  core.mutationStates[index] = {
    ...previous,
    result,
    settledAt: result.waiting || result._tag === "Initial" ? Option.none() : Option.some(Date.now())
  }
  if (!result.waiting && result._tag !== "Initial") {
    core.emit({
      _tag: "MutationSettled",
      name: previous.mutation.name,
      id,
      success: result._tag === "Success"
    })
  }
  core.touch()
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
