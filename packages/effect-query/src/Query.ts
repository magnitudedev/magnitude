import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import type * as Reactivity from "@effect/experimental/Reactivity"
import * as Cause from "effect/Cause"
import * as Deferred from "effect/Deferred"
import * as Duration from "effect/Duration"
import * as EffectData from "effect/Data"
import * as Effect from "effect/Effect"
import type * as Equivalence from "effect/Equivalence"
import * as Equal from "effect/Equal"
import * as Exit from "effect/Exit"
import * as Hash from "effect/Hash"
import * as Option from "effect/Option"
import * as Schedule from "effect/Schedule"
import type * as Scope from "effect/Scope"
import * as Stream from "effect/Stream"
import {
  getClientCore,
  QueryEntryTypeId,
  registerEntry,
  type QueryEntry,
  type QueryEntryCarrier,
  type QueryFilter
} from "./internal.js"
import {
  QueryDefinitionTypeId,
  type QueryDefinition,
  type QueryKey
} from "./Model.js"

export const TypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Query")
const SetInputTypeId: unique symbol = Symbol("@magnitudedev/effect-query/Query/setInput")
const OptionsTypeId: unique symbol = Symbol("@magnitudedev/effect-query/Query/options")

export interface State<Data, Error> {
  readonly result: AtomResult.Result<Data, Error>
  readonly fetchStatus: "idle" | "fetching" | "paused"
  readonly isStale: boolean
  readonly dataUpdatedAt: Option.Option<number>
  readonly failureCount: number
}

export interface QueryAtom<Input, Data, Error, Requirements>
  extends Atom.Atom<State<Data, Error>>, QueryEntryCarrier<Data> {
  readonly definition: QueryDefinition
  readonly input: Input
}

export interface Query<Input, Data, Error, Requirements> extends QueryDefinition {
  readonly [TypeId]?: {
    readonly input: Input
    readonly data: Data
    readonly error: Error
    readonly requirements: Requirements
  }
  readonly name: string
  readonly [OptionsTypeId]: unknown
  readonly match: {
    (): QueryFilter
    (input: Input): QueryFilter
  }
}

export type Any = QueryDefinition
export type Input<Q> = Q extends Query<infer I, infer _D, infer _E, infer _R> ? I : never
export type Data<Q> = Q extends Query<infer _I, infer D, infer _E, infer _R> ? D : never
export type Error<Q> = Q extends Query<infer _I, infer _D, infer E, infer _R> ? E : never
export type Requirements<Q> = Q extends Query<infer _I, infer _D, infer _E, infer R> ? R : never

export namespace QueryAtom {
  export type Any = Atom.Atom<import("./Model.js").QueryEntryState> & {
    readonly definition: QueryDefinition
  }
  export type Input<Q> = Q extends QueryAtom<infer I, infer _D, infer _E, infer _R> ? I : never
  export type Data<Q> = Q extends QueryAtom<infer _I, infer D, infer _E, infer _R> ? D : never
  export type Error<Q> = Q extends QueryAtom<infer _I, infer _D, infer E, infer _R> ? E : never
  export type Requirements<Q> = Q extends QueryAtom<infer _I, infer _D, infer _E, infer R> ? R : never
  export type State<Q> = Q extends QueryAtom<infer _I, infer D, infer E, infer _R> ?
    import("./Query.js").State<D, E> : never
}

interface CommonOptions<Input, Error> {
  readonly key: (input: Input) => QueryKey
  readonly staleTime?: Duration.DurationInput
  readonly gcTime?: Duration.DurationInput
  readonly retry?: Schedule.Schedule<unknown, Error, never>
  readonly refresh?: Schedule.Schedule<unknown, void, never>
}

/** Options for a request/response query. */
export type Options<Input, Data, Error, Requirements> = CommonOptions<Input, Error> & {
  readonly effect: (input: Input) => Effect.Effect<Data, Error, Requirements>
}

/**
 * Options for a query whose data source is a stream (TanStack `streamedQuery`):
 * the entry becomes successful on the first element and every later element is
 * reduced into the current data. Refetch, invalidation, and reset reopen the
 * stream. Freshness is owned by the stream, so there is no `staleTime`.
 */
export interface FromStreamOptions<Input, Event, Data, Error, Requirements> {
  readonly key: (input: Input) => QueryKey
  readonly stream: (input: Input) => Stream.Stream<Event, Error, Requirements>
  readonly reduce: (previous: Option.Option<Data>, event: Event) => Data
  /** Retry policy applied to the stream while it is open; defaults to no retry. */
  readonly reconnect?: Schedule.Schedule<unknown, Error, never>
  readonly gcTime?: Duration.DurationInput
}

type Source<Input, Data, Error, Requirements> =
  | {
    readonly _tag: "Effect"
    readonly effect: (input: Input) => Effect.Effect<Data, Error, Requirements>
    readonly retry: Schedule.Schedule<unknown, Error, never> | undefined
  }
  | {
    readonly _tag: "Fold"
    readonly fold: (input: Input) => Stream.Stream<(previous: Option.Option<Data>) => Data, Error, Requirements>
    readonly reconnect: Schedule.Schedule<unknown, Error, never> | undefined
  }

interface InternalOptions<Input, Data, Error, Requirements> {
  readonly key: (input: Input) => QueryKey
  readonly staleTime: Duration.DurationInput | undefined
  readonly gcTime: Duration.DurationInput | undefined
  readonly refresh: Schedule.Schedule<unknown, void, never> | undefined
  readonly source: Source<Input, Data, Error, Requirements>
}

interface StreamFailure<Error> {
  readonly cause: Cause.Cause<Error>
  readonly request: number
}

interface Control<Data, Error> {
  readonly invalidation: number
  readonly acceptedInvalidation: number
  readonly override: Option.Option<Data>
  readonly overrideUpdatedAt: Option.Option<number>
  readonly overrideRequest: number
  readonly failureCount: number
  readonly cancelled: boolean
  readonly refetchRequest: Option.Option<number>
  /** Terminal stream failure after data was produced (fold source only). */
  readonly streamFailure: Option.Option<StreamFailure<Error>>
}

interface FetchedValue<Data> {
  readonly data: Data
  readonly invalidation: number
  readonly request: number
}

const resultTimestamp = <A, E>(result: AtomResult.Result<A, E>): Option.Option<number> => {
  if (result._tag === "Success") return Option.some(result.timestamp)
  if (result._tag === "Failure") return Option.map(result.previousSuccess, (success) => success.timestamp)
  return Option.none()
}

const normalizeInterrupted = <A, E>(result: AtomResult.Result<A, E>): AtomResult.Result<A, E> => {
  if (!AtomResult.isInterrupted(result)) return result
  return Option.match(result.previousSuccess, {
    onNone: () => AtomResult.initial(),
    onSome: (success) => AtomResult.success(success.value, success)
  })
}

const validatedKey = (name: string, key: QueryKey): QueryKey => {
  if ((typeof key === "object" && key !== null) && !Equal.isEqual(key)) {
    throw new TypeError(`Query ${name} returned a structured key without Effect Equal semantics`)
  }
  return key
}

const makeDefinition = <Input, Data, Error, Requirements>(
  name: string,
  options: InternalOptions<Input, Data, Error, Requirements>
): Query<Input, Data, Error, Requirements> => {
  let definition!: Query<Input, Data, Error, Requirements>
  definition = {
    [QueryDefinitionTypeId]: true,
    [OptionsTypeId]: options,
    name,
    match: (input?: Input): QueryFilter => input === undefined
      ? { definition }
      : { definition, key: validatedKey(name, options.key(input)), exact: true }
  }
  return definition
}

export const make = <Input, Data, Error, Requirements>(
  name: string,
  options: Options<Input, Data, Error, Requirements>
): Query<Input, Data, Error, Requirements> => makeDefinition(name, {
  key: options.key,
  staleTime: options.staleTime,
  gcTime: options.gcTime,
  refresh: options.refresh,
  source: { _tag: "Effect", effect: options.effect, retry: options.retry }
})

export const fromStream = <Input, Event, Data, Error, Requirements>(
  name: string,
  options: FromStreamOptions<Input, Event, Data, Error, Requirements>
): Query<Input, Data, Error, Requirements> => makeDefinition(name, {
  key: options.key,
  staleTime: Number.POSITIVE_INFINITY,
  gcTime: options.gcTime,
  refresh: undefined,
  source: {
    _tag: "Fold",
    fold: (input) => Stream.map(options.stream(input), (event) => (previous: Option.Option<Data>) =>
      options.reduce(previous, event)),
    reconnect: options.reconnect
  }
})

export const makeAtomFamily = <Provided, RuntimeError, Input, Data, Error, Required extends Provided | Reactivity.Reactivity>(
  runtime: Atom.AtomRuntime<Provided, RuntimeError>,
  definition: Query<Input, Data, Error, Required>
): ((input: Input) => QueryAtom<Input, Data, Error | RuntimeError, Required>) => {
  const { name } = definition
  const options = definition[OptionsTypeId] as InternalOptions<Input, Data, Error, Required>
  const staleTime = Duration.toMillis(Duration.decode(options.staleTime ?? 0))
  const gcTime = options.gcTime ?? Duration.minutes(5)
  const keyFor = (input: Input): QueryKey => validatedKey(name, options.key(input))
  const source = options.source

  const family = Atom.family((identity: QueryKey) => {
    let canonicalInput!: Input
    let hasCanonicalInput = false
    const retain = <A extends Atom.Atom<any>>(atom: A): A => Atom.setIdleTTL(atom, gcTime)
    const control = retain(Atom.make<Control<Data, Error>>({
      invalidation: 0,
      acceptedInvalidation: -1,
      override: Option.none(),
      overrideUpdatedAt: Option.none(),
      overrideRequest: -1,
      failureCount: 0,
      cancelled: false,
      refetchRequest: Option.none(),
      streamFailure: Option.none()
    }))
    const request = retain(Atom.make(0))

    const loadEffect = (
      input: Input,
      invalidation: number,
      requestId: number,
      effect: (input: Input) => Effect.Effect<Data, Error, Required>,
      retry: Schedule.Schedule<unknown, Error, never> | undefined
    ): Effect.Effect<FetchedValue<Data>, Error, Required> => {
      let run = Effect.suspend(() => effect(input))
      if (retry !== undefined) run = Effect.retry(run, retry)
      return Effect.map(run, (data): FetchedValue<Data> => ({ data, invalidation, request: requestId }))
    }

    /**
     * Fold source: the first stream element resolves the fetch; later elements
     * are applied as data overrides while the stream stays open in the atom's
     * scope. A terminal failure after data was produced is surfaced as a
     * failure carrying the previous success.
     */
    const loadFold = (
      input: Input,
      invalidation: number,
      requestId: number,
      registry: AtomRegistry.Registry,
      fold: (input: Input) => Stream.Stream<(previous: Option.Option<Data>) => Data, Error, Required>,
      reconnect: Schedule.Schedule<unknown, Error, never> | undefined
    ): Effect.Effect<FetchedValue<Data>, Error, Required | Scope.Scope> => Effect.gen(function*() {
      // Leave the registry's synchronous evaluation before any element can write
      // atom state (a yield alone is drained synchronously by the registry).
      yield* Effect.async<void>((resume) => {
        queueMicrotask(() => resume(Effect.void))
      })
      const first = yield* Deferred.make<FetchedValue<Data>, Error>()
      let current = Option.none<Data>()
      let stream = fold(input)
      if (reconnect !== undefined) stream = Stream.retry(stream, reconnect)
      const isCurrentRequest = () => registry.getNodes().has(control) && registry.get(request) === requestId
      yield* stream.pipe(
        Stream.runForEach((step) => Effect.sync(() => {
          const next = step(current)
          const isFirst = Option.isNone(current)
          current = Option.some(next)
          if (isFirst) {
            Deferred.unsafeDone(first, Exit.succeed({ data: next, invalidation, request: requestId }))
            return
          }
          if (!isCurrentRequest()) return
          registry.update(control, (state) => ({
            ...state,
            override: Option.some(next),
            overrideUpdatedAt: Option.some(Date.now()),
            overrideRequest: requestId,
            acceptedInvalidation: state.invalidation
          }))
        })),
        Effect.onExit((exit) => Effect.sync(() => {
          if (Option.isNone(current)) {
            Deferred.unsafeDone(
              first,
              Exit.isFailure(exit)
                ? Exit.failCause(exit.cause)
                : Exit.die(new Error(`Query ${name}: stream completed before producing data`))
            )
            return
          }
          if (Exit.isSuccess(exit) || Cause.isInterruptedOnly(exit.cause) || !isCurrentRequest()) return
          registry.update(control, (state) => ({
            ...state,
            streamFailure: Option.some({ cause: exit.cause, request: requestId })
          }))
        })),
        Effect.forkScoped
      )
      return yield* Deferred.await(first)
    })

    const fetched = retain(runtime.atom((get) => {
      const requestId = get(request)
      const captured = get.once(control)
      const registry = get.registry
      const core = getClientCore(registry)
      if (captured.cancelled) return Effect.interrupt
      core.emit({ _tag: "FetchStarted", name, keyHash: Hash.hash(identity) })
      core.touch()
      const load = source._tag === "Effect"
        ? loadEffect(canonicalInput, captured.invalidation, requestId, source.effect, source.retry)
        : loadFold(canonicalInput, captured.invalidation, requestId, registry, source.fold, source.reconnect)
      return load.pipe(
        Effect.onExit((exit) => Effect.sync(() => {
          core.emit({
            _tag: "FetchSettled",
            name,
            keyHash: Hash.hash(identity),
            success: exit._tag === "Success"
          })
          core.touch()
          queueMicrotask(() => {
            if (!registry.getNodes().has(control)) return
            if (registry.get(request) !== requestId) return
            let refetch = false
            registry.update(control, (current) => {
              refetch = !current.cancelled && current.invalidation > captured.invalidation
              return {
                ...current,
                failureCount: exit._tag === "Success" ? 0 : current.failureCount + 1,
                refetchRequest: refetch
                  ? Option.some(requestId + 1)
                  : Option.none()
              }
            })
            if (refetch) registry.update(request, (value) => value + 1)
          })
        }))
      )
    }))

    const scheduler = options.refresh === undefined ? undefined : runtime.atom((get) => {
      const registry = get.registry
      const loop = (driver: Schedule.ScheduleDriver<unknown, void>): Effect.Effect<void> =>
        driver.next(undefined).pipe(
          Effect.tap(() => Effect.sync(() => entry.start(registry))),
          Effect.flatMap(() => loop(driver)),
          Effect.catchAll(() => Effect.void)
        )
      return Effect.yieldNow().pipe(
        Effect.zipRight(Schedule.driver(options.refresh!)),
        Effect.flatMap(loop)
      )
    })

    let entry!: QueryEntry<Data>
    let atom = Atom.readable<State<Data, Error | RuntimeError>>((get) => {
      const unregister = registerEntry(get.registry, entry)
      get.addFinalizer(unregister)
      if (scheduler !== undefined) get(scheduler)
      const current = get(control)
      const currentRequest = get(request)
      const fetchedResult = normalizeInterrupted(get(fetched))
      const accepted = AtomResult.value(fetchedResult)
      const hasOverride = Option.isSome(current.override)
        && (Option.isNone(accepted) || current.overrideRequest >= accepted.value.request)
      let result: AtomResult.Result<Data, Error | RuntimeError> = AtomResult.map(fetchedResult, (value) => value.data)
      if (hasOverride && Option.isSome(current.override)) {
        result = AtomResult.success(current.override.value, {
          timestamp: Option.getOrElse(current.overrideUpdatedAt, () => Date.now()),
          waiting: result.waiting
        })
      }
      if (
        result._tag === "Success"
        && Option.isSome(current.streamFailure)
        && current.streamFailure.value.request === currentRequest
      ) {
        result = AtomResult.failure(current.streamFailure.value.cause, { previousSuccess: Option.some(result) })
      }
      const updatedAt = hasOverride && Option.isSome(current.overrideUpdatedAt)
        ? current.overrideUpdatedAt
        : resultTimestamp(result)
      const acceptedInvalidation = hasOverride
        ? current.acceptedInvalidation
        : Option.match(accepted, { onNone: () => -1, onSome: (value) => value.invalidation })
      const ageFresh = staleTime === Number.POSITIVE_INFINITY
        || Option.exists(updatedAt, (timestamp) => Date.now() - timestamp < staleTime)
      const isStale = acceptedInvalidation < current.invalidation || !ageFresh
      if (!isStale && staleTime > 0 && staleTime !== Number.POSITIVE_INFINITY && Option.isSome(updatedAt)) {
        const timeout = setTimeout(() => get.refreshSelf(), Math.max(0, updatedAt.value + staleTime - Date.now()))
        get.addFinalizer(() => clearTimeout(timeout))
      }
      return EffectData.struct({
        result,
        fetchStatus: current.cancelled ? "paused" : result.waiting ? "fetching" : "idle",
        isStale,
        dataUpdatedAt: updatedAt,
        failureCount: current.failureCount
      })
    }, (refresh) => refresh(fetched))
    atom = retain(atom)

    entry = {
      stateAtom: atom,
      definition,
      name,
      key: identity,
      keyHash: Hash.hash(identity),
      state: (registry) => {
        const state = registry.get(atom)
        return { fetchStatus: state.fetchStatus, isStale: state.isStale }
      },
      failureCause: (registry) => {
        const result = registry.get(atom).result
        return result._tag === "Failure" ? Option.some(result.cause) : Option.none()
      },
      start: (registry, options) => {
        const status = entry.state(registry).fetchStatus
        const nextRequest = registry.get(request) + 1
        let shouldStart = status !== "fetching"
        registry.update(control, (current) => {
          if (status === "fetching" && options?.cancelRefetch === true && Option.isNone(current.refetchRequest)) {
            shouldStart = true
          }
          return {
            ...current,
            cancelled: false,
            refetchRequest: shouldStart && status === "fetching"
              ? Option.some(nextRequest)
              : current.refetchRequest
          }
        })
        if (shouldStart) registry.update(request, () => nextRequest)
      },
      cancel: (registry) => {
        registry.update(control, (current) => ({
          ...current,
          cancelled: true,
          refetchRequest: Option.none()
        }))
        registry.update(request, (value) => value + 1)
      },
      invalidate: (registry) => {
        registry.update(control, (current) => ({ ...current, invalidation: current.invalidation + 1 }))
      },
      remove: (registry) => {
        const core = getClientCore(registry)
        core.removed.add(entry)
        core.entries.delete(entry)
        entry.cancel(registry)
        core.touch()
      },
      setData: (registry, update) => {
        const currentState = registry.get(atom)
        const currentData = AtomResult.value(currentState.result)
        registry.update(control, (current) => ({
          ...current,
          override: Option.some(update(currentData)),
          overrideUpdatedAt: Option.some(Date.now()),
          overrideRequest: registry.get(request),
          acceptedInvalidation: current.invalidation
        }))
      },
      hasData: (registry) => Option.isSome(AtomResult.value(registry.get(atom).result)),
      getData: (registry) => AtomResult.value(registry.get(atom).result)
    }
    const atomWithInput = Object.defineProperty(atom, "input", {
      get: () => canonicalInput,
      enumerable: true,
      configurable: false
    }) as typeof atom & { readonly input: Input }
    return Object.assign(atomWithInput, {
      [QueryEntryTypeId]: entry,
      definition,
      [SetInputTypeId]: (input: Input) => {
        if (!hasCanonicalInput) {
          canonicalInput = input
          hasCanonicalInput = true
        }
      }
    })
  })

  return (input: Input) => {
    const identity = keyFor(input)
    const atom = family(identity)
    atom[SetInputTypeId](input)
    return atom
  }
}

export const select = <Input, Data, Error, Requirements, Selected>(
  query: QueryAtom<Input, Data, Error, Requirements>,
  selectValue: (data: Data) => Selected,
  equivalence?: Equivalence.Equivalence<Selected>
): Atom.Atom<State<Selected, Error>> =>
  Atom.readable((get) => {
    const state = get(query)
    const previous = get.self<State<Selected, Error>>()
    const result = AtomResult.map(state.result, (data) => {
      const selected = selectValue(data)
      if (equivalence === undefined || Option.isNone(previous)) return selected
      const previousValue = AtomResult.value(previous.value.result)
      return Option.isSome(previousValue) && equivalence(previousValue.value, selected)
        ? previousValue.value
        : selected
    })
    return EffectData.struct({ ...state, result })
  }, query.refresh)

export const when = <Input, Data, Error, Requirements>(
  query: Option.Option<QueryAtom<Input, Data, Error, Requirements>>
): Atom.Atom<Option.Option<State<Data, Error>>> =>
  Atom.readable((get) => Option.map(query, (atom) => get(atom)))

export const isQueryAtom = (value: unknown): value is QueryAtom.Any =>
  typeof value === "object" && value !== null && QueryEntryTypeId in value

export type { QueryFilter } from "./internal.js"
