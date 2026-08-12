import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import type * as Reactivity from "@effect/experimental/Reactivity"
import * as Duration from "effect/Duration"
import * as EffectData from "effect/Data"
import * as Effect from "effect/Effect"
import type * as Equivalence from "effect/Equivalence"
import * as Equal from "effect/Equal"
import * as Hash from "effect/Hash"
import * as Option from "effect/Option"
import * as Schedule from "effect/Schedule"
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

export type Options<Input, Data, Error, Requirements> = CommonOptions<Input, Error> & {
  readonly effect: (input: Input) => Effect.Effect<Data, Error, Requirements>
}

interface Control<Data> {
  readonly invalidation: number
  readonly acceptedInvalidation: number
  readonly override: Option.Option<Data>
  readonly overrideUpdatedAt: Option.Option<number>
  readonly overrideRequest: number
  readonly failureCount: number
  readonly cancelled: boolean
  readonly refetchRequest: Option.Option<number>
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

export const make = <Input, Data, Error, Requirements>(
  name: string,
  options: Options<Input, Data, Error, Requirements>
): Query<Input, Data, Error, Requirements> => {
  let definition!: Query<Input, Data, Error, Requirements>
  const keyFor = (input: Input): QueryKey => {
    const key = options.key(input)
    if ((typeof key === "object" && key !== null) && !Equal.isEqual(key)) {
      throw new TypeError(`Query ${name} returned a structured key without Effect Equal semantics`)
    }
    return key
  }
  definition = {
    [QueryDefinitionTypeId]: true,
    [OptionsTypeId]: options,
    name,
    match: (input?: Input): QueryFilter => input === undefined
      ? { definition }
      : { definition, key: keyFor(input), exact: true }
  }
  return definition
}

export const makeAtomFamily = <Provided, RuntimeError, Input, Data, Error, Required extends Provided | Reactivity.Reactivity>(
  runtime: Atom.AtomRuntime<Provided, RuntimeError>,
  definition: Query<Input, Data, Error, Required>
): ((input: Input) => QueryAtom<Input, Data, Error | RuntimeError, Required>) => {
  const { name } = definition
  const options = definition[OptionsTypeId] as Options<Input, Data, Error, Required>
  const staleTime = Duration.toMillis(Duration.decode(options.staleTime ?? 0))
  const gcTime = options.gcTime ?? Duration.minutes(5)
  const keyFor = (input: Input): QueryKey => {
    const key = options.key(input)
    if ((typeof key === "object" && key !== null) && !Equal.isEqual(key)) {
      throw new TypeError(`Query ${name} returned a structured key without Effect Equal semantics`)
    }
    return key
  }

  const load = (input: Input, invalidation: number, request: number) => {
    let effect = Effect.suspend(() => options.effect(input))
    if (options.retry !== undefined) effect = Effect.retry(effect, options.retry)
    return Effect.map(effect, (data): FetchedValue<Data> => ({ data, invalidation, request }))
  }

  const family = Atom.family((identity: QueryKey) => {
    let canonicalInput!: Input
    let hasCanonicalInput = false
    const retain = <A extends Atom.Atom<any>>(atom: A): A => Atom.setIdleTTL(atom, gcTime)
    const control = retain(Atom.make<Control<Data>>({
      invalidation: 0,
      acceptedInvalidation: -1,
      override: Option.none(),
      overrideUpdatedAt: Option.none(),
      overrideRequest: -1,
      failureCount: 0,
      cancelled: false,
      refetchRequest: Option.none()
    }))
    const request = retain(Atom.make(0))

    const fetched = retain(runtime.atom((get) => {
      const requestId = get(request)
      const captured = get.once(control)
      const registry = get.registry
      const core = getClientCore(registry)
      if (captured.cancelled) return Effect.interrupt
      core.emit({ _tag: "FetchStarted", name, keyHash: Hash.hash(identity) })
      core.touch()
      return load(canonicalInput, captured.invalidation, requestId).pipe(
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
      const fetchedResult = normalizeInterrupted(get(fetched))
      const accepted = AtomResult.value(fetchedResult)
      const hasOverride = Option.isSome(current.override)
        && (Option.isNone(accepted) || current.overrideRequest >= accepted.value.request)
      let result = AtomResult.map(fetchedResult, (value) => value.data)
      if (hasOverride && Option.isSome(current.override)) {
        result = AtomResult.success(current.override.value, {
          timestamp: Option.getOrElse(current.overrideUpdatedAt, () => Date.now()),
          waiting: result.waiting
        })
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
