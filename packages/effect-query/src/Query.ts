import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import type * as Reactivity from "@effect/experimental/Reactivity"
import * as Cause from "effect/Cause"
import * as Clock from "effect/Clock"
import * as Deferred from "effect/Deferred"
import * as Duration from "effect/Duration"
import * as EffectData from "effect/Data"
import * as Effect from "effect/Effect"
import type * as Equivalence from "effect/Equivalence"
import * as Equal from "effect/Equal"
import * as Exit from "effect/Exit"
import * as FiberId from "effect/FiberId"
import * as Hash from "effect/Hash"
import * as Option from "effect/Option"
import * as Schedule from "effect/Schedule"
import * as Schema from "effect/Schema"
import * as Stream from "effect/Stream"
import {
  QueryCacheTypeId,
  QueryClient,
  QueryEntryTypeId,
  type QueryCache,
  type QueryEntry,
  type QueryEntryCarrier,
  type QueryFilter
} from "./internal.js"
import {
  QueryDefinitionTypeId,
  type QueryDefinition,
  type QueryKey
} from "./Model.js"
import * as Operation from "./Operation.js"

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

export type DeclarationOptions<
  Payload extends Operation.PayloadInput,
  Success extends Schema.Schema.Any,
  Error extends Schema.Schema.All,
  Policy extends object,
  Input,
  QueryError,
> = Operation.Shape<Payload, Success, Error, Policy> & {
  readonly key?: (input: Input) => QueryKey
  readonly staleTime?: Duration.DurationInput
  readonly gcTime?: Duration.DurationInput
  readonly retry?: Schedule.Schedule<unknown, QueryError, never>
  readonly refresh?: Schedule.Schedule<unknown, void, never>
}

export type StreamDeclarationOptions<
  Payload extends Operation.PayloadInput,
  Success extends Schema.Schema.Any,
  Error extends Schema.Schema.All,
  Policy extends object,
  Input,
  Event,
  Data,
  QueryError,
> = Operation.Shape<Payload, Success, Error, Policy> & {
  readonly key?: (input: Input) => QueryKey
  readonly reduce: (previous: Option.Option<Data>, event: Event) => Data
  readonly reconnect?: Schedule.Schedule<unknown, QueryError, never>
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

/** Position of one fetch attempt: a generation is started explicitly, attempts within it are trailing refetches. */
interface Sequence {
  readonly generation: number
  readonly attempt: number
}

const after = (a: Sequence, b: Sequence): boolean =>
  a.generation > b.generation || (a.generation === b.generation && a.attempt > b.attempt)

const sameSequence = (a: Sequence, b: Sequence): boolean =>
  a.generation === b.generation && a.attempt === b.attempt

/** Elements of the fetch stream; the atom graph derives everything else from the latest one. */
type FetchEvent<Data, Error> =
  | { readonly _tag: "Started"; readonly sequence: Sequence }
  | {
    readonly _tag: "Settled"
    readonly sequence: Sequence
    readonly data: Data
    readonly invalidation: number
    readonly updatedAt: number
    readonly terminal: boolean
  }
  | {
    readonly _tag: "Failed"
    readonly sequence: Sequence
    readonly cause: Cause.Cause<Error>
    readonly terminal: boolean
  }

interface Override<Data> {
  readonly data: Data
  readonly updatedAt: number
  readonly sequence: Sequence
  readonly invalidation: number
}

interface Control<Data> {
  readonly invalidation: number
  readonly cancelled: boolean
  /** Generation started as a replacement while a fetch was active; later cancel-refetch starts coalesce into it. */
  readonly replacement: Option.Option<number>
  readonly cancellation: Deferred.Deferred<never, FetchCancelled>
  readonly override: Option.Option<Override<Data>>
}

interface FetchCancelled {
  readonly _tag: "FetchCancelled"
}

const fetchCancelled: FetchCancelled = { _tag: "FetchCancelled" }

interface Accepted<Data> {
  readonly sequence: Sequence
  readonly data: Data
  readonly invalidation: number
  readonly updatedAt: number
}

interface Failures<Error> {
  readonly count: number
  readonly latest: Option.Option<{ readonly sequence: Sequence; readonly cause: Cause.Cause<Error> }>
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

export function make<Input, Data, Error, Requirements>(
  name: string,
  options: Options<Input, Data, Error, Requirements>
): Query<Input, Data, Error, Requirements>
export function make<
  const Name extends string,
  Payload extends Operation.PayloadInput = typeof Schema.Void,
  Success extends Schema.Schema.Any = typeof Schema.Void,
  Error extends Schema.Schema.All = typeof Schema.Never,
  Policy extends object = {},
>(
  name: Name,
  options: DeclarationOptions<
    Payload,
    Success,
    Error,
    Policy,
    Operation.PayloadConstructor<Payload>,
    Schema.Schema.Type<Error>
  >,
): Query<
  Operation.PayloadConstructor<Payload>,
  Schema.Schema.Type<Success>,
  Schema.Schema.Type<Error>,
  Operation.Implementations<any>
> & Operation.Declared<Name, "query", Payload, Success, Error, Policy>
export function make(
  name: string,
  options: Options<unknown, unknown, unknown, unknown> | (DeclarationOptions<any, any, any, any, any, any> & object),
): Query<any, any, any, any> {
  if (!("effect" in options)) {
    const declaration = options as DeclarationOptions<any, any, any, any, any, any>
    let definition!: Query<any, any, any, any> & Operation.Any
    definition = makeDefinition(name, {
      key: declaration.key ?? Operation.payloadKey(declaration.payload ?? Schema.Void),
      staleTime: declaration.staleTime,
      gcTime: declaration.gcTime,
      refresh: declaration.refresh,
      source: {
        _tag: "Effect",
        effect: (input) => Operation.execute(definition, input) as never,
        retry: declaration.retry,
      },
    }) as never
    return Operation.attach(definition, name, "query", declaration as never) as never
  }
  const local = options as Options<any, any, any, any>
  return makeDefinition(name, {
    key: local.key,
    staleTime: local.staleTime,
    gcTime: local.gcTime,
    refresh: local.refresh,
    source: { _tag: "Effect", effect: local.effect, retry: local.retry }
  })
}

export function fromStream<Input, Event, Data, Error, Requirements>(
  name: string,
  options: FromStreamOptions<Input, Event, Data, Error, Requirements>
): Query<Input, Data, Error, Requirements>
export function fromStream<
  const Name extends string,
  Data,
  Payload extends Operation.PayloadInput = typeof Schema.Void,
  Success extends Schema.Schema.Any = typeof Schema.Void,
  Error extends Schema.Schema.All = typeof Schema.Never,
  Policy extends object = {},
>(
  name: Name,
  options: StreamDeclarationOptions<
    Payload,
    Success,
    Error,
    Policy,
    Operation.PayloadConstructor<Payload>,
    Schema.Schema.Type<Success>,
    Data,
    Schema.Schema.Type<Error>
  >,
): Query<
  Operation.PayloadConstructor<Payload>,
  Data,
  Schema.Schema.Type<Error>,
  Operation.Implementations<any>
> & Operation.Declared<Name, "queryFromStream", Payload, Success, Error, Policy>
export function fromStream(
  name: string,
  options: FromStreamOptions<any, any, any, any, any> | (StreamDeclarationOptions<any, any, any, any, any, any, any, any> & object),
): Query<any, any, any, any> {
  if (!("stream" in options)) {
    const declaration = options as StreamDeclarationOptions<any, any, any, any, any, any, any, any>
    let definition!: Query<any, any, any, any> & Operation.Any
    definition = makeDefinition(name, {
      key: declaration.key ?? Operation.payloadKey(declaration.payload ?? Schema.Void),
      staleTime: Number.POSITIVE_INFINITY,
      gcTime: declaration.gcTime,
      refresh: undefined,
      source: {
        _tag: "Fold",
        fold: (input) => Stream.map(Operation.stream(definition, input) as Stream.Stream<any, any, any>, (event) =>
          (previous: Option.Option<any>) => declaration.reduce(previous, event)),
        reconnect: declaration.reconnect,
      },
    }) as never
    return Operation.attach(definition, name, "queryFromStream", declaration as never) as never
  }
  const local = options as FromStreamOptions<any, any, any, any, any>
  return makeDefinition(name, {
    key: local.key,
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: local.gcTime,
    refresh: undefined,
    source: {
      _tag: "Fold",
      fold: (input) => Stream.map(local.stream(input), (event) => (previous: Option.Option<any>) =>
        local.reduce(previous, event)),
      reconnect: local.reconnect
    }
  })
}

export const makeAtomFamily = <Provided, RuntimeError, Input, Data, Error, Required extends Provided | Reactivity.Reactivity>(
  runtime: Atom.AtomRuntime<Provided | QueryClient, RuntimeError>,
  definition: Query<Input, Data, Error, Required>
): ((input: Input) => QueryAtom<Input, Data, Error | RuntimeError, Required>) => {
  const { name } = definition
  const options = definition[OptionsTypeId] as InternalOptions<Input, Data, Error, Required>
  const staleTime = Duration.toMillis(Duration.decode(options.staleTime ?? 0))
  const gcTime = options.gcTime ?? Duration.minutes(5)
  const keyFor = (input: Input): QueryKey => validatedKey(name, options.key(input))
  const source = options.source
  type Event = FetchEvent<Data, Error>

  const family = Atom.family((identity: QueryKey) => {
    let canonicalInput!: Input
    let hasCanonicalInput = false
    const keyHash = Hash.hash(identity)
    const retain = <A extends Atom.Atom<any>>(atom: A): A => Atom.setIdleTTL(atom, gcTime)
    const control = retain(Atom.make<Control<Data>>({
      invalidation: 0,
      cancelled: false,
      replacement: Option.none(),
      cancellation: Deferred.unsafeMake<never, FetchCancelled>(FiberId.none),
      override: Option.none()
    }))
    /** Explicitly started fetch generation; bumping it reopens the fetch stream. */
    const generation = retain(Atom.make(0))

    const cacheOf = Effect.map(QueryClient, (client): QueryCache => client[QueryCacheTypeId])

    /**
     * One fetch attempt of the request/response source, followed by a trailing
     * attempt whenever the entry was invalidated while this one was in flight.
     */
    const loadEffect = (
      registry: AtomRegistry.Registry,
      effect: (input: Input) => Effect.Effect<Data, Error, Required>,
      retry: Schedule.Schedule<unknown, Error, never> | undefined,
      sequence: Sequence,
      capturedInvalidation: number
    ): Stream.Stream<Event, never, Required | QueryClient> => {
      const started: Event = { _tag: "Started", sequence }
      const run = retry === undefined
        ? Effect.suspend(() => effect(canonicalInput))
        : Effect.retry(Effect.suspend(() => effect(canonicalInput)), retry)
      const outcome = Stream.unwrap(Effect.gen(function*() {
        const cache = yield* cacheOf
        yield* cache.emit({ _tag: "FetchStarted", name, keyHash })
        const exit = yield* Effect.exit(run)
        yield* cache.emit({ _tag: "FetchSettled", name, keyHash, success: Exit.isSuccess(exit) })
        const current = registry.get(control)
        const needsTrailing = !current.cancelled && current.invalidation > capturedInvalidation
        const event: Event = Exit.isSuccess(exit)
          ? {
            _tag: "Settled",
            sequence,
            data: exit.value,
            invalidation: capturedInvalidation,
            updatedAt: yield* Clock.currentTimeMillis,
            terminal: !needsTrailing
          }
          : { _tag: "Failed", sequence, cause: exit.cause, terminal: !needsTrailing }
        const trailing = needsTrailing
          ? loadEffect(
            registry,
            effect,
            retry,
            { generation: sequence.generation, attempt: sequence.attempt + 1 },
            current.invalidation
          )
          : undefined
        return trailing === undefined
          ? Stream.make(event)
          : Stream.concat(Stream.make(event), trailing)
      }))
      return Stream.concat(Stream.make(started), outcome)
    }

    /**
     * Fold source: the first element settles the fetch and every later element
     * settles it again with reduced data while the stream stays open. A terminal
     * failure after data was produced is a `Failed` event whose previous success
     * is retained by the derivation below.
     */
    const loadFold = (
      registry: AtomRegistry.Registry,
      fold: (input: Input) => Stream.Stream<(previous: Option.Option<Data>) => Data, Error, Required>,
      reconnect: Schedule.Schedule<unknown, Error, never> | undefined,
      sequence: Sequence,
      capturedInvalidation: number
    ): Stream.Stream<Event, never, Required | QueryClient> => {
      const started: Event = { _tag: "Started", sequence }
      const outcome = Stream.unwrap(Effect.gen(function*() {
        const cache = yield* cacheOf
        yield* cache.emit({ _tag: "FetchStarted", name, keyHash })
        let produced = false
        let stream = fold(canonicalInput)
        if (reconnect !== undefined) stream = Stream.retry(stream, reconnect)
        const settled: Stream.Stream<Event, never, Required> = stream.pipe(
          Stream.mapAccum(Option.none<Data>(), (previous, step) => {
            const next = step(previous)
            return [Option.some(next), next]
          }),
          Stream.mapEffect((data) => Effect.map(Clock.currentTimeMillis, (updatedAt): Extract<Event, { _tag: "Settled" }> => {
            produced = true
            return {
              _tag: "Settled",
              sequence,
              data,
              invalidation: capturedInvalidation,
              updatedAt,
              terminal: registry.get(control).invalidation <= capturedInvalidation
            }
          })),
          Stream.catchAllCause((cause): Stream.Stream<Event> => {
            if (Cause.isInterruptedOnly(cause)) return Stream.empty
            const failed: Event = {
              _tag: "Failed",
              sequence,
              cause,
              terminal: true
            }
            return Stream.succeed(failed)
          })
        )
        const closing = Stream.unwrap(Effect.gen(function*() {
          yield* cache.emit({ _tag: "FetchSettled", name, keyHash, success: produced })
          if (!produced) {
            const failed: Event = {
              _tag: "Failed",
              sequence,
              cause: Cause.die(new Error(`Query ${name}: stream completed before producing data`)),
              terminal: true
            }
            return Stream.make(failed)
          }
          const current = registry.get(control)
          return !current.cancelled && current.invalidation > capturedInvalidation
            ? loadFold(
              registry,
              fold,
              reconnect,
              { generation: sequence.generation, attempt: sequence.attempt + 1 },
              current.invalidation
            )
            : Stream.empty
        }))
        return Stream.concat(settled, closing)
      }))
      return Stream.concat(Stream.make(started), outcome)
    }

    /**
     * The fetch stream of the current generation. Elements are applied to the
     * graph by the atom runtime; nothing in the stream writes atoms. A cancelled
     * generation is an interrupted result so retained data stays visible.
     */
    const fetched = retain(runtime.atom((get) => {
      const currentGeneration = get(generation)
      const captured = get.once(control)
      const registry = get.registry
      if (captured.cancelled) return Stream.failCause(Cause.interrupt(FiberId.none))
      const sequence: Sequence = { generation: currentGeneration, attempt: 0 }
      return source._tag === "Effect"
        ? loadEffect(registry, source.effect, source.retry, sequence, captured.invalidation)
        : loadFold(registry, source.fold, source.reconnect, sequence, captured.invalidation)
    }))

    /** Await a concrete fetch generation and its sanctioned replacements, never presentation state. */
    const awaitGeneration = (
      registry: AtomRegistry.Registry,
      target: number,
      cancellation: Deferred.Deferred<never, FetchCancelled>
    ): Effect.Effect<Data, unknown> => {
      const completed = AtomRegistry.toStream(registry, fetched).pipe(
        Stream.filterMap((result): Option.Option<Exit.Exit<Data, unknown>> => {
          if (result._tag === "Failure") return Option.some(Exit.failCause(result.cause))
          const event = AtomResult.value(result)
          if (Option.isNone(event)) return Option.none()
          if (event.value.sequence.generation < target) return Option.none()
          switch (event.value._tag) {
            case "Started":
              return Option.none()
            case "Settled":
              return event.value.terminal
                ? Option.some(Exit.succeed(event.value.data))
                : Option.none()
            case "Failed":
              return event.value.terminal
                ? Option.some(Exit.failCause(event.value.cause))
                : Option.none()
          }
        }),
        Stream.runHead,
        Effect.flatMap(Option.match({
          onNone: () => Effect.dieMessage(`Query ${name}: fetch generation ended before completion`),
          onSome: (exit) => Exit.match(exit, {
            onFailure: (cause) => Effect.failCause(cause),
            onSuccess: (data) => Effect.succeed(data)
          })
        }))
      )
      return Effect.raceFirst(completed, Deferred.await(cancellation)).pipe(
        Effect.catchAll((error) => error === fetchCancelled ? Effect.interrupt : Effect.fail(error))
      )
    }

    const latestEvent = (get: Atom.Context): Option.Option<Event> => AtomResult.value(get(fetched))

    const ticket = (
      registry: AtomRegistry.Registry,
      target: number,
      cancellation: Deferred.Deferred<never, FetchCancelled>
    ): import("./internal.js").QueryFetch<Data> => ({
      await: awaitGeneration(registry, target, cancellation).pipe(
        // Query state is a lazy projection. Crossing the imperative fetch
        // boundary materializes the accepted terminal event before return.
        Effect.ensuring(Effect.sync(() => {
          registry.get(atom)
        }))
      )
    })

    /** The most recent settled data; remembered across later attempts and generations. */
    const accepted = retain(Atom.readable<Option.Option<Accepted<Data>>>((get) => {
      const previous = Option.flatten(get.self<Option.Option<Accepted<Data>>>())
      const event = latestEvent(get)
      if (Option.isSome(event) && event.value._tag === "Settled") {
        const { sequence, data, invalidation, updatedAt } = event.value
        return Option.some({ sequence, data, invalidation, updatedAt })
      }
      return previous
    }))

    /** Consecutive failed attempts, and the latest failure; reset by a settled attempt. */
    const failures = retain(Atom.readable<Failures<Error | RuntimeError>>((get) => {
      const previous = Option.getOrElse(
        get.self<Failures<Error | RuntimeError>>(),
        (): Failures<Error | RuntimeError> => ({ count: 0, latest: Option.none() })
      )
      const result = get(fetched)
      if (result._tag === "Failure" && !Cause.isInterruptedOnly(result.cause)) {
        // The runtime failed to build, or the stream itself failed.
        return {
          count: previous.count + 1,
          latest: Option.some({ sequence: { generation: -1, attempt: previous.count }, cause: result.cause })
        }
      }
      const event = AtomResult.value(result)
      if (Option.isNone(event)) return previous
      switch (event.value._tag) {
        case "Settled":
          return { count: 0, latest: Option.none() }
        case "Failed": {
          const counted = Option.exists(previous.latest, (latest) => sameSequence(latest.sequence, event.value.sequence))
          return {
            count: counted ? previous.count : previous.count + 1,
            latest: Option.some({ sequence: event.value.sequence, cause: event.value.cause })
          }
        }
        case "Started":
          return previous
      }
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
    // Membership in the client's cache lasts exactly as long as this node: the
    // registration node depends only on the runtime, so it is evaluated once and
    // released when it loses its last child (the state atom below).
    const registration = Atom.setIdleTTL(
      runtime.atom(Effect.gen(function*() {
        const cache = yield* cacheOf
        yield* Effect.acquireRelease(cache.registerQuery(entry), () => cache.unregisterQuery(entry))
      })),
      0
    )
    let atom = Atom.readable<State<Data, Error | RuntimeError>>((get) => {
      const registered = get(registration)
      if (registered._tag === "Failure") throw Cause.squash(registered.cause)
      if (scheduler !== undefined) get(scheduler)
      const current = get(control)
      const fetchedResult = get(fetched)
      const acceptedValue = get(accepted)
      const failed = get(failures)
      // A fetch is in flight from `Started` until that attempt settles or fails;
      // a fold source stays open after its first element without "fetching".
      const fetching = Option.match(AtomResult.value(fetchedResult), {
        onNone: () => fetchedResult.waiting,
        onSome: (event) => event._tag === "Started"
      })

      const supersededBy = (sequence: Sequence): boolean =>
        Option.exists(acceptedValue, (value) => after(value.sequence, sequence))
      const override = Option.filter(current.override, (value) => !supersededBy(value.sequence))
      const failure = Option.filter(failed.latest, (value) => !supersededBy(value.sequence))
      const previousSuccess = Option.map(acceptedValue, (value) =>
        AtomResult.success(value.data, { timestamp: value.updatedAt }))
      const result: AtomResult.Result<Data, Error | RuntimeError> = Option.match(override, {
        onSome: (value) => AtomResult.success(value.data, { timestamp: value.updatedAt, waiting: fetching }),
        onNone: () => Option.match(failure, {
          onSome: (value) => AtomResult.failure<Data, Error | RuntimeError>(value.cause, { previousSuccess, waiting: fetching }),
          onNone: () => Option.match(previousSuccess, {
            onNone: () => AtomResult.initial<Data, Error | RuntimeError>(fetching),
            onSome: (success) => AtomResult.success(success.value, { timestamp: success.timestamp, waiting: fetching })
          })
        })
      })

      const updatedAt = Option.match(override, {
        onSome: (value) => Option.some(value.updatedAt),
        onNone: () => Option.map(acceptedValue, (value) => value.updatedAt)
      })
      const acceptedInvalidation = Option.match(override, {
        onSome: (value) => value.invalidation,
        onNone: () => Option.match(acceptedValue, { onNone: () => -1, onSome: (value) => value.invalidation })
      })
      const ageFresh = staleTime === Number.POSITIVE_INFINITY
        || Option.exists(updatedAt, (timestamp) => Date.now() - timestamp < staleTime)
      const isStale = acceptedInvalidation < current.invalidation || !ageFresh
      if (!isStale && staleTime > 0 && staleTime !== Number.POSITIVE_INFINITY && Option.isSome(updatedAt)) {
        const timeout = setTimeout(() => get.refreshSelf(), Math.max(0, updatedAt.value + staleTime - Date.now()))
        get.addFinalizer(() => clearTimeout(timeout))
      }
      return EffectData.struct({
        result,
        fetchStatus: current.cancelled ? "paused" : fetching ? "fetching" : "idle",
        isStale,
        dataUpdatedAt: updatedAt,
        failureCount: failed.count
      })
    }, (refresh) => refresh(fetched))
    atom = retain(atom)

    entry = {
      stateAtom: atom,
      definition,
      name,
      key: identity,
      keyHash,
      state: (registry) => {
        const state = registry.get(atom)
        return { fetchStatus: state.fetchStatus, isStale: state.isStale }
      },
      start: (registry, options) => {
        const status = entry.state(registry).fetchStatus
        const currentGeneration = registry.get(generation)
        let shouldStart = status !== "fetching"
        let cancellation = registry.get(control).cancellation
        registry.update(control, (current) => {
          if (
            status === "fetching"
            && options?.cancelRefetch === true
            && !Option.contains(current.replacement, currentGeneration)
          ) {
            shouldStart = true
          }
          if (shouldStart && status !== "fetching") {
            cancellation = Deferred.unsafeMake<never, FetchCancelled>(FiberId.none)
          }
          return {
            ...current,
            cancelled: false,
            cancellation,
            replacement: shouldStart && status === "fetching"
              ? Option.some(currentGeneration + 1)
              : shouldStart ? Option.none() : current.replacement
          }
        })
        const target = shouldStart ? currentGeneration + 1 : currentGeneration
        if (shouldStart) registry.update(generation, () => target)
        return ticket(registry, target, cancellation)
      },
      join: (registry) => {
        const current = registry.get(control)
        return ticket(registry, registry.get(generation), current.cancellation)
      },
      cancel: (registry) => Effect.sync(() => {
        Deferred.unsafeDone(registry.get(control).cancellation, Effect.fail(fetchCancelled))
        registry.update(control, (current) => ({ ...current, cancelled: true }))
        registry.update(generation, (value) => value + 1)
      }),
      invalidate: (registry) => {
        registry.update(control, (current) => ({ ...current, invalidation: current.invalidation + 1 }))
      },
      setData: (registry, update) => Effect.gen(function*() {
        const updatedAt = yield* Clock.currentTimeMillis
        const currentData = AtomResult.value(registry.get(atom).result)
        const sequence = Option.match(AtomResult.value(registry.get(fetched)), {
          onNone: (): Sequence => ({ generation: registry.get(generation), attempt: 0 }),
          onSome: (event) => event.sequence
        })
        registry.update(control, (current) => ({
          ...current,
          override: Option.some({
            data: update(currentData),
            updatedAt,
            sequence,
            invalidation: current.invalidation
          })
        }))
      }),
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

export const isQuery = (value: unknown): value is Any =>
  typeof value === "object" && value !== null && QueryDefinitionTypeId in value

export const isQueryAtom = (value: unknown): value is QueryAtom.Any =>
  typeof value === "object" && value !== null && QueryEntryTypeId in value

export type { QueryFilter } from "./internal.js"
