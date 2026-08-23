import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import type * as Reactivity from "@effect/experimental/Reactivity"
import * as Cause from "effect/Cause"
import * as EffectData from "effect/Data"
import * as Duration from "effect/Duration"
import * as Effect from "effect/Effect"
import * as Equal from "effect/Equal"
import * as Hash from "effect/Hash"
import * as Option from "effect/Option"
import * as PubSub from "effect/PubSub"
import * as Schedule from "effect/Schedule"
import * as Schema from "effect/Schema"
import * as Stream from "effect/Stream"
import {
  QueryCacheTypeId,
  QueryClient,
  SubscriptionEntryTypeId,
  subscriptionEntry,
  type SubscriptionEntry,
  type SubscriptionEntryCarrier
} from "./internal.js"
import {
  SubscriptionDefinitionTypeId,
  type QueryKey,
  type SubscriptionDefinition,
  type SubscriptionFilter,
  type SubscriptionStatus
} from "./Model.js"
import * as Operation from "./Operation.js"

export const TypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Subscription")
const SetInputTypeId: unique symbol = Symbol("@magnitudedev/effect-query/Subscription/setInput")
const OptionsTypeId: unique symbol = Symbol("@magnitudedev/effect-query/Subscription/options")

export type Status = SubscriptionStatus

/**
 * Observable state of one shared subscription.
 *
 * - `connecting`: the stream is being established for the first time in this generation.
 * - `active`: at least one event has been received in the current attempt.
 * - `reconnecting`: the stream failed and the reconnect schedule is retrying; `latest` and
 *   `failure` are retained.
 * - `failed`: the reconnect schedule gave up (or the runtime could not be built).
 * - `completed`: the stream ended normally.
 * - `idle`: closed via `Atom.Interrupt`, or not started.
 */
export interface State<Event, Error> {
  readonly status: Status
  readonly latest: Option.Option<Event>
  readonly failure: Option.Option<Cause.Cause<Error>>
  readonly attempt: number
}

export interface SubscriptionAtom<Input, Event, Error, Requirements>
  extends Atom.Writable<State<Event, Error>, Atom.Reset | Atom.Interrupt>, SubscriptionEntryCarrier<Event> {
  readonly definition: SubscriptionDefinition
  readonly input: Input
}

export interface Subscription<Input, Event, Error, Requirements> extends SubscriptionDefinition {
  readonly [TypeId]?: {
    readonly input: Input
    readonly event: Event
    readonly error: Error
    readonly requirements: Requirements
  }
  readonly name: string
  readonly [OptionsTypeId]: unknown
  readonly match: {
    (): SubscriptionFilter
    (input: Input): SubscriptionFilter
  }
}

export type Any = SubscriptionDefinition
export type Input<S> = S extends Subscription<infer I, infer _E, infer _Err, infer _R> ? I : never
export type Event<S> = S extends Subscription<infer _I, infer E, infer _Err, infer _R> ? E : never
export type Error<S> = S extends Subscription<infer _I, infer _E, infer Err, infer _R> ? Err : never
export type Requirements<S> = S extends Subscription<infer _I, infer _E, infer _Err, infer R> ? R : never

export namespace SubscriptionAtom {
  export type Input<S> = S extends SubscriptionAtom<infer I, infer _E, infer _Err, infer _R> ? I : never
  export type Event<S> = S extends SubscriptionAtom<infer _I, infer E, infer _Err, infer _R> ? E : never
  export type Error<S> = S extends SubscriptionAtom<infer _I, infer _E, infer Err, infer _R> ? Err : never
}

export interface Options<Input, Event, Error, Requirements> {
  readonly key: (input: Input) => QueryKey
  readonly stream: (input: Input) => Stream.Stream<Event, Error, Requirements>
  /** Retry policy applied when the stream fails; defaults to capped exponential backoff with jitter. */
  readonly reconnect?: Schedule.Schedule<unknown, Error, never>
  /** How long the shared stream stays open after its last observer leaves. */
  readonly gcTime?: Duration.DurationInput
}

export type DeclarationOptions<
  Payload extends Operation.PayloadInput,
  Success extends Schema.Schema.Any,
  Error extends Schema.Schema.All,
  Policy extends object,
  Input,
  StreamError,
> = Operation.Shape<Payload, Success, Error, Policy> & {
  readonly key?: (input: Input) => QueryKey
  readonly reconnect?: Schedule.Schedule<unknown, StreamError, never>
  readonly gcTime?: Duration.DurationInput
}

export const defaultReconnect: Schedule.Schedule<unknown, unknown, never> = Schedule.exponential("100 millis").pipe(
  Schedule.modifyDelay((_, delay) => Duration.min(delay, Duration.seconds(5))),
  Schedule.jittered
)

interface Control {
  readonly generation: number
  readonly closed: boolean
}

export function make<Input, Event, Error, Requirements>(
  name: string,
  options: Options<Input, Event, Error, Requirements>
): Subscription<Input, Event, Error, Requirements>
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
): Subscription<
  Operation.PayloadConstructor<Payload>,
  Schema.Schema.Type<Success>,
  Schema.Schema.Type<Error>,
  Operation.Implementations<any>
> & Operation.Declared<Name, "subscription", Payload, Success, Error, Policy>
export function make(
  name: string,
  options: Options<any, any, any, any> | (DeclarationOptions<any, any, any, any, any, any> & object),
): Subscription<any, any, any, any> {
  const declared = !("stream" in options)
  let definition!: Subscription<any, any, any, any>
  const resolvedOptions: Options<any, any, any, any> = declared
    ? {
        key: options.key ?? Operation.payloadKey(options.payload ?? Schema.Void),
        stream: (input) => Operation.stream(definition as Subscription<any, any, any, any> & Operation.Any, input) as never,
        reconnect: options.reconnect,
        gcTime: options.gcTime,
      }
    : options as Options<any, any, any, any>
  const keyFor = (input: any): QueryKey => {
    const key = resolvedOptions.key(input)
    if ((typeof key === "object" && key !== null) && !Equal.isEqual(key)) {
      throw new TypeError(`Subscription ${name} returned a structured key without Effect Equal semantics`)
    }
    return key
  }
  definition = {
    [SubscriptionDefinitionTypeId]: true,
    [OptionsTypeId]: resolvedOptions,
    name,
    match: (input?: any): SubscriptionFilter => input === undefined
      ? { definition }
      : { definition, key: keyFor(input) }
  }
  return declared
    ? Operation.attach(definition, name, "subscription", options as never) as never
    : definition
}

export const makeAtomFamily = <Provided, RuntimeError, Input, Event, Error, Required extends Provided | Reactivity.Reactivity>(
  runtime: Atom.AtomRuntime<Provided | QueryClient, RuntimeError>,
  definition: Subscription<Input, Event, Error, Required>
): ((input: Input) => SubscriptionAtom<Input, Event, Error | RuntimeError, Required>) => {
  const { name } = definition
  const options = definition[OptionsTypeId] as Options<Input, Event, Error, Required>
  const gcTime = options.gcTime ?? Duration.minutes(5)
  const reconnect = options.reconnect ?? defaultReconnect
  const keyFor = (input: Input): QueryKey => {
    const key = options.key(input)
    if ((typeof key === "object" && key !== null) && !Equal.isEqual(key)) {
      throw new TypeError(`Subscription ${name} returned a structured key without Effect Equal semantics`)
    }
    return key
  }
  type PublicState = State<Event, Error | RuntimeError>

  const family = Atom.family((identity: QueryKey) => {
    let canonicalInput!: Input
    let hasCanonicalInput = false
    const retain = <A extends Atom.Atom<any>>(atom: A): A => Atom.setIdleTTL(atom, gcTime)
    const initialState: PublicState = {
      status: "idle",
      latest: Option.none(),
      failure: Option.none(),
      attempt: 0
    }
    const control = retain(Atom.make<Control>({ generation: 0, closed: false }))
    const pubsub = Effect.runSync(PubSub.unbounded<Event>())

    /**
     * The subscription's lifecycle as a stream of public states: connecting,
     * active per event, reconnecting per failure (driven by the reconnect
     * schedule), failed when that schedule is exhausted or the source dies,
     * completed when the source ends, idle when closed. Elements are applied to
     * the graph by the atom runtime; nothing here writes atoms.
     */
    const runner = retain(runtime.atom((get) => {
      const { closed } = get(control)
      if (closed) return Stream.succeed<PublicState>(initialState)
      const attempt = (
        driver: Schedule.ScheduleDriver<unknown, Error, never>,
        index: number,
        latest: Option.Option<Event>
      ): Stream.Stream<PublicState, never, Required> => {
        let seen = latest
        const opening: PublicState = {
          status: index === 0 ? "connecting" : "reconnecting",
          latest,
          failure: Option.none(),
          attempt: index + 1
        }
        const events: Stream.Stream<PublicState, Error, Required> = options.stream(canonicalInput).pipe(
          Stream.mapEffect((event) => Effect.map(PubSub.publish(pubsub, event), (): PublicState => {
            seen = Option.some(event)
            return { status: "active", latest: seen, failure: Option.none(), attempt: index + 1 }
          }))
        )
        const completed = Stream.unwrap(Effect.sync(() => Stream.succeed<PublicState>({
          status: "completed",
          latest: seen,
          failure: Option.none(),
          attempt: index + 1
        })))
        return Stream.concat(Stream.succeed(opening), Stream.concat(events, completed)).pipe(
          Stream.catchAllCause((cause): Stream.Stream<PublicState, never, Required> => {
            if (Cause.isInterruptedOnly(cause)) return Stream.empty
            const failed = Stream.succeed<PublicState>({
              status: "failed",
              latest: seen,
              failure: Option.some(cause),
              attempt: index + 1
            })
            return Option.match(Cause.failureOption(cause), {
              onNone: () => failed,
              onSome: (error) => Stream.concat(
                Stream.succeed<PublicState>({
                  status: "reconnecting",
                  latest: seen,
                  failure: Option.some(cause),
                  attempt: index + 1
                }),
                Stream.unwrap(driver.next(error).pipe(
                  Effect.map(() => attempt(driver, index + 1, seen)),
                  Effect.catchAll(() => Effect.succeed(failed))
                ))
              )
            })
          })
        )
      }
      return Stream.unwrap(Effect.map(Schedule.driver(reconnect), (driver) => attempt(driver, 0, Option.none())))
    }))

    let entry!: SubscriptionEntry<Event>
    // Cache membership lasts exactly as long as this node (see Query.ts).
    const registration = Atom.setIdleTTL(
      runtime.atom(Effect.gen(function*() {
        const cache = (yield* QueryClient)[QueryCacheTypeId]
        yield* Effect.acquireRelease(cache.registerSubscription(entry), () => cache.unregisterSubscription(entry))
      })),
      0
    )
    let atom = Atom.writable<PublicState, Atom.Reset | Atom.Interrupt>(
      (get) => {
        const registered = get(registration)
        if (registered._tag === "Failure") throw Cause.squash(registered.cause)
        const run = get(runner)
        const previous: PublicState = Option.getOrElse(get.self<PublicState>(), () => initialState)
        if (run._tag === "Failure" && !run.waiting && !Cause.isInterruptedOnly(run.cause)) {
          const failed: PublicState = { ...previous, status: "failed", failure: Option.some(run.cause) }
          return EffectData.struct(failed)
        }
        const state: PublicState = Option.getOrElse(AtomResult.value(run), () => previous)
        return EffectData.struct(state)
      },
      (ctx, value) => {
        const current = ctx.get(control)
        if (value === Atom.Reset) {
          ctx.set(control, { generation: current.generation + 1, closed: false })
          return
        }
        ctx.set(control, { ...current, closed: true })
      }
    )
    atom = retain(atom)

    entry = {
      stateAtom: atom,
      definition,
      name,
      key: identity,
      keyHash: Hash.hash(identity),
      reconnect: (registry) => registry.set(atom, Atom.Reset),
      close: (registry) => registry.set(atom, Atom.Interrupt),
      events: (registry) => Stream.unwrapScoped(Effect.gen(function*() {
        const dequeue = yield* PubSub.subscribe(pubsub)
        yield* Effect.acquireRelease(
          Effect.sync(() => registry.mount(atom)),
          (unmount) => Effect.sync(unmount)
        )
        return Stream.fromQueue(dequeue)
      }))
    }

    const atomWithInput = Object.defineProperty(atom, "input", {
      get: () => canonicalInput,
      enumerable: true,
      configurable: false
    }) as typeof atom & { readonly input: Input }
    return Object.assign(atomWithInput, {
      [SubscriptionEntryTypeId]: entry,
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

/**
 * The shared stream's events. Consuming this keeps the subscription mounted for
 * the lifetime of the consumer; the subscription closes `gcTime` after the last
 * observer leaves.
 */
export const events = <Input, Event, Error, Requirements>(
  subscription: SubscriptionAtom<Input, Event, Error, Requirements>
): Stream.Stream<Event, never, AtomRegistry.AtomRegistry> =>
  Stream.unwrap(Effect.map(AtomRegistry.AtomRegistry, (registry) => subscriptionEntry(subscription).events(registry)))

/** Current state of one materialized subscription, read outside the Atom graph. */
export const state = <Input, Event, Error, Requirements>(
  subscription: SubscriptionAtom<Input, Event, Error, Requirements>
): Effect.Effect<State<Event, Error>, never, AtomRegistry.AtomRegistry> =>
  Effect.map(AtomRegistry.AtomRegistry, (registry) => registry.get(subscription))

export const isSubscription = (value: unknown): value is Any =>
  typeof value === "object" && value !== null && SubscriptionDefinitionTypeId in value

export const isSubscriptionAtom = (value: unknown): value is SubscriptionAtom<unknown, unknown, unknown, unknown> =>
  typeof value === "object" && value !== null && SubscriptionEntryTypeId in value
