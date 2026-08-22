import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
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
import * as Stream from "effect/Stream"
import {
  registerSubscriptionEntry,
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

export const defaultReconnect: Schedule.Schedule<unknown, unknown, never> = Schedule.exponential("100 millis").pipe(
  Schedule.modifyDelay((_, delay) => Duration.min(delay, Duration.seconds(5))),
  Schedule.jittered
)

interface Control {
  readonly generation: number
  readonly closed: boolean
}

/** A real asynchronous boundary: continues in a microtask, after the registry finished evaluating. */
const afterEvaluation: Effect.Effect<void> = Effect.async<void>((resume) => {
  queueMicrotask(() => resume(Effect.void))
})

export const make = <Input, Event, Error, Requirements>(
  name: string,
  options: Options<Input, Event, Error, Requirements>
): Subscription<Input, Event, Error, Requirements> => {
  let definition!: Subscription<Input, Event, Error, Requirements>
  const keyFor = (input: Input): QueryKey => {
    const key = options.key(input)
    if ((typeof key === "object" && key !== null) && !Equal.isEqual(key)) {
      throw new TypeError(`Subscription ${name} returned a structured key without Effect Equal semantics`)
    }
    return key
  }
  definition = {
    [SubscriptionDefinitionTypeId]: true,
    [OptionsTypeId]: options,
    name,
    match: (input?: Input): SubscriptionFilter => input === undefined
      ? { definition }
      : { definition, key: keyFor(input) }
  }
  return definition
}

export const makeAtomFamily = <Provided, RuntimeError, Input, Event, Error, Required extends Provided | Reactivity.Reactivity>(
  runtime: Atom.AtomRuntime<Provided, RuntimeError>,
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
    const state = retain(Atom.make<PublicState>(initialState))
    const control = retain(Atom.make<Control>({ generation: 0, closed: false }))
    const pubsub = Effect.runSync(PubSub.unbounded<Event>())

    const runner = retain(runtime.atom((get) => {
      const { closed } = get(control)
      const registry = get.registry
      const update = (f: (current: PublicState) => PublicState) => Effect.sync(() => {
        if (!registry.getNodes().has(state)) return
        registry.update(state, f)
      })
      // Atom state is never written while this atom is being evaluated: every
      // update below runs after `afterEvaluation`, outside the registry's
      // synchronous evaluation (a yield alone is drained synchronously).
      if (closed) return afterEvaluation.pipe(Effect.zipRight(update((current) => ({ ...current, status: "idle" }))))
      const attempt = Effect.suspend(() =>
        update((current) => ({
          ...current,
          status: current.attempt === 0 ? "connecting" : "reconnecting",
          attempt: current.attempt + 1
        })).pipe(
          Effect.zipRight(options.stream(canonicalInput).pipe(
            Stream.runForEach((event) => PubSub.publish(pubsub, event).pipe(
              Effect.zipRight(update((current) => ({
                ...current,
                status: "active",
                latest: Option.some(event),
                failure: Option.none()
              })))
            ))
          ))
        )
      )
      return afterEvaluation.pipe(
        Effect.zipRight(attempt),
        Effect.tapErrorCause((cause) => update((current) => ({
          ...current,
          status: "reconnecting",
          failure: Option.some(cause)
        }))),
        Effect.retry(reconnect),
        Effect.matchCauseEffect({
          onFailure: (cause) => Cause.isInterruptedOnly(cause)
            ? Effect.void
            : update((current) => ({ ...current, status: "failed", failure: Option.some(cause) })),
          onSuccess: () => update((current) => ({ ...current, status: "completed" }))
        })
      )
    }))

    let entry!: SubscriptionEntry<Event>
    let atom = Atom.writable<PublicState, Atom.Reset | Atom.Interrupt>(
      (get) => {
        const unregister = registerSubscriptionEntry(get.registry, entry)
        get.addFinalizer(unregister)
        const current = get(state)
        const run = get(runner)
        if (run._tag === "Failure" && !run.waiting) {
          return EffectData.struct<PublicState>({ ...current, status: "failed", failure: Option.some(run.cause) })
        }
        return EffectData.struct(current)
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
