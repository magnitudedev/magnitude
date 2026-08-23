import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import type * as Reactivity from "@effect/experimental/Reactivity"
import * as Chunk from "effect/Chunk"
import * as Clock from "effect/Clock"
import * as Context from "effect/Context"
import * as Data from "effect/Data"
import * as Duration from "effect/Duration"
import * as Effect from "effect/Effect"
import * as Option from "effect/Option"
import * as Schedule from "effect/Schedule"
import * as Schema from "effect/Schema"
import {
  MutationInternalTypeId,
  mutationMatches,
  mutationController,
  QueryCacheTypeId,
  QueryClient,
  type MutationController,
  type MutationControllerCarrier
} from "./internal.js"
import {
  MutationDefinitionTypeId,
  MutationStateId,
  MutationScope,
  type AnyMutationState,
  type MutationDefinition,
  type MutationState,
  type MutationFilter
} from "./Model.js"
import * as Operation from "./Operation.js"

export const TypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Mutation")
const OptionsTypeId: unique symbol = Symbol("@magnitudedev/effect-query/Mutation/options")

export { MutationScope }

export class MutationSynchronizationError<Output, SynchronizationError> extends Data.TaggedError(
  "MutationSynchronizationError"
)<{
  readonly output: Output
  readonly error: SynchronizationError
}> {}

export interface Mutation<Input, Output, CommandError, Requirements, SynchronizationError = never>
  extends MutationDefinition {
  readonly [TypeId]?: {
    readonly input: Input
    readonly output: Output
    readonly commandError: CommandError
    readonly requirements: Requirements
    readonly synchronizationError: SynchronizationError
  }
  readonly [OptionsTypeId]: unknown
  readonly name: string
  readonly match: () => MutationFilter
}

export interface MutationAtom<Input, Output, CommandError, Requirements, SynchronizationError = never>
  extends Atom.Writable<
    AtomResult.Result<Output, CommandError | MutationSynchronizationError<Output, SynchronizationError>>,
    Input | Atom.Reset | Atom.Interrupt
  >,
    MutationControllerCarrier<
      Input,
      Output,
      CommandError | MutationSynchronizationError<Output, SynchronizationError>
    > {
  readonly definition: MutationDefinition
}

export type Any = MutationDefinition
export type Input<M> = M extends Mutation<infer I, infer _O, infer _CE, infer _R, infer _SE> ? I : never
export type Output<M> = M extends Mutation<infer _I, infer O, infer _CE, infer _R, infer _SE> ? O : never
export type CommandError<M> = M extends Mutation<infer _I, infer _O, infer E, infer _R, infer _SE> ? E : never
export type Requirements<M> = M extends Mutation<infer _I, infer _O, infer _CE, infer R, infer _SE> ? R : never
export type SynchronizationError<M> = M extends Mutation<infer _I, infer _O, infer _CE, infer _R, infer E> ? E : never
export type Error<M> = M extends Mutation<infer _I, infer O, infer CE, infer _R, infer SE>
  ? CE | MutationSynchronizationError<O, SE>
  : never
export type State<M> = M extends Mutation<infer I, infer O, infer CE, infer _R, infer SE>
  ? MutationState<I, O, CE | MutationSynchronizationError<O, SE>>
  : AnyMutationState

export interface Filters<M extends AnyMutation = AnyMutation> {
  readonly mutation?: M
  readonly scope?: MutationScope
  readonly status?: "pending" | "success" | "error"
  readonly predicate?: (state: State<M>) => boolean
}

type AnyMutation = MutationDefinition

const mutationFilter = <M extends AnyMutation>(
  filters: Filters<M> | undefined,
): MutationFilter | undefined => filters === undefined ? undefined : ({
    mutation: filters.mutation,
    scope: filters.scope,
    status: filters.status,
    predicate: filters.predicate as MutationFilter["predicate"],
  })

export interface StateOptions<M extends AnyMutation = AnyMutation, Selected = State<M>> {
  readonly filters?: Filters<M>
  readonly select?: (state: State<M>) => Selected
}

/**
 * Reactive mutation-history selection over the contextual QueryClient's cache,
 * equivalent to TanStack Query's useMutationState.
 */
export function state<M extends AnyMutation, Selected>(
  options: { readonly filters?: Filters<M>; readonly select: (state: State<M>) => Selected },
): Effect.Effect<Atom.Atom<ReadonlyArray<Selected>>, never, QueryClient>
export function state<M extends AnyMutation = AnyMutation>(
  options?: { readonly filters?: Filters<M> },
): Effect.Effect<Atom.Atom<ReadonlyArray<State<M>>>, never, QueryClient>
export function state<M extends AnyMutation, Selected>(
  options: StateOptions<M, Selected> = {},
): Effect.Effect<Atom.Atom<ReadonlyArray<State<M> | Selected>>, never, QueryClient> {
  const filter = mutationFilter(options.filters)
  const select = options.select
  return Effect.map(QueryClient, (client) => {
    const history = Atom.subscriptionRef(client[QueryCacheTypeId].mutations)
    return Atom.readable((get) => {
      const matching = Chunk.toReadonlyArray(get(history))
        .filter((entry): entry is State<M> => mutationMatches(entry, filter))
      return select === undefined ? matching : matching.map(select)
    })
  })
}

/** Number of matching pending mutations, equivalent to TanStack Query's useIsMutating. */
export const isMutating = <M extends AnyMutation = AnyMutation>(
  filters?: Filters<M>,
): Effect.Effect<Atom.Atom<number>, never, QueryClient> =>
  Effect.map(
    state({ filters: { ...filters, status: "pending" } }),
    (matches) => Atom.map(matches, (states) => states.length)
  )

export interface Options<Input, Output, CommandError, CommandRequirements, SynchronizationError, SynchronizationRequirements> {
  readonly effect: (input: Input) => Effect.Effect<Output, CommandError, CommandRequirements>
  readonly synchronize?: (
    output: Output,
    input: Input
  ) => Effect.Effect<void, SynchronizationError, SynchronizationRequirements>
  readonly scope?: (input: Input) => MutationScope
  readonly retry?: Schedule.Schedule<unknown, CommandError, never>
  readonly gcTime?: Duration.DurationInput
}

export type DeclarationOptions<
  Payload extends Operation.PayloadInput,
  Success extends Schema.Schema.Any,
  Error extends Schema.Schema.All,
  Policy extends object,
  Input,
  Output,
  CommandError,
  SynchronizationError,
  SynchronizationRequirements,
> = Operation.Shape<Payload, Success, Error, Policy> & {
  readonly synchronize?: (
    output: Output,
    input: Input,
  ) => Effect.Effect<void, SynchronizationError, SynchronizationRequirements>
  readonly scope?: (input: Input) => MutationScope
  readonly retry?: Schedule.Schedule<unknown, CommandError, never>
  readonly gcTime?: Duration.DurationInput
}

let nextMutationStateId = 0

export function make<
  Input,
  Output,
  CommandError,
  CommandRequirements,
  SynchronizationError = never,
  SynchronizationRequirements = never
>(
  name: string,
  options: Options<
    Input,
    Output,
    CommandError,
    CommandRequirements,
    SynchronizationError,
    SynchronizationRequirements
  >,
): Mutation<
  Input,
  Output,
  CommandError,
  CommandRequirements | SynchronizationRequirements,
  SynchronizationError
>
export function make<
  const Name extends string,
  Payload extends Operation.PayloadInput = typeof Schema.Void,
  Success extends Schema.Schema.Any = typeof Schema.Void,
  Error extends Schema.Schema.All = typeof Schema.Never,
  Policy extends object = {},
  SynchronizationError = never,
  SynchronizationRequirements = never,
>(
  name: Name,
  options: DeclarationOptions<
    Payload,
    Success,
    Error,
    Policy,
    Operation.PayloadConstructor<Payload>,
    Schema.Schema.Type<Success>,
    Schema.Schema.Type<Error>,
    SynchronizationError,
    SynchronizationRequirements
  >,
): Mutation<
  Operation.PayloadConstructor<Payload>,
  Schema.Schema.Type<Success>,
  Schema.Schema.Type<Error>,
  Operation.Implementations<any> | SynchronizationRequirements,
  SynchronizationError
> & Operation.Declared<Name, "mutation", Payload, Success, Error, Policy>
export function make(
  name: string,
  options: Options<any, any, any, any, any, any> | (DeclarationOptions<any, any, any, any, any, any, any, any, any> & object),
): Mutation<any, any, any, any, any> {
  const declared = !("effect" in options)
  let definition!: Mutation<any, any, any, any, any>
  const effect = declared
    ? (input: unknown) => Operation.execute(definition as Mutation<any, any, any, any, any> & Operation.Any, input) as never
    : (options as Options<any, any, any, any, any, any>).effect
  definition = {
    [MutationDefinitionTypeId]: true,
    [OptionsTypeId]: { ...options, effect },
    name,
    match: (): MutationFilter => ({ mutation: definition })
  }
  return declared
    ? Operation.attach(definition, name, "mutation", options as never) as never
    : definition
}

type Command<Input> =
  | { readonly _tag: "Submit"; readonly id: MutationStateId; readonly input: Input }
  | { readonly _tag: "Interrupt" }

export const makeAtom = <
  Provided,
  RuntimeError,
  Input,
  Output,
  CommandError,
  Required extends Provided | Reactivity.Reactivity,
  SynchronizationError,
>(
  runtime: Atom.AtomRuntime<Provided | QueryClient, RuntimeError>,
  definition: Mutation<
    Input,
    Output,
    CommandError,
    Required,
    SynchronizationError
  >
): MutationAtom<
  Input,
  Output,
  CommandError | RuntimeError,
  Required,
  SynchronizationError
> => {
  const { name } = definition
  const options = definition[OptionsTypeId] as Options<
    Input,
    Output,
    CommandError,
    Required,
    SynchronizationError,
    Required
  >
  type PublicError = CommandError | RuntimeError | MutationSynchronizationError<Output, SynchronizationError>
  type PublicResult = AtomResult.Result<Output, PublicError>
  const gcTime = options.gcTime ?? Duration.minutes(5)

  /**
   * The invocation this atom presents. Atom state is per registry by
   * construction; kept alive so the presented invocation survives the atom
   * being unobserved, as it did with a per-registry map.
   */
  const latestId = Atom.keepAlive(Atom.make(Option.none<MutationStateId>()))

  /**
   * Submission and interruption run as Effects in the client's context. A
   * submitted command becomes a fiber of the cache scope, so it outlives this
   * evaluation, is serialized by its semantic scope, and is interrupted by
   * `Atom.Interrupt` or by disposing the registry.
   */
  const run = runtime.fn<Command<Input>>()((command) => Effect.gen(function*() {
    const cache = (yield* QueryClient)[QueryCacheTypeId]
    if (command._tag === "Interrupt") return yield* cache.interruptInvocations(definition)
    const { id, input } = command
    const scope = Option.fromNullable(options.scope?.(input))
    yield* cache.addMutation({
      id,
      mutation: definition,
      input,
      result: AtomResult.initial(true),
      scope,
      submittedAt: yield* Clock.currentTimeMillis,
      settledAt: Option.none()
    })
    yield* cache.emit({ _tag: "MutationStarted", name, id })
    const command$ = options.retry === undefined
      ? Effect.suspend(() => options.effect(input))
      : Effect.retry(Effect.suspend(() => options.effect(input)), options.retry)
    let operation: Effect.Effect<Output, PublicError, Required> = command$.pipe(
      Effect.flatMap((output) => options.synchronize === undefined
        ? Effect.succeed(output)
        : options.synchronize(output, input).pipe(
          Effect.mapError((error) => new MutationSynchronizationError({ output, error })),
          Effect.as(output)
        ))
    )
    if (Option.isSome(scope)) {
      const semaphore = yield* cache.scopeSemaphore(scope.value)
      operation = semaphore.withPermits(1)(operation)
    }
    const context = yield* Effect.context<Required>()
    yield* cache.runInvocation(
      id,
      definition,
      operation.pipe(
        Effect.provide(context),
        Effect.onExit((exit) => cache.settleMutation(id, AtomResult.fromExit(exit)))
      )
    )
  }))

  /**
   * The history entry of one invocation: `None` while the runtime is still
   * building, otherwise whether the entry is present. A fresh node reads the
   * current history synchronously and follows changes thereafter, so the entry
   * is visible in the same evaluation that submitted it; absence means the
   * history has forgotten it.
   */
  const historyEntry = Atom.family((id: MutationStateId) =>
    runtime.subscriptionRef(Effect.map(QueryClient, (client) => client[QueryCacheTypeId].mutations)).pipe(
      Atom.map((history) => history._tag === "Success"
        ? Option.some(Chunk.findFirst(history.value, (entry) => entry.id === id))
        : Option.none<Option.Option<AnyMutationState>>())
    ))

  /**
   * History retention for one invocation: the entry stays in the cache while
   * this node is observed and `gcTime` after it loses its last observer.
   */
  const retention = Atom.family((id: MutationStateId) => Atom.setIdleTTL(
    runtime.atom(Effect.gen(function*() {
      const cache = (yield* QueryClient)[QueryCacheTypeId]
      yield* Effect.acquireRelease(Effect.void, () => cache.forgetMutation(id))
    })),
    gcTime
  ))

  const writable = Atom.writable<PublicResult, Input | Atom.Reset | Atom.Interrupt>(
    (get) => {
      const runResult = get(run)
      const fallback = (): PublicResult => runResult._tag === "Failure"
        ? AtomResult.failure(runResult.cause)
        : AtomResult.initial()
      const latest = get(latestId)
      if (Option.isNone(latest)) return fallback()
      get(retention(latest.value))
      const history = get(historyEntry(latest.value))
      if (Option.isNone(history)) return runResult._tag === "Failure" ? fallback() : AtomResult.initial(true)
      return Option.match(history.value, {
        onNone: fallback,
        onSome: (entry) => entry.result as PublicResult
      })
    },
    (ctx, value: Input | Atom.Reset | Atom.Interrupt) => {
      if (value === Atom.Reset) {
        ctx.set(latestId, Option.none())
        ctx.set(run, Atom.Reset)
        return
      }
      if (value === Atom.Interrupt) {
        ctx.set(run, { _tag: "Interrupt" })
        return
      }
      const id = MutationStateId(`${name}:${nextMutationStateId++}`)
      ctx.set(latestId, Option.some(id))
      ctx.set(run, { _tag: "Submit", id, input: value })
    }
  )

  let mutation!: MutationAtom<Input, Output, CommandError | RuntimeError, Required, SynchronizationError>
  const internal: MutationController<Input, Output, PublicError> = {
    submit: (registry, input) => {
      registry.set(mutation, input)
      return Option.getOrThrow(registry.get(latestId))
    },
    await: (registry, id) => Effect.gen(function*() {
      const built = yield* AtomRegistry.getResult(registry, runtime)
      const cache = Context.get(built.context, QueryClient)[QueryCacheTypeId]
      const settled = yield* cache.awaitMutation(id)
      const result = settled.result as PublicResult
      return result._tag === "Success"
        ? result.value
        : result._tag === "Failure"
          ? yield* Effect.failCause(result.cause)
          : yield* Effect.dieMessage(`Mutation ${id} settled without a result`)
    })
  }
  mutation = Object.assign(writable, {
    [MutationInternalTypeId]: internal,
    definition
  })
  return mutation
}

export const execute = <Input, Output, CommandError, Requirements, SynchronizationError>(
  mutation: MutationAtom<Input, Output, CommandError, Requirements, SynchronizationError>,
  input: Input
): Effect.Effect<
  Output,
  CommandError | MutationSynchronizationError<Output, SynchronizationError>,
  AtomRegistry.AtomRegistry
> => Effect.gen(function*() {
  const registry = yield* AtomRegistry.AtomRegistry
  const controller = mutationController(mutation)
  return yield* Effect.acquireUseRelease(
    Effect.sync(() => registry.mount(mutation)),
    () => controller.await(registry, controller.submit(registry, input)),
    (unmount) => Effect.sync(unmount)
  )
})

export const isMutation = (value: unknown): value is Any =>
  typeof value === "object" && value !== null && MutationDefinitionTypeId in value

export const isMutationAtom = (value: unknown): value is MutationAtom<any, any, any, any, any> =>
  typeof value === "object" && value !== null && MutationInternalTypeId in value
