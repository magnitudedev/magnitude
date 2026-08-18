import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import type * as Reactivity from "@effect/experimental/Reactivity"
import * as Data from "effect/Data"
import * as Deferred from "effect/Deferred"
import * as Duration from "effect/Duration"
import * as Effect from "effect/Effect"
import * as Option from "effect/Option"
import * as Schedule from "effect/Schedule"
import {
  addMutationState,
  getClientCore,
  MutationInternalTypeId,
  mutationMatches,
  mutationController,
  settleMutationState,
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

/** Reactive mutation-cache selection, equivalent to TanStack Query's useMutationState. */
export const state = <M extends AnyMutation = AnyMutation, Selected = State<M>>(
  options: StateOptions<M, Selected> = {},
): Atom.Atom<ReadonlyArray<Selected>> => Atom.readable((get) => {
  const core = getClientCore(get.registry)
  get(core.revision)
  const states = core.mutationStates.filter((state) =>
    mutationMatches(state, mutationFilter(options.filters))) as unknown as ReadonlyArray<State<M>>
  return options.select === undefined
    ? states as unknown as ReadonlyArray<Selected>
    : states.map(options.select)
})

/** Number of matching pending mutation states, equivalent to TanStack Query's useIsMutating. */
export const isMutating = <M extends AnyMutation = AnyMutation>(
  filters?: Filters<M>,
): Atom.Atom<number> => state({
  filters: { ...filters, status: "pending" },
  select: () => 1,
}).pipe(Atom.map((matches) => matches.length))

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

interface Invocation<Input, Output, Error> {
  readonly id: MutationStateId
  readonly input: Input
  readonly registry: AtomRegistry.Registry
  readonly deferred: Deferred.Deferred<Output, Error>
  readonly cancellation: Deferred.Deferred<void>
  result: AtomResult.Result<Output, Error>
  started: boolean
}

let nextMutationStateId = 0

export const make = <
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
  >
): Mutation<
  Input,
  Output,
  CommandError,
  CommandRequirements | SynchronizationRequirements,
  SynchronizationError
> => {
  let definition!: Mutation<
    Input,
    Output,
    CommandError,
    CommandRequirements | SynchronizationRequirements,
    SynchronizationError
  >
  definition = {
    [MutationDefinitionTypeId]: true,
    [OptionsTypeId]: options,
    name,
    match: (): MutationFilter => ({ mutation: definition })
  }
  return definition
}

export const makeAtom = <
  Provided,
  RuntimeError,
  Input,
  Output,
  CommandError,
  Required extends Provided | Reactivity.Reactivity,
  SynchronizationError,
>(
  runtime: Atom.AtomRuntime<Provided, RuntimeError>,
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
  type CurrentInvocation = Invocation<Input, Output, PublicError>
  const registryAtom = Atom.readable((get) => get.registry)
  const activeByRegistry = new WeakMap<AtomRegistry.Registry, Map<MutationStateId, CurrentInvocation>>()
  const activeFor = (registry: AtomRegistry.Registry): Map<MutationStateId, CurrentInvocation> => {
    const existing = activeByRegistry.get(registry)
    if (existing !== undefined) return existing
    const active = new Map<MutationStateId, CurrentInvocation>()
    activeByRegistry.set(registry, active)
    return active
  }
  const gcTime = Duration.toMillis(Duration.decode(options.gcTime ?? Duration.minutes(5)))
  const completeInvocation = (
    core: ReturnType<typeof getClientCore>,
    invocation: CurrentInvocation,
    result: AtomResult.Result<Output, PublicError>
  ) => {
    invocation.result = result
    settleMutationState(core, invocation.id, result)
    if (gcTime === Number.POSITIVE_INFINITY) return
    const timer = setTimeout(() => {
      const index = core.mutationStates.findIndex((state) => state.id === invocation.id)
      if (index >= 0) core.mutationStates.splice(index, 1)
      if (core.registry.getNodes().has(core.revision)) core.touch()
    }, gcTime)
    timer.unref()
  }
  let mutation!: MutationAtom<Input, Output, CommandError | RuntimeError, Required, SynchronizationError>

  const run = runtime.fn<CurrentInvocation>()((invocation) => {
    if (invocation.started) {
      return Deferred.await(invocation.deferred)
    }
    invocation.started = true
    const core = getClientCore(invocation.registry)
    const command = options.retry === undefined
      ? Effect.suspend(() => options.effect(invocation.input))
      : Effect.retry(Effect.suspend(() => options.effect(invocation.input)), options.retry)
    let operation = command.pipe(
        Effect.flatMap((output) => options.synchronize === undefined
          ? Effect.succeed(output)
          : options.synchronize(output, invocation.input).pipe(
            Effect.mapError((error) => new MutationSynchronizationError({ output, error })),
            Effect.as(output)
          ))
      )

    const scope = options.scope?.(invocation.input)
    if (scope !== undefined) {
      let semaphore = core.mutationScopes.get(scope)
      if (semaphore === undefined) {
        semaphore = Effect.unsafeMakeSemaphore(1)
        core.mutationScopes.set(scope, semaphore)
      }
      operation = semaphore.withPermits(1)(operation)
    }

    return Effect.raceFirst(
      operation,
      Deferred.await(invocation.cancellation).pipe(Effect.zipRight(Effect.interrupt))
    ).pipe(
      Effect.onExit((exit) => Deferred.done(invocation.deferred, exit).pipe(
        Effect.zipRight(Effect.sync(() => {
          activeFor(invocation.registry).delete(invocation.id)
          completeInvocation(
            core,
            invocation,
            AtomResult.fromExit(exit)
          )
        }))
      ))
    )
  }, { concurrent: true })

  const latest = new WeakMap<AtomRegistry.Registry, CurrentInvocation>()
  const writable = Atom.writable(
    (get) => {
      const core = getClientCore(get.registry)
      get(core.revision)
      const result = run.read(get)
      const invocation = latest.get(get.registry)
      if (invocation !== undefined && (invocation.result._tag === "Initial" || invocation.result.waiting)
        && result._tag === "Failure" && !result.waiting) {
        activeFor(get.registry).delete(invocation.id)
        Effect.runSync(Deferred.failCause(invocation.deferred, result.cause))
        completeInvocation(getClientCore(get.registry), invocation, result)
      }
      return invocation === undefined
        ? result
        : invocation.result
    },
    (ctx, value: Input | Atom.Reset | Atom.Interrupt) => {
      if (value === Atom.Reset) {
        latest.delete(ctx.get(registryAtom))
        run.write(ctx, Atom.Reset)
        return
      }
      if (value === Atom.Interrupt) {
        for (const invocation of activeFor(ctx.get(registryAtom)).values()) {
          Effect.runSync(Deferred.succeed(invocation.cancellation, undefined))
        }
        run.write(ctx, Atom.Interrupt)
        return
      }
      const registry = ctx.get(registryAtom)
      const core = getClientCore(registry)
      const id = MutationStateId(`${name}:${Date.now()}:${nextMutationStateId++}`)
      const state: MutationState<Input, Output, PublicError> = {
        id,
        mutation: definition,
        input: value,
        result: AtomResult.initial(true),
        scope: Option.fromNullable(options.scope?.(value)),
        submittedAt: Date.now(),
        settledAt: Option.none()
      }
      addMutationState(core, state)
      core.emit({ _tag: "MutationStarted", name, id })
      const invocation: CurrentInvocation = {
        id,
        input: value,
        registry,
        deferred: Effect.runSync(Deferred.make<Output, PublicError>()),
        cancellation: Effect.runSync(Deferred.make<void>()),
        result: AtomResult.initial(true),
        started: false
      }
      latest.set(registry, invocation)
      activeFor(registry).set(id, invocation)
      run.write(ctx, invocation)
    }
  )

  const internal: MutationController<Input, Output, PublicError> = {
    invoke: (registry, input) => {
      registry.get(mutation)
      registry.set(mutation, input)
      const invocation = latest.get(registry)
      if (invocation === undefined) throw new Error(`Mutation ${name} did not create invocation state`)
      return { id: invocation.id, await: Deferred.await(invocation.deferred) }
    }
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
    () => {
      return controller.invoke(registry, input).await
    },
    (unmount) => Effect.sync(unmount)
  )
})

export const isMutation = (value: unknown): value is Any =>
  typeof value === "object" && value !== null && MutationDefinitionTypeId in value

export const isMutationAtom = (value: unknown): value is MutationAtom<any, any, any, any, any> =>
  typeof value === "object" && value !== null && MutationInternalTypeId in value
