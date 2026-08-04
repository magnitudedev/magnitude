import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import * as Data from "effect/Data"
import * as Deferred from "effect/Deferred"
import * as Duration from "effect/Duration"
import * as Effect from "effect/Effect"
import * as Option from "effect/Option"
import * as Schedule from "effect/Schedule"
import {
  addExecution,
  getClientCore,
  MutationInternalTypeId,
  mutationController,
  settleExecution,
  type MutationController,
  type MutationControllerCarrier
} from "./internal.js"
import {
  MutationDefinitionTypeId,
  MutationExecutionId,
  MutationScope,
  type AnyMutationExecution,
  type MutationDefinition,
  type MutationExecution,
  type MutationFilter
} from "./Model.js"

export const TypeId: unique symbol = Symbol.for("@magnitudedev/effect-query/Mutation")

export { MutationExecutionId, MutationScope }
export type { AnyMutationExecution, MutationExecution, MutationFilter }

export class MutationSynchronizationError<Output, SynchronizationError> extends Data.TaggedError(
  "MutationSynchronizationError"
)<{
  readonly output: Output
  readonly error: SynchronizationError
}> {}

export interface Mutation<Input, Output, CommandError, Requirements, SynchronizationError = never>
  extends Atom.Writable<
    AtomResult.Result<Output, CommandError | MutationSynchronizationError<Output, SynchronizationError>>,
    Input | Atom.Reset | Atom.Interrupt
  >,
    MutationDefinition,
    MutationControllerCarrier<
      Input,
      Output,
      CommandError | MutationSynchronizationError<Output, SynchronizationError>
    > {
  readonly [TypeId]?: {
    readonly input: Input
    readonly output: Output
    readonly commandError: CommandError
    readonly requirements: Requirements
    readonly synchronizationError: SynchronizationError
  }
  readonly name: string
  readonly match: () => MutationFilter
}

export type Any = MutationDefinition
export type Input<M> = M extends Mutation<infer I, infer _O, infer _CE, infer _R, infer _SE> ? I : never
export type Output<M> = M extends Mutation<infer _I, infer O, infer _CE, infer _R, infer _SE> ? O : never
export type CommandError<M> = M extends Mutation<infer _I, infer _O, infer E, infer _R, infer _SE> ? E : never
export type Requirements<M> = M extends Mutation<infer _I, infer _O, infer _CE, infer R, infer _SE> ? R : never
export type SynchronizationError<M> = M extends Mutation<infer _I, infer _O, infer _CE, infer _R, infer E> ? E : never

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

export interface Factory<Provided, RuntimeError> {
  readonly make: <
    Input,
    Output,
    CommandError,
    CommandRequirements extends Provided,
    SynchronizationError = never,
    SynchronizationRequirements extends Provided = never
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
  ) => Mutation<
    Input,
    Output,
    CommandError | RuntimeError,
    CommandRequirements | SynchronizationRequirements,
    SynchronizationError
  >
}

interface Invocation<Input, Output, Error> {
  readonly id: MutationExecutionId
  readonly input: Input
  readonly registry: AtomRegistry.Registry
  readonly deferred: Deferred.Deferred<Output, Error>
  readonly cancellation: Deferred.Deferred<void>
  result: AtomResult.Result<Output, Error>
  started: boolean
}

let nextExecutionId = 0

const makeDefinition = <
  Provided,
  RuntimeError,
  Input,
  Output,
  CommandError,
  CommandRequirements extends Provided,
  SynchronizationError,
  SynchronizationRequirements extends Provided
>(
  runtime: Atom.AtomRuntime<Provided, RuntimeError>,
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
  CommandError | RuntimeError,
  CommandRequirements | SynchronizationRequirements,
  SynchronizationError
> => {
  type PublicError = CommandError | RuntimeError | MutationSynchronizationError<Output, SynchronizationError>
  type CurrentInvocation = Invocation<Input, Output, PublicError>
  const registryAtom = Atom.readable((get) => get.registry)
  const semaphoresByRegistry = new WeakMap<AtomRegistry.Registry, Map<MutationScope, Effect.Semaphore>>()
  const semaphoresFor = (registry: AtomRegistry.Registry): Map<MutationScope, Effect.Semaphore> => {
    const existing = semaphoresByRegistry.get(registry)
    if (existing !== undefined) return existing
    const semaphores = new Map<MutationScope, Effect.Semaphore>()
    semaphoresByRegistry.set(registry, semaphores)
    return semaphores
  }
  const activeByRegistry = new WeakMap<AtomRegistry.Registry, Map<MutationExecutionId, CurrentInvocation>>()
  const activeFor = (registry: AtomRegistry.Registry): Map<MutationExecutionId, CurrentInvocation> => {
    const existing = activeByRegistry.get(registry)
    if (existing !== undefined) return existing
    const active = new Map<MutationExecutionId, CurrentInvocation>()
    activeByRegistry.set(registry, active)
    return active
  }
  const gcTime = Duration.toMillis(Duration.decode(options.gcTime ?? Duration.minutes(5)))
  const completeExecution = (
    core: ReturnType<typeof getClientCore>,
    invocation: CurrentInvocation,
    result: AtomResult.Result<Output, PublicError>
  ) => {
    invocation.result = result
    settleExecution(core, invocation.id, result)
    if (gcTime === Number.POSITIVE_INFINITY) return
    const timer = setTimeout(() => {
      const index = core.executions.findIndex((execution) => execution.id === invocation.id)
      if (index >= 0) core.executions.splice(index, 1)
      if (core.registry.getNodes().has(core.revision)) core.touch()
    }, gcTime)
    timer.unref()
  }
  let mutation!: Mutation<Input, Output, CommandError | RuntimeError, CommandRequirements | SynchronizationRequirements, SynchronizationError>

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
      const semaphores = semaphoresFor(invocation.registry)
      let semaphore = semaphores.get(scope)
      if (semaphore === undefined) {
        semaphore = Effect.unsafeMakeSemaphore(1)
        semaphores.set(scope, semaphore)
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
          completeExecution(
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
        completeExecution(getClientCore(get.registry), invocation, result)
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
      const id = MutationExecutionId(`${name}:${Date.now()}:${nextExecutionId++}`)
      const execution: MutationExecution<Input, Output, PublicError> = {
        id,
        mutation,
        input: value,
        result: AtomResult.initial(true),
        scope: Option.fromNullable(options.scope?.(value)),
        submittedAt: Date.now(),
        settledAt: Option.none()
      }
      addExecution(core, execution)
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
      if (invocation === undefined) throw new Error(`Mutation ${name} did not create an execution`)
      return { id: invocation.id, await: Deferred.await(invocation.deferred) }
    }
  }
  mutation = Object.assign(writable, {
    [MutationDefinitionTypeId]: true as const,
    [MutationInternalTypeId]: internal,
    name,
    match: (): MutationFilter => ({ mutation })
  })
  return mutation
}

export const bind = <Provided, RuntimeError>(
  runtime: Atom.AtomRuntime<Provided, RuntimeError>
): Factory<Provided, RuntimeError> => ({
  make: ((name: string, options: Options<unknown, unknown, unknown, Provided, unknown, Provided>) =>
    makeDefinition(runtime, name, options)) as Factory<Provided, RuntimeError>["make"]
})

export const execute = <Input, Output, CommandError, Requirements, SynchronizationError>(
  mutation: Mutation<Input, Output, CommandError, Requirements, SynchronizationError>,
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
  typeof value === "object" && value !== null && MutationInternalTypeId in value
