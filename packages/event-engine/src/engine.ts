import {
  Cause,
  Context,
  Data,
  Deferred,
  Effect,
  ExecutionStrategy,
  Exit,
  Fiber,
  FiberRef,
  HashMap,
  Layer,
  Option,
  PubSub,
  Queue,
  Ref,
  Scope,
  Stream,
  DateTime,
} from "effect"
import {
  AddressInternal,
  AddressMatch,
  AddressRoot,
  type AddressEncodedValue,
  type AddressRef,
  type AddressSpace,
  type AddressSpaceBase,
  type AddressTarget,
} from "./address"
import type { AmbientDefinition, AmbientReader } from "./ambient"
import {
  type DiscriminatorKey,
  makeEventSelection,
} from "./event"
import type { EventStore } from "./event-store"
import {
  makeProjectionBuilder,
  ProjectionDefinitionBase,
  ProjectionDefinitionInternal,
  ProjectionInternal,
  CurrentAddressUnavailable,
  isProjectionReadError,
  ProjectionNotInEngine,
  type PartitionUpdate,
  type ProjectionReadError,
  ProjectionReader,
  type ProjectionFor,
  UndeclaredProjectionRead,
} from "./projection"
import {
  Emit,
  makeWorkerBuilder,
  PartitionedWorkerHandler,
  PartitionedWorkerSignalHandler,
  type UpdateAmbient,
  WorkerDefinitionBase,
  WorkerDefinitionInternal,
  WorkerInternal,
  type WorkerRequirements,
  WorkerHandler,
  type WorkerInterruption,
  WorkerSignalHandler,
  type WorkerHandlerError,
  type WorkerFor,
} from "./worker"
import { makeFacadeBuilder } from "./facade"
import {
  makeSignalBuilder,
  SignalInternal,
  type SignalDefinition,
  type SignalEmitter,
  type SignalOccurrence,
  type SignalObserver,
  SignalCascadeExceeded,
} from "./signal"
import {
  EngineRuntimeObservationTypeId,
  type ObservableEngineRuntime,
} from "./runtime-internal"

export class DuplicateComponentName extends Data.TaggedError("DuplicateComponentName")<{
  readonly name: string
}> {}

export class ProjectionDependencyCycle extends Data.TaggedError("ProjectionDependencyCycle")<{
  readonly projection: string
}> {}

export class MissingProjectionDependency extends Data.TaggedError("MissingProjectionDependency")<{
  readonly projection: string
  readonly dependency: string
}> {}

export class DuplicateAddressSpaceName extends Data.TaggedError("DuplicateAddressSpaceName")<{
  readonly name: string
}> {}

export class AddressRouteMismatch extends Data.TaggedError("AddressRouteMismatch")<{
  readonly expectedSpace: string
}> {}

export class ForeignEngineComponent extends Data.TaggedError("ForeignEngineComponent")<{
  readonly component: string
}> {}

export class MissingWorkerProjectionDependency extends Data.TaggedError("MissingWorkerProjectionDependency")<{
  readonly worker: string
  readonly dependency: string
}> {}

export class DuplicateSignalName extends Data.TaggedError("DuplicateSignalName")<{
  readonly name: string
}> {}

export class ForeignEngineSignal extends Data.TaggedError("ForeignEngineSignal")<{
  readonly name: string
}> {}

export class MissingSignalPublisher extends Data.TaggedError("MissingSignalPublisher")<{
  readonly signal: string
  readonly subscriber: string
}> {}

export type EngineDefinitionError =
  | DuplicateComponentName
  | ProjectionDependencyCycle
  | MissingProjectionDependency
  | DuplicateAddressSpaceName
  | ForeignEngineComponent
  | MissingWorkerProjectionDependency
  | DuplicateSignalName
  | ForeignEngineSignal
  | MissingSignalPublisher

export class EngineAlreadyActivated extends Data.TaggedError("EngineAlreadyActivated")<{}> {}

export type EngineActivationError = EngineAlreadyActivated
export type EventSettlementError = ProjectionReadError | SignalCascadeExceeded

/** Whether a value is a settlement's error. */
export const isEventSettlementError = (value: unknown): value is EventSettlementError =>
  isProjectionReadError(value) || value instanceof SignalCascadeExceeded

/** An accept on an engine that is closing or closed. */
export class EngineClosed extends Data.TaggedError("EngineClosed")<{
  readonly closure: Closure<unknown>
}> {}

/** How an engine ended: closed, closed for removal, or closed because a worker handler failed. */
export type Closure<StoreError> =
  | { readonly _tag: "Closed" }
  | { readonly _tag: "Removed" }
  | { readonly _tag: "Failed"; readonly cause: Cause.Cause<WorkerHandlerError | StoreError> }

type SignalEnqueue = (signal: SignalDefinition<unknown>, payload: unknown, address: AddressRef | undefined) => void

interface ProjectionRuntime<Event> {
  readonly definition: ProjectionDefinitionBase
  // Folds a trigger through the projection's handlers of it, in their order.
  readonly handle: (trigger: ProjectionTrigger<Event>, enqueue: SignalEnqueue) => Effect.Effect<void, ProjectionReadError>
  readonly get: Effect.Effect<unknown>
  readonly getAt: (address: AddressRef) => Effect.Effect<Option.Option<unknown>>
  readonly entries: Effect.Effect<HashMap.HashMap<AddressRef, unknown>>
  readonly addresses: Effect.Effect<readonly AddressRef[]>
  // Captures the state, and returns its restoration.
  readonly capture: Effect.Effect<Effect.Effect<void>>
}

type WorkerTrigger<Event> =
  | { readonly _tag: "Event"; readonly event: Event }
  | { readonly _tag: "Signal"; readonly signal: SignalOccurrence<unknown> }

// The engine provides every handler's requirements where it is built.
type LaneHandlers<Event, E> = readonly (
  | WorkerHandler<Event, E, any>
  | PartitionedWorkerHandler<Event, AddressSpaceBase, E, any>
  | WorkerSignalHandler<Event, E, any>
  | PartitionedWorkerSignalHandler<Event, AddressSpaceBase, E, any>
)[]

interface LaneTask<Event, E> {
  readonly _tag: "Event"
  readonly trigger: WorkerTrigger<Event>
  readonly handlers: LaneHandlers<Event, E>
  // The first interruption it received, while queued or running.
  interruption: Option.Option<WorkerInterruption>
}

interface LaneBarrier {
  readonly _tag: "Barrier"
  readonly done: Deferred.Deferred<void>
}

// Ends the lane once the items before it have been taken.
interface LaneEnd {
  readonly _tag: "End"
}

type LaneItem<Event, E> = LaneTask<Event, E> | LaneBarrier | LaneEnd

interface WorkerLane<Event, E> {
  readonly address: AddressRef | undefined
  readonly offer: (trigger: WorkerTrigger<Event>, handlers: LaneHandlers<Event, E>) => Effect.Effect<void>
  // Gives every queued task `interruption`, which it observes when it runs, and interrupts the
  // running handler with it, without waiting: its interruption handling may emit through the
  // acceptance lock its caller holds. A task keeps the first interruption it receives.
  readonly interrupt: (interruption: WorkerInterruption) => Effect.Effect<void>
  // Interrupts the lane, and ends it once its running handler has finished.
  readonly close: (interruption: WorkerInterruption) => Effect.Effect<void>
  // Queues a barrier behind the lane's work, and returns its completion. The caller holds the
  // acceptance lock, so the barrier is queued before any `End` a later close queues.
  readonly barrier: Effect.Effect<Effect.Effect<void>>
  readonly done: Effect.Effect<void>
}

interface WorkerRuntime<Event, E> {
  readonly definition: WorkerDefinitionBase
  // Opens the worker's initial lanes, whose handlers emit through `emit`.
  readonly activate: (emit: Emit<Event, E>) => Effect.Effect<void>
  // Resolves every route an occurrence takes through this worker, and returns its release.
  readonly prepare: (trigger: WorkerTrigger<Event>) => Effect.Effect<Effect.Effect<void>>
  readonly interrupt: (target: InterruptTarget) => Effect.Effect<void>
  readonly close: (interruption: WorkerInterruption) => Effect.Effect<void>
  // Queues a barrier on every open lane, and returns the completions of those barriers and of the
  // lanes still finishing. The caller holds the acceptance lock.
  readonly barriers: Effect.Effect<readonly Effect.Effect<void>[]>
  readonly done: Effect.Effect<void>
}

const targetAddresses = <A extends AddressRef>(
  target: AddressTarget<A>,
  existing: readonly A[],
  expectedSpace?: AddressSpaceBase,
): readonly A[] => {
  const addresses = (() => {
    switch (target._tag) {
    case "None": return []
    case "One": return [target.address]
    case "Many": return target.addresses
    case "All": return existing
    }
  })()
  if (expectedSpace !== undefined) {
    for (const address of addresses) {
      if (!AddressInternal.belongsTo(address, expectedSpace)) {
        throw new AddressRouteMismatch({ expectedSpace: expectedSpace.name })
      }
    }
  }
  return addresses
}

const sortProjections = (
  projections: readonly ProjectionDefinitionBase[],
  publishers: ReadonlyMap<SignalDefinition<unknown>, readonly ProjectionDefinitionBase[]>,
): readonly ProjectionDefinitionBase[] => {
  const declared = new Set(projections)
  const visiting = new Set<ProjectionDefinitionBase>()
  const visited = new Set<ProjectionDefinitionBase>()
  const sorted: ProjectionDefinitionBase[] = []

  const visit = (projection: ProjectionDefinitionBase): void => {
    if (visited.has(projection)) return
    if (visiting.has(projection)) {
      throw new ProjectionDependencyCycle({ projection: projection.name })
    }
    visiting.add(projection)
    const signalDependencies = projection.signalSubscriptions.flatMap((signal) => publishers.get(signal) ?? [])
    for (const dependency of [...projection.dependencies, ...signalDependencies]) {
      if (!declared.has(dependency)) {
        throw new MissingProjectionDependency({ projection: projection.name, dependency: dependency.name })
      }
      visit(dependency)
    }
    visiting.delete(projection)
    visited.add(projection)
    sorted.push(projection)
  }

  for (const projection of projections) visit(projection)
  return sorted
}

const validateNames = (
  projections: readonly ProjectionDefinitionBase[],
  workers: readonly WorkerDefinitionBase[],
): void => {
  const names = new Set<string>()
  for (const component of [...projections, ...workers]) {
    if (names.has(component.name)) throw new DuplicateComponentName({ name: component.name })
    names.add(component.name)
  }
}

const validateWorkerDependencies = (
  projections: readonly ProjectionDefinitionBase[],
  workers: readonly WorkerDefinitionBase[],
): void => {
  const declared = new Set(projections)
  for (const worker of workers) {
    for (const dependency of worker.dependencies) {
      if (!declared.has(dependency)) {
        throw new MissingWorkerProjectionDependency({ worker: worker.name, dependency: dependency.name })
      }
    }
  }
}

const validateSignals = (
  family: object,
  projections: readonly ProjectionDefinitionBase[],
  workers: readonly WorkerDefinitionBase[],
): ReadonlyMap<SignalDefinition<unknown>, readonly ProjectionDefinitionBase[]> => {
  const publishers = new Map<SignalDefinition<unknown>, ProjectionDefinitionBase[]>()
  const names = new Map<string, SignalDefinition<unknown>>()
  const allSignals = projections.flatMap((projection) => [
    ...projection.emits,
    ...projection.signalSubscriptions,
  ]).concat(workers.flatMap((worker) => worker.signalSubscriptions))

  for (const signal of allSignals) {
    if (!SignalInternal.belongsTo(signal, family)) {
      throw new ForeignEngineSignal({ name: signal.name })
    }
    const previous = names.get(signal.name)
    if (previous !== undefined && previous !== signal) {
      throw new DuplicateSignalName({ name: signal.name })
    }
    names.set(signal.name, signal)
  }

  for (const projection of projections) {
    for (const signal of projection.emits) {
      const existing = publishers.get(signal) ?? []
      publishers.set(signal, [...existing, projection])
    }
  }
  for (const projection of projections) {
    for (const signal of projection.signalSubscriptions) {
      if ((publishers.get(signal)?.length ?? 0) === 0) {
        throw new MissingSignalPublisher({ signal: signal.name, subscriber: projection.name })
      }
    }
  }
  for (const worker of workers) {
    for (const signal of worker.signalSubscriptions) {
      if ((publishers.get(signal)?.length ?? 0) === 0) {
        throw new MissingSignalPublisher({ signal: signal.name, subscriber: worker.name })
      }
    }
  }
  return publishers
}

const validateSpaces = (
  projections: readonly ProjectionDefinitionBase[],
  workers: readonly WorkerDefinitionBase[],
): void => {
  const spaces = [...projections, ...workers].flatMap((component) =>
    component.topology._tag === "Partitioned" ? [component.topology.by] : [],
  )
  const siblings = new Map<AddressSpaceBase | typeof AddressRoot, Map<string, AddressSpaceBase>>()
  for (const leaf of spaces) {
    let current: AddressSpaceBase | typeof AddressRoot = leaf
    while (current !== AddressRoot) {
      const concrete = current as AddressSpace<AddressSpaceBase | typeof AddressRoot, unknown, AddressEncodedValue>
      const byName = siblings.get(concrete.parentSpace) ?? new Map<string, AddressSpaceBase>()
      const previous = byName.get(concrete.name)
      if (previous !== undefined && previous !== concrete) {
        throw new DuplicateAddressSpaceName({ name: concrete.name })
      }
      byName.set(concrete.name, concrete)
      siblings.set(concrete.parentSpace, byName)
      current = concrete.parentSpace
    }
  }
}

const makeLane = <Event, E>(
  scope: Scope.Scope,
  address: AddressRef | undefined,
  execute: (
    trigger: WorkerTrigger<Event>,
    handlers: LaneHandlers<Event, E>,
    address: AddressRef | undefined,
    interruption: Effect.Effect<Option.Option<WorkerInterruption>>,
  ) => Effect.Effect<void, WorkerHandlerError | E>,
  fail: (cause: Cause.Cause<WorkerHandlerError | E>) => Effect.Effect<void>,
  offered: { count: number },
): Effect.Effect<WorkerLane<Event, E>> => Effect.gen(function* () {
  const queue = yield* Queue.unbounded<LaneItem<Event, E>>()
  const coordination = yield* Effect.makeSemaphore(1)
  const queued = new Set<LaneTask<Event, E>>()
  let current: { readonly task: LaneTask<Event, E>; readonly fiber: Fiber.RuntimeFiber<void, WorkerHandlerError | E> } | undefined

  const consumer = yield* Effect.gen(function* () {
    while (true) {
      const item = yield* Queue.take(queue)
      if (item._tag === "End") return
      if (item._tag === "Barrier") {
        yield* Deferred.succeed(item.done, undefined)
        continue
      }
      const fiber = yield* coordination.withPermits(1)(Effect.gen(function* () {
        queued.delete(item)
        const fiber = yield* Effect.fork(execute(item.trigger, item.handlers, address, Effect.sync(() => item.interruption)))
        current = { task: item, fiber }
        return fiber
      }))
      const exit = yield* Fiber.await(fiber)
      yield* coordination.withPermits(1)(Effect.sync(() => {
        current = undefined
      }))
      // A handler's failure is the engine's: it closes the engine as failed.
      if (Exit.isFailure(exit) && !Cause.isInterruptedOnly(exit.cause)) yield* fail(exit.cause)
    }
  // Lanes open while an acceptance releases uninterruptibly; their work stays interruptible.
  }).pipe(Effect.interruptible, Effect.forkIn(scope))

  const mark = (task: LaneTask<Event, E>, interruption: WorkerInterruption) => {
    task.interruption = Option.orElse(task.interruption, () => Option.some(interruption))
  }
  const interrupt = (interruption: WorkerInterruption) => coordination.withPermits(1)(Effect.gen(function* () {
    for (const task of queued) mark(task, interruption)
    if (current === undefined) return
    mark(current.task, interruption)
    yield* Fiber.interruptFork(current.fiber)
  }))

  const barrier = Effect.flatMap(Deferred.make<void>(), (done) =>
    Effect.as(Queue.offer(queue, { _tag: "Barrier", done }), Deferred.await(done)))

  const offer = (trigger: WorkerTrigger<Event>, handlers: LaneHandlers<Event, E>): Effect.Effect<void> =>
    coordination.withPermits(1)(Effect.suspend(() => {
      const task: LaneTask<Event, E> = { _tag: "Event", trigger, handlers, interruption: Option.none() }
      queued.add(task)
      offered.count += 1
      return Queue.offer(queue, task)
    }).pipe(Effect.asVoid))

  return {
    address,
    offer,
    interrupt,
    close: (interruption) => Effect.zipRight(interrupt(interruption), Queue.offer(queue, { _tag: "End" })),
    barrier,
    done: Fiber.await(consumer).pipe(Effect.asVoid),
  }
})

/**
 * What a transaction interrupts: the lanes at an address or under a match, or every lane of a
 * worker, partitioned or not.
 */
export type InterruptTarget = AddressRef | AddressMatch<AddressRef> | WorkerDefinitionBase

/**
 * What a transaction decided: events to accept as one, lanes to interrupt once they are committed,
 * which reach the work from before the transaction and not the work its events release, and its result.
 */
export interface Transaction<Event, A> {
  readonly events: readonly Event[]
  readonly interrupts?: readonly InterruptTarget[]
  readonly result: A
}

export interface EngineRuntime<Event, StoreError = never> extends ObservableEngineRuntime {
  readonly accept: (event: Event) => Effect.Effect<void, EventSettlementError | StoreError | EngineClosed>
  /**
   * Reads settled state and decides, under the acceptance lock: its events settle, commit in one
   * store commit and release together, or not at all, and its interrupts apply in the same
   * uninterruptible step, between the commit and the release. The body's context has only a reader.
   */
  readonly transact: <A, E = never, R = never>(
    body: (context: { readonly read: ProjectionReader }) => Effect.Effect<Transaction<Event, A>, E, R>,
  ) => Effect.Effect<A, E | EventSettlementError | StoreError | EngineClosed, R>
  /**
   * Activates the workers with `emit`, or with the engine's own emit, which commits to the engine's
   * store, if any, and is never rejected. Activating again with the same emit does nothing.
   */
  readonly activate: (options?: { readonly emit: Emit<Event, StoreError> }) => Effect.Effect<void, EngineActivationError>
  readonly interrupt: (target: InterruptTarget) => Effect.Effect<void>
  /** Sets an ambient's value, like a handler's `update`, and rejects with `EngineClosed` once closing has begun or a handler has failed. */
  readonly update: <Value>(ambient: AmbientDefinition<Value>, value: Value) => Effect.Effect<void, EventSettlementError | EngineClosed>
  /**
   * Every read is of one settled state. A fiber that holds the acceptance lock, as a transaction's
   * body does, reads directly. Every other call that takes the lock (`accept`, `transact`,
   * `activate`, `interrupt`, `update`, `awaitIdle`, `close`, the engine's own emit, and workers'
   * `update`) dies as a defect when made by a fiber that holds it, instead of waiting on itself.
   */
  readonly read: ProjectionReader
  readonly signals: SignalObserver
  /** Completes once no lane holds work, including work that work admitted meanwhile. */
  readonly awaitIdle: Effect.Effect<void>
  /**
   * Starts the close, unless one has started, and waits for `closed`. The close rejects further
   * accepts, transactions and public updates with `EngineClosed`, interrupts every lane with `Closed`
   * or `Removed` (handlers may still emit and update while they finish), waits for the lanes, ends
   * change observation and signal streams, and completes `closed`. It runs once, in its own fiber:
   * interrupting a caller stops only its wait. Closing the engine's scope closes it as `Closed`.
   */
  readonly close: (reason: "closed" | "removed") => Effect.Effect<void>
  /** The engine's closure, once it has closed: a worker handler failure supersedes a close. */
  readonly closed: Effect.Effect<Closure<StoreError>>
}

export interface EngineDefinition<Event, StoreError = never, R = never> {
  readonly name: string
  readonly projections: readonly ProjectionDefinitionBase[]
  readonly workers: readonly WorkerDefinitionBase[]
  readonly Tag: Context.Tag<EngineRuntime<Event, StoreError>, EngineRuntime<Event, StoreError>>
  readonly Layer: Layer.Layer<EngineRuntime<Event, StoreError>, never, R>
  /** An engine without a store: accepted events are settled and released, and kept nowhere. */
  readonly build: Effect.Effect<EngineRuntime<Event, StoreError>, never, Scope.Scope | R>
  /**
   * An engine over a store: every accepted event is committed before it is settled for good and
   * released. Opening replays the store with workers inactive and no signals released, runs
   * `recover`, then activates workers with an emit that commits.
   */
  readonly open: <E = never, R2 = never>(options: {
    readonly store: EventStore<Event, StoreError>
    readonly recover?: (context: {
      readonly read: ProjectionReader
      readonly emit: Emit<Event, StoreError>
    }) => Effect.Effect<void, E, R2>
  }) => Effect.Effect<EngineRuntime<Event, StoreError>, EventSettlementError | StoreError | E, Scope.Scope | R | R2>
}

const makePendingSignalEmitter = () => {
  const pending: Array<readonly [SignalDefinition<unknown>, unknown]> = []
  const emit = ((signal: SignalDefinition<unknown>, payload: unknown) => {
    pending.push([signal, payload])
  }) as SignalEmitter<readonly SignalDefinition<unknown>[]>
  const release = (
    enqueue: SignalEnqueue,
    address: AddressRef | undefined,
  ): void => {
    for (const [signal, payload] of pending) enqueue(signal, payload, address)
  }
  return { emit, release }
}

const isWorker = (target: InterruptTarget): target is WorkerDefinitionBase => WorkerInternal.typeId in target

type Settled<Event> =
  | { readonly _tag: "Event"; readonly event: Event }
  | { readonly _tag: "Ambient"; readonly ambient: AmbientDefinition<unknown>; readonly value: unknown }

// What projection handlers fold: a settled event or ambient value, or a signal of its cascade.
type ProjectionTrigger<Event> = Settled<Event> | WorkerTrigger<Event>

type ProjectionHandler<Event> = ProjectionDefinitionInternal<Event>["_handlers"][number]

// Whether a handler handles a trigger.
const triggers = <Event>(descriptor: ProjectionHandler<Event>, trigger: ProjectionTrigger<Event>): boolean => {
  switch (trigger._tag) {
  case "Event": return descriptor._tag === "Event" && descriptor.matcher.matches(trigger.event)
  case "Signal": return descriptor._tag === "Signal" && descriptor.signal === trigger.signal.signal
  case "Ambient": return descriptor._tag === "Ambient" && descriptor.ambient === trigger.ambient
  }
}

// What a trigger's handlers route on, and the context field that carries it to them.
const triggerInput = <Event>(trigger: ProjectionTrigger<Event>): { readonly route: unknown; readonly context: object } => {
  switch (trigger._tag) {
  case "Event": return { route: trigger.event, context: { event: trigger.event } }
  case "Signal": return { route: trigger.signal, context: { signal: trigger.signal } }
  case "Ambient": return { route: trigger.value, context: { value: trigger.value } }
  }
}

const makeRuntime = <Event, StoreError, R>(
  projections: readonly ProjectionDefinitionBase[],
  workers: readonly WorkerDefinitionBase[],
  store: EventStore<Event, StoreError> | undefined,
) => Effect.gen(function* () {
  const scope = yield* Effect.scope
  const context = yield* Effect.context<R>()
  // Lanes and the engine's close run here. It closes after the engine's own close finalizer, which
  // a later finalizer runs, has awaited `closed`, so nothing interrupts the close.
  const laneScope = yield* Scope.fork(scope, ExecutionStrategy.sequential)
  const acceptance = yield* Effect.makeSemaphore(1)
  // The acceptance lock's current hold, and each fiber's hold, which the fibers it forks inherit. A
  // fiber holds the lock while its hold is the current one, so a fiber that outlives the hold it
  // inherited does not.
  let holder: object | undefined
  const hold = yield* FiberRef.make<object | undefined>(undefined)
  const holdsLock = Effect.map(FiberRef.get(hold), (held) => held !== undefined && held === holder)
  // Runs `effect`, which waits on the acceptance lock. A fiber that holds it would wait on itself
  // forever, so it dies instead.
  const unheld = <A, E, R>(effect: Effect.Effect<A, E, R>): Effect.Effect<A, E, R> =>
    Effect.flatMap(holdsLock, (held) => held
      ? Effect.dieMessage("The acceptance lock was taken by a fiber that holds it")
      : effect)
  // Runs `effect` holding the acceptance lock.
  const locked = <A, E, R>(effect: Effect.Effect<A, E, R>): Effect.Effect<A, E, R> =>
    unheld(acceptance.withPermits(1)(Effect.acquireUseRelease(
        Effect.sync(() => {
          holder = {}
          return holder
        }),
        (held) => Effect.locally(effect, hold, held),
        () => Effect.sync(() => {
          holder = undefined
        }),
      )))
  // Runs `effect` over one settled state: under the acceptance lock, or directly when this fiber holds it.
  const settled = <A, E, R>(effect: Effect.Effect<A, E, R>): Effect.Effect<A, E, R> =>
    Effect.flatMap(holdsLock, (held) => held ? effect : locked(effect))
  // Observers read after each settlement, latest first: a slow one has at most one pending.
  const occurrences = yield* PubSub.sliding<void>(1)
  const signalPubSub = yield* PubSub.unbounded<SignalOccurrence<unknown>>()
  const closed = yield* Deferred.make<Closure<StoreError>>()
  const projectionRuntimes = new Map<ProjectionDefinitionBase, ProjectionRuntime<Event>>()
  const ambientValues = new Map<AmbientDefinition<unknown>, unknown>()
  const ambientReader: AmbientReader = {
    get: <Value>(ambient: AmbientDefinition<Value>) =>
      (ambientValues.has(ambient) ? ambientValues.get(ambient) : ambient.initial) as Value,
  }
  const offered = { count: 0 }
  // Worker emits are never rejected: closing waits for the lanes, whose handlers may still emit.
  type HandlerError = StoreError
  // The close's closure, from when it takes the acceptance lock: then nothing more is released to
  // workers.
  let ending: Closure<StoreError> | undefined
  // The first worker handler failure, recorded as it happens, ahead of the close it starts: from
  // then the engine rejects accepts, and its closure is this failure.
  let failed: Closure<StoreError> | undefined
  // The engine's final closure, once every lane has ended.
  let ended: Closure<StoreError> | undefined
  // Whether the close has started: it runs once.
  let closing = false

  // Accepts, transactions and public updates are rejected once a handler has failed or closing has
  // begun.
  const rejectClosed = Effect.suspend(() => {
    const closure = failed ?? ending
    return closure === undefined ? Effect.void : Effect.fail(new EngineClosed({ closure }))
  })
  // Work still running may emit and update until the engine has ended. After, no lane runs, so only
  // a fiber that escaped its handler can, which is a defect: nothing it emits is stored.
  const unlessEnded = (call: string) => Effect.suspend(() => ended === undefined
    ? Effect.void
    : Effect.dieMessage(`A worker ${call} after the engine ended, from a fiber that outlived its handler`))

  // Every read is of one settled state.
  const makeReader = (
    currentAddress: AddressRef | undefined,
    allowed: ReadonlySet<ProjectionDefinitionBase> | undefined,
  ): ProjectionReader => {
    const ensure = (definition: ProjectionDefinitionBase): Effect.Effect<ProjectionRuntime<Event>, ProjectionReadError> => {
      if (allowed !== undefined && !allowed.has(definition)) {
        return Effect.fail(new UndeclaredProjectionRead({ projection: definition.name }))
      }
      const runtime = projectionRuntimes.get(definition)
      return runtime === undefined
        ? Effect.fail(new ProjectionNotInEngine({ projection: definition.name }))
        : Effect.succeed(runtime)
    }
    return {
      get: (definition) => settled(ensure(definition).pipe(Effect.flatMap((runtime) => runtime.get))) as never,
      at: (definition, address) => settled(ensure(definition).pipe(Effect.flatMap((runtime) => runtime.getAt(address)))) as never,
      current: (definition) => currentAddress === undefined
        ? Effect.fail(new CurrentAddressUnavailable()) as never
        : settled(ensure(definition).pipe(Effect.flatMap((runtime) => runtime.getAt(currentAddress)))) as never,
      entries: (definition) => settled(ensure(definition).pipe(Effect.flatMap((runtime) => runtime.entries))) as never,
    }
  }

  for (const definition of projections) {
    const internal = definition as ProjectionDefinitionInternal<Event>
    const allowed = new Set(definition.dependencies)
    const handlers = internal._handlers
    // Runs one handler on its state for a trigger, and releases the signals it emitted.
    const apply = (
      descriptor: ProjectionHandler<Event>,
      trigger: ProjectionTrigger<Event>,
      state: unknown,
      address: AddressRef | undefined,
      enqueue: SignalEnqueue,
    ) => Effect.gen(function* () {
      const handle = descriptor.handle as unknown as (context: object) => PartitionUpdate<unknown> | Effect.Effect<PartitionUpdate<unknown>, ProjectionReadError>
      const pending = makePendingSignalEmitter()
      const result = yield* ProjectionInternal.toEffect(handle({
        ...triggerInput(trigger).context,
        ...(address === undefined ? {} : { address }),
        state,
        read: makeReader(address, allowed),
        ambient: ambientReader,
        emit: pending.emit,
      }))
      pending.release(enqueue, address)
      return result
    })

    if (definition.topology._tag === "Singleton") {
      const config = internal._config
      if (!("initial" in config) || typeof config.initial === "function") continue
      const state = yield* Ref.make<unknown>(config.initial)
      projectionRuntimes.set(definition, {
        definition,
        handle: (trigger, enqueue) => Effect.forEach(handlers, (descriptor) =>
          "route" in descriptor || !triggers(descriptor, trigger) ? Effect.void : Effect.gen(function* () {
            const result = yield* apply(descriptor, trigger, yield* Ref.get(state), undefined, enqueue)
            if (result._tag === "Set") yield* Ref.set(state, result.state)
          }), { discard: true }),
        get: Ref.get(state),
        getAt: () => Effect.succeed(Option.none()),
        entries: Effect.succeed(HashMap.empty()),
        addresses: Effect.succeed([]),
        capture: Effect.map(Ref.get(state), (captured) => Ref.set(state, captured)),
      })
      continue
    }

    const config = internal._config
    if (!("by" in config) || typeof config.initial !== "function") continue
    const initial = config.initial as unknown as (context: { readonly address: AddressRef }) => unknown
    const partitionSpace = definition.topology.by
    const state = yield* Ref.make(HashMap.empty<AddressRef, unknown>())
    projectionRuntimes.set(definition, {
      definition,
      handle: (trigger, enqueue) => Effect.forEach(handlers, (descriptor) =>
        !("route" in descriptor) || !triggers(descriptor, trigger) ? Effect.void : Effect.gen(function* () {
          const route = descriptor.route as (input: unknown) => AddressTarget<AddressRef>
          const existing = Array.from(HashMap.keys(yield* Ref.get(state)))
          for (const address of targetAddresses(route(triggerInput(trigger).route), existing, partitionSpace)) {
            const current = Option.getOrElse(HashMap.get(yield* Ref.get(state), address), () => initial({ address }))
            const result = yield* apply(descriptor, trigger, current, address, enqueue)
            if (result._tag === "Set") yield* Ref.update(state, HashMap.set(address, result.state))
            if (result._tag === "Remove") yield* Ref.update(state, HashMap.remove(address))
          }
        }), { discard: true }),
      get: Effect.void,
      getAt: (address) => Ref.get(state).pipe(Effect.map(HashMap.get(address))),
      entries: Ref.get(state),
      addresses: Ref.get(state).pipe(Effect.map((entries) => Array.from(HashMap.keys(entries)))),
      capture: Effect.map(Ref.get(state), (captured) => Ref.set(state, captured)),
    })
  }

  const publicReader = makeReader(undefined, undefined)

  let activationEmit: Emit<Event, HandlerError> | undefined

  // The close, which the first request starts: it closes every lane, waits for them, ends
  // observation and completes `closed`. It runs in its own fiber in `laneScope`, outside every
  // caller and every hold of the lock, so interrupting a caller does not abandon it, and a handler
  // that requests it is interrupted by it rather than waited on.
  const close = (closure: Closure<StoreError>): Effect.Effect<void> => Effect.gen(function* () {
    yield* locked(Effect.gen(function* () {
      ending = closure
      const interruption: WorkerInterruption = closure._tag === "Removed" ? { _tag: "Removed" } : { _tag: "Closed" }
      for (const worker of workerRuntimes) yield* worker.close(interruption)
    }))
    yield* Effect.forEach(workerRuntimes, (worker) => worker.done, { discard: true })
    // A handler failure supersedes a close.
    const final = yield* locked(Effect.sync(() => ended = failed ?? closure))
    yield* PubSub.shutdown(occurrences)
    yield* PubSub.shutdown(signalPubSub)
    yield* Deferred.succeed(closed, final)
  })
  const startClose = (closure: Closure<StoreError>): Effect.Effect<void> => Effect.uninterruptible(Effect.suspend(() => {
    if (closing) return Effect.void
    closing = true
    return Effect.asVoid(Effect.forkIn(Effect.locally(close(closure), hold, undefined), laneScope))
  }))
  const fail = (cause: Cause.Cause<WorkerHandlerError | HandlerError>) => Effect.suspend(() => {
    failed ??= { _tag: "Failed", cause }
    return startClose(failed)
  })

  // Settles an ambient's new value: the caller holds the acceptance lock. Nothing is stored for an ambient.
  const settleAmbient = <Value>(ambient: AmbientDefinition<Value>, value: Value) => settleAndRelease([{
    _tag: "Ambient",
    ambient: ambient as AmbientDefinition<unknown>,
    value,
  }], Effect.void, [])

  // Handlers' update, which like their emit is never rejected (see `unlessEnded`).
  const update: UpdateAmbient = (ambient, value) => locked(Effect.zipRight(unlessEnded("updated an ambient"), settleAmbient(ambient, value)))

  const workerRuntimes: WorkerRuntime<Event, HandlerError>[] = []
  for (const definition of workers) {
    const internal = definition as WorkerDefinitionInternal<Event, HandlerError>
    const allowed = new Set(definition.dependencies)
    let unpartitioned: WorkerLane<Event, HandlerError> | undefined
    let lanes = HashMap.empty<AddressRef, WorkerLane<Event, HandlerError>>()
    // Lanes closed by admission or by the engine, until their running handler has finished.
    const finishing = new Set<WorkerLane<Event, HandlerError>>()

    const execute = (emit: Emit<Event, HandlerError>) => (
      trigger: WorkerTrigger<Event>,
      handlers: LaneHandlers<Event, HandlerError>,
      address: AddressRef | undefined,
      interruption: Effect.Effect<Option.Option<WorkerInterruption>>,
    ): Effect.Effect<void, WorkerHandlerError | HandlerError> => Effect.gen(function* () {
      const read = makeReader(address, allowed)
      for (const descriptor of handlers) {
        if (trigger._tag === "Event" && descriptor._tag === "Event") {
          if ("route" in descriptor) {
            if (address === undefined) continue
            const handle = descriptor.handle as unknown as (context: {
              readonly event: Event
              readonly address: AddressRef
              readonly emit: Emit<Event, HandlerError>
              readonly read: ProjectionReader
              readonly update: UpdateAmbient
              readonly interruption: Effect.Effect<Option.Option<WorkerInterruption>>
            }) => Effect.Effect<void, WorkerHandlerError | HandlerError>
            yield* handle({ event: trigger.event, address, emit, read, update, interruption })
          } else {
            yield* descriptor.handle({ event: trigger.event, emit, read, update, interruption })
          }
        }
        if (trigger._tag === "Signal" && descriptor._tag === "Signal") {
          if ("route" in descriptor) {
            if (address === undefined) continue
            const handle = descriptor.handle as unknown as (context: {
              readonly signal: SignalOccurrence<unknown>
              readonly address: AddressRef
              readonly emit: Emit<Event, HandlerError>
              readonly read: ProjectionReader
              readonly update: UpdateAmbient
              readonly interruption: Effect.Effect<Option.Option<WorkerInterruption>>
            }) => Effect.Effect<void, WorkerHandlerError | HandlerError>
            yield* handle({ signal: trigger.signal, address, emit, read, update, interruption })
          } else {
            yield* descriptor.handle({ signal: trigger.signal, emit, read, update, interruption })
          }
        }
      }
    // Every handler's requirements are the engine's `R`, which `context` provides.
    }).pipe(Effect.provide(context)) as Effect.Effect<void, WorkerHandlerError | HandlerError>

    const retire = (lane: WorkerLane<Event, HandlerError>, interruption: WorkerInterruption) => Effect.gen(function* () {
      finishing.add(lane)
      yield* lane.close(interruption)
      yield* lane.done.pipe(
        Effect.zipRight(Effect.sync(() => finishing.delete(lane))),
        Effect.interruptible,
        Effect.forkIn(laneScope),
      )
    })

    // A lane's fibers do not hold the lock of the acceptance that opens it.
    const newLane = (address: AddressRef | undefined, emit: Emit<Event, HandlerError>) =>
      Effect.locally(makeLane(laneScope, address, execute(emit), fail, offered), hold, undefined)

    const openLane = (address: AddressRef, emit: Emit<Event, HandlerError>): Effect.Effect<void> => Effect.gen(function* () {
      if (Option.isSome(HashMap.get(lanes, address))) return
      const lane = yield* newLane(address, emit)
      lanes = HashMap.set(lanes, address, lane)
    })

    const closeLane = (address: AddressRef): Effect.Effect<void> => Effect.gen(function* () {
      const lane = HashMap.get(lanes, address)
      if (Option.isNone(lane)) return
      lanes = HashMap.remove(lanes, address)
      yield* retire(lane.value, { _tag: "Interrupted" })
    })

    // A handler or admission's route, resolved and checked against the worker's space now, and
    // expanded against the open lanes when the occurrence is released.
    const resolve = (target: AddressTarget<AddressRef>) => {
      if (definition.topology._tag === "Partitioned") targetAddresses(target, [], definition.topology.by)
      return () => targetAddresses(target, Array.from(HashMap.keys(lanes)))
    }

    const prepare = (trigger: WorkerTrigger<Event>): Effect.Effect<Effect.Effect<void>> => Effect.sync(() => {
      if (activationEmit === undefined || ending !== undefined) return Effect.void
      const emit = activationEmit
      const triggered = (handler: LaneHandlers<Event, HandlerError>[number] | WorkerDefinitionInternal<Event, HandlerError>["_admission"][number]) =>
        trigger._tag === "Event"
          ? handler._tag === "Event" && handler.matcher.matches(trigger.event)
          : handler._tag === "Signal" && handler.signal === trigger.signal.signal
      const routeOf = (handler: { readonly route: (input: never) => unknown }) =>
        resolve((trigger._tag === "Event"
          ? (handler.route as (event: Event) => unknown)(trigger.event)
          : (handler.route as (signal: SignalOccurrence<unknown>) => unknown)(trigger.signal)) as AddressTarget<AddressRef>)

      if (definition.topology._tag === "Unpartitioned") {
        const matching = internal._handlers.filter(triggered)
        return matching.length > 0 && unpartitioned !== undefined
          ? unpartitioned.offer(trigger, matching)
          : Effect.void
      }

      const admissions = internal._admission.filter(triggered).map((admission) => ({ admission, addresses: routeOf(admission) }))
      const tasks = internal._handlers.filter(triggered).flatMap((handler) =>
        "route" in handler ? [{ handler, addresses: routeOf(handler) }] : [])
      return Effect.gen(function* () {
        for (const { admission, addresses } of admissions) {
          for (const address of addresses()) {
            if (admission.update._tag === "Admit") yield* openLane(address, emit)
            else yield* closeLane(address)
          }
        }
        for (const { handler, addresses } of tasks) {
          for (const address of addresses()) {
            const lane = HashMap.get(lanes, address)
            if (Option.isSome(lane)) yield* lane.value.offer(trigger, [handler])
          }
        }
      })
    })

    const runtime: WorkerRuntime<Event, HandlerError> = {
      definition,
      activate: (emit) => Effect.gen(function* () {
        if (definition.topology._tag === "Unpartitioned") {
          unpartitioned = yield* newLane(undefined, emit)
          return
        }
        for (const address of internal._initial) yield* openLane(address, emit)
      }),
      prepare,
      interrupt: (target) => Effect.gen(function* () {
        if (isWorker(target)) {
          if (target !== definition) return
          for (const lane of [...(unpartitioned === undefined ? [] : [unpartitioned]), ...HashMap.values(lanes)]) {
            yield* lane.interrupt({ _tag: "Interrupted" })
          }
          return
        }
        const match = "_tag" in target ? target : AddressMatch.exact(target)
        for (const [address, lane] of HashMap.entries(lanes)) {
          if (AddressMatch.includes(match, address)) yield* lane.interrupt({ _tag: "Interrupted" })
        }
      }),
      close: (interruption) => Effect.gen(function* () {
        const open = [...(unpartitioned === undefined ? [] : [unpartitioned]), ...HashMap.values(lanes)]
        unpartitioned = undefined
        lanes = HashMap.empty()
        for (const lane of open) yield* retire(lane, interruption)
        for (const lane of finishing) yield* lane.interrupt(interruption)
      }),
      barriers: Effect.suspend(() => Effect.map(
        Effect.forEach([...(unpartitioned === undefined ? [] : [unpartitioned]), ...HashMap.values(lanes)], (lane) => lane.barrier),
        (queued) => [...queued, ...Array.from(finishing, (lane) => lane.done)],
      )),
      done: Effect.suspend(() => Effect.forEach(Array.from(finishing), (lane) => lane.done, { discard: true })),
    }
    workerRuntimes.push(runtime)
  }

  // Settles an event or an ambient's new value and the signal cascade it starts, through every
  // projection in dependency order. The caller holds the acceptance lock.
  const settle = (settled: Settled<Event>): Effect.Effect<SignalOccurrence<unknown>[], EventSettlementError> =>
    Effect.gen(function* () {
      const pendingSignals: SignalOccurrence<unknown>[] = []
      const timestamp = DateTime.unsafeNow()
      const enqueueFrom = (projection: ProjectionDefinitionBase): SignalEnqueue => (signal, payload, address) => {
        pendingSignals.push({
          signal,
          payload,
          source: { projection, address: Option.fromNullable(address) },
          timestamp,
        })
      }
      if (settled._tag === "Ambient") ambientValues.set(settled.ambient, settled.value)
      for (const projection of projections) {
        const runtime = projectionRuntimes.get(projection)
        if (runtime === undefined) continue
        yield* runtime.handle(settled, enqueueFrom(projection))
      }

      const cascadeLimit = 1_000
      for (let cursor = 0; cursor < pendingSignals.length; cursor += 1) {
        if (cursor >= cascadeLimit) {
          return yield* new SignalCascadeExceeded({ limit: cascadeLimit })
        }
        const occurrence = pendingSignals[cursor]
        for (const projection of projections) {
          const runtime = projectionRuntimes.get(projection)
          if (runtime === undefined) continue
          yield* runtime.handle({ _tag: "Signal", signal: occurrence }, enqueueFrom(projection))
        }
      }
      return pendingSignals
    })

  // Captures every projection's state, and returns its restoration.
  const capture = Effect.map(
    Effect.forEach(projectionRuntimes.values(), (runtime) => runtime.capture),
    (restores) => Effect.all(restores, { discard: true }),
  )

  // Settles each occurrence in order, resolves every worker route, commits once, applies
  // `interrupts`, then releases. The interrupts reach the work that existed before this step, not
  // the work it releases, so one step can stop running work and start new work. A failure before the
  // commit completes restores every projection's state and ambient value, so a failed settlement or
  // commit leaves no trace. The whole step, the commit included, is uninterruptible, so a committed
  // event is always followed by its interrupts and released. Its release reaches workers only if they
  // were active and closing had not begun when it settled: an event settled before activation, such
  // as one `recover` emits, or once closing has begun, reaches none. The lanes it opens and the
  // handlers it releases to run interruptibly.
  function settleAndRelease<E>(
    items: readonly Settled<Event>[],
    commit: Effect.Effect<void, E>,
    interrupts: readonly InterruptTarget[],
  ): Effect.Effect<void, EventSettlementError | E> {
    return Effect.uninterruptible(Effect.gen(function* () {
      const restore = yield* capture
      const ambientsBefore = new Map(ambientValues)
      const released = yield* Effect.gen(function* () {
        const steps: { readonly signals: readonly SignalOccurrence<unknown>[]; readonly releases: readonly Effect.Effect<void>[] }[] = []
        for (const occurrence of items) {
          const signals = yield* settle(occurrence)
          const triggers: WorkerTrigger<Event>[] = [
            ...(occurrence._tag === "Event" ? [{ _tag: "Event" as const, event: occurrence.event }] : []),
            ...signals.map((signal) => ({ _tag: "Signal" as const, signal })),
          ]
          const releases = yield* Effect.forEach(triggers, (trigger) =>
            Effect.forEach(workerRuntimes, (worker) => worker.prepare(trigger)))
          steps.push({ signals, releases: releases.flat() })
        }
        yield* commit
        return steps
      }).pipe(Effect.onError(() => Effect.zipRight(restore, Effect.sync(() => {
        ambientValues.clear()
        for (const [ambient, value] of ambientsBefore) ambientValues.set(ambient, value)
      }))))
      for (const target of interrupts) yield* interrupt(target)
      for (const { signals, releases } of released) {
        yield* PubSub.publish(occurrences, undefined)
        yield* Effect.all(releases, { discard: true })
        for (const occurrence of signals) yield* PubSub.publish(signalPubSub, occurrence)
      }
    }))
  }

  // Folds a stored event: nothing is committed, and nothing is released.
  const replay = (event: Event): Effect.Effect<void, EventSettlementError> =>
    locked(Effect.asVoid(settle({ _tag: "Event", event })))

  // Commits events to the store. Without a store, or without
  // events, which change no state, nothing is committed.
  const commitEvents = (events: readonly Event[]): Effect.Effect<void, StoreError> =>
    store === undefined || events.length === 0 ? Effect.void : store.commit(events)

  // Accepts events as one, in one commit, then applies interrupts: the caller holds the acceptance
  // lock.
  const acceptAll = (
    events: readonly Event[],
    interrupts: readonly InterruptTarget[],
  ): Effect.Effect<void, EventSettlementError | StoreError> =>
    settleAndRelease(events.map((event) => ({ _tag: "Event" as const, event })), commitEvents(events), interrupts)

  // The engine's own emit (see `HandlerError` and `unlessEnded`).
  const emit: Emit<Event, HandlerError> = (event) => locked(Effect.zipRight(unlessEnded("emitted"), acceptAll([event], [])))

  // The caller holds the acceptance lock.
  const interrupt = (target: InterruptTarget) =>
    Effect.forEach(workerRuntimes, (worker) => worker.interrupt(target), { discard: true })

  // The caller holds the acceptance lock.
  const activateWith = (emit: Emit<Event, HandlerError>) => Effect.gen(function* () {
    activationEmit = emit
    for (const worker of workerRuntimes) yield* worker.activate(emit)
  })

  const runtime: EngineRuntime<Event, StoreError> = {
    accept: (event) => locked(Effect.zipRight(rejectClosed, acceptAll([event], []))),
    transact: <A, E, R>(
      body: (context: { readonly read: ProjectionReader }) => Effect.Effect<Transaction<Event, A>, E, R>,
    ) => locked(Effect.gen(function* () {
      yield* rejectClosed
      const transaction = yield* body({ read: publicReader })
      yield* acceptAll(transaction.events, transaction.interrupts ?? [])
      return transaction.result
    })),
    activate: (options) => locked(Effect.gen(function* () {
      const chosen = options?.emit ?? emit
      if (activationEmit !== undefined) {
        if (activationEmit === chosen) return
        return yield* new EngineAlreadyActivated()
      }
      yield* activateWith(chosen)
    })),
    interrupt: (target) => locked(interrupt(target)),
    update: (ambient, value) => locked(Effect.zipRight(rejectClosed, settleAmbient(ambient, value))),
    read: publicReader,
    signals: {
      stream: (signal) => Stream.fromPubSub(signalPubSub).pipe(
        Stream.filter((occurrence): occurrence is SignalOccurrence<never> => occurrence.signal === signal),
      ) as never,
      subscribe: (signal) => PubSub.subscribe(signalPubSub).pipe(
        Effect.map((queue) => Stream.fromQueue(queue).pipe(
          Stream.filter((occurrence): occurrence is SignalOccurrence<never> => occurrence.signal === signal),
        )),
      ) as never,
    },
    awaitIdle: Effect.gen(function* () {
      while (true) {
        // Every barrier is queued under the lock, ahead of the end of any lane that closes after.
        const pending = yield* locked(Effect.gen(function* () {
          if (ended !== undefined) return Option.none<readonly [number, readonly Effect.Effect<void>[]]>()
          const result: Effect.Effect<void>[] = []
          for (const worker of workerRuntimes) result.push(...yield* worker.barriers)
          return Option.some([offered.count, result] as const)
        }))
        if (Option.isNone(pending)) return
        const [mark, barriers] = pending.value
        yield* Effect.all(barriers, { concurrency: "unbounded", discard: true })
        if ((yield* locked(Effect.sync(() => offered.count))) === mark) return
      }
    }),
    close: (reason) => unheld(Effect.zipRight(
      startClose(reason === "closed" ? { _tag: "Closed" } : { _tag: "Removed" }),
      Deferred.await(closed),
    )),
    closed: Deferred.await(closed),
    [EngineRuntimeObservationTypeId]: {
      subscribe: PubSub.subscribe(occurrences),
      settled,
    },
  }
  yield* Effect.addFinalizer(() => Effect.zipRight(startClose({ _tag: "Closed" }), Deferred.await(closed)))

  return {
    runtime,
    replay,
    emit,
    activate: locked(activateWith(emit)),
  }
})

export interface EngineFamily<Event, Key extends DiscriminatorKey<Event> | undefined, StoreError = never> {
  readonly Projection: ReturnType<typeof makeProjectionBuilder<Event, Key>>
  readonly Worker: ReturnType<typeof makeWorkerBuilder<Event, Key, StoreError>>
  readonly Facade: ReturnType<typeof makeFacadeBuilder<Event, StoreError>>
  readonly Signal: ReturnType<typeof makeSignalBuilder>
  readonly make: <const Workers extends readonly WorkerFor<Event>[]>(config: {
    readonly name: string
    readonly projections: readonly ProjectionFor<Event>[]
    /** Their handlers' requirements are the engine's, provided where it is built or opened. */
    readonly workers: Workers
  }) => EngineDefinition<Event, StoreError, WorkerRequirements<Workers[number]>>
}

function define<Event, StoreError = never>(): EngineFamily<Event, undefined, StoreError>
function define<
  Event,
  Key extends DiscriminatorKey<Event> = DiscriminatorKey<Event>,
  StoreError = never,
>(options: { readonly discriminator: Key }): EngineFamily<Event, Key, StoreError>
function define<
  Event,
  Key extends DiscriminatorKey<Event> | undefined,
  StoreError = never,
>(options?: { readonly discriminator: Key }): EngineFamily<Event, Key, StoreError> {
  const selection = makeEventSelection<Event, Key>(options?.discriminator as Key)
  const family = {}
  const Projection = makeProjectionBuilder(selection)
  const Worker = makeWorkerBuilder<Event, Key, StoreError>(selection)
  const Facade = makeFacadeBuilder<Event, StoreError>()
  const Signal = makeSignalBuilder(family)

  return {
    Projection,
    Worker,
    Facade,
    Signal,
    make: <const Workers extends readonly WorkerFor<Event>[]>(config: {
      readonly name: string
      readonly projections: readonly ProjectionFor<Event>[]
      readonly workers: Workers
    }): EngineDefinition<Event, StoreError, WorkerRequirements<Workers[number]>> => {
      type R = WorkerRequirements<Workers[number]>
      const declared = config.projections
      for (const component of declared) {
        if ((component as unknown as ProjectionDefinitionInternal<Event>)._eventSelection !== selection) {
          throw new ForeignEngineComponent({ component: component.name })
        }
      }
      for (const component of config.workers) {
        if ((component as unknown as WorkerDefinitionInternal<Event>)._selection !== selection) {
          throw new ForeignEngineComponent({ component: component.name })
        }
      }
      validateNames(declared, config.workers)
      const publishers = validateSignals(family, declared, config.workers)
      const projections = sortProjections(declared, publishers)
      validateWorkerDependencies(projections, config.workers)
      validateSpaces(projections, config.workers)
      const build = Effect.map(
        makeRuntime<Event, StoreError, R>(projections, config.workers, undefined),
        ({ runtime }) => runtime,
      )
      const Tag = Context.GenericTag<EngineRuntime<Event, StoreError>>(`@magnitudedev/event-engine/${config.name}`)
      return {
        name: config.name,
        projections,
        workers: config.workers,
        Tag,
        Layer: Layer.scoped(Tag, build),
        build,
        open: ({ store, recover }) => Effect.gen(function* () {
          const engine = yield* makeRuntime<Event, StoreError, R>(projections, config.workers, store)
          yield* Stream.runForEach(store.read, engine.replay)
          if (recover !== undefined) yield* recover({ read: engine.runtime.read, emit: engine.emit })
          yield* engine.activate
          return engine.runtime
        }),
      }
    },
  }
}

export const EventEngine = { define }
