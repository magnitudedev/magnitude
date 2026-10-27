import type { Effect, Option } from "effect"
import type { Address, AddressRef, AddressSpaceBase, AddressTarget } from "./address"
import type { AmbientDefinition } from "./ambient"
import type {
  DiscriminatorKey,
  DiscriminatorValue,
  EventRefinement,
  EventSelection,
  EventVariant,
} from "./event"
import { EventAny } from "./event"
import type { ProjectionFor, ProjectionReader, ProjectionReadError } from "./projection"
import type { SignalCascadeExceeded, SignalDefinition, SignalOccurrence } from "./signal"

const WorkerTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/Worker")
const WorkerEventTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/Worker/Event")
const WorkerRequirementsTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/Worker/Requirements")

export type WorkerHandlerError = ProjectionReadError | SignalCascadeExceeded

export type Emit<Event, E = never> = (event: Event) => Effect.Effect<void, WorkerHandlerError | E>

/** Why a running handler's fiber was interrupted: its lane was interrupted, or its engine closed or was removed. */
export type WorkerInterruption =
  | { readonly _tag: "Interrupted" }
  | { readonly _tag: "Closed" }
  | { readonly _tag: "Removed" }

/** Sets an ambient's value, which settles like an event: ambient handlers fold it and their signals are released. */
export type UpdateAmbient = <Value>(ambient: AmbientDefinition<Value>, value: Value) => Effect.Effect<void, WorkerHandlerError>

export type WorkerPartitionUpdate =
  | { readonly _tag: "Admit" }
  | { readonly _tag: "Close" }

export const Worker = {
  admit: { _tag: "Admit" } as const,
  close: { _tag: "Close" } as const,
}

// `Trigger` is the handled event; `emit` takes any event of the engine.
export interface WorkerHandlerContext<Event, E = never, Trigger = Event> {
  readonly event: Trigger
  readonly emit: Emit<Event, E>
  readonly read: ProjectionReader
  readonly update: UpdateAmbient
  /** The interruption of this handler's fiber, once it is interrupted. */
  readonly interruption: Effect.Effect<Option.Option<WorkerInterruption>>
}

export interface PartitionedWorkerHandlerContext<
  Event,
  Space extends AddressSpaceBase,
  E = never,
  Trigger = Event,
> extends WorkerHandlerContext<Event, E, Trigger> {
  readonly address: Address<Space>
  readonly read: ProjectionReader<Space>
}

export interface WorkerHandler<Event, E = never, R = never> {
  readonly _tag: "Event"
  readonly matcher: { readonly name: string; readonly matches: (event: Event) => boolean }
  readonly handle: (context: WorkerHandlerContext<Event, E>) => Effect.Effect<void, WorkerHandlerError | E, R>
}

export interface PartitionedWorkerHandler<Event, Space extends AddressSpaceBase, E = never, R = never> {
  readonly _tag: "Event"
  readonly matcher: { readonly name: string; readonly matches: (event: Event) => boolean }
  readonly route: (event: Event) => AddressTarget<Address<Space>>
  readonly handle: (context: PartitionedWorkerHandlerContext<Event, Space, E>) => Effect.Effect<void, WorkerHandlerError | E, R>
}

export interface WorkerSignalHandlerContext<Event, Payload, E = never> {
  readonly signal: SignalOccurrence<Payload>
  readonly emit: Emit<Event, E>
  readonly read: ProjectionReader
  readonly update: UpdateAmbient
  /** The interruption of this handler's fiber, once it is interrupted. */
  readonly interruption: Effect.Effect<Option.Option<WorkerInterruption>>
}

export interface PartitionedWorkerSignalHandlerContext<Event, Payload, Space extends AddressSpaceBase, E = never>
  extends WorkerSignalHandlerContext<Event, Payload, E> {
  readonly address: Address<Space>
  readonly read: ProjectionReader<Space>
}

export interface WorkerSignalHandler<Event, E = never, R = never> {
  readonly _tag: "Signal"
  readonly signal: SignalDefinition<unknown>
  readonly handle: (context: WorkerSignalHandlerContext<Event, unknown, E>) => Effect.Effect<void, WorkerHandlerError | E, R>
}

export interface PartitionedWorkerSignalHandler<Event, Space extends AddressSpaceBase, E = never, R = never> {
  readonly _tag: "Signal"
  readonly signal: SignalDefinition<unknown>
  readonly route: (signal: SignalOccurrence<unknown>) => AddressTarget<Address<Space>>
  readonly handle: (context: PartitionedWorkerSignalHandlerContext<Event, unknown, Space, E>) => Effect.Effect<void, WorkerHandlerError | E, R>
}

export interface WorkerSignalHandlerBuilder<Event, E = never> {
  <Payload, R = never>(
    signal: SignalDefinition<Payload>,
    handler: (context: WorkerSignalHandlerContext<Event, Payload, E>) => Effect.Effect<void, WorkerHandlerError | E, R>,
  ): WorkerSignalHandler<Event, E, R>
}

export interface PartitionedWorkerSignalHandlerBuilder<Event, Space extends AddressSpaceBase, E = never> {
  <Payload, R = never>(
    signal: SignalDefinition<Payload>,
    options: { readonly route: (signal: SignalOccurrence<Payload>) => AddressTarget<Address<Space>> },
    handler: (context: PartitionedWorkerSignalHandlerContext<Event, Payload, Space, E>) => Effect.Effect<void, WorkerHandlerError | E, R>,
  ): PartitionedWorkerSignalHandler<Event, Space, E, R>
}

export interface WorkerEventHandlerBuilder<
  Event,
  Key extends DiscriminatorKey<Event> | undefined,
  E = never,
> {
  <Value extends DiscriminatorValue<Event, Key>, R = never>(
    selector: Value,
    handler: (context: WorkerHandlerContext<Event, E, EventVariant<Event, Key, Value>>) => Effect.Effect<void, WorkerHandlerError | E, R>,
  ): WorkerHandler<Event, E, R>
  <Selected extends Event, R = never>(
    selector: EventRefinement<Event, Selected>,
    handler: (context: WorkerHandlerContext<Event, E, Selected>) => Effect.Effect<void, WorkerHandlerError | E, R>,
  ): WorkerHandler<Event, E, R>
  <R = never>(
    selector: typeof EventAny,
    handler: (context: WorkerHandlerContext<Event, E>) => Effect.Effect<void, WorkerHandlerError | E, R>,
  ): WorkerHandler<Event, E, R>
}

export interface PartitionedWorkerEventHandlerBuilder<
  Event,
  Key extends DiscriminatorKey<Event> | undefined,
  Space extends AddressSpaceBase,
  E = never,
> {
  <Value extends DiscriminatorValue<Event, Key>, R = never>(
    selector: Value,
    options: { readonly route: (event: EventVariant<Event, Key, Value>) => AddressTarget<Address<Space>> },
    handler: (context: PartitionedWorkerHandlerContext<Event, Space, E, EventVariant<Event, Key, Value>>) => Effect.Effect<void, WorkerHandlerError | E, R>,
  ): PartitionedWorkerHandler<Event, Space, E, R>
  <Selected extends Event, R = never>(
    selector: EventRefinement<Event, Selected>,
    options: { readonly route: (event: Selected) => AddressTarget<Address<Space>> },
    handler: (context: PartitionedWorkerHandlerContext<Event, Space, E, Selected>) => Effect.Effect<void, WorkerHandlerError | E, R>,
  ): PartitionedWorkerHandler<Event, Space, E, R>
  <R = never>(
    selector: typeof EventAny,
    options: { readonly route: (event: Event) => AddressTarget<Address<Space>> },
    handler: (context: PartitionedWorkerHandlerContext<Event, Space, E>) => Effect.Effect<void, WorkerHandlerError | E, R>,
  ): PartitionedWorkerHandler<Event, Space, E, R>
}

export interface WorkerAdmissionHandler<Event, Space extends AddressSpaceBase> {
  readonly _tag: "Event"
  readonly matcher: { readonly name: string; readonly matches: (event: Event) => boolean }
  readonly route: (event: Event) => AddressTarget<Address<Space>>
  readonly update: WorkerPartitionUpdate
}

export interface WorkerSignalAdmissionHandler<Space extends AddressSpaceBase> {
  readonly _tag: "Signal"
  readonly signal: SignalDefinition<unknown>
  readonly route: (signal: SignalOccurrence<unknown>) => AddressTarget<Address<Space>>
  readonly update: WorkerPartitionUpdate
}

export interface WorkerSignalAdmissionHandlerBuilder<Space extends AddressSpaceBase> {
  <Payload>(
    signal: SignalDefinition<Payload>,
    options: { readonly route: (signal: SignalOccurrence<Payload>) => AddressTarget<Address<Space>> },
    update: WorkerPartitionUpdate,
  ): WorkerSignalAdmissionHandler<Space>
}

export interface WorkerAdmissionHandlerBuilder<
  Event,
  Key extends DiscriminatorKey<Event> | undefined,
  Space extends AddressSpaceBase,
> {
  <Value extends DiscriminatorValue<Event, Key>>(
    selector: Value,
    options: { readonly route: (event: EventVariant<Event, Key, Value>) => AddressTarget<Address<Space>> },
    update: WorkerPartitionUpdate,
  ): WorkerAdmissionHandler<Event, Space>
  <Selected extends Event>(
    selector: EventRefinement<Event, Selected>,
    options: { readonly route: (event: Selected) => AddressTarget<Address<Space>> },
    update: WorkerPartitionUpdate,
  ): WorkerAdmissionHandler<Event, Space>
  (
    selector: typeof EventAny,
    options: { readonly route: (event: Event) => AddressTarget<Address<Space>> },
    update: WorkerPartitionUpdate,
  ): WorkerAdmissionHandler<Event, Space>
}

export interface WorkerDefinitionBase {
  readonly [WorkerTypeId]: typeof WorkerTypeId
  readonly name: string
  readonly topology:
    | { readonly _tag: "Unpartitioned" }
    | { readonly _tag: "Partitioned"; readonly by: AddressSpaceBase }
  readonly dependencies: readonly import("./projection").ProjectionDefinitionBase[]
  readonly signalSubscriptions: readonly SignalDefinition<unknown>[]
  readonly [WorkerEventTypeId]: (...args: never[]) => unknown
  readonly [WorkerRequirementsTypeId]: () => unknown
}

export type WorkerFor<Event> = WorkerDefinitionBase & {
  readonly [WorkerEventTypeId]: (event: Event) => Event
}

/** The services a worker's handlers require, which the engine is given where it is built. */
export type WorkerRequirements<Worker extends WorkerDefinitionBase> = ReturnType<Worker[typeof WorkerRequirementsTypeId]>

type HandlerRequirements<Handler> = Handler extends { readonly handle: (context: never) => Effect.Effect<void, any, infer R> } ? R : never

export interface UnpartitionedWorkerDefinition<Event = unknown, R = never> extends WorkerDefinitionBase {
  readonly topology: { readonly _tag: "Unpartitioned" }
  readonly [WorkerEventTypeId]: (event: Event) => Event
  readonly [WorkerRequirementsTypeId]: () => R
}

export interface PartitionedWorkerDefinition<Space extends AddressSpaceBase, Event = unknown, R = never> extends WorkerDefinitionBase {
  readonly topology: { readonly _tag: "Partitioned"; readonly by: Space }
  readonly [WorkerEventTypeId]: (event: Event) => Event
  readonly [WorkerRequirementsTypeId]: () => R
}

export interface UnpartitionedWorkerConfig<
  Event,
  Key extends DiscriminatorKey<Event> | undefined,
  Name extends string,
  E = never,
  Handlers extends readonly (WorkerHandler<Event, E, any> | WorkerSignalHandler<Event, E, any>)[] = readonly (WorkerHandler<Event, E, any> | WorkerSignalHandler<Event, E, any>)[],
> {
  readonly name: Name
  readonly reads?: readonly ProjectionFor<Event>[]
  readonly handlers: (
    on: {
      readonly event: WorkerEventHandlerBuilder<Event, Key, E>
      readonly signal: WorkerSignalHandlerBuilder<Event, E>
    },
  ) => Handlers
}

export interface PartitionedWorkerConfig<
  Event,
  Key extends DiscriminatorKey<Event> | undefined,
  Name extends string,
  Space extends AddressSpaceBase,
  E = never,
  Handlers extends readonly (PartitionedWorkerHandler<Event, Space, E, any> | PartitionedWorkerSignalHandler<Event, Space, E, any>)[] = readonly (PartitionedWorkerHandler<Event, Space, E, any> | PartitionedWorkerSignalHandler<Event, Space, E, any>)[],
> {
  readonly name: Name
  readonly reads?: readonly ProjectionFor<Event>[]
  readonly partition: {
    readonly by: Space
    readonly initial?: readonly Address<Space>[]
    readonly admission: (
      on: {
        readonly event: WorkerAdmissionHandlerBuilder<Event, Key, Space>
        readonly signal: WorkerSignalAdmissionHandlerBuilder<Space>
      },
    ) => readonly (WorkerAdmissionHandler<Event, Space> | WorkerSignalAdmissionHandler<Space>)[]
  }
  readonly handlers: (
    on: {
      readonly event: PartitionedWorkerEventHandlerBuilder<Event, Key, Space, E>
      readonly signal: PartitionedWorkerSignalHandlerBuilder<Event, Space, E>
    },
  ) => Handlers
}

export interface WorkerDefinitionInternal<Event, E = never> extends WorkerDefinitionBase {
  readonly _selection: EventSelection<Event, DiscriminatorKey<Event> | undefined>
  readonly _handlers: readonly (
    | WorkerHandler<Event, E, any>
    | PartitionedWorkerHandler<Event, AddressSpaceBase, E, any>
    | WorkerSignalHandler<Event, E, any>
    | PartitionedWorkerSignalHandler<Event, AddressSpaceBase, E, any>
  )[]
  readonly _admission: readonly (
    | WorkerAdmissionHandler<Event, AddressSpaceBase>
    | WorkerSignalAdmissionHandler<AddressSpaceBase>
  )[]
  readonly _initial: readonly AddressRef[]
}

export const WorkerInternal = { typeId: WorkerTypeId }

export const makeWorkerBuilder = <
  Event,
  Key extends DiscriminatorKey<Event> | undefined,
  E = never,
>(selection: EventSelection<Event, Key>) => {
  function define<
    Name extends string,
    Space extends AddressSpaceBase,
    const Handlers extends readonly (PartitionedWorkerHandler<Event, Space, E, any> | PartitionedWorkerSignalHandler<Event, Space, E, any>)[],
  >(
    config: PartitionedWorkerConfig<Event, Key, Name, Space, E, Handlers>,
  ): PartitionedWorkerDefinition<Space, Event, HandlerRequirements<Handlers[number]>>
  function define<
    Name extends string,
    const Handlers extends readonly (WorkerHandler<Event, E, any> | WorkerSignalHandler<Event, E, any>)[],
  >(
    config: UnpartitionedWorkerConfig<Event, Key, Name, E, Handlers>,
  ): UnpartitionedWorkerDefinition<Event, HandlerRequirements<Handlers[number]>>
  function define<Name extends string, Space extends AddressSpaceBase>(
    config: UnpartitionedWorkerConfig<Event, Key, Name, E> | PartitionedWorkerConfig<Event, Key, Name, Space, E>,
  ): UnpartitionedWorkerDefinition<Event, unknown> | PartitionedWorkerDefinition<Space, Event, unknown> {
    if (!("partition" in config)) {
      const event = ((selector: PropertyKey | typeof EventAny | EventRefinement<Event, Event>, handler: WorkerHandler<Event, E, any>["handle"]) => ({
        _tag: "Event",
        matcher: selection.matcher(selector),
        handle: handler,
      })) as unknown as WorkerEventHandlerBuilder<Event, Key, E>
      const signal = ((definition: SignalDefinition<unknown>, handler: WorkerSignalHandler<Event, E, any>["handle"]) => ({
        _tag: "Signal",
        signal: definition,
        handle: handler,
      })) as unknown as WorkerSignalHandlerBuilder<Event, E>
      const handlers = config.handlers({ event, signal })
      return {
        [WorkerTypeId]: WorkerTypeId,
        [WorkerEventTypeId]: undefined as unknown as (event: Event) => Event,
        [WorkerRequirementsTypeId]: undefined as unknown as () => unknown,
        name: config.name,
        topology: { _tag: "Unpartitioned" },
        dependencies: config.reads ?? [],
        signalSubscriptions: handlers.flatMap((handler) => handler._tag === "Signal" ? [handler.signal] : []),
        _selection: selection,
        _handlers: handlers,
        _admission: [],
        _initial: [],
      } as UnpartitionedWorkerDefinition<Event, unknown>
    }

    const event = ((
      selector: PropertyKey | typeof EventAny | EventRefinement<Event, Event>,
      options: { readonly route: (event: Event) => AddressTarget<Address<Space>> },
      handler: PartitionedWorkerHandler<Event, Space, E, any>["handle"],
    ) => ({ _tag: "Event", matcher: selection.matcher(selector), route: options.route, handle: handler })) as unknown as PartitionedWorkerEventHandlerBuilder<Event, Key, Space, E>
    const signal = ((
      definition: SignalDefinition<unknown>,
      options: { readonly route: (signal: SignalOccurrence<unknown>) => AddressTarget<Address<Space>> },
      handler: PartitionedWorkerSignalHandler<Event, Space, E, any>["handle"],
    ) => ({ _tag: "Signal", signal: definition, route: options.route, handle: handler })) as unknown as PartitionedWorkerSignalHandlerBuilder<Event, Space, E>
    const admissionEvent = ((
      selector: PropertyKey | typeof EventAny | EventRefinement<Event, Event>,
      options: { readonly route: (event: Event) => AddressTarget<Address<Space>> },
      update: WorkerPartitionUpdate,
    ) => ({ _tag: "Event", matcher: selection.matcher(selector), route: options.route, update })) as WorkerAdmissionHandlerBuilder<Event, Key, Space>
    const admissionSignal = ((
      definition: SignalDefinition<unknown>,
      options: { readonly route: (signal: SignalOccurrence<unknown>) => AddressTarget<Address<Space>> },
      update: WorkerPartitionUpdate,
    ) => ({ _tag: "Signal", signal: definition, route: options.route, update })) as WorkerSignalAdmissionHandlerBuilder<Space>

    const handlers = config.handlers({ event, signal })
    const admissions = config.partition.admission({ event: admissionEvent, signal: admissionSignal })
    return {
      [WorkerTypeId]: WorkerTypeId,
      [WorkerEventTypeId]: undefined as unknown as (event: Event) => Event,
      [WorkerRequirementsTypeId]: undefined as unknown as () => unknown,
      name: config.name,
      topology: { _tag: "Partitioned", by: config.partition.by },
      dependencies: config.reads ?? [],
      signalSubscriptions: [
        ...handlers.flatMap((handler) => handler._tag === "Signal" ? [handler.signal] : []),
        ...admissions.flatMap((admission) => admission._tag === "Signal" ? [admission.signal] : []),
      ],
      _selection: selection,
      _handlers: handlers,
      _admission: admissions,
      _initial: config.partition.initial ?? [],
    } as PartitionedWorkerDefinition<Space, Event, unknown>
  }

  return { define }
}
