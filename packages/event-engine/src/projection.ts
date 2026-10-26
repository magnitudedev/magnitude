import { Data, Effect, HashMap, Option, Schema } from "effect"
import type { Address, AddressSpaceBase, AddressTarget } from "./address"
import type { AmbientDefinition, AmbientReader } from "./ambient"
import type {
  DiscriminatorKey,
  DiscriminatorValue,
  EventRefinement,
  EventSelection,
  EventVariant,
} from "./event"
import { EventAny } from "./event"
import type {
  SignalDefinition,
  SignalEmitter,
  SignalOccurrence,
} from "./signal"

const ProjectionTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/Projection")
const ProjectionStateTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/Projection/State")
const ProjectionSpaceTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/Projection/Space")
const ProjectionEventTypeId: unique symbol = Symbol.for("@magnitudedev/event-engine/Projection/Event")

export type ProjectionUpdate<State> =
  | { readonly _tag: "Set"; readonly state: State }
  | { readonly _tag: "Keep" }

export type PartitionUpdate<State> = ProjectionUpdate<State> | { readonly _tag: "Remove" }

export const Projection = {
  set: <State>(state: State): ProjectionUpdate<State> => ({ _tag: "Set", state }),
  keep: { _tag: "Keep" } as const,
  remove: { _tag: "Remove" } as const,
}

export class UndeclaredProjectionRead extends Data.TaggedError("UndeclaredProjectionRead")<{
  readonly projection: string
}> {}

export class ProjectionNotInEngine extends Data.TaggedError("ProjectionNotInEngine")<{
  readonly projection: string
}> {}

export class CurrentAddressUnavailable extends Data.TaggedError("CurrentAddressUnavailable")<{}> {}

const projectionReadErrors = [UndeclaredProjectionRead, ProjectionNotInEngine, CurrentAddressUnavailable] as const

export type ProjectionReadError = InstanceType<(typeof projectionReadErrors)[number]>

/** Whether a value is a projection read's error. */
export const isProjectionReadError = (value: unknown): value is ProjectionReadError =>
  projectionReadErrors.some((error) => value instanceof error)

export interface ProjectionDefinitionBase {
  readonly [ProjectionTypeId]: typeof ProjectionTypeId
  readonly name: string
  readonly topology:
    | { readonly _tag: "Singleton" }
    | { readonly _tag: "Partitioned"; readonly by: AddressSpaceBase }
  readonly dependencies: readonly ProjectionDefinitionBase[]
  readonly emits: readonly SignalDefinition<unknown>[]
  readonly signalSubscriptions: readonly SignalDefinition<unknown>[]
  readonly [ProjectionStateTypeId]: unknown
  readonly [ProjectionSpaceTypeId]: AddressSpaceBase | undefined
  readonly [ProjectionEventTypeId]: (...args: never[]) => unknown
}

export type ProjectionFor<Event> = ProjectionDefinitionBase & {
  readonly [ProjectionEventTypeId]: (event: Event) => Event
}

export interface SingletonProjectionDefinition<State, EncodedState = unknown, Event = unknown> extends ProjectionDefinitionBase {
  readonly topology: { readonly _tag: "Singleton" }
  readonly stateSchema: Schema.Schema<State, EncodedState, never>
  readonly [ProjectionStateTypeId]: State
  readonly [ProjectionSpaceTypeId]: undefined
  readonly [ProjectionEventTypeId]: (event: Event) => Event
}

export interface PartitionedProjectionDefinition<
  Space extends AddressSpaceBase,
  State,
  EncodedState = unknown,
  Event = unknown,
> extends ProjectionDefinitionBase {
  readonly topology: { readonly _tag: "Partitioned"; readonly by: Space }
  readonly stateSchema: Schema.Schema<State, EncodedState, never>
  readonly [ProjectionStateTypeId]: State
  readonly [ProjectionSpaceTypeId]: Space
  readonly [ProjectionEventTypeId]: (event: Event) => Event
}

export type StateOf<Definition extends ProjectionDefinitionBase> =
  Definition[typeof ProjectionStateTypeId]

export type SpaceOf<Definition extends ProjectionDefinitionBase> =
  Definition[typeof ProjectionSpaceTypeId]

export interface ProjectionReader<CurrentSpace extends AddressSpaceBase | undefined = undefined> {
  readonly get: <Definition extends ProjectionDefinitionBase & { readonly topology: { readonly _tag: "Singleton" } }>(
    definition: Definition,
  ) => Effect.Effect<StateOf<Definition>, ProjectionReadError>
  readonly at: <Definition extends ProjectionDefinitionBase & { readonly topology: { readonly _tag: "Partitioned" } }>(
    definition: Definition,
    address: Address<NonNullable<SpaceOf<Definition>>>,
  ) => Effect.Effect<Option.Option<StateOf<Definition>>, ProjectionReadError>
  readonly current: <Definition extends ProjectionDefinitionBase & {
    readonly topology: { readonly _tag: "Partitioned"; readonly by: NonNullable<CurrentSpace> }
  }>(definition: CurrentSpace extends AddressSpaceBase ? Definition : never,
  ) => Effect.Effect<Option.Option<StateOf<Definition>>, ProjectionReadError>
  readonly entries: <Definition extends ProjectionDefinitionBase & { readonly topology: { readonly _tag: "Partitioned" } }>(
    definition: Definition,
  ) => Effect.Effect<HashMap.HashMap<Address<NonNullable<SpaceOf<Definition>>>, StateOf<Definition>>, ProjectionReadError>
}

export interface SingletonProjectionHandlerContext<
  Event,
  State,
  Emits extends readonly SignalDefinition<unknown>[],
> {
  readonly event: Event
  readonly state: State
  readonly read: ProjectionReader
  readonly ambient: AmbientReader
  readonly emit: SignalEmitter<Emits>
}

export interface PartitionedProjectionHandlerContext<
  Event,
  Space extends AddressSpaceBase,
  State,
  Emits extends readonly SignalDefinition<unknown>[],
> {
  readonly event: Event
  readonly address: Address<Space>
  readonly state: State
  readonly read: ProjectionReader<Space>
  readonly ambient: AmbientReader
  readonly emit: SignalEmitter<Emits>
}

export interface SingletonSignalHandlerContext<
  Payload,
  State,
  Emits extends readonly SignalDefinition<unknown>[],
> {
  readonly signal: SignalOccurrence<Payload>
  readonly state: State
  readonly read: ProjectionReader
  readonly ambient: AmbientReader
  readonly emit: SignalEmitter<Emits>
}

export interface SingletonAmbientHandlerContext<
  Value,
  State,
  Emits extends readonly SignalDefinition<unknown>[],
> {
  readonly value: Value
  readonly state: State
  readonly read: ProjectionReader
  readonly ambient: AmbientReader
  readonly emit: SignalEmitter<Emits>
}

export interface PartitionedAmbientHandlerContext<
  Value,
  Space extends AddressSpaceBase,
  State,
  Emits extends readonly SignalDefinition<unknown>[],
> extends SingletonAmbientHandlerContext<Value, State, Emits> {
  readonly address: Address<Space>
  readonly read: ProjectionReader<Space>
}

export interface PartitionedSignalHandlerContext<
  Payload,
  Space extends AddressSpaceBase,
  State,
  Emits extends readonly SignalDefinition<unknown>[],
> extends SingletonSignalHandlerContext<Payload, State, Emits> {
  readonly address: Address<Space>
  readonly read: ProjectionReader<Space>
}

type HandlerResult<Update> = Update | Effect.Effect<Update, ProjectionReadError, never>

export interface SingletonEventHandlerBuilder<
  Event,
  Key extends DiscriminatorKey<Event> | undefined,
  State,
  Emits extends readonly SignalDefinition<unknown>[],
> {
  <Value extends DiscriminatorValue<Event, Key>>(
    selector: Value,
    handler: (context: SingletonProjectionHandlerContext<EventVariant<Event, Key, Value>, State, Emits>) => HandlerResult<ProjectionUpdate<State>>,
  ): SingletonHandler<Event, State>
  <Selected extends Event>(
    selector: EventRefinement<Event, Selected>,
    handler: (context: SingletonProjectionHandlerContext<Selected, State, Emits>) => HandlerResult<ProjectionUpdate<State>>,
  ): SingletonHandler<Event, State>
  (
    selector: typeof EventAny,
    handler: (context: SingletonProjectionHandlerContext<Event, State, Emits>) => HandlerResult<ProjectionUpdate<State>>,
  ): SingletonHandler<Event, State>
}

export interface PartitionedEventHandlerBuilder<
  Event,
  Key extends DiscriminatorKey<Event> | undefined,
  Space extends AddressSpaceBase,
  State,
  Emits extends readonly SignalDefinition<unknown>[],
> {
  <Value extends DiscriminatorValue<Event, Key>>(
    selector: Value,
    options: { readonly route: (event: EventVariant<Event, Key, Value>) => AddressTarget<Address<Space>> },
    handler: (context: PartitionedProjectionHandlerContext<EventVariant<Event, Key, Value>, Space, State, Emits>) => HandlerResult<PartitionUpdate<State>>,
  ): PartitionedHandler<Event, Space, State>
  <Selected extends Event>(
    selector: EventRefinement<Event, Selected>,
    options: { readonly route: (event: Selected) => AddressTarget<Address<Space>> },
    handler: (context: PartitionedProjectionHandlerContext<Selected, Space, State, Emits>) => HandlerResult<PartitionUpdate<State>>,
  ): PartitionedHandler<Event, Space, State>
  (
    selector: typeof EventAny,
    options: { readonly route: (event: Event) => AddressTarget<Address<Space>> },
    handler: (context: PartitionedProjectionHandlerContext<Event, Space, State, Emits>) => HandlerResult<PartitionUpdate<State>>,
  ): PartitionedHandler<Event, Space, State>
}

export interface SingletonHandler<Event, State, Emits extends readonly SignalDefinition<unknown>[] = readonly SignalDefinition<unknown>[]> {
  readonly _tag: "Event"
  readonly matcher: { readonly name: string; readonly matches: (event: Event) => boolean }
  readonly handle: (context: SingletonProjectionHandlerContext<Event, State, Emits>) => HandlerResult<ProjectionUpdate<State>>
}

export interface PartitionedHandler<Event, Space extends AddressSpaceBase, State, Emits extends readonly SignalDefinition<unknown>[] = readonly SignalDefinition<unknown>[]> {
  readonly _tag: "Event"
  readonly matcher: { readonly name: string; readonly matches: (event: Event) => boolean }
  readonly route: (event: Event) => AddressTarget<Address<Space>>
  readonly handle: (context: PartitionedProjectionHandlerContext<Event, Space, State, Emits>) => HandlerResult<PartitionUpdate<State>>
}

export interface SingletonSignalHandler<State, Emits extends readonly SignalDefinition<unknown>[] = readonly SignalDefinition<unknown>[]> {
  readonly _tag: "Signal"
  readonly signal: SignalDefinition<unknown>
  readonly handle: (context: SingletonSignalHandlerContext<unknown, State, Emits>) => HandlerResult<ProjectionUpdate<State>>
}

export interface PartitionedSignalHandler<Space extends AddressSpaceBase, State, Emits extends readonly SignalDefinition<unknown>[] = readonly SignalDefinition<unknown>[]> {
  readonly _tag: "Signal"
  readonly signal: SignalDefinition<unknown>
  readonly route: (signal: SignalOccurrence<unknown>) => AddressTarget<Address<Space>>
  readonly handle: (context: PartitionedSignalHandlerContext<unknown, Space, State, Emits>) => HandlerResult<PartitionUpdate<State>>
}

export interface SingletonAmbientHandler<State, Emits extends readonly SignalDefinition<unknown>[] = readonly SignalDefinition<unknown>[]> {
  readonly _tag: "Ambient"
  readonly ambient: AmbientDefinition<unknown>
  readonly handle: (context: SingletonAmbientHandlerContext<unknown, State, Emits>) => HandlerResult<ProjectionUpdate<State>>
}

export interface PartitionedAmbientHandler<Space extends AddressSpaceBase, State, Emits extends readonly SignalDefinition<unknown>[] = readonly SignalDefinition<unknown>[]> {
  readonly _tag: "Ambient"
  readonly ambient: AmbientDefinition<unknown>
  readonly route: (value: unknown) => AddressTarget<Address<Space>>
  readonly handle: (context: PartitionedAmbientHandlerContext<unknown, Space, State, Emits>) => HandlerResult<PartitionUpdate<State>>
}

export interface SingletonAmbientHandlerBuilder<State, Emits extends readonly SignalDefinition<unknown>[]> {
  <Value>(
    ambient: AmbientDefinition<Value>,
    handler: (context: SingletonAmbientHandlerContext<Value, State, Emits>) => HandlerResult<ProjectionUpdate<State>>,
  ): SingletonAmbientHandler<State, Emits>
}

export interface PartitionedAmbientHandlerBuilder<Space extends AddressSpaceBase, State, Emits extends readonly SignalDefinition<unknown>[]> {
  <Value>(
    ambient: AmbientDefinition<Value>,
    options: { readonly route: (value: Value) => AddressTarget<Address<Space>> },
    handler: (context: PartitionedAmbientHandlerContext<Value, Space, State, Emits>) => HandlerResult<PartitionUpdate<State>>,
  ): PartitionedAmbientHandler<Space, State, Emits>
}

export interface SingletonSignalHandlerBuilder<State, Emits extends readonly SignalDefinition<unknown>[]> {
  <Payload>(
    signal: SignalDefinition<Payload>,
    handler: (context: SingletonSignalHandlerContext<Payload, State, Emits>) => HandlerResult<ProjectionUpdate<State>>,
  ): SingletonSignalHandler<State, Emits>
}

export interface PartitionedSignalHandlerBuilder<Space extends AddressSpaceBase, State, Emits extends readonly SignalDefinition<unknown>[]> {
  <Payload>(
    signal: SignalDefinition<Payload>,
    options: { readonly route: (signal: SignalOccurrence<Payload>) => AddressTarget<Address<Space>> },
    handler: (context: PartitionedSignalHandlerContext<Payload, Space, State, Emits>) => HandlerResult<PartitionUpdate<State>>,
  ): PartitionedSignalHandler<Space, State, Emits>
}

export interface SingletonProjectionConfig<
  Event,
  Key extends DiscriminatorKey<Event> | undefined,
  Name extends string,
  State,
  EncodedState,
  Emits extends readonly SignalDefinition<unknown>[],
> {
  readonly name: Name
  readonly state: Schema.Schema<State, EncodedState, never>
  readonly initial: State
  readonly emits?: Emits
  readonly reads?: readonly ProjectionFor<Event>[]
  readonly handlers: (
    on: {
      readonly event: SingletonEventHandlerBuilder<Event, Key, State, Emits>
      readonly signal: SingletonSignalHandlerBuilder<State, Emits>
      readonly ambient: SingletonAmbientHandlerBuilder<State, Emits>
    },
  ) => readonly (SingletonHandler<Event, State, Emits> | SingletonSignalHandler<State, Emits> | SingletonAmbientHandler<State, Emits>)[]
}

export interface PartitionedProjectionConfig<
  Event,
  Key extends DiscriminatorKey<Event> | undefined,
  Name extends string,
  Space extends AddressSpaceBase,
  State,
  EncodedState,
  Emits extends readonly SignalDefinition<unknown>[],
> {
  readonly name: Name
  readonly by: Space
  readonly state: Schema.Schema<State, EncodedState, never>
  readonly initial: (context: { readonly address: Address<Space> }) => State
  readonly emits?: Emits
  readonly reads?: readonly ProjectionFor<Event>[]
  readonly handlers: (
    on: {
      readonly event: PartitionedEventHandlerBuilder<Event, Key, Space, State, Emits>
      readonly signal: PartitionedSignalHandlerBuilder<Space, State, Emits>
      readonly ambient: PartitionedAmbientHandlerBuilder<Space, State, Emits>
    },
  ) => readonly (
    | PartitionedHandler<Event, Space, State, Emits>
    | PartitionedSignalHandler<Space, State, Emits>
    | PartitionedAmbientHandler<Space, State, Emits>
  )[]
}

export interface ProjectionDefinitionInternal<Event> extends ProjectionDefinitionBase {
  readonly _eventSelection: EventSelection<Event, DiscriminatorKey<Event> | undefined>
  readonly _config: SingletonProjectionConfig<Event, DiscriminatorKey<Event> | undefined, string, unknown, unknown, readonly SignalDefinition<unknown>[]>
    | PartitionedProjectionConfig<Event, DiscriminatorKey<Event> | undefined, string, AddressSpaceBase, unknown, unknown, readonly SignalDefinition<unknown>[]>
  readonly _handlers: readonly (
    | SingletonHandler<Event, unknown>
    | PartitionedHandler<Event, AddressSpaceBase, unknown>
    | SingletonSignalHandler<unknown>
    | PartitionedSignalHandler<AddressSpaceBase, unknown>
    | SingletonAmbientHandler<unknown>
    | PartitionedAmbientHandler<AddressSpaceBase, unknown>
  )[]
}

const toEffect = <Update>(result: HandlerResult<Update>): Effect.Effect<Update, ProjectionReadError> =>
  Effect.isEffect(result) ? result : Effect.succeed(result)

export const ProjectionInternal = {
  typeId: ProjectionTypeId,
  stateTypeId: ProjectionStateTypeId,
  spaceTypeId: ProjectionSpaceTypeId,
  toEffect,
}

export const makeProjectionBuilder = <
  Event,
  Key extends DiscriminatorKey<Event> | undefined,
>(selection: EventSelection<Event, Key>) => {
  const singleton = <
    Name extends string,
    State,
    EncodedState,
    const Emits extends readonly SignalDefinition<unknown>[] = readonly [],
  >(
    config: SingletonProjectionConfig<Event, Key, Name, State, EncodedState, Emits>,
  ): SingletonProjectionDefinition<State, EncodedState, Event> => {
    const event = ((selector: PropertyKey | typeof EventAny | EventRefinement<Event, Event>, handler: SingletonHandler<Event, State, Emits>["handle"]) => ({
      _tag: "Event",
      matcher: selection.matcher(selector),
      handle: handler,
    })) as SingletonEventHandlerBuilder<Event, Key, State, Emits>
    const signal = ((definition: SignalDefinition<unknown>, handler: SingletonSignalHandler<State, Emits>["handle"]) => ({
      _tag: "Signal",
      signal: definition,
      handle: handler,
    })) as SingletonSignalHandlerBuilder<State, Emits>
    const ambient = ((definition: AmbientDefinition<unknown>, handler: SingletonAmbientHandler<State, Emits>["handle"]) => ({
      _tag: "Ambient",
      ambient: definition,
      handle: handler,
    })) as SingletonAmbientHandlerBuilder<State, Emits>
    const handlers = config.handlers({ event, signal, ambient })
    return {
      [ProjectionTypeId]: ProjectionTypeId,
      [ProjectionStateTypeId]: undefined as State,
      [ProjectionSpaceTypeId]: undefined,
      [ProjectionEventTypeId]: undefined as unknown as (event: Event) => Event,
      name: config.name,
      topology: { _tag: "Singleton" },
      dependencies: config.reads ?? [],
      emits: config.emits ?? [],
      signalSubscriptions: handlers.flatMap((handler) => handler._tag === "Signal" ? [handler.signal] : []),
      stateSchema: config.state,
      _eventSelection: selection,
      _config: config,
      _handlers: handlers,
    } as SingletonProjectionDefinition<State, EncodedState, Event>
  }

  const partitioned = <
    Name extends string,
    Space extends AddressSpaceBase,
    State,
    EncodedState,
    const Emits extends readonly SignalDefinition<unknown>[] = readonly [],
  >(
    config: PartitionedProjectionConfig<Event, Key, Name, Space, State, EncodedState, Emits>,
  ): PartitionedProjectionDefinition<Space, State, EncodedState, Event> => {
    const event = ((
      selector: PropertyKey | typeof EventAny | EventRefinement<Event, Event>,
      options: { readonly route: (event: Event) => AddressTarget<Address<Space>> },
      handler: PartitionedHandler<Event, Space, State, Emits>["handle"],
    ) => ({ _tag: "Event", matcher: selection.matcher(selector), route: options.route, handle: handler })) as PartitionedEventHandlerBuilder<Event, Key, Space, State, Emits>
    const signal = ((
      definition: SignalDefinition<unknown>,
      options: { readonly route: (signal: SignalOccurrence<unknown>) => AddressTarget<Address<Space>> },
      handler: PartitionedSignalHandler<Space, State, Emits>["handle"],
    ) => ({ _tag: "Signal", signal: definition, route: options.route, handle: handler })) as PartitionedSignalHandlerBuilder<Space, State, Emits>
    const ambient = ((
      definition: AmbientDefinition<unknown>,
      options: { readonly route: (value: unknown) => AddressTarget<Address<Space>> },
      handler: PartitionedAmbientHandler<Space, State, Emits>["handle"],
    ) => ({ _tag: "Ambient", ambient: definition, route: options.route, handle: handler })) as PartitionedAmbientHandlerBuilder<Space, State, Emits>
    const handlers = config.handlers({ event, signal, ambient })
    return {
      [ProjectionTypeId]: ProjectionTypeId,
      [ProjectionStateTypeId]: undefined as State,
      [ProjectionSpaceTypeId]: config.by,
      [ProjectionEventTypeId]: undefined as unknown as (event: Event) => Event,
      name: config.name,
      topology: { _tag: "Partitioned", by: config.by },
      dependencies: config.reads ?? [],
      emits: config.emits ?? [],
      signalSubscriptions: handlers.flatMap((handler) => handler._tag === "Signal" ? [handler.signal] : []),
      stateSchema: config.state,
      _eventSelection: selection,
      _config: config,
      _handlers: handlers,
    } as PartitionedProjectionDefinition<Space, State, EncodedState, Event>
  }

  return { singleton, partitioned }
}
