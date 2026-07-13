// /**
//  * Flow Types
//  *
//  * Core type definitions for the Flow API.
//  */

// import { Effect, Context } from 'effect'
// import type {
//   FSMInstance,
//   TransitionMatrix,
//   AnyStateClass,
//   StateUnion,
//   StateNames,
//   TaggedItem,
//   PropsOf,
//   StateClassRecord,
//   TransitionUpdates
// } from '../fsm/define'
// import type { BaseEvent } from '../core/event-bus-core'

// // ---------------------------------------------------------------------------
// // Flow State Constraint
// // ---------------------------------------------------------------------------

// /**
//  * All flow states must have an id field
//  */
// export type FlowState = TaggedItem & { id: string }

// // ---------------------------------------------------------------------------
// // Flow Handle
// // ---------------------------------------------------------------------------

// /**
//  * Handle to a child flow, returned by ctx.spawn()
//  */
// export interface FlowHandle<TState extends FlowState> {
//   /**
//    * The child flow's instance ID
//    */
//   readonly id: string

//   /**
//    * Wait for child to reach a terminal state
//    */
//   readonly await: () => Effect.Effect<TState>

//   /**
//    * Get child's current state
//    */
//   readonly state: Effect.Effect<TState | undefined>
// }

// // ---------------------------------------------------------------------------
// // Flow Context
// // ---------------------------------------------------------------------------

// /**
//  * Options for waitFor
//  */
// export interface WaitForOptions<TEvent> {
//   /**
//    * Custom matcher function. If provided, overrides default flowInstanceId matching.
//    */
//   readonly match?: (event: TEvent) => boolean
// }

// /**
//  * Flow context service - provides ctx methods inside run()
//  *
//  * TClasses is needed for properly typed transition()
//  */
// export interface FlowContextService<
//   TEvent extends BaseEvent,
//   T extends TransitionMatrix,
//   TClasses extends readonly AnyStateClass[]
// > {
//   /**
//    * Execute a side effect with automatic replay.
//    * First run: executes and persists result
//    * Replay: returns persisted result
//    */
//   readonly effect: <TResult, R>(
//     eff: Effect.Effect<TResult, never, R>
//   ) => Effect.Effect<TResult, never, R>

//   /**
//    * Spawn a child flow.
//    * First run: creates child, returns handle
//    * Replay: returns handle to existing child
//    */
//   readonly spawn: <TChildName extends string, TChildInput>(
//     flow: {
//       readonly name: TChildName
//       readonly flowStartedType: FlowStartedEventType<TChildName>
//       readonly entry: (id: string, input: TChildInput) => FlowState
//     },
//     input: TChildInput
//   ) => Effect.Effect<FlowHandle<FlowState>>

//   /**
//    * Wait for an external event.
//    * First run: subscribes and waits
//    * Replay: returns from history if exists, otherwise subscribes
//    */
//   readonly waitFor: <TEventTypes extends readonly TEvent['type'][]>(
//     eventTypes: TEventTypes,
//     options?: WaitForOptions<Extract<TEvent, { type: TEventTypes[number] }>>
//   ) => Effect.Effect<Extract<TEvent, { type: TEventTypes[number] }>>

//   /**
//    * Transition to a new state. Emits signal to update projection state.
//    * Fields new to target are required; fields shared with source are optional.
//    */
//   readonly transition: <
//     From extends StateUnion<TClasses>,
//     To extends StateNames<T> & keyof StateClassRecord<TClasses>
//   >(
//     from: From,
//     targetState: To,
//     data: TransitionUpdates<From, PropsOf<StateClassRecord<TClasses>[To]>>
//   ) => Effect.Effect<void>

//   /**
//    * Emit an event for UI/external consumption.
//    */
//   readonly emit: (event: TEvent) => Effect.Effect<void>
// }

// /**
//  * FlowContext tag - used to access context in run()
//  */
// export class FlowContext extends Context.Tag('FlowContext')<
//   FlowContext,
//   FlowContextService<BaseEvent, TransitionMatrix, readonly AnyStateClass[]>
// >() {}

// // ---------------------------------------------------------------------------
// // Flow Definition
// // ---------------------------------------------------------------------------

// /**
//  * Input type for entry function, excluding id which is provided by framework
//  */
// export type EntryInput<TClasses extends readonly AnyStateClass[]> =
//   Omit<PropsOf<TClasses[0]>, 'id'>

// /**
//  * Flow definition - the config object passed to Flow.define()
//  *
//  * TEvent must include InternalEvent (users should union their events with Flow.InternalEvent)
//  */
// export interface FlowConfig<
//   TName extends string,
//   T extends TransitionMatrix,
//   TClasses extends readonly AnyStateClass[],
//   TEvent extends BaseEvent,
//   TInput
// > {
//   /**
//    * Flow name - used for logging, service names, etc.
//    */
//   readonly name: TName

//   /**
//    * The FSM defining states and transitions
//    */
//   readonly fsm: FSMInstance<T, TClasses>

//   /**
//    * Entry function - creates initial state from spawn input.
//    * Framework provides the id.
//    */
//   readonly entry: (id: string, input: TInput) => StateUnion<TClasses> & { id: string }

//   /**
//    * Run function - imperative orchestration logic.
//    * Executes linearly from start to completion.
//    */
//   readonly run: (
//     state: StateUnion<TClasses> & { id: string },
//     ctx: FlowContextService<TEvent, T, TClasses>
//   ) => Effect.Effect<void>
// }

// /**
//  * Flow definition result - what Flow.define() returns
//  */
// export interface FlowDefinition<
//   TName extends string,
//   T extends TransitionMatrix,
//   TClasses extends readonly AnyStateClass[],
//   TEvent extends BaseEvent,
//   TInput
// > extends FlowConfig<TName, T, TClasses, TEvent, TInput> {
//   /**
//    * Projection Tag for dependency injection
//    */
//   readonly ProjectionTag: Context.Tag<
//     FlowProjectionService<StateUnion<TClasses> & { id: string }>,
//     FlowProjectionService<StateUnion<TClasses> & { id: string }>
//   >

//   /**
//    * Combined Layer (projection + worker)
//    */
//   // Layer type will be refined during implementation
// }

// /**
//  * Flow projection service interface
//  */
// export interface FlowProjectionService<TState extends FlowState> {
//   readonly getById: (id: string) => Effect.Effect<TState | undefined>
//   readonly getAll: () => Effect.Effect<TState[]>
//   readonly getByState: (state: string) => Effect.Effect<TState[]>
// }

// // ---------------------------------------------------------------------------
// // Flow Events (Template Literal Types)
// // ---------------------------------------------------------------------------

// /**
//  * Convert flow name to event type prefix
//  * e.g., 'OAuth' -> 'oauth', 'UserAuth' -> 'userauth'
//  */
// export type FlowEventPrefix<TName extends string> = Lowercase<TName>

// /**
//  * Flow started event type for a specific flow
//  * e.g., 'oauth_flow_started'
//  */
// export type FlowStartedEventType<TName extends string> = `${FlowEventPrefix<TName>}_flow_started`

// /**
//  * Flow effect result event type for a specific flow
//  * e.g., 'oauth_flow_effect_result'
//  */
// export type FlowEffectResultEventType<TName extends string> = `${FlowEventPrefix<TName>}_flow_effect_result`

// /**
//  * Event emitted when an effect completes
//  */
// export interface FlowEffectResultEvent<
//   TName extends string,
//   TResult = unknown
// > {
//   readonly type: FlowEffectResultEventType<TName>
//   readonly flowInstanceId: string
//   readonly effectIndex: number
//   readonly result: TResult
// }

// /**
//  * Event emitted when a flow is started (root or child)
//  */
// export interface FlowStartedEvent<
//   TName extends string,
//   TState extends FlowState = FlowState
// > {
//   readonly type: FlowStartedEventType<TName>
//   readonly flowInstanceId: string
//   readonly initialState: TState
//   readonly parent?: {
//     readonly flowInstanceId: string
//     readonly spawnIndex: number
//   }
// }

// /**
//  * All events for a specific flow
//  */
// export type FlowEvents<
//   TName extends string,
//   TState extends FlowState = FlowState
// > =
//   | FlowStartedEvent<TName, TState>
//   | FlowEffectResultEvent<TName>
