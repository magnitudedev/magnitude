// /**
//  * Projection.fsmSingleton() - Single FSM instance projection with auto-signals
//  *
//  * Creates a projection that manages a single FSM instance with:
//  * - Type-safe transitions via instance.to()
//  * - Automatic signal emission per FSM state
//  * - Immutable state updates (handlers return new state)
//  */
// 
// import { Effect, SubscriptionRef, Context, Layer, PubSub } from 'effect'
// import { ProjectionBusTag, type ProjectionBusService } from '../core/projection-bus'
// import { type BaseEvent, type Timestamped } from '../core/event-bus-core'
// import {
//   type Signal,
//   createSignal,
// } from '../signal/define'
// import {
//   type FSMInstance,
//   type TransitionMatrix,
//   type StateNames,
//   type AnyStateClass,
//   type StateUnion,
//   type TaggedItem
// } from '../fsm/define'
// 
// // ---------------------------------------------------------------------------
// // Types
// // ---------------------------------------------------------------------------
// 
// /**
//  * Projection instance for FSM singleton
//  */
// export interface FSMSingletonInstance<TState extends TaggedItem> {
//   readonly state: SubscriptionRef.SubscriptionRef<TState>
//   readonly get: Effect.Effect<TState>
// }
// 
// /**
//  * Signals generated per FSM state.
//  * Each signal emits the full state union type (the value when entering that state).
//  */
// export type FSMSingletonSignals<
//   T extends TransitionMatrix,
//   TState extends TaggedItem
// > = {
//   readonly [K in StateNames<T>]: Signal<TState>
// }
// 
// /**
//  * Extract PubSub types from FSM signals for Layer output
//  */
// export type FSMSingletonSignalPubSubs<
//   T extends TransitionMatrix,
//   TState extends TaggedItem
// > = {
//   [K in StateNames<T>]: PubSub.PubSub<TState>
// }[StateNames<T>]
// 
// /**
//  * Event handler - receives current state, returns new state
//  */
// export type FSMSingletonEventHandler<TState extends TaggedItem, TEvent extends BaseEvent> = (
//   instance: TState,
//   event: Timestamped<TEvent>
// ) => TState
// 
// // ---------------------------------------------------------------------------
// // Config and Result Types
// // ---------------------------------------------------------------------------
// 
// export interface FSMSingletonConfig<
//   TName extends string,
//   T extends TransitionMatrix,
//   Classes extends readonly AnyStateClass[],
//   TEvent extends BaseEvent
// > {
//   readonly name: TName
//   readonly fsm: FSMInstance<T, Classes>
//   readonly initial: StateUnion<Classes>
// 
//   readonly eventHandlers: {
//     [E in TEvent['type']]?: FSMSingletonEventHandler<
//       StateUnion<Classes>,
//       Timestamped<Extract<TEvent, { type: E }>>
//     >
//   }
// }
// 
// export interface FSMSingletonResult<
//   T extends TransitionMatrix,
//   Classes extends readonly AnyStateClass[],
//   TEvent extends BaseEvent
// > {
//   readonly name: string
//   readonly Tag: Context.Tag<FSMSingletonInstance<StateUnion<Classes>>, FSMSingletonInstance<StateUnion<Classes>>>
//   readonly Layer: Layer.Layer<
//     FSMSingletonInstance<StateUnion<Classes>> | FSMSingletonSignalPubSubs<T, StateUnion<Classes>>,
//     never,
//     ProjectionBusService<TEvent>
//   >
//   readonly signals: FSMSingletonSignals<T, StateUnion<Classes>>
// }
// 
// // ---------------------------------------------------------------------------
// // Implementation
// // ---------------------------------------------------------------------------
// 
// /**
//  * Curried version - specify event type first for better inference
//  */
// export function fsmSingleton<TEvent extends BaseEvent>() {
//   return <
//     TName extends string,
//     T extends TransitionMatrix,
//     Classes extends readonly AnyStateClass[]
//   >(
//     config: FSMSingletonConfig<TName, T, Classes, TEvent>
//   ): FSMSingletonResult<T, Classes, TEvent> => fsmSingletonImpl(config)
// }
// 
// function fsmSingletonImpl<
//   TName extends string,
//   T extends TransitionMatrix,
//   Classes extends readonly AnyStateClass[],
//   TEvent extends BaseEvent
// >(
//   config: FSMSingletonConfig<TName, T, Classes, TEvent>
// ): FSMSingletonResult<T, Classes, TEvent> {
//   type TState = StateUnion<Classes>
// 
//   const serviceName = `${config.name}Projection`
//   const Tag = Context.GenericTag<FSMSingletonInstance<TState>>(serviceName)
//   const BusTag = ProjectionBusTag<TEvent>()
// 
//   // Create signals for each state
//   const stateSignalsMap = new Map<StateNames<T>, Signal<TState>>()
//   for (const state of config.fsm.states) {
//     stateSignalsMap.set(state, createSignal<TState>(`${config.name}/${state}`, config.name))
//   }
// 
//   // Create signal layers
//   const StateSignalLayers = config.fsm.states.map((state: StateNames<T>) => {
//     const signal = stateSignalsMap.get(state)!
//     return Layer.scoped(signal.tag, PubSub.unbounded<TState>())
//   })
//   const AllSignalLayers = StateSignalLayers.reduce((acc: Layer.Layer<PubSub.PubSub<TState>, never, never>, layer: Layer.Layer<PubSub.PubSub<TState>, never, never>) => Layer.merge(acc, layer))
// 
//   // Collect all event types we need to listen to
//   const allEventTypes = Object.keys(config.eventHandlers) as TEvent['type'][]
// 
//   // Main projection layer
//   const ProjectionLayer = Layer.scoped(
//     Tag,
//     Effect.gen(function* () {
//       const bus = yield* BusTag
// 
//       const stateRef = yield* SubscriptionRef.make<TState>(config.initial)
// 
//       // Get PubSubs for signals
//       const stateHubs = new Map<string, PubSub.PubSub<TState>>()
//       for (const state of config.fsm.states) {
//         const signal = stateSignalsMap.get(state)!
//         stateHubs.set(state, yield* signal.tag)
//       }
// 
//       // Helper to emit state signal
//       const emitStateSignal = (state: string, value: TState) => {
//         const hub = stateHubs.get(state)
//         return hub ? PubSub.publish(hub, value) : Effect.void
//       }
// 
//       // Event handler
//       const eventHandler = (event: TEvent) =>
//         Effect.gen(function* () {
//           const eventType = event.type as TEvent['type']
//           const handler = config.eventHandlers[eventType]
// 
//           if (!handler) return
// 
//           const currentState = yield* SubscriptionRef.get(stateRef)
//           const previousTag = currentState._tag
// 
//           // Call handler - returns new state
//           const newState = handler(currentState, event as Timestamped<Extract<TEvent, { type: typeof eventType }>>)
// 
//           // Only update if state changed
//           if (newState !== currentState) {
//             yield* SubscriptionRef.set(stateRef, newState)
// 
//             // Emit signal if tag changed
//             if (newState._tag !== previousTag) {
//               yield* emitStateSignal(newState._tag, newState)
//             }
//           }
//         })
// 
//       // Register with bus
//       if (allEventTypes.length > 0) {
//         yield* bus.register(eventHandler, allEventTypes, serviceName)
//       }
// 
//       return {
//         state: stateRef,
//         get: SubscriptionRef.get(stateRef)
//       }
//     })
//   )
// 
//   // Combine layers
//   const FullLayer = Layer.provideMerge(ProjectionLayer, AllSignalLayers)
// 
//   // Build signals object
//   const signals = Object.fromEntries(
//     config.fsm.states.map((state: StateNames<T>) => [state, stateSignalsMap.get(state)!])
//   ) as FSMSingletonSignals<T, TState>
// 
//   return {
//     name: config.name,
//     Tag,
//     Layer: FullLayer,
//     signals
//   }
// }
