// /**
//  * Projection.fsmCollection() - Collection of FSM instances with auto-signals
//  *
//  * Creates a projection that manages a Map of FSM instances with:
//  * - Spawners for creating new instances
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
//  * Collection state - Map of id to FSM state instance
//  */
// export interface FSMCollectionState<TState extends TaggedItem & { id: string }> {
//   readonly instances: Map<string, TState>
// }
// 
// /**
//  * Projection instance for FSM collection
//  */
// export interface FSMCollectionInstance<TState extends TaggedItem & { id: string }> {
//   readonly state: SubscriptionRef.SubscriptionRef<FSMCollectionState<TState>>
//   readonly get: Effect.Effect<FSMCollectionState<TState>>
//   readonly getById: (id: string) => Effect.Effect<TState | undefined>
//   readonly getByState: (state: string) => Effect.Effect<TState[]>
//   readonly getAll: () => Effect.Effect<TState[]>
// }
// 
// /**
//  * Signals generated per FSM state plus a change signal.
//  * Each signal emits the full state union type.
//  */
// export type FSMCollectionSignals<
//   T extends TransitionMatrix,
//   TState extends TaggedItem & { id: string }
// > = {
//   readonly change: Signal<TState>
// } & {
//   readonly [K in StateNames<T>]: Signal<TState>
// }
// 
// /**
//  * Extract PubSub types from FSM collection signals for Layer output
//  */
// export type FSMCollectionSignalPubSubs<
//   T extends TransitionMatrix,
//   TState extends TaggedItem & { id: string }
// > = PubSub.PubSub<TState> | {
//   [K in StateNames<T>]: PubSub.PubSub<TState>
// }[StateNames<T>]
// 
// /**
//  * Spawner function - returns a new state instance
//  */
// export type FSMCollectionSpawner<TState extends TaggedItem & { id: string }, TEvent extends BaseEvent> = (
//   event: Timestamped<TEvent>
// ) => TState
// 
// /**
//  * Event handler - receives current state, returns new state (immutable)
//  */
// export type FSMCollectionEventHandler<TState extends TaggedItem & { id: string }, TEvent extends BaseEvent> = (
//   instance: TState,
//   event: Timestamped<TEvent>
// ) => TState
// 
// // ---------------------------------------------------------------------------
// // Config and Result Types
// // ---------------------------------------------------------------------------
// 
// export interface FSMCollectionConfig<
//   TName extends string,
//   T extends TransitionMatrix,
//   Classes extends readonly AnyStateClass[],
//   TEvent extends BaseEvent
// > {
//   readonly name: TName
//   readonly fsm: FSMInstance<T, Classes>
// 
//   /** Extract instance id from event */
//   readonly instanceId: (event: Timestamped<TEvent>) => string
// 
//   /** Events that spawn new instances - handler returns constructed state */
//   readonly spawners?: {
//     [E in TEvent['type']]?: FSMCollectionSpawner<
//       StateUnion<Classes> & { id: string },
//       Timestamped<Extract<TEvent, { type: E }>>
//     >
//   }
// 
//   /** Events that operate on existing instances - handler returns new state */
//   readonly eventHandlers?: {
//     [E in TEvent['type']]?: FSMCollectionEventHandler<
//       StateUnion<Classes> & { id: string },
//       Timestamped<Extract<TEvent, { type: E }>>
//     >
//   }
// }
// 
// export interface FSMCollectionResult<
//   T extends TransitionMatrix,
//   Classes extends readonly AnyStateClass[],
//   TEvent extends BaseEvent
// > {
//   readonly name: string
//   readonly Tag: Context.Tag<
//     FSMCollectionInstance<StateUnion<Classes> & { id: string }>,
//     FSMCollectionInstance<StateUnion<Classes> & { id: string }>
//   >
//   readonly Layer: Layer.Layer<
//     FSMCollectionInstance<StateUnion<Classes> & { id: string }> | FSMCollectionSignalPubSubs<T, StateUnion<Classes> & { id: string }>,
//     never,
//     ProjectionBusService<TEvent>
//   >
//   readonly signals: FSMCollectionSignals<T, StateUnion<Classes> & { id: string }>
//   readonly changeSignal: Signal<StateUnion<Classes> & { id: string }>
// }
// 
// // ---------------------------------------------------------------------------
// // Implementation
// // ---------------------------------------------------------------------------
// 
// /**
//  * Curried version - specify event type first for better inference
//  */
// export function fsmCollection<TEvent extends BaseEvent>() {
//   return <
//     TName extends string,
//     T extends TransitionMatrix,
//     Classes extends readonly AnyStateClass[]
//   >(
//     config: FSMCollectionConfig<TName, T, Classes, TEvent>
//   ): FSMCollectionResult<T, Classes, TEvent> => fsmCollectionImpl(config)
// }
// 
// function fsmCollectionImpl<
//   TName extends string,
//   T extends TransitionMatrix,
//   Classes extends readonly AnyStateClass[],
//   TEvent extends BaseEvent
// >(
//   config: FSMCollectionConfig<TName, T, Classes, TEvent>
// ): FSMCollectionResult<T, Classes, TEvent> {
//   type TState = StateUnion<Classes> & { id: string }
// 
//   const serviceName = `${config.name}Projection`
//   const Tag = Context.GenericTag<FSMCollectionInstance<TState>>(serviceName)
//   const BusTag = ProjectionBusTag<TEvent>()
// 
//   // Create signals for each state plus change signal
//   const changeSignal = createSignal<TState>(`${config.name}/change`, config.name)
//   const stateSignalsMap = new Map<StateNames<T>, Signal<TState>>()
//   for (const state of config.fsm.states) {
//     stateSignalsMap.set(state, createSignal<TState>(`${config.name}/${state}`, config.name))
//   }
// 
//   // Create signal layers
//   const ChangeSignalLayer = Layer.scoped(changeSignal.tag, PubSub.unbounded<TState>())
//   const StateSignalLayers = config.fsm.states.map((state: StateNames<T>) => {
//     const signal = stateSignalsMap.get(state)!
//     return Layer.scoped(signal.tag, PubSub.unbounded<TState>())
//   })
//   const AllSignalLayers = [ChangeSignalLayer, ...StateSignalLayers].reduce(
//     (acc: Layer.Layer<PubSub.PubSub<TState>, never, never>, layer: Layer.Layer<PubSub.PubSub<TState>, never, never>) =>
//       Layer.merge(acc, layer)
//   )
// 
//   // Collect all event types we need to listen to
//   const allEventTypes = new Set<string>()
//   if (config.spawners) {
//     for (const eventType of Object.keys(config.spawners)) {
//       allEventTypes.add(eventType)
//     }
//   }
//   if (config.eventHandlers) {
//     for (const eventType of Object.keys(config.eventHandlers)) {
//       allEventTypes.add(eventType)
//     }
//   }
// 
//   // Main projection layer
//   const ProjectionLayer = Layer.scoped(
//     Tag,
//     Effect.gen(function* () {
//       const bus = yield* BusTag
// 
//       const stateRef = yield* SubscriptionRef.make<FSMCollectionState<TState>>({
//         instances: new Map()
//       })
// 
//       // Get PubSubs for signals
//       const changeHub = yield* changeSignal.tag
//       const stateHubs = new Map<string, PubSub.PubSub<TState>>()
//       for (const state of config.fsm.states) {
//         const signal = stateSignalsMap.get(state)!
//         stateHubs.set(state, yield* signal.tag)
//       }
// 
//       // Helper to emit signals
//       const emitChange = (item: TState) =>
//         PubSub.publish(changeHub, item)
// 
//       const emitStateSignal = (state: string, item: TState) => {
//         const hub = stateHubs.get(state)
//         return hub ? PubSub.publish(hub, item) : Effect.void
//       }
// 
//       // Event handler
//       const eventHandler = (event: TEvent) =>
//         Effect.gen(function* () {
//           const eventType = event.type as TEvent['type']
//           const currentState = yield* SubscriptionRef.get(stateRef)
//           const instances = new Map(currentState.instances)
// 
//           // Pending signals to emit
//           const pendingSignals: Array<{ type: 'change' | 'state'; state?: string; item: TState }> = []
// 
//           // Check if this is a spawner event
//           const spawner = config.spawners?.[eventType]
//           if (spawner) {
//             const newItem = spawner(event as Timestamped<Extract<TEvent, { type: typeof eventType }>>)
//             instances.set(newItem.id, newItem)
//             pendingSignals.push({ type: 'change', item: newItem })
//             pendingSignals.push({ type: 'state', state: newItem._tag, item: newItem })
//           }
// 
//           // Check if this is an event handler
//           const handler = config.eventHandlers?.[eventType]
//           if (handler) {
//             const id = config.instanceId(event as Timestamped<TEvent>)
//             const existingItem = instances.get(id)
// 
//             if (existingItem) {
//               const previousTag = existingItem._tag
// 
//               // Call handler - returns new state
//               const newItem = handler(existingItem, event as Timestamped<Extract<TEvent, { type: typeof eventType }>>)
// 
//               // Only update if state changed
//               if (newItem !== existingItem) {
//                 instances.set(id, newItem)
//                 pendingSignals.push({ type: 'change', item: newItem })
// 
//                 // If tag changed, emit state signal
//                 if (newItem._tag !== previousTag) {
//                   pendingSignals.push({ type: 'state', state: newItem._tag, item: newItem })
//                 }
//               }
//             }
//           }
// 
//           // Update state if changed
//           if (instances.size !== currentState.instances.size || pendingSignals.length > 0) {
//             yield* SubscriptionRef.set(stateRef, { instances })
//           }
// 
//           // Emit signals
//           for (const signal of pendingSignals) {
//             if (signal.type === 'change') {
//               yield* emitChange(signal.item)
//             } else if (signal.type === 'state' && signal.state) {
//               yield* emitStateSignal(signal.state, signal.item)
//             }
//           }
//         })
// 
//       // Register with bus
//       if (allEventTypes.size > 0) {
//         yield* bus.register(eventHandler, [...allEventTypes], serviceName)
//       }
// 
//       return {
//         state: stateRef,
//         get: SubscriptionRef.get(stateRef),
//         getById: (id: string) => Effect.map(SubscriptionRef.get(stateRef), (s) => s.instances.get(id)),
//         getByState: (state: string) => Effect.map(SubscriptionRef.get(stateRef), (s) =>
//           Array.from(s.instances.values()).filter(item => item._tag === state)
//         ),
//         getAll: () => Effect.map(SubscriptionRef.get(stateRef), (s) => Array.from(s.instances.values()))
//       }
//     })
//   )
// 
//   // Combine layers
//   const FullLayer = Layer.provideMerge(ProjectionLayer, AllSignalLayers)
// 
//   // Build signals object
//   const signals = {
//     change: changeSignal,
//     ...Object.fromEntries(
//       config.fsm.states.map((state: StateNames<T>) => [state, stateSignalsMap.get(state)!])
//     )
//   } as FSMCollectionSignals<T, TState>
// 
//   return {
//     name: config.name,
//     Tag,
//     Layer: FullLayer,
//     signals,
//     changeSignal
//   }
// }
