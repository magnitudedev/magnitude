// /**
//  * Flow.define() - Main API for defining flows
//  *
//  * Creates a flow with:
//  * - Projection (FSM state via signals, not events)
//  * - Worker (single fiber per flow instance, runs linearly)
//  */

// import { Effect, Context, Layer, Ref, Fiber, Stream, PubSub, SubscriptionRef } from 'effect'
// import { WorkerBusTag, type WorkerBusService } from '../core/worker-bus'
// import { type ProjectionBusService } from '../core/projection-bus'
// import { HydrationContext } from '../core/hydration-context'
// import { type BaseEvent } from '../core/event-bus-core'
// import { type Signal, createSignal } from '../signal/define'
// import {
//   type FSMInstance,
//   type TransitionMatrix,
//   type StateNames,
//   type AnyStateClass,
//   type StateUnion,
//   type PropsOf,
//   type StateClassRecord,
//   type TransitionUpdates
// } from '../fsm/define'
// import {
//   type FlowState,
//   type FlowConfig,
//   type FlowStartedEvent,
//   type FlowEffectResultEvent,
//   type FlowStartedEventType,
//   type FlowEffectResultEventType,
//   type FlowEvents,
//   type FlowHandle,
//   type FlowContextService,
//   type WaitForOptions,
//   type FlowProjectionService
// } from './types'

// // ---------------------------------------------------------------------------
// // Flow.define()
// // ---------------------------------------------------------------------------

// export function define<
//   TName extends string,
//   T extends TransitionMatrix,
//   TClasses extends readonly AnyStateClass[],
//   TEvent extends BaseEvent,
//   TInput
// >(
//   config: FlowConfig<TName, T, TClasses, TEvent, TInput>
// ): FlowDefinitionResult<TName, T, TClasses, TEvent, TInput> {
//   type TState = StateUnion<TClasses> & { id: string }
//   type TFlowStartedEvent = FlowStartedEvent<TName, TState>
//   type TFlowEffectResultEvent = FlowEffectResultEvent<TName>

//   const { name, fsm, entry, run } = config

//   // Compute event type strings at runtime
//   const flowStartedType = `${name.toLowerCase()}_flow_started` as FlowStartedEventType<TName>
//   const flowEffectResultType = `${name.toLowerCase()}_flow_effect_result` as FlowEffectResultEventType<TName>

//   // ---------------------------------------------------------------------------
//   // Create Projection (FSM state management via signals)
//   // ---------------------------------------------------------------------------

//   const projectionName = `${name}Projection`
//   const ProjectionTag = Context.GenericTag<FlowProjectionInstance<TState>>(projectionName)

//   // Signal for state changes
//   const changeSignal = createSignal<TState>(`${name}/change`, name)

//   // Signals per FSM state
//   const stateSignals = new Map<StateNames<T>, Signal<TState>>()
//   for (const stateName of fsm.states) {
//     stateSignals.set(stateName, createSignal<TState>(`${name}/${stateName}`, name))
//   }

//   // Build signals object
//   const signals = {
//     change: changeSignal,
//     ...Object.fromEntries(
//       fsm.states.map((stateName) => [stateName, stateSignals.get(stateName)!])
//     )
//   } as FlowSignals<T, TState>

//   // Projection layer
//   const ProjectionLayer = Layer.scoped(
//     ProjectionTag,
//     Effect.gen(function* () {
//       const stateRef = yield* SubscriptionRef.make<Map<string, TState>>(new Map())

//       // Get hubs for signals
//       const changeHub = yield* changeSignal.tag
//       const stateHubs = new Map<StateNames<T>, PubSub.PubSub<TState>>()
//       for (const stateName of fsm.states) {
//         const signal = stateSignals.get(stateName)!
//         stateHubs.set(stateName, yield* signal.tag)
//       }

//       const updateState = (id: string, newState: TState) =>
//         Effect.gen(function* () {
//           const current = yield* SubscriptionRef.get(stateRef)
//           const prev = current.get(id)
//           const updated = new Map(current)
//           updated.set(id, newState)
//           yield* SubscriptionRef.set(stateRef, updated)

//           // Emit change signal
//           yield* PubSub.publish(changeHub, newState)

//           // Emit state-specific signal if state tag changed
//           if (!prev || prev._tag !== newState._tag) {
//             const hub = stateHubs.get(newState._tag as StateNames<T>)
//             if (hub) {
//               yield* PubSub.publish(hub, newState)
//             }
//           }
//         })

//       return {
//         state: stateRef,
//         getById: (id: string) =>
//           Effect.map(SubscriptionRef.get(stateRef), (m) => m.get(id)),
//         getAll: () =>
//           Effect.map(SubscriptionRef.get(stateRef), (m) => Array.from(m.values())),
//         getByState: (stateName: string) =>
//           Effect.map(SubscriptionRef.get(stateRef), (m) =>
//             Array.from(m.values()).filter((s) => s._tag === stateName)
//           ),
//         updateState
//       }
//     })
//   )

//   // Signal layers
//   const ChangeSignalLayer = Layer.scoped(changeSignal.tag, PubSub.unbounded<TState>())
//   const StateSignalLayers = fsm.states.map((stateName) => {
//     const signal = stateSignals.get(stateName)!
//     return Layer.scoped(signal.tag, PubSub.unbounded<TState>())
//   })

//   const AllSignalLayers = [ChangeSignalLayer, ...StateSignalLayers].reduce(
//     (acc, layer) => Layer.merge(acc, layer)
//   )

//   const FullProjectionLayer = Layer.provideMerge(ProjectionLayer, AllSignalLayers)

//   // ---------------------------------------------------------------------------
//   // Create Worker (manages fibers per flow instance)
//   // ---------------------------------------------------------------------------

//   const workerName = `${name}FlowWorker`
//   const WorkerTag = Context.GenericTag<void>(workerName)
//   const BusTag = WorkerBusTag<TEvent>()

//   type FiberMap = Map<string, Fiber.RuntimeFiber<void, unknown>>
//   type EffectHistory = Map<string, Map<number, unknown>>
//   type SpawnHistory = Map<string, Map<number, { childId: string; flowType: string }>>
//   type EventHistory = Map<string, TEvent[]>

//   const WorkerLayer = Layer.scoped(
//     WorkerTag,
//     Effect.gen(function* () {
//       const hydration = yield* HydrationContext
//       const isHydrating = yield* hydration.isHydrating()

//       if (isHydrating) {
//         return
//       }

//       const bus = yield* BusTag
//       const proj = yield* ProjectionTag
//       const runningFlows = yield* Ref.make<FiberMap>(new Map())

//       // History tracking for replay
//       const effectHistory = yield* Ref.make<EffectHistory>(new Map())
//       const spawnHistory = yield* Ref.make<SpawnHistory>(new Map())
//       const eventHistory = yield* Ref.make<EventHistory>(new Map())

//       // Listen for events to build history
//       // SAFETY: TEvent is generic, but user must include `typeof Flow.Events` in their union.
//       // We check event.type before narrowing, so cast through unknown is safe.
//       type TFlowEvent = (TFlowEffectResultEvent | TFlowStartedEvent) & { flowInstanceId?: string }
//       yield* Effect.forkScoped(
//         Stream.runForEach(
//           bus.stream,
//           (event) =>
//             Effect.gen(function* () {
//               const e = event as unknown as TFlowEvent

//               if (e.type === flowEffectResultType) {
//                 const eff = e as TFlowEffectResultEvent
//                 yield* Ref.update(effectHistory, (h) => {
//                   const m = new Map(h.get(eff.flowInstanceId) ?? [])
//                   m.set(eff.effectIndex, eff.result)
//                   return new Map(h).set(eff.flowInstanceId, m)
//                 })
//               }

//               if (e.type === flowStartedType) {
//                 const started = e as TFlowStartedEvent
//                 if (started.parent) {
//                   yield* Ref.update(spawnHistory, (h) => {
//                     const m = new Map(h.get(started.parent!.flowInstanceId) ?? [])
//                     m.set(started.parent!.spawnIndex, {
//                       childId: started.flowInstanceId,
//                       flowType: name
//                     })
//                     return new Map(h).set(started.parent!.flowInstanceId, m)
//                   })
//                 }
//               }

//               const flowInstanceId = e.flowInstanceId
//               if (flowInstanceId) {
//                 yield* Ref.update(eventHistory, (h) => {
//                   const events = [...(h.get(flowInstanceId) ?? []), event]
//                   return new Map(h).set(flowInstanceId, events)
//                 })
//               }
//             })
//         )
//       )

//       // Listen for flow_started events to spawn fibers
//       yield* Effect.forkScoped(
//         Stream.runForEach(
//           Stream.filter(
//             bus.stream,
//             (e): e is TEvent & TFlowStartedEvent => e.type === flowStartedType
//           ),
//           (event) =>
//             Effect.gen(function* () {
//               const flowId = event.flowInstanceId
//               const initialState = event.initialState

//               // Set initial state in projection
//               yield* proj.updateState(flowId, initialState)

//               // Load history for this instance
//               const effects = (yield* Ref.get(effectHistory)).get(flowId) ?? new Map()
//               const spawns = (yield* Ref.get(spawnHistory)).get(flowId) ?? new Map()
//               const events = (yield* Ref.get(eventHistory)).get(flowId) ?? []

//               // Create current state ref for this flow
//               const currentStateRef = yield* Ref.make<TState>(initialState)

//               // Create context
//               const ctx = createFlowCtx<TName, TEvent, T, TClasses>({
//                 flowName: name,
//                 flowInstanceId: flowId,
//                 flowEffectResultType,
//                 bus,
//                 fsm,
//                 proj,
//                 currentStateRef,
//                 effectResults: effects,
//                 spawnedChildren: spawns,
//                 receivedEvents: events
//               })

//               // Fork fiber for this flow
//               const fiber = yield* Effect.forkScoped(
//                 run(initialState, ctx).pipe(
//                   Effect.catchAllCause((cause) =>
//                     Effect.logError(`Flow ${name}:${flowId} failed`, cause).pipe(
//                       Effect.andThen(Effect.die(cause))
//                     )
//                   )
//                 )
//               )

//               yield* Ref.update(runningFlows, (m) => new Map(m).set(flowId, fiber))
//             })
//         )
//       )
//     })
//   )

//   // ---------------------------------------------------------------------------
//   // Combine Layers
//   // ---------------------------------------------------------------------------

//   const CombinedLayer = Layer.provideMerge(WorkerLayer, FullProjectionLayer)

//   // ---------------------------------------------------------------------------
//   // Return
//   // ---------------------------------------------------------------------------

//   return {
//     name,
//     fsm,
//     entry,
//     run,
//     flowStartedType,
//     flowEffectResultType,
//     // Phantom value for `typeof OAuthFlow.Events`
//     Events: null as unknown as FlowEvents<TName, TState>,
//     ProjectionTag,
//     WorkerTag,
//     Layer: CombinedLayer,
//     signals,
//     changeSignal,

//     start: (id: string, input: TInput) =>
//       Effect.gen(function* () {
//         const bus = yield* BusTag
//         const initialState = entry(id, input)

//         // SAFETY: This is a FlowStartedEvent which is part of FlowEvents<TName>.
//         // TS can't verify TEvent includes FlowEvents because TEvent is an unconstrained generic.
//         // The user's event union must include `typeof Flow.Events` for the system to work.
//         yield* bus.publish({
//           type: flowStartedType,
//           flowInstanceId: id,
//           initialState
//         } as unknown as TEvent)

//         return id
//       })
//   }
// }

// // ---------------------------------------------------------------------------
// // Flow Context Factory
// // ---------------------------------------------------------------------------

// interface FlowCtxInput<
//   TName extends string,
//   TEvent extends BaseEvent,
//   T extends TransitionMatrix,
//   TClasses extends readonly AnyStateClass[]
// > {
//   flowName: TName
//   flowInstanceId: string
//   flowEffectResultType: FlowEffectResultEventType<TName>
//   bus: WorkerBusService<TEvent>
//   fsm: FSMInstance<T, TClasses>
//   proj: FlowProjectionInstance<StateUnion<TClasses> & { id: string }>
//   currentStateRef: Ref.Ref<StateUnion<TClasses> & { id: string }>
//   effectResults: Map<number, unknown>
//   spawnedChildren: Map<number, { childId: string; flowType: string }>
//   receivedEvents: TEvent[]
// }

// function createFlowCtx<
//   TName extends string,
//   TEvent extends BaseEvent,
//   T extends TransitionMatrix,
//   TClasses extends readonly AnyStateClass[]
// >(
//   input: FlowCtxInput<TName, TEvent, T, TClasses>
// ): FlowContextService<TEvent, T, TClasses> {
//   const {
//     flowInstanceId,
//     flowEffectResultType,
//     bus,
//     fsm,
//     proj,
//     currentStateRef,
//     effectResults,
//     spawnedChildren,
//     receivedEvents
//   } = input

//   let effectIndex = 0
//   let spawnIndex = 0
//   let waitForIndex = 0

//   return {
//     effect: <TResult, R>(eff: Effect.Effect<TResult, never, R>) =>
//       Effect.gen(function* () {
//         const idx = effectIndex++

//         if (effectResults.has(idx)) {
//           return effectResults.get(idx) as TResult
//         }

//         const result = yield* eff

//         // SAFETY: This is a FlowEffectResultEvent which is part of FlowEvents<TName>.
//         // TS can't verify TEvent includes FlowEvents because TEvent is an unconstrained generic.
//         // The user's event union must include `typeof Flow.Events` for the system to work.
//         yield* bus.publish({
//           type: flowEffectResultType,
//           flowInstanceId,
//           effectIndex: idx,
//           result
//         } as unknown as TEvent)

//         return result
//       }),

//     spawn: <TChildName extends string, TChildInput>(
//       flow: {
//         readonly name: TChildName
//         readonly flowStartedType: FlowStartedEventType<TChildName>
//         readonly entry: (id: string, input: TChildInput) => FlowState
//       },
//       childInput: TChildInput
//     ) =>
//       Effect.gen(function* () {
//         const idx = spawnIndex++

//         const existing = spawnedChildren.get(idx)
//         const terminalStates = fsm.getTerminalStates()

//         if (existing) {
//           return {
//             id: existing.childId,
//             await: () => waitForTerminal(existing.childId, proj, terminalStates),
//             state: proj.getById(existing.childId)
//           } satisfies FlowHandle<FlowState>
//         }

//         const childId = `${flowInstanceId}:child:${idx}`
//         const initialState = flow.entry(childId, childInput)

//         // SAFETY: This is the child flow's FlowStartedEvent.
//         // TS can't verify TEvent includes the child's events because TEvent is an unconstrained generic.
//         // The user's event union must include child flow events for spawning to work.
//         yield* bus.publish({
//           type: flow.flowStartedType,
//           flowInstanceId: childId,
//           initialState,
//           parent: {
//             flowInstanceId,
//             spawnIndex: idx
//           }
//         } as unknown as TEvent)

//         return {
//           id: childId,
//           await: () => waitForTerminal(childId, proj, terminalStates),
//           state: proj.getById(childId)
//         } satisfies FlowHandle<FlowState>
//       }),

//     waitFor: <TEventTypes extends readonly TEvent['type'][]>(
//       eventTypes: TEventTypes,
//       options?: WaitForOptions<Extract<TEvent, { type: TEventTypes[number] }>>
//     ) =>
//       Effect.gen(function* () {
//         type TMatchedEvent = Extract<TEvent, { type: TEventTypes[number] }>
//         const typeSet = new Set<string>(eventTypes)

//         const defaultMatch = (e: TMatchedEvent): boolean =>
//           'flowInstanceId' in e &&
//           (e as { flowInstanceId: string }).flowInstanceId === flowInstanceId

//         const match = options?.match ?? defaultMatch

//         // Check history
//         for (let i = waitForIndex; i < receivedEvents.length; i++) {
//           const event = receivedEvents[i]
//           if (typeSet.has(event.type)) {
//             const matched = event as TMatchedEvent
//             if (match(matched)) {
//               waitForIndex = i + 1
//               return matched
//             }
//           }
//         }

//         // Subscribe for new events
//         const stream = bus.subscribeToTypes(eventTypes as readonly TEvent['type'][])

//         const filtered = Stream.filter(stream, (e): e is TMatchedEvent =>
//           typeSet.has(e.type) && match(e as TMatchedEvent)
//         )

//         const result = yield* Stream.runHead(filtered)

//         if (result._tag === 'None') {
//           return yield* Effect.die(
//             new Error(`waitFor: stream ended without match for ${flowInstanceId}`)
//           )
//         }

//         return result.value
//       }),

//     transition: <
//       From extends StateUnion<TClasses>,
//       To extends StateNames<T> & keyof StateClassRecord<TClasses>
//     >(
//       from: From,
//       targetState: To,
//       data: TransitionUpdates<From, PropsOf<StateClassRecord<TClasses>[To]>>
//     ) =>
//       Effect.gen(function* () {
//         type TState = StateUnion<TClasses> & { id: string }

//         // fsm.transition spreads current instance first, so id is preserved
//         const newState = fsm.transition(from, targetState, data) as TState

//         yield* Ref.set(currentStateRef, newState)
//         yield* proj.updateState(flowInstanceId, newState)
//       }),

//     emit: (event: TEvent) => bus.publish(event)
//   }
// }

// function waitForTerminal<TState extends FlowState>(
//   id: string,
//   proj: FlowProjectionInstance<TState>,
//   terminalStates: readonly string[]
// ): Effect.Effect<TState> {
//   return Effect.gen(function* () {
//     const current = yield* proj.getById(id)
//     const terminalSet = new Set(terminalStates)

//     if (current && terminalSet.has(current._tag)) {
//       return current
//     }

//     const stream = proj.state.changes.pipe(
//       Stream.map((m: Map<string, TState>) => m.get(id)),
//       Stream.filter((s): s is TState => s !== undefined && terminalSet.has(s._tag))
//     )

//     const result = yield* Stream.runHead(stream)

//     if (result._tag === 'None') {
//       return yield* Effect.die(new Error(`Child ${id} ended without terminal state`))
//     }

//     return result.value
//   })
// }

// // ---------------------------------------------------------------------------
// // Types
// // ---------------------------------------------------------------------------

// interface FlowProjectionInstance<TState extends FlowState> {
//   state: SubscriptionRef.SubscriptionRef<Map<string, TState>>
//   getById: (id: string) => Effect.Effect<TState | undefined>
//   getAll: () => Effect.Effect<TState[]>
//   getByState: (state: string) => Effect.Effect<TState[]>
//   updateState: (id: string, state: TState) => Effect.Effect<void>
// }

// type FlowSignals<T extends TransitionMatrix, TState extends FlowState> = {
//   change: Signal<TState>
// } & {
//   [K in StateNames<T>]: Signal<TState>
// }

// /**
//  * Extract PubSub types from flow signals for Layer output
//  */
// type FlowSignalPubSubs<T extends TransitionMatrix, TState extends FlowState> =
//   | PubSub.PubSub<TState>  // change signal
//   | { [K in StateNames<T>]: PubSub.PubSub<TState> }[StateNames<T>]  // per-state signals

// export interface FlowDefinitionResult<
//   TName extends string,
//   T extends TransitionMatrix,
//   TClasses extends readonly AnyStateClass[],
//   TEvent extends BaseEvent,
//   TInput
// > extends FlowConfig<TName, T, TClasses, TEvent, TInput> {
//   /** Event type for flow_started - used by spawn */
//   readonly flowStartedType: FlowStartedEventType<TName>
//   /** Event type for flow_effect_result */
//   readonly flowEffectResultType: FlowEffectResultEventType<TName>
//   /** All events this flow emits - include in your event union */
//   readonly Events: FlowEvents<TName, StateUnion<TClasses> & { id: string }>
//   readonly ProjectionTag: Context.Tag<
//     FlowProjectionInstance<StateUnion<TClasses> & { id: string }>,
//     FlowProjectionInstance<StateUnion<TClasses> & { id: string }>
//   >
//   readonly WorkerTag: Context.Tag<void, void>
//   readonly Layer: Layer.Layer<
//     FlowProjectionInstance<StateUnion<TClasses> & { id: string }> | void | FlowSignalPubSubs<T, StateUnion<TClasses> & { id: string }>,
//     never,
//     WorkerBusService<TEvent> | HydrationContext
//   >
//   readonly signals: FlowSignals<T, StateUnion<TClasses> & { id: string }>
//   readonly changeSignal: Signal<StateUnion<TClasses> & { id: string }>
//   readonly start: (
//     id: string,
//     input: TInput
//   ) => Effect.Effect<string, never, WorkerBusService<TEvent>>
// }

// export const Flow = {
//   define
// }
