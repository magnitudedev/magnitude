/**
 * sage-core Public API
 *
 * Core Concepts:
 * - Event: Persisted fact, replayed during hydration
 * - Signal: Ephemeral notification, NOT replayed
 * - FSM: Type-safe state machine with Data.TaggedClass
 * - Projection: Stateful handler for events + signals
 * - Worker: Stateless effect handler for events or signals
 * - Agent: Composition of projections and workers
 */

// Core types
export * from './types'
export { HydrationContext } from './core/hydration-context'
export { EventSinkTag, makeEventSinkLayer, type EventSinkService } from './core/event-sink'
export { InterruptCoordinator, InterruptCoordinatorLive, type InterruptBaseline, type InterruptCoordinator as InterruptCoordinatorService } from './core/interrupt-coordinator'
export { ProjectionBusTag, makeProjectionBusLayer, type ProjectionBusService } from './core/projection-bus'
export { WorkerBusTag, makeWorkerBusLayer, type WorkerBusService } from './core/worker-bus'
export { EventBusCoreTag, makeEventBusCoreLayer, type EventBusCoreService, type BaseEvent, type Timestamped } from './core/event-bus-core'
export {
  FrameworkError,
  FrameworkErrorPubSub,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporter,
  FrameworkErrorReporterLive,
  type FrameworkErrorReporterService
} from './core/framework-error'

// Main API modules
export * as Signal from './signal/index'
export * as FSM from './fsm/index'
export * as Projection from './projection/index'
export * as Worker from './worker/index'
export * as Agent from './agent/index'
export * as Display from './display/index'
// Flow is unused legacy — commented out
// export * as Flow from './flow/index'
export * as Fork from './fork/index'

// Convenience exports
export { type PublishFn, type WorkerReadFn } from './worker/index'
