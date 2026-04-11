/**
 * Worker.defineForked() - Worker with per-fork concurrent fibers
 *
 * Mirrors Projection.defineForked but for workers:
 * - Spawns a separate fiber for each fork (concurrent execution)
 * - Automatically manages fiber lifecycle (spawn on activate, interrupt on complete)
 * - Per-fork interrupt isolation (completing one fork doesn't affect others)
 * - Root fork (forkId = null) always runs, never interrupted by fork lifecycle
 *
 * Events must have a `forkId: string | null` field.
 *
 * The consumer specifies which event type represents fork activation via
 * `forkLifecycle.activateOn`. Optional `forkLifecycle.completeOn` can be provided
 * to interrupt and remove a fork fiber when a terminal event occurs.
 */

import { Effect, Context, Layer, Stream, PubSub, Queue, Ref, Fiber, Cause } from 'effect'
import { WorkerBusTag, type WorkerBusService } from '../core/worker-bus'
import { HydrationContext } from '../core/hydration-context'
import { InterruptCoordinator } from '../core/interrupt-coordinator'
import { extractForkIdFromEvent, extractForkIdFromSignal } from './util'
import { type BaseEvent, type Timestamped } from '../core/event-bus-core'
import { Signal, type SignalValue } from '../signal/define'
import type { ProjectionInstance, ProjectionResult } from '../projection/define'
import type { ForkedProjectionInstance, ForkableEvent, ForkedProjectionResult } from '../projection/defineForked'
import type { PublishFn, WorkerReadFn, WorkerSignalHandlerBuilder, WorkerSignalHandlerPair } from './define'
import { FrameworkErrorReporter, FrameworkError, type FrameworkErrorReporterService } from '../core/framework-error'


// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Event handler for forked workers.
 * Same signature as regular worker handlers.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ForkedWorkerEventHandler<TEvent extends BaseEvent, E extends TEvent['type'], R = any> = (
  event: Timestamped<Extract<TEvent, { type: E }>>,
  publish: PublishFn<TEvent>,
  read: WorkerReadFn<TEvent>
) => Effect.Effect<void, never, R>

/**
 * Event handlers record for forked workers.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ForkedWorkerEventHandlers<TEvent extends BaseEvent, R = any> = {
  [E in TEvent['type']]?: ForkedWorkerEventHandler<TEvent, E, R>
}

/**
 * Extract Effect requirements from event handlers.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ExtractHandlerRequirements<THandlers> = THandlers extends ForkedWorkerEventHandlers<any, infer R> ? R : never

/**
 * Fork lifecycle configuration.
 * Tells the framework which event types to use for fork spawn/teardown.
 */
export interface ForkLifecycle<TEvent extends BaseEvent> {
  /** Event type that triggers spawning a new fork fiber */
  readonly activateOn: TEvent['type']
  /** Optional event type(s) that trigger interrupting a fork fiber */
  readonly completeOn?: TEvent['type'] | readonly TEvent['type'][]
}

/**
 * Forked worker config.
 */
export interface ForkedWorkerConfig<
  TName extends string,
  TEvent extends BaseEvent,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  THandlers extends ForkedWorkerEventHandlers<TEvent> = ForkedWorkerEventHandlers<TEvent, any>
> {
  readonly name: TName

  /** Fork lifecycle - which events spawn/teardown fork fibers */
  readonly forkLifecycle: ForkLifecycle<TEvent>

  /** Event handlers - dispatched to the correct fork's fiber */
  readonly eventHandlers?: THandlers

  /**
   * Event types that should NOT be automatically interrupted.
   */
  readonly ignoreInterrupt?: readonly TEvent['type'][]

  /**
   * Signal handlers - subscribe to signals from projections.
   * Use the builder pattern: `signalHandlers: (on) => [on(signal, handler)]`
   */
  readonly signalHandlers?: (
    on: WorkerSignalHandlerBuilder<TEvent>
  ) => readonly WorkerSignalHandlerPair<TEvent>[]
}

export interface ForkedWorkerResult<
  TEvent extends BaseEvent,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  THandlers extends ForkedWorkerEventHandlers<TEvent> = ForkedWorkerEventHandlers<TEvent, any>
> {
  readonly Tag: Context.Tag<void, void>
  readonly Layer: Layer.Layer<
    void,
    never,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ExtractHandlerRequirements<THandlers> | WorkerBusService<TEvent> | HydrationContext | PubSub.PubSub<any> | FrameworkErrorReporterService
  >
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

interface ForkFiber<TEvent> {
  readonly fiber: Fiber.RuntimeFiber<void, never>
  readonly queue: Queue.Queue<TEvent>
}

// ---------------------------------------------------------------------------
// Read helper (same as Worker.define)
// ---------------------------------------------------------------------------

function makeWorkerReadFn<TEvent extends BaseEvent>(
  forkId: string | null
): WorkerReadFn<TEvent> {
  const impl = ((
    projection: ProjectionResult<any, unknown, any, any> | ForkedProjectionResult<any, unknown, any, any>,
    overrideForkId?: string | null
  ) => {
    const targetForkId = overrideForkId !== undefined ? overrideForkId : forkId
    if (projection.isForked) {
      return Effect.flatMap(
        projection.Tag,
        (instance: ForkedProjectionInstance<unknown>) => instance.getFork(targetForkId)
      )
    } else {
      return Effect.flatMap(
        projection.Tag,
        (instance: ProjectionInstance<unknown>) => instance.get
      )
    }
  }) as WorkerReadFn<TEvent>

  impl.allForks = ((<TForkState>(projection: ForkedProjectionResult<any, TForkState, any, any>) =>
    Effect.flatMap(
      projection.Tag,
      (instance: ForkedProjectionInstance<TForkState>) => instance.getAllForks()
    )) as WorkerReadFn<TEvent>['allForks'])

  return impl
}

// ---------------------------------------------------------------------------
// defineForked
// ---------------------------------------------------------------------------

export function defineForked<TEvent extends ForkableEvent>() {
  return <
    TName extends string,
    THandlers extends ForkedWorkerEventHandlers<TEvent>
  >(
    config: ForkedWorkerConfig<TName, TEvent, THandlers>
  ): ForkedWorkerResult<TEvent, THandlers> => {
    const serviceName = `${config.name}ForkedWorker`
    const Tag = Context.GenericTag<void>(serviceName)
    const BusTag = WorkerBusTag<TEvent>()

    const Live = Layer.scoped(Tag, Effect.gen(function* () {
      const bus = yield* BusTag
      const hydration = yield* HydrationContext
      const interruptCoordinator = yield* InterruptCoordinator
      const reporter = yield* FrameworkErrorReporter

      if (yield* hydration.isHydrating()) {
        return
      }

      const publish: PublishFn<TEvent> = (event) => bus.publish(event)

      const withInterrupt = <A, RH>(
        handler: Effect.Effect<A, never, RH>,
        targetForkId: string | null
      ) =>
        Effect.gen(function* () {
          const baseline = yield* interruptCoordinator.current(targetForkId)
          return yield* Effect.raceFirst(
            handler,
            interruptCoordinator.waitForInterrupt(targetForkId, baseline)
          )
        })

      // Track active fork fibers: forkId → ForkFiber
      const forkFibers = yield* Ref.make<Map<string | null, ForkFiber<TEvent>>>(new Map())

      // Collect event types we care about (handler types + lifecycle types)
      const handlerEventTypes = config.eventHandlers
        ? new Set(Object.keys(config.eventHandlers) as TEvent['type'][])
        : new Set<TEvent['type']>()

      const ignoreInterrupt = config.ignoreInterrupt ?? []

      // All event types we need from the bus
      const allEventTypes = new Set<TEvent['type']>([
        ...handlerEventTypes,
        config.forkLifecycle.activateOn,
      ])
      const completeOnTypes = config.forkLifecycle.completeOn === undefined
        ? []
        : Array.isArray(config.forkLifecycle.completeOn)
          ? config.forkLifecycle.completeOn
          : [config.forkLifecycle.completeOn]
      for (const completeType of completeOnTypes) {
        allEventTypes.add(completeType)
      }

      // -----------------------------------------------------------------------
      // Per-fork fiber spawner
      // -----------------------------------------------------------------------
      // Creates a queue + fiber for a fork. The queue is created synchronously
      // so events can be pushed to it immediately. The fiber consumes from
      // the queue asynchronously.

      const spawnForkFiber = (forkId: string | null) =>
        Effect.gen(function* () {
          const forkQueue = yield* Queue.unbounded<TEvent>()
          if (forkId !== null) {
            yield* interruptCoordinator.beginExecution(forkId)
          }

          const read = makeWorkerReadFn<TEvent>(forkId)

          const fiber = yield* Effect.forkScoped(
            Stream.runForEach(
              Stream.fromQueue(forkQueue),
              (event) => {
                const handler = config.eventHandlers![event.type as TEvent['type']]
                if (!handler) return Effect.void

                const handlerEffect = handler(
                  event as Timestamped<Extract<TEvent, { type: typeof event.type }>>,
                  publish,
                  read
                )

                const withErrorBoundary = (effect: Effect.Effect<unknown, never, unknown>) =>
                  effect.pipe(
                    Effect.catchAllCause((cause) => {
                      if (Cause.isInterruptedOnly(cause)) return Effect.void
                      return reporter.report(FrameworkError.WorkerEventHandlerError({
                        workerName: config.name,
                        eventType: event.type,
                        cause
                      }))
                    })
                  )

                if (ignoreInterrupt.includes(event.type)) {
                  return withErrorBoundary(handlerEffect)
                }

                return withErrorBoundary(withInterrupt(handlerEffect, forkId))
              }
            )
          )

          // beginExecution is fork-lifecycle-scoped, but interrupt baselines are
          // invocation-scoped. The root fork is long-lived, so each handler run must
          // read the current baseline immediately before racing against interrupts.
          const forkFiberEntry: ForkFiber<TEvent> = { fiber, queue: forkQueue }
          yield* Ref.update(forkFibers, (m) => new Map(m).set(forkId, forkFiberEntry))
          return forkFiberEntry
        })

      // -----------------------------------------------------------------------
      // Spawn root fork fiber (always running)
      // -----------------------------------------------------------------------

      yield* spawnForkFiber(null)

      // -----------------------------------------------------------------------
      // Single dispatch loop
      // -----------------------------------------------------------------------
      // One ordered stream from the bus. Handles lifecycle (spawn/complete)
      // and dispatches handler events to the correct fork's queue.
      // Because it's a single sequential loop, the fork's queue is guaranteed
      // to exist before subsequent events (like turn_started) are dispatched.

      // -----------------------------------------------------------------------
      // Signal handlers
      // -----------------------------------------------------------------------

      if (config.signalHandlers) {
        const on: WorkerSignalHandlerBuilder<TEvent> = (signal, handler) => ({
          signal,
          handler
        })

        const handlerPairs = config.signalHandlers(on)

        for (const { signal, handler } of handlerPairs) {
          const pubsub = yield* signal.tag
          yield* Effect.forkScoped(
            Stream.runForEach(
              Stream.fromPubSub(pubsub),
              (value) => Effect.gen(function* () {
                if (yield* hydration.isHydrating()) return

                const signalForkId = extractForkIdFromSignal(value)
                const read = makeWorkerReadFn<TEvent>(signalForkId)
                yield* withInterrupt(handler(value, publish, read), signalForkId)
              }).pipe(
                Effect.catchAllCause((cause) => {
                  if (Cause.isInterruptedOnly(cause)) return Effect.void
                  return reporter.report(FrameworkError.WorkerSignalHandlerError({
                    workerName: config.name,
                    signalName: signal.name,
                    cause
                  }))
                })
              )
            )
          )
        }
      }

      // -----------------------------------------------------------------------
      // Single dispatch loop
      // -----------------------------------------------------------------------

      yield* Effect.forkScoped(
        Stream.runForEach(
          bus.subscribeToTypes([...allEventTypes]),
          (event) => Effect.gen(function* () {
            const forkId = extractForkIdFromEvent(event)

            // --- Lifecycle: activate ---
            if (event.type === config.forkLifecycle.activateOn) {
              if (forkId !== null) {
                const existing = yield* Ref.get(forkFibers)
                if (!existing.has(forkId)) {
                  yield* spawnForkFiber(forkId)
                }
              }
            }

            // --- Lifecycle: complete ---
            if (completeOnTypes.includes(event.type as TEvent['type'])) {
              if (forkId !== null) {
                const fibers = yield* Ref.get(forkFibers)
                const forkFiber = fibers.get(forkId)
                if (forkFiber) {
                  yield* Fiber.interrupt(forkFiber.fiber)
                  yield* Ref.update(forkFibers, (m) => {
                    const newMap = new Map(m)
                    newMap.delete(forkId)
                    return newMap
                  })
                }
              }
            }

            // --- Dispatch to fork's queue ---
            if (handlerEventTypes.has(event.type as TEvent['type'])) {
              const fibers = yield* Ref.get(forkFibers)
              const forkFiber = fibers.get(forkId)
              if (forkFiber) {
                yield* Queue.offer(forkFiber.queue, event)
              }
            }
          }).pipe(
            Effect.catchAllCause((cause) => {
              if (Cause.isInterruptedOnly(cause)) return Effect.void
              return reporter.report(FrameworkError.WorkerLifecycleError({
                workerName: config.name,
                eventType: event.type,
                cause
              }))
            })
          )
        )
      )
    }))

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    type LayerInput = ExtractHandlerRequirements<THandlers> | WorkerBusService<TEvent> | HydrationContext | PubSub.PubSub<any> | FrameworkErrorReporterService

    return {
      Tag,
      Layer: Live as Layer.Layer<void, never, LayerInput>
    }
  }
}
