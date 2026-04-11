/**
 * Worker.define() - Event and Signal Triggered Workers
 *
 * Workers are async handlers that run in separate fibers.
 * They can read projection state via the `read` helper.
 */

import { Effect, Context, Layer, Stream, PubSub, Cause } from 'effect'
import { WorkerBusTag, type WorkerBusService } from '../core/worker-bus'
import { HydrationContext } from '../core/hydration-context'
import { InterruptCoordinator } from '../core/interrupt-coordinator'
import { extractForkIdFromEvent, extractForkIdFromSignal } from './util'
import { type BaseEvent, type Timestamped } from '../core/event-bus-core'
import { Signal, type SignalValue } from '../signal/define'
import type { ProjectionInstance, ProjectionResult } from '../projection/define'
import type { ForkedProjectionInstance, ForkableEvent, ForkedProjectionResult } from '../projection/defineForked'
import { FrameworkErrorReporter, FrameworkError, type FrameworkErrorReporterService } from '../core/framework-error'


// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type PublishFn<E extends BaseEvent> = (event: E) => Effect.Effect<void>

// ---------------------------------------------------------------------------
// Read Helper Types
// ---------------------------------------------------------------------------

/**
 * Worker read function - returns Effect that resolves to projection state.
 * For forked projections, automatically resolves to the forkId from the event.
 *
 * Uses `any` for acceptance (projection parameter), StateOfProjection for extraction.
 */
export type WorkerReadFn<TEvent extends BaseEvent> = {
  <TState>(projection: ProjectionResult<any, TState, any, any>): Effect.Effect<TState>
  <TForkState>(projection: ForkedProjectionResult<any, TForkState, any, any, any>): Effect.Effect<TForkState>
  <TForkState>(projection: ForkedProjectionResult<any, TForkState, any, any, any>, forkId: string | null): Effect.Effect<TForkState>
  allForks: <TForkState>(projection: ForkedProjectionResult<any, TForkState, any, any, any>) => Effect.Effect<Map<string | null, TForkState>>
}

// ---------------------------------------------------------------------------
// Signal Handler Builder Types
// ---------------------------------------------------------------------------

/**
 * Builder function type for creating signal handlers with full type inference.
 * The `on` function infers TSignal from the signal argument and uses it to type the handler.
 */
export type WorkerSignalHandlerBuilder<TEvent extends BaseEvent> =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  <TSignal extends Signal<any, any>, R = never>(
    signal: TSignal,
    handler: (
      value: SignalValue<TSignal>,
      publish: PublishFn<TEvent>,
      read: WorkerReadFn<TEvent>
    ) => Effect.Effect<void, never, R>
  ) => { signal: TSignal; handler: typeof handler }

/**
 * Return type of signal handler builder - the { signal, handler } pair
 */
export type WorkerSignalHandlerPair<TEvent extends BaseEvent> =
  ReturnType<WorkerSignalHandlerBuilder<TEvent>>

/**
 * Worker event handler type - typed parameters with Effect return.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type WorkerEventHandler<TEvent extends BaseEvent, E extends TEvent['type'], R = any> = (
  event: Timestamped<Extract<TEvent, { type: E }>>,
  publish: PublishFn<TEvent>,
  read: WorkerReadFn<TEvent>
) => Effect.Effect<void, never, R>

/**
 * Worker event handlers record type.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type WorkerEventHandlers<TEvent extends BaseEvent, R = any> = {
  [E in TEvent['type']]?: WorkerEventHandler<TEvent, E, R>
}

/**
 * Extract Effect requirements from event handlers.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ExtractHandlerRequirements<THandlers> = THandlers extends WorkerEventHandlers<any, infer R> ? R : never

type ExtractSettledRequirements<TSettled> =
  TSettled extends ((...args: any[]) => Effect.Effect<any, any, infer R>) ? R : never

/**
 * Worker config with properly typed event handlers.
 */
export interface WorkerConfig<
  TName extends string,
  TEvent extends BaseEvent,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  THandlers extends WorkerEventHandlers<TEvent> = WorkerEventHandlers<TEvent, any>,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSettledHandler extends (ctx: { publish: PublishFn<TEvent>; read: WorkerReadFn<TEvent> }) => Effect.Effect<void, never, any> = (ctx: { publish: PublishFn<TEvent>; read: WorkerReadFn<TEvent> }) => Effect.Effect<void>
> {
  readonly name: TName

  /** Event handlers - typed with event, publish, and read parameters */
  readonly eventHandlers?: THandlers

  /**
   * Event types that should NOT be automatically interrupted.
   * Useful for short atomic handlers or handlers that manage interrupt themselves.
   */
  readonly ignoreInterrupt?: readonly TEvent['type'][]

  /**
   * Signal handlers - subscribe to signals from projections.
   * Use the builder pattern: `signalHandlers: (on) => [on(signal, handler)]`
   */
  readonly signalHandlers?: (
    on: WorkerSignalHandlerBuilder<TEvent>
  ) => readonly WorkerSignalHandlerPair<TEvent>[]

  readonly onProjectionsSettled?: TSettledHandler
}

export interface WorkerResult<
  TEvent extends BaseEvent,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  THandlers extends WorkerEventHandlers<TEvent> = WorkerEventHandlers<TEvent, any>,
  TSettledHandler = undefined
> {
  readonly Tag: Context.Tag<void, void>
  readonly Layer: Layer.Layer<
    void,
    never,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ExtractHandlerRequirements<THandlers> | ExtractSettledRequirements<TSettledHandler> | WorkerBusService<TEvent> | HydrationContext | PubSub.PubSub<any> | FrameworkErrorReporterService
  >
}

/**
 * Create a read function for a specific forkId context.
 * For forked projections, resolves to the specified fork.
 * For standard projections, returns the global state.
 *
 * Implementation accepts any and returns any - the WorkerReadFn type
 * provides the correct generic signature for callers.
 */
function makeWorkerReadFn<TEvent extends BaseEvent>(forkId: string | null): WorkerReadFn<TEvent> {
  const impl = ((
    projection: ProjectionResult<any, unknown, any, any, any> | ForkedProjectionResult<any, unknown, any, any, any>,
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

  impl.allForks = ((<TForkState>(projection: ForkedProjectionResult<any, TForkState, any, any, any>) =>
    Effect.flatMap(
      projection.Tag,
      (instance: ForkedProjectionInstance<TForkState>) => instance.getAllForks()
    )) as WorkerReadFn<TEvent>['allForks'])

  return impl
}

export function define<TEvent extends BaseEvent>() {
  return <
    TName extends string,
    THandlers extends WorkerEventHandlers<TEvent> = WorkerEventHandlers<TEvent, any>,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    TSettledHandler extends (ctx: { publish: PublishFn<TEvent>; read: WorkerReadFn<TEvent> }) => Effect.Effect<void, never, any> = (ctx: { publish: PublishFn<TEvent>; read: WorkerReadFn<TEvent> }) => Effect.Effect<void>
  >(
    config: WorkerConfig<TName, TEvent, THandlers, TSettledHandler>
  ): WorkerResult<TEvent, THandlers, TSettledHandler> => {
    const serviceName = `${config.name}Worker`
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

      const withInterrupt = <A, RH>(handler: Effect.Effect<A, never, RH>, targetForkId: string | null) =>
        Effect.gen(function* () {
          const baseline = yield* interruptCoordinator.current(targetForkId)
          return yield* Effect.raceFirst(
            handler,
            interruptCoordinator.waitForInterrupt(targetForkId, baseline)
          )
        })

      // Set up event handlers
      if (config.eventHandlers) {
        const eventTypes = Object.keys(config.eventHandlers) as TEvent['type'][]
        const ignoreInterrupt = config.ignoreInterrupt ?? []

        if (eventTypes.length > 0) {
          yield* Effect.forkScoped(
            Stream.runForEach(
              bus.subscribeToTypes(eventTypes),
              (event) => {
                const handler = config.eventHandlers![event.type as TEvent['type']]
                if (handler) {
                  // Create read function with forkId from event (if available)
                  const forkId = extractForkIdFromEvent(event)
                  const read = makeWorkerReadFn<TEvent>(forkId)

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

                  // Skip interrupt wrapping for ignored event types, but always apply error boundary
                  if (ignoreInterrupt.includes(event.type)) {
                    return withErrorBoundary(handlerEffect)
                  }
                  return withErrorBoundary(withInterrupt(handlerEffect, forkId))
                }
                return Effect.void
              }
            )
          )
        }
      }

      if (config.onProjectionsSettled) {
        const settledRead = makeWorkerReadFn<TEvent>(null)
        const settledStream = yield* bus.subscribe()

        yield* Effect.forkScoped(
          Stream.runForEach(settledStream, () =>
            Effect.gen(function* () {
              yield* config.onProjectionsSettled!({ publish, read: settledRead })
            }).pipe(
              Effect.catchAllCause((cause) => {
                if (Cause.isInterruptedOnly(cause)) return Effect.void
                return reporter.report(FrameworkError.WorkerSettledHandlerError({
                  workerName: config.name,
                  cause
                }))
              })
            )
          )
        )
      }

      // Set up signal handlers
      if (config.signalHandlers) {
        // Create the `on` builder function
        const on: WorkerSignalHandlerBuilder<TEvent> = (signal, handler) => ({
          signal,
          handler
        })

        // Call the builder to get the handler pairs
        const handlerPairs = config.signalHandlers(on)

        for (const { signal, handler } of handlerPairs) {
          const pubsub = yield* signal.tag
          yield* Effect.forkScoped(
            Stream.runForEach(
              Stream.fromPubSub(pubsub),
              (value) => Effect.gen(function* () {
                if (yield* hydration.isHydrating()) return

                // For signal handlers, extract forkId from signal value if present
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
    }))

    // TypeScript can't track Signal<T> types through loops, so cast the Layer
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    type LayerInput = ExtractHandlerRequirements<THandlers> | ExtractSettledRequirements<TSettledHandler> | WorkerBusService<TEvent> | HydrationContext | PubSub.PubSub<any> | FrameworkErrorReporterService

    return {
      Tag,
      Layer: Live as Layer.Layer<void, never, LayerInput>
    }
  }
}
