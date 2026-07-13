/**
 * FrameworkError — typed error reporting for the event-core framework.
 *
 * Provides a tagged union of all recoverable framework errors and a
 * PubSub-backed reporter service. Projections, workers, and the bus
 * access the reporter via the Effect context.
 *
 * Fatal errors (queue infrastructure, layer construction) are NOT
 * caught here — they remain defects that crash the runtime.
 */

import { Data, Effect, Context, Layer, PubSub, Cause } from 'effect'

// ---------------------------------------------------------------------------
// FrameworkError tagged union
// ---------------------------------------------------------------------------

export type FrameworkError = Data.TaggedEnum<{
  ProjectionEventHandlerError: {
    readonly projectionName: string
    readonly eventType: string
    readonly cause: Cause.Cause<unknown>
  }
  ProjectionSignalHandlerError: {
    readonly projectionName: string
    readonly signalName: string
    readonly cause: Cause.Cause<unknown>
  }
  WorkerEventHandlerError: {
    readonly workerName: string
    readonly eventType: string
    readonly cause: Cause.Cause<unknown>
  }
  WorkerSignalHandlerError: {
    readonly workerName: string
    readonly signalName: string
    readonly cause: Cause.Cause<unknown>
  }
  WorkerSettledHandlerError: {
    readonly workerName: string
    readonly cause: Cause.Cause<unknown>
  }
  WorkerLifecycleError: {
    readonly workerName: string
    readonly eventType: string
    readonly cause: Cause.Cause<unknown>
  }
  SinkError: {
    readonly eventType: string
    readonly cause: Cause.Cause<unknown>
  }
  BroadcastError: {
    readonly eventType: string
    readonly cause: Cause.Cause<unknown>
  }
  SubscriptionError: {
    readonly subscriptionName: string
    readonly cause: Cause.Cause<unknown>
  }
}>

export const FrameworkError = Data.taggedEnum<FrameworkError>()

// ---------------------------------------------------------------------------
// PubSub for framework errors
// ---------------------------------------------------------------------------

export const FrameworkErrorPubSub = Context.GenericTag<PubSub.PubSub<FrameworkError>>('FrameworkErrorPubSub')

// ---------------------------------------------------------------------------
// Reporter service
// ---------------------------------------------------------------------------

export interface FrameworkErrorReporterService {
  readonly report: (error: FrameworkError) => Effect.Effect<void>
}

export const FrameworkErrorReporter = Context.GenericTag<FrameworkErrorReporterService>('FrameworkErrorReporter')

// ---------------------------------------------------------------------------
// Reporter layer (backed by PubSub)
// ---------------------------------------------------------------------------

export const FrameworkErrorReporterLive: Layer.Layer<
  FrameworkErrorReporterService,
  never,
  PubSub.PubSub<FrameworkError>
> = Layer.effect(
  FrameworkErrorReporter,
  Effect.gen(function* () {
    const pubsub = yield* FrameworkErrorPubSub

    return {
      report: (error: FrameworkError) =>
        PubSub.publish(pubsub, error).pipe(
          // The reporter must never fail — swallow errors in the reporter itself
          Effect.catchAllCause(() => Effect.void)
        )
    }
  })
)

// ---------------------------------------------------------------------------
// PubSub layer
// ---------------------------------------------------------------------------

export const FrameworkErrorPubSubLive: Layer.Layer<PubSub.PubSub<FrameworkError>> =
  Layer.scoped(FrameworkErrorPubSub, PubSub.unbounded<FrameworkError>())
