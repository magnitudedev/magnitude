import { Context, Effect, Layer } from 'effect'
import type { TraceInput, TraceData } from '@magnitudedev/tracing'

export type { TraceInput, TraceData }

/**
 * Driver-facing trace emitter — accepts transport-level trace input.
 * The driver emits TraceInput; withTraceScope enriches it to TraceData
 * before forwarding to the TracePersister.
 */
export class TraceEmitter extends Context.Tag('TraceEmitter')<
  TraceEmitter,
  {
    readonly emit: (trace: TraceInput) => Effect.Effect<void>
  }
>() {}

/**
 * Agent-facing trace persister — accepts fully enriched TraceData.
 * Created by makeTracePersister, consumed by withTraceScope.
 */
export class TracePersister extends Context.Tag('TracePersister')<
  TracePersister,
  {
    readonly emit: (trace: TraceData) => Effect.Effect<void>
  }
>() {}

/**
 * Create a TracePersister layer that calls the given callback on each enriched trace.
 */
export function makeTracePersister(cb: (trace: TraceData) => void): Layer.Layer<TracePersister> {
  return Layer.succeed(TracePersister, {
    emit: (trace) => Effect.sync(() => cb(trace)),
  })
}

/**
 * Create no-op layers for both TraceEmitter and TracePersister.
 */
export function makeNoopTracer(): Layer.Layer<TraceEmitter | TracePersister> {
  return Layer.mergeAll(
    Layer.succeed(TraceEmitter, { emit: () => Effect.void }),
    Layer.succeed(TracePersister, { emit: () => Effect.void }),
  )
}

export interface TraceStore {
  readonly traces: TraceData[]
}

/**
 * Create test layers that capture enriched traces for assertions.
 */
export function makeTestTracer(): { layer: Layer.Layer<TraceEmitter | TracePersister>; store: TraceStore } {
  const store: TraceStore = { traces: [] }
  const layer = Layer.mergeAll(
    Layer.succeed(TraceEmitter, { emit: () => Effect.void }),
    Layer.succeed(TracePersister, {
      emit: (trace: TraceData) =>
        Effect.sync(() => {
          store.traces.push(trace)
        }),
    }),
  )
  return { layer, store }
}