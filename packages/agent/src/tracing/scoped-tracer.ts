import { Effect } from 'effect'
import { TraceEmitter, TracePersister } from '@magnitudedev/providers'
import type { TraceInput, AgentTraceMeta } from '@magnitudedev/tracing'

export interface TraceScope {
  readonly metadata: AgentTraceMeta
  readonly strategyId?: string | null
  readonly systemPrompt?: string | null
}

/**
 * Wraps an effect with a scoped tracer that enriches TraceInput with agent metadata.
 * Reads the TraceWriter from context, provides a Tracer to the driver that
 * enriches TraceInput into full TraceData before forwarding to the TraceWriter.
 */
export function withTraceScope<A, E, R>(
  scope: TraceScope,
  effect: Effect.Effect<A, E, R>,
): Effect.Effect<A, E, R | TracePersister> {
  return Effect.gen(function* () {
    const persister = yield* TracePersister
    const emitter = {
      emit: (base: TraceInput) =>
        persister.emit({
          ...base,
          callType: scope.metadata.callType,
          metadata: scope.metadata,
          strategyId: scope.strategyId ?? null,
          systemPrompt: scope.systemPrompt ?? null,
        }),
    }
    return yield* Effect.provideService(effect, TraceEmitter, emitter)
  })
}