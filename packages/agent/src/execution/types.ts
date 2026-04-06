/**
 * Execution Types
 *
 * Types for the turn event stream between execution-manager and cortex.
 * Relocated from strategies/types.ts after strategy abstraction removal.
 */

import { Effect, Stream, Queue, Deferred, Scope, Data } from 'effect'
import type { XmlRuntimeCrash, ToolCallEvent } from '@magnitudedev/xml-act'
import type { ResponsePart, MessageDestination } from '../events'
import type { ExecuteResult } from './execution-manager'
import type { CallUsage, CollectorData } from '@magnitudedev/providers'


// =============================================================================
// Turn Events
// =============================================================================

/**
 * Events yielded during turn execution.
 * Cortex maps these to AppEvents and publishes them to the event bus.
 *
 * These carry only the data the execution manager naturally has.
 * Cortex decorates with forkId, turnId, chainId when publishing.
 */
export type TurnEvent =
  // --- Message/thinking content ---
  | { readonly _tag: 'MessageStart'; readonly id: string; readonly destination: MessageDestination }
  | { readonly _tag: 'MessageChunk'; readonly id: string; readonly text: string }
  | { readonly _tag: 'MessageEnd'; readonly id: string }
  | { readonly _tag: 'ThinkingDelta'; readonly text: string }
  | { readonly _tag: 'ThinkingEnd'; readonly about: string | null }
  | { readonly _tag: 'LensStarted'; readonly name: string }
  | { readonly _tag: 'LensDelta'; readonly text: string }
  | { readonly _tag: 'LensEnded'; readonly name: string }

  // --- Tool events (forwarded xml-act ToolCallEvent with agent metadata) ---
  | { readonly _tag: 'ToolEvent'; readonly toolCallId: string; readonly toolKey: string; readonly event: ToolCallEvent }

  // --- Terminal (always last event in the stream) ---
  | { readonly _tag: 'TurnResult'; readonly value: TurnStrategyResult }

// =============================================================================
// Turn Error
// =============================================================================

/**
 * Typed errors from turn execution.
 * Auth failures, LLM API errors, and stream read errors.
 */
export type TurnError = Data.TaggedEnum<{
  /** OAuth / API key validation failure */
  readonly AuthFailed: { readonly message: string; readonly cause?: unknown }
  /** LLM API returned an error (HTTP error, validation error, etc.) */
  readonly LLMFailed: { readonly message: string; readonly cause?: unknown }
  /** Error reading from the LLM response stream */
  readonly StreamFailed: { readonly message: string; readonly cause?: unknown }
}>

export const TurnError = Data.taggedEnum<TurnError>()

// =============================================================================
// Turn Stream Helper
// =============================================================================

/**
 * Create a turn event stream from an effectful producer.
 *
 * The producer receives a Queue to offer events into.
 * The stream completes when the producer returns.
 * Errors propagate through the stream via a Deferred.
 *
 * The producer MUST offer a TurnResult event before returning.
 */
export function createTurnStream<R>(
  producer: (queue: Queue.Queue<TurnEvent>) => Effect.Effect<void, XmlRuntimeCrash | TurnError, R>
): Stream.Stream<TurnEvent, XmlRuntimeCrash | TurnError, R | Scope.Scope> {
  return Stream.unwrapScoped(
    Effect.gen(function* () {
      const queue = yield* Queue.unbounded<TurnEvent>()
      const done = yield* Deferred.make<void, XmlRuntimeCrash | TurnError>()

      yield* Effect.forkScoped(
        producer(queue).pipe(
          Effect.exit,
          Effect.flatMap((exit) => Deferred.done(done, exit)),
          Effect.ensuring(Effect.yieldNow().pipe(Effect.andThen(Queue.shutdown(queue)))),
        )
      )

      const queueStream = Stream.fromQueue(queue)
      const doneCheck = Stream.fromEffect(Deferred.await(done)).pipe(Stream.drain)
      return Stream.concat(queueStream, doneCheck)
    })
  )
}

// =============================================================================
// Turn Result
// =============================================================================

export interface TurnStrategyResult {
  readonly executeResult: ExecuteResult
  readonly usage: CallUsage
  readonly collectorData?: CollectorData
  /** Provider-native representation of the model's response */
  readonly responseParts: readonly ResponsePart[]
  /** Raw XML chunks accumulated during streaming (for interrupt preservation) */
  readonly rawCodeChunks: string[]
}
