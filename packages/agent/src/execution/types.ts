/**
 * Execution Types
 *
 * Types for the turn event stream between execution-manager and cortex.
 * Relocated from strategies/types.ts after strategy abstraction removal.
 */

import { Effect, Data } from 'effect'
import type { ToolCallEvent } from '@magnitudedev/xml-act'
import type { MessageDestination } from '../events'
import type { ExecuteResult } from './execution-manager'
import type { CallUsage } from '@magnitudedev/providers'


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
  | { readonly _tag: 'RawResponseChunk'; readonly text: string }
  | { readonly _tag: 'LensStarted'; readonly name: string }
  | { readonly _tag: 'LensDelta'; readonly text: string }
  | { readonly _tag: 'LensEnded'; readonly name: string }

  // --- Tool events (forwarded xml-act ToolCallEvent with agent metadata) ---
  | { readonly _tag: 'ToolEvent'; readonly toolCallId: string; readonly toolKey: string; readonly event: ToolCallEvent }

  // --- Terminal (always last event in the stream) ---
  | { readonly _tag: 'TurnResult'; readonly value: TurnStrategyResult }

export interface TurnEventSink {
  readonly emit: (event: TurnEvent) => Effect.Effect<void>
}

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
// Turn Result
// =============================================================================

export interface TurnStrategyResult {
  readonly executeResult: ExecuteResult
  readonly usage: CallUsage
}
