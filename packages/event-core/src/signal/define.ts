/**
 * Signal System with Context Tags
 *
 * Signals are ephemeral derived notifications - NOT persisted.
 * Uses Effect's Context Tags to provide type-safe access to PubSubs.
 */

import { Effect, Context, PubSub, Stream } from 'effect'

// ---------------------------------------------------------------------------
// Signal Definition (input to Projection.define)
// ---------------------------------------------------------------------------

/**
 * Signal definition - the input type for Projection.define's signals config.
 * Contains just the name and value type. Source state is attached by Projection.define.
 */
export interface SignalDef<T> {
  readonly name: string
  /** Phantom field to carry the value type */
  readonly _T?: T
}

/**
 * Create a Signal definition.
 * Used in Projection.define's signals config.
 *
 * @param name Unique identifier for the signal (used for Tag creation).
 */
export function create<T>(name: string): SignalDef<T> {
  return { name }
}

/**
 * Create a full Signal directly (for internal framework use).
 * Use this when you need a Signal outside of Projection.define, e.g., in FSM projections.
 *
 * @param name Unique identifier for the signal
 * @param sourceProjectionName The name of the projection that emits this signal
 */
export function createSignal<T, TSourceState = unknown>(
  name: string,
  sourceProjectionName: string
): Signal<T, TSourceState> {
  return new Signal<T, TSourceState>(name, sourceProjectionName)
}

// ---------------------------------------------------------------------------
// Signal Class (output from Projection.define)
// ---------------------------------------------------------------------------

/**
 * A Signal identifies a stream of values of type T from a source projection with state TSourceState.
 * It wraps a Context.Tag which is used to locate the underlying PubSub at runtime.
 *
 * Created by Projection.define from SignalDef<T>, with TSourceState attached.
 *
 * @typeParam T - The value type carried by the signal
 * @typeParam TSourceState - The state type of the projection that emits this signal (defaults to unknown for internal use)
 */
export class Signal<T, TSourceState = unknown> {
  readonly tag: Context.Tag<PubSub.PubSub<T>, PubSub.PubSub<T>>

  constructor(
    readonly name: string,
    readonly sourceProjectionName: string
  ) {
    this.tag = Context.GenericTag<PubSub.PubSub<T>>(name)
  }
}

/**
 * Convert a SignalDef to a Signal with source state attached.
 * Used internally by Projection.define.
 */
export function fromDef<T, TSourceState>(
  def: SignalDef<T>,
  sourceProjectionName: string
): Signal<T, TSourceState> {
  return new Signal<T, TSourceState>(def.name, sourceProjectionName)
}

// ---------------------------------------------------------------------------
// Type Transformations
// ---------------------------------------------------------------------------

/**
 * Transform a record of SignalDefs to a record of Signals with source state attached.
 * Used in Projection.define's return type.
 */
export type AttachSourceState<
  TSignalDefs extends Record<string, SignalDef<unknown>>,
  TSourceState
> = {
  [K in keyof TSignalDefs]: TSignalDefs[K] extends SignalDef<infer T>
    ? Signal<T, TSourceState>
    : never
}

/**
 * Extract the value type from a Signal.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type SignalValue<T> = T extends Signal<infer V, any> ? V : never

/**
 * Extract the source state type from a Signal.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type SignalSourceState<T> = T extends Signal<any, infer S> ? S : never

// ---------------------------------------------------------------------------
// Emit Functions
// ---------------------------------------------------------------------------

/**
 * Type-safe emit function for a specific signal.
 * Sync callback - framework handles Effect wrapping internally.
 */
export type SignalEmit<T> = (value: T) => void

/**
 * Maps a record of signal definitions to their emit functions.
 */
export type SignalEmitters<TSignalDefs extends Record<string, SignalDef<unknown>>> = {
  [K in keyof TSignalDefs]: TSignalDefs[K] extends SignalDef<infer V>
    ? SignalEmit<V>
    : never
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/**
 * Helper to get the stream from a signal.
 * Requires the Signal's PubSub to be in the environment.
 */
export function stream<T, TSourceState>(
  signal: Signal<T, TSourceState>
): Stream.Stream<T, never, PubSub.PubSub<T>> {
  return Stream.unwrap(
    Effect.map(signal.tag, (pubsub) => Stream.fromPubSub(pubsub))
  )
}

/**
 * Helper to emit to a signal.
 * Requires the Signal's PubSub to be in the environment.
 */
export function emit<T, TSourceState>(
  signal: Signal<T, TSourceState>,
  value: T
): Effect.Effect<void, never, PubSub.PubSub<T>> {
  return Effect.flatMap(signal.tag, (pubsub) => PubSub.publish(pubsub, value))
}
