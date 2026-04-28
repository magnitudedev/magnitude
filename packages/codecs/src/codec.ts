import { Effect, Schema, Stream } from 'effect'
import type { DriverError } from '@magnitudedev/drivers'
import type { ResponseStreamEvent } from './events/turn-part-event'
import type { ToolDef } from './tools/tool-def'

// =============================================================================
// Errors
// =============================================================================

export class CodecEncodeError extends Schema.TaggedError<CodecEncodeError>()(
  'CodecEncodeError',
  {
    reason:  Schema.String,
    /** Any partial state captured at encode time for diagnostics. */
    context: Schema.Unknown,
  },
) {}

export class CodecDecodeError extends Schema.TaggedError<CodecDecodeError>()(
  'CodecDecodeError',
  {
    reason:  Schema.String,
    /** The partially-decoded state at the point of failure. */
    partial: Schema.Unknown,
  },
) {}

// =============================================================================
// Encode options
// =============================================================================

/**
 * EncodeOptions — per-turn inference parameters.
 *
 * All fields optional; the codec or bound model supplies defaults.
 *
 * thinkingLevel  — controls the depth/budget of the model's reasoning.
 *                  Maps to provider-specific parameters at encode time.
 *                  Defaults to 'medium'.
 * maxTokens      — maximum output tokens. Provider default if absent.
 * stopSequences  — additional stop sequences beyond the model's default.
 */
export interface EncodeOptions {
  readonly thinkingLevel?: 'low' | 'medium' | 'high'
  readonly maxTokens?:     number
  readonly stopSequences?: readonly string[]
}

// =============================================================================
// Message type (forward-declared here for the Codec interface signature)
// =============================================================================
// NOTE: The full Message union is defined in packages/agent (Phase 4).
// The codec interface uses `unknown` here to avoid a circular dependency;
// the concrete NativeChatCompletionsCodec (Phase 2) will import the real
// Message type and cast appropriately.
//
// This is an intentional Phase 0 limitation — Phase 4 extracts inbox-types
// into a shared package so both agent and codecs can depend on it cleanly.

// =============================================================================
// Codec interface
// =============================================================================

/**
 * Codec<WireRequest, WireChunk>
 *
 * The codec layer sits between agent memory and the wire.
 *
 * encode: Memory → WireRequest
 *   Takes the full conversation history (as typed messages) and the declared
 *   tools, and produces a provider-ready request object. Pure Effect; no I/O.
 *
 * decode: Stream<WireChunk> → Stream<ResponseStreamEvent>
 *   Transforms the raw SSE chunk stream from the driver into the canonical
 *   ResponseStreamEvent vocabulary. Stateful (accumulates partial tool-call JSON);
 *   state is managed via Stream.mapAccum (pure, immutable rebuilds) so the
 *   stream is safe to consume from a single fiber.
 *
 * Type parameters:
 *   WireRequest — the request object type produced by encode
 *                 (e.g. ChatCompletionsRequest)
 *   WireChunk   — the decoded chunk type consumed by decode
 *                 (e.g. ChatCompletionsStreamChunk)
 */
export interface Codec<WireRequest, WireChunk> {
  readonly id: string

  readonly encode: (
    memory:  readonly unknown[],
    tools:   readonly ToolDef[],
    options: EncodeOptions,
  ) => Effect.Effect<WireRequest, CodecEncodeError>

  readonly decode: (
    chunks: Stream.Stream<WireChunk, DriverError>,
  ) => Stream.Stream<ResponseStreamEvent, CodecDecodeError | DriverError>
}
