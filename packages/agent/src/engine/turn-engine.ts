/**
 * TurnEngine — Effect service that orchestrates one agent turn.
 *
 * Pipeline: codec.encode → driver.send → codec.decode → createTurnEngine
 *
 * The service is a thin composition layer. All format and wire concerns live
 * in the NativeBoundModel's transport. All parsing and tool dispatch live in
 * the turn-engine library.
 */

import { Context, Effect, Layer, Schema, Stream } from 'effect'
import { HttpClient } from '@effect/platform'
import {
  createTurnEngine,
  type TurnEngineEvent,
  type RegisteredTool,
  TurnEngineCrash,
} from '@magnitudedev/turn-engine'
import {
  CodecEncodeError,
  CodecDecodeError,
  type EncodeOptions,
  type ToolDef,
} from '@magnitudedev/codecs'
import { DriverError } from '@magnitudedev/drivers'
import { type NativeBoundModel, extractAuthToken } from './native-bound-model'

// =============================================================================
// TurnEngineError
// =============================================================================

/**
 * Single tagged error type covering all turn-execution failure modes.
 *
 * `phase` identifies where in the pipeline the failure occurred:
 *   - 'encode'  — codec.encode rejected the request before sending
 *   - 'send'    — driver.send / DriverError (could be pre-stream or mid-stream)
 *   - 'decode'  — codec.decode failed on a stream chunk
 *   - 'engine'  — TurnEngine crashed during event processing / tool dispatch
 *   - 'unknown' — catch-all for unexpected errors
 *
 * `cause` carries the original error object for diagnostics / logging.
 */
export class TurnEngineError extends Schema.TaggedError<TurnEngineError>()(
  'TurnEngineError',
  {
    message: Schema.String,
    phase:   Schema.Literal('encode', 'send', 'decode', 'engine', 'unknown'),
    cause:   Schema.Unknown,
  },
) {}

// =============================================================================
// Service shape
// =============================================================================

export interface TurnEngineRunParams {
  /** The bound model — carries codec, driver, auth, and wire config. */
  readonly model: NativeBoundModel
  /** Conversation memory (codec-specific opaque shape). */
  readonly memory: readonly unknown[]
  /** Engine tool map (name → registered handler) for the dispatcher. */
  readonly tools: ReadonlyMap<string, RegisteredTool>
  /** Codec tool definitions used during encode. */
  readonly toolDefs: readonly ToolDef[]
  /** Encode options (thinking level, max tokens, etc.). */
  readonly options: EncodeOptions
  /** Destination injected into every MessageStart event. Default: 'user'. */
  readonly messageDestination?: string
  /** Kind injected into ThoughtStart events. Default: 'reasoning'. */
  readonly thoughtKind?: string
}

export interface TurnEngineShape {
  /**
   * Run one turn. Returns an Effect that resolves to a Stream of engine events.
   *
   * The Effect itself can fail with TurnEngineError for encode / initial-send
   * failures (pre-stream). The Stream can fail with TurnEngineError for
   * decode / engine failures (mid-stream).
   *
   * Requires HttpClient in the environment (used by the driver).
   */
  readonly runTurn: (params: TurnEngineRunParams) => Effect.Effect<
    Stream.Stream<TurnEngineEvent, TurnEngineError>,
    TurnEngineError,
    HttpClient.HttpClient
  >
}

export class TurnEngine extends Context.Tag('TurnEngine')<TurnEngine, TurnEngineShape>() {}

// =============================================================================
// Error helpers
// =============================================================================

function wrapEncodeError(err: CodecEncodeError): TurnEngineError {
  return new TurnEngineError({
    message: `Encode failed: ${err.reason ?? String(err)}`,
    phase: 'encode',
    cause: err,
  })
}

function wrapSendError(err: DriverError): TurnEngineError {
  return new TurnEngineError({
    message: `Driver send failed: ${err.message ?? String(err)}`,
    phase: 'send',
    cause: err,
  })
}

function wrapStreamError(err: CodecDecodeError | DriverError): TurnEngineError {
  if (err._tag === 'CodecDecodeError') {
    return new TurnEngineError({
      message: `Decode failed: ${(err as CodecDecodeError).reason ?? String(err)}`,
      phase: 'decode',
      cause: err,
    })
  }
  // DriverError mid-stream
  return new TurnEngineError({
    message: `Driver stream error: ${(err as DriverError).message ?? String(err)}`,
    phase: 'send',
    cause: err,
  })
}

function wrapCrash(crash: TurnEngineCrash): TurnEngineError {
  return new TurnEngineError({
    message: `Engine crashed: ${crash.message}`,
    phase: 'engine',
    cause: crash,
  })
}

// =============================================================================
// Live layer
// =============================================================================

export const TurnEngineLive = Layer.succeed(TurnEngine, {
  runTurn: ({ model, memory, tools, toolDefs, options, messageDestination, thoughtKind }) =>
    Effect.gen(function* () {
      const authToken = extractAuthToken(model.auth)
      const call = {
        endpoint:  model.wireConfig.endpoint,
        authToken,
      }

      // ── 1. Encode + send (Effect channel: CodecEncodeError | DriverError) ─
      const partStream = yield* model.transport.run(memory, toolDefs, options, call).pipe(
        Effect.mapError((err: CodecEncodeError | DriverError) => {
          if (err._tag === 'CodecEncodeError') return wrapEncodeError(err as CodecEncodeError)
          return wrapSendError(err as DriverError)
        }),
      )

      // ── 2. Map stream-level decode / driver errors ──────────────────────
      const safePartStream = partStream.pipe(
        Stream.mapError(wrapStreamError),
      )

      // ── 3. Run engine over codec response events ───────────────────────
      const engine = createTurnEngine({
        tools,
        messageDestination: messageDestination ?? 'user',
        thoughtKind,
      })
      const output = engine.streamWith(safePartStream).pipe(
        Stream.mapError(wrapCrash),
      )

      return output
    }),
})
