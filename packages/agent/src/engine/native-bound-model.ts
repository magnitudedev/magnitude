/**
 * NativeBoundModel — the native-paradigm model handle.
 *
 * The bound model carries everything the TurnEngine needs to run one turn:
 *   - identity   (model + auth + endpoint config)
 *   - transport  (a closed-over codec+driver pair, exposed as a single
 *                 encode→send→decode function)
 *
 * The codec and driver are not exposed as separate fields. They are paired
 * by `<W, C>` at construction time and immediately bundled into a
 * `NativeTransport` whose internals erase those generics by closure. The
 * consumer (TurnEngine) never sees `<W, C>`, never needs to spell them, and
 * never needs a cast — because the factory is the only place where a
 * concrete `(Codec<W, C>, Driver<W, C>)` pair lives.
 *
 * Why this shape:
 *   - Storing `Codec<unknown, unknown>` + `Driver<unknown, unknown>` is
 *     unsound — Codec.decode is contravariant in C, so concrete codecs are
 *     not assignable to the unknown-erased form. The "fix" of casting
 *     discards the relationship between codec and driver entirely.
 *   - Bundling at construction preserves the (W, C) pairing inside the
 *     closure and exposes only the composed function, which is exactly
 *     the surface the consumer needs.
 */

import { Effect, Stream } from 'effect'
import { HttpClient } from '@effect/platform'
import type { Codec, EncodeOptions, ResponseStreamEvent, ToolDef } from '@magnitudedev/codecs'
import type { Driver, DriverError } from '@magnitudedev/drivers'
import { CodecEncodeError, CodecDecodeError } from '@magnitudedev/codecs'
import type { AuthInfo } from '@magnitudedev/storage'
import type { ProviderModel } from '@magnitudedev/providers'

// =============================================================================
// Wire config
// =============================================================================

export interface NativeWireConfig {
  readonly endpoint:        string
  readonly wireModelName:   string
  readonly defaultMaxTokens: number
}

// =============================================================================
// Transport bundle
// =============================================================================

/**
 * NativeTransport — type-erased encode→send→decode bundle.
 *
 * The (W, C) generics from the codec+driver pair are closed over by the
 * factory below; consumers never need to spell them.
 *
 * `run` performs the full turn pipeline:
 *   memory + tools + options
 *     → codec.encode → wireRequest
 *     → driver.send  → Stream<wireChunk, DriverError>
 *     → codec.decode → Stream<ResponseStreamEvent, CodecDecodeError | DriverError>
 */
export interface NativeTransport {
  readonly run: (
    memory:  readonly unknown[],
    tools:   readonly ToolDef[],
    options: EncodeOptions,
    call:    { readonly endpoint: string; readonly authToken: string },
  ) => Effect.Effect<
    Stream.Stream<ResponseStreamEvent, CodecDecodeError | DriverError>,
    CodecEncodeError | DriverError,
    HttpClient.HttpClient
  >
}

/**
 * Factory: pair a concrete Codec<W, C> with a concrete Driver<W, C> and
 * return a NativeTransport. The (W, C) parameters are inferred at the call
 * site from the codec+driver argument types and never escape the closure.
 */
export const makeNativeTransport = <W, C>(
  codec:  Codec<W, C>,
  driver: Driver<W, C>,
): NativeTransport => ({
  run: (memory, tools, options, call) =>
    Effect.gen(function* () {
      const wireRequest = yield* codec.encode(memory, tools, options)
      const wireStream  = yield* driver.send(wireRequest, call)
      return codec.decode(wireStream)
    }),
})

// =============================================================================
// Bound model
// =============================================================================

export interface NativeBoundModel {
  readonly model:      ProviderModel
  readonly auth:       AuthInfo
  readonly transport:  NativeTransport
  readonly wireConfig: NativeWireConfig
}

/**
 * Extract the bearer token from AuthInfo.
 * API key → key string.
 * OAuth   → accessToken.
 */
export function extractAuthToken(auth: AuthInfo): string {
  switch (auth.type) {
    case 'api':   return auth.key
    case 'oauth': return auth.accessToken
    default:      return ''
  }
}
