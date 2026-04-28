import { Effect, Stream } from 'effect'
import { HttpClient } from '@effect/platform'
import type { DriverError } from './errors'

/**
 * Options passed to Driver.send on every call.
 *
 * endpoint  — full base URL of the provider endpoint
 *             e.g. "https://api.fireworks.ai/inference/v1"
 * authToken — bearer token or API key (already resolved by auth layer)
 *
 * No signal / timeoutMs — Effect interruption is the standard cancellation
 * path. Callers wrap Driver.send in Effect.timeout / Effect.interrupt.
 */
export interface DriverCallOptions {
  readonly endpoint:  string
  readonly authToken: string
}

/**
 * Driver<WireRequest, WireChunk>
 *
 * Pure transport layer. Knows nothing about prompts, tools, or memory.
 * Takes a pre-encoded wire request, opens an SSE stream, and yields
 * parsed wire-level chunk objects.
 *
 * R channel is HttpClient.HttpClient (provided by FetchHttpClient.layer
 * at the app root — Bun ships native fetch so no bespoke layer needed).
 */
export interface Driver<WireRequest, WireChunk> {
  readonly id: string
  readonly send: (
    request: WireRequest,
    options: DriverCallOptions,
  ) => Effect.Effect<
    Stream.Stream<WireChunk, DriverError>,
    DriverError,
    HttpClient.HttpClient
  >
}
