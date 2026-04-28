import { Effect, Stream, pipe } from 'effect'
import { DriverError } from '../errors'

const SSE_DATA_PREFIX = 'data: '
const SSE_DONE_SIGNAL = '[DONE]'

/**
 * sseChunks
 *
 * Parses a raw SSE byte stream into a stream of JSON-decoded data payloads.
 *
 * Protocol:
 *   - Lines beginning with `:` are SSE comments — skipped.
 *   - Blank lines are separators — skipped.
 *   - Lines beginning with `data: ` are emitted as parsed JSON objects.
 *   - `data: [DONE]` is the stream terminator — consumed silently (stream ends).
 *   - Any other field (event:, id:, retry:) is silently ignored.
 *   - JSON parse failures surface as DriverError { reason: 'sse_parse_failed' }.
 */
export const sseChunks = (
  byteStream: Stream.Stream<Uint8Array, DriverError>,
): Stream.Stream<unknown, DriverError> =>
  pipe(
    byteStream,
    Stream.decodeText('utf-8'),
    Stream.splitLines,
    // Filter comments and blank lines before any further processing
    Stream.filter((line) => line.length > 0 && !line.startsWith(':')),
    // Only process data lines; ignore event:/id:/retry: fields
    Stream.filter((line) => line.startsWith(SSE_DATA_PREFIX)),
    // Strip the "data: " prefix
    Stream.map((line) => line.slice(SSE_DATA_PREFIX.length)),
    // Terminate stream at [DONE] marker (takeUntil includes the terminator element)
    Stream.takeUntil((data) => data === SSE_DONE_SIGNAL),
    // Filter out the [DONE] sentinel itself
    Stream.filter((data) => data !== SSE_DONE_SIGNAL),
    // Parse JSON — surface malformed payloads as DriverError
    Stream.mapEffect((data) =>
      Effect.try({
        try: () => JSON.parse(data) as unknown,
        catch: (cause) =>
          new DriverError({
            reason: `sse_parse_failed: ${String(cause)}`,
            status: null,
            body: data,
          }),
      }),
    ),
  )
