import { Effect, Stream } from "effect"

const DEFAULT_DONE = "[DONE]"

function dataPayload(line: string): string | null {
  if (!line.startsWith("data:")) return null
  const remainder = line.slice("data:".length)
  return remainder.startsWith(" ") ? remainder.slice(1) : remainder
}

/**
 * Parse a byte stream of SSE events into typed values.
 *
 * Pipeline: decode UTF-8 → split lines → filter blanks/comments →
 * extract `data:` payloads → stop at done signal → decode each payload.
 */
export function sseStream<T, E, E2>(
  byteStream: Stream.Stream<Uint8Array, E>,
  decodePayload: (raw: string) => Effect.Effect<T, E2>,
  doneSignal: string = DEFAULT_DONE,
): Stream.Stream<T, E | E2> {
  return byteStream.pipe(
    Stream.decodeText("utf-8"),
    Stream.splitLines,
    Stream.filter((line) => line.length > 0 && !line.startsWith(":")),
    Stream.map(dataPayload),
    Stream.filter((payload): payload is string => payload !== null),
    Stream.takeUntil((payload) => payload === doneSignal),
    Stream.filter((payload) => payload !== doneSignal),
    Stream.mapEffect(decodePayload),
  )
}
