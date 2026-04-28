import { Effect, Stream } from "effect"
import { ParseError } from "../errors/model-error"

const DONE = "[DONE]"

function dataPayload(line: string): string | null {
  if (!line.startsWith("data:")) {
    return null
  }
  const remainder = line.slice("data:".length)
  return remainder.startsWith(" ") ? remainder.slice(1) : remainder
}

export function sseChunks<E>(
  providerId: string,
  byteStream: Stream.Stream<Uint8Array, E>,
): Stream.Stream<unknown, E | ParseError> {
  return byteStream.pipe(
    Stream.decodeText("utf-8"),
    Stream.splitLines,
    Stream.filter((line) => line.length > 0 && !line.startsWith(":")),
    Stream.map(dataPayload),
    Stream.filter((payload): payload is string => payload !== null),
    Stream.takeUntil((payload) => payload === DONE),
    Stream.filter((payload) => payload !== DONE),
    Stream.mapEffect((payload) =>
      Effect.try({
        try: () => JSON.parse(payload) as unknown,
        catch: () =>
          new ParseError({
            providerId,
            message: `Failed to parse SSE payload: ${payload}`,
          }),
      }),
    ),
  )
}
