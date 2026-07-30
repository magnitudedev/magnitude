import { Effect, Option } from "effect"

const DEFAULT_DIAGNOSTIC_LIMIT_BYTES = 64 * 1024

const appendBounded = (
  current: string,
  chunk: string,
  limitBytes: number,
): string => {
  const bytes = new TextEncoder().encode(`${current}${chunk}`)
  if (bytes.byteLength <= limitBytes) return `${current}${chunk}`
  let start = bytes.byteLength - limitBytes
  while (start < bytes.byteLength && (bytes[start]! & 0xc0) === 0x80) {
    start += 1
  }
  return new TextDecoder().decode(bytes.subarray(start))
}

export interface SpawnDiagnosticCapture {
  readonly diagnostic: Effect.Effect<Option.Option<string>>
}

export const captureSpawnDiagnostics = (
  outputs: ReadonlyArray<AsyncIterable<Uint8Array | string>>,
  limitBytes = DEFAULT_DIAGNOSTIC_LIMIT_BYTES,
): SpawnDiagnosticCapture => {
  let tail = ""
  const drains = outputs.map(async (output) => {
    const decoder = new TextDecoder()
    try {
      for await (const chunk of output) {
        const text =
          typeof chunk === "string"
            ? chunk
            : decoder.decode(chunk, { stream: true })
        tail = appendBounded(tail, text, limitBytes)
      }
      tail = appendBounded(tail, decoder.decode(), limitBytes)
    } catch {
      // Process exit and parent shutdown may close a pipe abruptly. Any bytes
      // already captured remain useful diagnostics.
    }
  })
  const drained = Promise.all(drains).then(() => undefined)

  return {
    diagnostic: Effect.promise(() => drained).pipe(
      Effect.map(() => tail.trim()),
      Effect.map((value) =>
        value.length === 0 ? Option.none() : Option.some(value),
      ),
    ),
  }
}
