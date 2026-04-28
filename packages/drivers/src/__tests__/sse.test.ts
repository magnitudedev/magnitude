import { describe, it, expect } from 'vitest'
import { Effect, Stream } from 'effect'
import { sseChunks } from '../openai-chat-completions/sse'
import { DriverError } from '../errors'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Encode a string as Uint8Array (UTF-8). */
const encode = (s: string): Uint8Array => new TextEncoder().encode(s)

/** Collect all items from a stream into an array, or throw the error. */
const collect = <A>(stream: Stream.Stream<A, DriverError>): Promise<A[]> =>
  Effect.runPromise(Stream.runCollect(stream).pipe(Effect.map((chunk) => Array.from(chunk))))

/** Build an sseChunks stream from a list of byte-chunk strings. */
const parse = (chunks: string[]): Stream.Stream<unknown, DriverError> => {
  const byteStream = Stream.fromIterable(chunks.map(encode)) as Stream.Stream<
    Uint8Array,
    DriverError
  >
  return sseChunks(byteStream)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('sseChunks', () => {
  it('parses a single complete event', async () => {
    const payload = { foo: 'bar', n: 42 }
    const sse = `data: ${JSON.stringify(payload)}\n\n`
    const results = await collect(parse([sse]))
    expect(results).toHaveLength(1)
    expect(results[0]).toEqual(payload)
  })

  it('parses multiple events', async () => {
    const p1 = { seq: 1 }
    const p2 = { seq: 2 }
    const sse = `data: ${JSON.stringify(p1)}\n\ndata: ${JSON.stringify(p2)}\n\n`
    const results = await collect(parse([sse]))
    expect(results).toHaveLength(2)
    expect(results[0]).toEqual(p1)
    expect(results[1]).toEqual(p2)
  })

  it('reassembles events split across byte chunks', async () => {
    const payload = { id: 'xyz', value: 'hello world' }
    const full = `data: ${JSON.stringify(payload)}\n\ndata: [DONE]\n`
    // Split after the first 10 bytes
    const chunk1 = full.slice(0, 10)
    const chunk2 = full.slice(10)
    const results = await collect(parse([chunk1, chunk2]))
    expect(results).toHaveLength(1)
    expect(results[0]).toEqual(payload)
  })

  it('terminates cleanly on [DONE]', async () => {
    const p = { done: false }
    const sse = `data: ${JSON.stringify(p)}\n\ndata: [DONE]\n`
    const results = await collect(parse([sse]))
    expect(results).toHaveLength(1)
    expect(results[0]).toEqual(p)
  })

  it('skips SSE comment lines (starting with :)', async () => {
    const p = { x: 1 }
    const sse = `: this is a comment\ndata: ${JSON.stringify(p)}\n\ndata: [DONE]\n`
    const results = await collect(parse([sse]))
    expect(results).toHaveLength(1)
    expect(results[0]).toEqual(p)
  })

  it('skips blank lines', async () => {
    const p = { x: 2 }
    const sse = `\n\ndata: ${JSON.stringify(p)}\n\ndata: [DONE]\n`
    const results = await collect(parse([sse]))
    expect(results).toHaveLength(1)
    expect(results[0]).toEqual(p)
  })

  it('emits DriverError on malformed JSON', async () => {
    const sse = 'data: {not valid json}\n\n'
    const result = await Effect.runPromise(
      Stream.runCollect(parse([sse])).pipe(
        Effect.map(() => ({ ok: true as const })),
        Effect.catchAll((err) => Effect.succeed({ ok: false as const, err })),
      ),
    )
    expect(result.ok).toBe(false)
    if (!result.ok) {
      expect(result.err).toBeInstanceOf(DriverError)
      expect((result.err as DriverError).reason).toMatch(/sse_parse_failed/)
    }
  })

  it('ignores non-data SSE fields (event:, id:, retry:)', async () => {
    const p = { ev: 'message' }
    const sse = `event: message\nid: 1\nretry: 3000\ndata: ${JSON.stringify(p)}\n\ndata: [DONE]\n`
    const results = await collect(parse([sse]))
    expect(results).toHaveLength(1)
    expect(results[0]).toEqual(p)
  })
})
