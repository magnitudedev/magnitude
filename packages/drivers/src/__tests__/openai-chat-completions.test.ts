import { describe, it, expect } from 'vitest'
import { Effect, Stream, Layer } from 'effect'
import {
  HttpClient,
  HttpClientRequest,
  HttpClientResponse,
} from '@effect/platform'
import { OpenAIChatCompletionsDriver } from '../openai-chat-completions/driver'
import {
  ChatCompletionsStreamChunk,
  type ChatCompletionsRequest,
} from '../wire/chat-completions'
import { DriverError } from '../errors'
import * as fs from 'node:fs'
import * as path from 'node:path'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const FIXTURES_DIR = path.join(new URL('.', import.meta.url).pathname, 'fixtures')

const loadFixture = (name: string): Uint8Array =>
  fs.readFileSync(path.join(FIXTURES_DIR, name))

const encode = (s: string): Uint8Array => new TextEncoder().encode(s)

/** Minimal valid chat completions request. */
const makeRequest = (): ChatCompletionsRequest => ({
  model: 'kimi-k2-5',
  messages: [{ role: 'user', content: 'Hello' }],
  stream: true,
})

const OPTIONS = {
  endpoint: 'https://api.example.com/v1',
  authToken: 'test-token',
}

// ---------------------------------------------------------------------------
// Mock HttpClient
//
// HttpClient.make builds a fully conformant HttpClient from a simple
// (req, url, signal, fiber) -> Effect<HttpClientResponse> function.
// We use HttpClientResponse.fromWeb to lift a native Response.
// ---------------------------------------------------------------------------

const makeHttpClientLayer = (
  handler: (req: HttpClientRequest.HttpClientRequest) => Response,
): Layer.Layer<HttpClient.HttpClient> => {
  const client = HttpClient.make((req) =>
    Effect.sync(() =>
      HttpClientResponse.fromWeb(req, handler(req)),
    ),
  )
  return Layer.succeed(HttpClient.HttpClient, client)
}

// ---------------------------------------------------------------------------
// Run driver, collect all chunks
// ---------------------------------------------------------------------------

const runWithLayer = <A>(
  layer: Layer.Layer<HttpClient.HttpClient>,
  f: Effect.Effect<A, DriverError, HttpClient.HttpClient>,
): Promise<A> => Effect.runPromise(f.pipe(Effect.provide(layer)))

const collectChunks = async (
  body: Uint8Array,
  status = 200,
): Promise<ChatCompletionsStreamChunk[]> => {
  const layer = makeHttpClientLayer(() => new Response(body as BodyInit, { status }))
  return runWithLayer(
    layer,
    Effect.gen(function* () {
      const stream = yield* OpenAIChatCompletionsDriver.send(makeRequest(), OPTIONS)
      const collected = yield* Stream.runCollect(stream)
      return Array.from(collected)
    }),
  )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('OpenAIChatCompletionsDriver', () => {
  describe('happy path', () => {
    it('decodes basic SSE fixture — reasoning + content + stop', async () => {
      const chunks = await collectChunks(loadFixture('sse-basic.txt'))
      expect(chunks).toHaveLength(3)

      expect(chunks[0]).toBeInstanceOf(ChatCompletionsStreamChunk)
      expect(chunks[0].choices[0].delta.reasoning_content).toBe('Let me think.')
      expect(chunks[0].choices[0].finish_reason).toBeNull()

      expect(chunks[1].choices[0].delta.content).toBe('Hello!')

      expect(chunks[2].choices[0].finish_reason).toBe('stop')
      expect(chunks[2].usage?.completion_tokens).toBe(5)
    })

    it('decodes tool-call SSE fixture', async () => {
      const chunks = await collectChunks(loadFixture('sse-tool-calls.txt'))
      expect(chunks).toHaveLength(5)

      const tc0 = chunks[0].choices[0].delta.tool_calls?.[0]
      expect(tc0?.id).toBe('call_abc')
      expect(tc0?.function?.name).toBe('read_file')
      expect(tc0?.function?.arguments).toBe('')

      expect(chunks[1].choices[0].delta.tool_calls?.[0]?.function?.arguments).toBe('{"pa')
      expect(chunks[2].choices[0].delta.tool_calls?.[0]?.function?.arguments).toBe('th":"')
      expect(chunks[3].choices[0].delta.tool_calls?.[0]?.function?.arguments).toBe('/tmp"}')

      expect(chunks[4].choices[0].finish_reason).toBe('tool_calls')
    })

    it('decodes reasoning-only SSE fixture', async () => {
      const chunks = await collectChunks(loadFixture('sse-reasoning.txt'))
      expect(chunks).toHaveLength(5)
      expect(chunks[0].choices[0].delta.reasoning_content).toBe('Step 1: analyze')
      expect(chunks[1].choices[0].delta.reasoning_content).toBe(' the problem')
      expect(chunks[2].choices[0].delta.reasoning_content).toBe('. Done.')
      expect(chunks[3].choices[0].delta.content).toBe('The answer is 42.')
      expect(chunks[4].choices[0].finish_reason).toBe('stop')
    })
  })

  describe('error handling', () => {
    it('fails with DriverError status=401 on unauthorized response', async () => {
      const layer = makeHttpClientLayer(() =>
        new Response('{"error":"Unauthorized"}', { status: 401 }),
      )

      const result = await Effect.runPromise(
        Effect.gen(function* () {
          const stream = yield* OpenAIChatCompletionsDriver.send(makeRequest(), OPTIONS)
          yield* Stream.runCollect(stream)
          return { ok: true as const }
        }).pipe(
          Effect.provide(layer),
          Effect.catchAll((err) =>
            Effect.succeed({ ok: false as const, err: err as DriverError }),
          ),
        ),
      )

      expect(result.ok).toBe(false)
      if (!result.ok) {
        expect(result.err).toBeInstanceOf(DriverError)
        expect(result.err.status).toBe(401)
        expect(result.err.reason).toBe('http_status')
      }
    })

    it('fails with DriverError status=429 on rate limit response', async () => {
      const layer = makeHttpClientLayer(() =>
        new Response('{"error":"Rate limit exceeded"}', { status: 429 }),
      )

      const result = await Effect.runPromise(
        Effect.gen(function* () {
          const stream = yield* OpenAIChatCompletionsDriver.send(makeRequest(), OPTIONS)
          yield* Stream.runCollect(stream)
          return { ok: true as const }
        }).pipe(
          Effect.provide(layer),
          Effect.catchAll((err) =>
            Effect.succeed({ ok: false as const, err: err as DriverError }),
          ),
        ),
      )

      expect(result.ok).toBe(false)
      if (!result.ok) {
        expect(result.err.status).toBe(429)
      }
    })

    it('surfaces DriverError in stream on malformed JSON chunk', async () => {
      // First chunk is valid, second is malformed JSON
      const sse =
        'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n' +
        'data: {not valid json}\n\n' +
        'data: [DONE]\n'

      const layer = makeHttpClientLayer(() => new Response(encode(sse) as BodyInit, { status: 200 }))

      const result = await Effect.runPromise(
        Effect.gen(function* () {
          const stream = yield* OpenAIChatCompletionsDriver.send(makeRequest(), OPTIONS).pipe(
            Effect.provide(layer),
          )
          yield* Stream.runCollect(stream)
          return { ok: true as const }
        }).pipe(
          Effect.catchAll((err) =>
            Effect.succeed({ ok: false as const, err: err as DriverError }),
          ),
        ),
      )

      expect(result.ok).toBe(false)
      if (!result.ok) {
        expect(result.err).toBeInstanceOf(DriverError)
        expect(result.err.reason).toMatch(/sse_parse_failed|chunk_decode_failed/)
      }
    })

    it('surfaces DriverError in stream on mid-stream connection drop', async () => {
      // Return a response that reads fine initially but the body errors mid-stream.
      // We simulate this by making the ReadableStream error after a few bytes.
      const validJson = '{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}'
      const partial = `data: ${validJson}\n\n`

      const readableStream = new ReadableStream({
        start(controller) {
          controller.enqueue(encode(partial))
          controller.error(new Error('connection reset'))
        },
      })

      const layer = makeHttpClientLayer(() => new Response(readableStream, { status: 200 }))

      const result = await Effect.runPromise(
        Effect.gen(function* () {
          const stream = yield* OpenAIChatCompletionsDriver.send(makeRequest(), OPTIONS).pipe(
            Effect.provide(layer),
          )
          yield* Stream.runCollect(stream)
          return { ok: true as const }
        }).pipe(
          Effect.catchAll((err) =>
            Effect.succeed({ ok: false as const, err }),
          ),
        ),
      )

      // Connection drops should surface as some kind of error
      expect(result.ok).toBe(false)
    })
  })
})
