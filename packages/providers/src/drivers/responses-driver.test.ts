import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { Effect } from 'effect'
import { ResponsesDriver } from './responses-driver'
import { Model } from '../model/model'
import { ModelConnection } from '../model/model-connection'
import type { DriverRequest } from './types'

vi.mock('./baml-dispatch', () => ({
  bamlStreamRequest: vi.fn(async () => ({
    body: {
      json: () => ({
        model: 'gpt-5',
        input: [{ role: 'user', content: 'hey' }],
      }),
    },
  })),
  bamlParse: vi.fn((_functionName: string, output: string) => ({ output })),
}))

const makeSsePayload = (
  events: unknown[],
  options?: { includeDone?: boolean; finalNewline?: boolean },
) => {
  const includeDone = options?.includeDone ?? true
  const finalNewline = options?.finalNewline ?? true
  const eventPayload = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('')
  const donePayload = includeDone ? 'data: [DONE]\n\n' : ''
  const payload = `${eventPayload}${donePayload}`
  return finalNewline ? payload : payload.replace(/\n+$/, '')
}

const makeChunkedSseResponse = (chunks: string[]) => {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(new TextEncoder().encode(chunk))
      }
      controller.close()
    },
  })

  return new Response(stream, { status: 200 })
}

const makeSseResponse = (
  events: unknown[],
  options?: { includeDone?: boolean; finalNewline?: boolean },
) => makeChunkedSseResponse([makeSsePayload(events, options)])

describe('ResponsesDriver usage normalization', () => {
  const originalFetch = globalThis.fetch

  beforeEach(() => {
    vi.restoreAllMocks()
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  const makeReq = (oauth = false): DriverRequest => ({
    slot: 'main',
    functionName: 'testFn',
    args: [],
    connection: ModelConnection.Responses({
      auth: oauth
        ? { type: 'oauth', accessToken: 'test-token', accountId: 'acct_123' }
        : null,
      endpoint: 'https://api.openai.com/v1/responses',
      headers: { Authorization: 'Bearer test' },
    }),
    model: new Model({
      id: 'gpt-5',
      providerId: 'openai',
      name: 'gpt-5',
      contextWindow: 400000,
      maxOutputTokens: 8192,
      costs: null,
    }),
    inference: {},
  })

  it('adds input_tokens_details.cached_tokens to inputTokens and cacheReadTokens when present', async () => {
    globalThis.fetch = vi.fn(async () => makeSseResponse([
      {
        type: 'response.output_text.delta',
        delta: 'ok',
      },
      {
        type: 'response.completed',
        response: {
          usage: {
            input_tokens: 134,
            output_tokens: 12,
            input_tokens_details: {
              cached_tokens: 900,
            },
          },
        },
      },
    ])) as typeof fetch

    const result = await Effect.runPromise(ResponsesDriver.complete(makeReq()))

    expect(result.usage.inputTokens).toBe(1034)
    expect(result.usage.outputTokens).toBe(12)
    expect(result.usage.cacheReadTokens).toBe(900)
  })

  it('keeps input_tokens as-is when cached_tokens is absent', async () => {
    globalThis.fetch = vi.fn(async () => makeSseResponse([
      {
        type: 'response.completed',
        response: {
          usage: {
            input_tokens: 200,
            output_tokens: 20,
          },
        },
      },
    ])) as typeof fetch

    const result = await Effect.runPromise(ResponsesDriver.complete(makeReq()))

    expect(result.usage.inputTokens).toBe(200)
    expect(result.usage.outputTokens).toBe(20)
  })

  it('uses top-level event.usage as pragmatic fallback when response.usage is absent', async () => {
    globalThis.fetch = vi.fn(async () => makeSseResponse([
      {
        type: 'response.completed',
        usage: {
          input_tokens: 150,
          output_tokens: 15,
          input_tokens_details: {
            cached_tokens: 50,
          },
        },
      },
    ])) as typeof fetch

    const result = await Effect.runPromise(ResponsesDriver.complete(makeReq()))

    expect(result.usage.inputTokens).toBe(200)
    expect(result.usage.outputTokens).toBe(15)
    expect(result.collectorData._tag).toBe('Responses')
    if (result.collectorData._tag !== 'Responses') throw new Error('Expected Responses collector data')
    expect(result.collectorData.diagnostics?.usagePath).toBe('response.completed.usage')
    expect(result.collectorData.diagnostics?.usageSource).toBe('response.completed.usage')
  })

  it('emits OpenAI OAuth diagnostics with terminal payload and usage source/path', async () => {
    const completed = {
      type: 'response.completed',
      usage: {
        input_tokens: 146,
        output_tokens: 9,
      },
      request_id: 'req_123',
    }

    globalThis.fetch = vi.fn(async () => makeSseResponse([completed])) as typeof fetch

    const result = await Effect.runPromise(ResponsesDriver.complete(makeReq(true)))

    expect(result.collectorData._tag).toBe('Responses')
    if (result.collectorData._tag !== 'Responses') throw new Error('Expected Responses collector data')

    expect(result.collectorData.diagnostics).toMatchObject({
      codexVariant: 'openai-codex',
      providerId: 'openai',
      modelId: 'gpt-5',
      authType: 'oauth',
      driverId: 'openai-responses',
      endpoint: 'https://api.openai.com/v1/responses',
      terminalEventType: 'response.completed',
      terminalEventPayload: completed,
      usageSource: 'response.completed.usage',
      usagePath: 'response.completed.usage',
      rawUsage: completed.usage,
      parsedInputTokens: 146,
      parsedOutputTokens: 9,
      parsedCacheReadTokens: null,
      parsedCacheWriteTokens: null,
      selectedUsageEventType: 'response.completed',
      usageRejectionReasons: [],
      usageAbsent: false,
      streamEndReason: 'done-sentinel',
      sawDoneSentinel: true,
      responseIdSeen: false,
    })
    expect(result.collectorData.diagnostics?.terminalCompletedCount).toBeGreaterThanOrEqual(1)
  })

  it('falls back to latest valid non-completed response.* usage when completed usage is absent', async () => {
    globalThis.fetch = vi.fn(async () => makeSseResponse([
      {
        type: 'response.in_progress',
        usage: {
          input_tokens: 111,
          output_tokens: 11,
        },
      },
      {
        type: 'response.output_text.delta',
        delta: 'ok',
      },
    ])) as typeof fetch

    const result = await Effect.runPromise(ResponsesDriver.complete(makeReq()))

    expect(result.usage.inputTokens).toBe(111)
    expect(result.usage.outputTokens).toBe(11)
    expect(result.collectorData._tag).toBe('Responses')
    if (result.collectorData._tag !== 'Responses') throw new Error('Expected Responses collector data')
    expect(result.collectorData.diagnostics?.usageSource).toBe('response.other')
  })

  it('prioritizes response.completed.response.usage over later non-terminal usage', async () => {
    globalThis.fetch = vi.fn(async () => makeSseResponse([
      {
        type: 'response.in_progress',
        usage: {
          input_tokens: 100,
          output_tokens: 10,
        },
      },
      {
        type: 'response.completed',
        response: {
          usage: {
            input_tokens: 250,
            output_tokens: 25,
            input_tokens_details: { cached_tokens: 5 },
          },
        },
      },
      {
        type: 'response.content_part.added',
        usage: {
          input_tokens: 300,
          output_tokens: 30,
          input_tokens_details: { cached_tokens: 7 },
        },
      },
    ])) as typeof fetch

    const result = await Effect.runPromise(ResponsesDriver.complete(makeReq()))

    expect(result.usage.inputTokens).toBe(255)
    expect(result.usage.outputTokens).toBe(25)
    expect(result.collectorData._tag).toBe('Responses')
    if (result.collectorData._tag !== 'Responses') throw new Error('Expected Responses collector data')
    expect(result.collectorData.diagnostics?.usageSource).toBe('response.completed.response.usage')
  })

  it('parses usage when response.completed is split across chunks', async () => {
    const completed = {
      type: 'response.completed',
      response: {
        usage: {
          input_tokens: 321,
          output_tokens: 33,
          input_tokens_details: { cached_tokens: 9 },
        },
      },
    }
    const payload = makeSsePayload([completed], { includeDone: true })
    const splitPoint = Math.floor(payload.length / 2)
    globalThis.fetch = vi.fn(async () =>
      makeChunkedSseResponse([payload.slice(0, splitPoint), payload.slice(splitPoint)]),
    ) as typeof fetch

    const result = await Effect.runPromise(ResponsesDriver.complete(makeReq()))

    expect(result.usage.inputTokens).toBe(330)
    expect(result.usage.outputTokens).toBe(33)
  })

  it('parses usage from trailing EOF buffer without final newline', async () => {
    globalThis.fetch = vi.fn(async () => makeSseResponse([
      {
        type: 'response.completed',
        response: {
          usage: {
            input_tokens: 210,
            output_tokens: 21,
          },
        },
      },
    ], { includeDone: false, finalNewline: false })) as typeof fetch

    const result = await Effect.runPromise(ResponsesDriver.complete(makeReq()))

    expect(result.usage.inputTokens).toBe(210)
    expect(result.usage.outputTokens).toBe(21)
  })

  it('parses usage when completed event is followed by [DONE]', async () => {
    globalThis.fetch = vi.fn(async () => makeSseResponse([
      {
        type: 'response.completed',
        response: {
          usage: {
            input_tokens: 400,
            output_tokens: 40,
            input_tokens_details: { cached_tokens: 100 },
          },
        },
      },
    ], { includeDone: true })) as typeof fetch

    const result = await Effect.runPromise(ResponsesDriver.complete(makeReq()))

    expect(result.usage.inputTokens).toBe(500)
    expect(result.usage.outputTokens).toBe(40)
  })

  it('supports camelCase usage on response.completed.response.usage', async () => {
    globalThis.fetch = vi.fn(async () => makeSseResponse([
      {
        type: 'response.completed',
        response: {
          usage: {
            inputTokens: 100,
            outputTokens: 10,
            cachedInputTokens: 25,
          },
        },
      },
    ])) as typeof fetch

    const result = await Effect.runPromise(ResponsesDriver.complete(makeReq()))
    expect(result.usage.inputTokens).toBe(125)
    expect(result.usage.outputTokens).toBe(10)
    expect(result.usage.cacheReadTokens).toBe(25)
    expect(result.collectorData._tag).toBe('Responses')
    if (result.collectorData._tag !== 'Responses') throw new Error('Expected Responses collector data')
    expect(result.collectorData.diagnostics?.usageSource).toBe('response.completed.response.usage')
  })

  it('supports camelCase usage in non-terminal fallback events', async () => {
    globalThis.fetch = vi.fn(async () => makeSseResponse([
      {
        type: 'response.in_progress',
        usage: {
          inputTokens: 80,
          outputTokens: 8,
          inputTokensDetails: { cachedTokens: 20 },
        },
      },
    ])) as typeof fetch

    const result = await Effect.runPromise(ResponsesDriver.complete(makeReq()))
    expect(result.usage.inputTokens).toBe(100)
    expect(result.usage.outputTokens).toBe(8)
    expect(result.usage.cacheReadTokens).toBe(20)
    expect(result.collectorData._tag).toBe('Responses')
    if (result.collectorData._tag !== 'Responses') throw new Error('Expected Responses collector data')
    expect(result.collectorData.diagnostics?.usageSource).toBe('response.other')
  })

  it('ignores invalid usage payloads and reports rejection reasons', async () => {
    globalThis.fetch = vi.fn(async () => makeSseResponse([
      {
        type: 'response.in_progress',
        usage: {
          input_tokens: 'bad',
          output_tokens: null,
        },
      },
      {
        type: 'response.completed',
        response: {
          usage: {
            input_tokens: 'bad',
          },
        },
      },
    ])) as typeof fetch

    const result = await Effect.runPromise(ResponsesDriver.complete(makeReq()))

    expect(result.usage.inputTokens).toBeNull()
    expect(result.usage.outputTokens).toBeNull()
    expect(result.collectorData._tag).toBe('Responses')
    if (result.collectorData._tag !== 'Responses') throw new Error('Expected Responses collector data')
    expect(result.collectorData.diagnostics?.usageSource).toBe('none')
    expect(result.collectorData.diagnostics?.usageRejectionReasons.length).toBeGreaterThan(0)
  })

  it('recovers usage via fallback retrieve for non-terminal stream with responseId', async () => {
    const fetchMock = vi.fn()
    fetchMock
      .mockResolvedValueOnce(makeSseResponse([
        {
          type: 'response.in_progress',
          response: { id: 'resp_123' },
        },
      ], { includeDone: false }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        usage: {
          input_tokens: 9000,
          output_tokens: 90,
          input_tokens_details: { cached_tokens: 1000 },
        },
      }), { status: 200, headers: { 'content-type': 'application/json' } }))

    globalThis.fetch = fetchMock as typeof fetch

    const result = await Effect.runPromise(ResponsesDriver.complete(makeReq()))

    expect(result.usage.inputTokens).toBe(10000)
    expect(result.usage.outputTokens).toBe(90)
    expect(result.collectorData._tag).toBe('Responses')
    if (result.collectorData._tag !== 'Responses') throw new Error('Expected Responses collector data')
    expect(result.collectorData.diagnostics?.fallbackRetrieveUsed).toBe(true)
    expect(result.collectorData.diagnostics?.fallbackRetrieveSucceeded).toBe(true)
    expect(result.collectorData.diagnostics?.fallbackRetrieveUsageFound).toBe(true)
    expect(result.collectorData.diagnostics?.usageSource).toBe('fallback-retrieve.usage')
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(String(fetchMock.mock.calls[1]?.[0])).toContain('/v1/responses/resp_123')
  })

  it('records fallback retrieve attempt when retrieval fails and leaves usage null', async () => {
    const fetchMock = vi.fn()
    fetchMock
      .mockResolvedValueOnce(makeSseResponse([
        {
          type: 'response.in_progress',
          response: { id: 'resp_404' },
        },
      ], { includeDone: false }))
      .mockResolvedValueOnce(new Response('not found', { status: 404 }))

    globalThis.fetch = fetchMock as typeof fetch

    const result = await Effect.runPromise(ResponsesDriver.complete(makeReq()))

    expect(result.usage.inputTokens).toBeNull()
    expect(result.usage.outputTokens).toBeNull()
    expect(result.collectorData._tag).toBe('Responses')
    if (result.collectorData._tag !== 'Responses') throw new Error('Expected Responses collector data')
    expect(result.collectorData.diagnostics?.fallbackRetrieveUsed).toBe(true)
    expect(result.collectorData.diagnostics?.fallbackRetrieveSucceeded).toBe(false)
    expect(result.collectorData.diagnostics?.fallbackRetrieveUsageFound).toBe(false)
    expect(result.collectorData.diagnostics?.usageSource).toBe('none')
    expect(result.collectorData.diagnostics?.usageRejectionReasons.some((r) => r.includes('fallback-retrieve:http-404'))).toBe(true)
  })
})
