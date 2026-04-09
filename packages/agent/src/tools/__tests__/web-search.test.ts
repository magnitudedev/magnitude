import { describe, expect, test } from 'bun:test'
import { Effect, Layer } from 'effect'
import { noopToolContext } from '@magnitudedev/tools'
import { ProviderAuth, ProviderState } from '@magnitudedev/providers'
import * as webSearchModule from '../web-search'
import { webSearchTool } from '../web-search-tool'
import { __testOnly, openrouterWebSearch } from '../web-search-openrouter'

type ProviderId = string | null
type SlotModels = Record<string, { providerId: ProviderId; modelId?: string } | undefined>

function makeProviderState(providerIdOrSlots: ProviderId | SlotModels, modelId?: string) {
  const slots: SlotModels =
    typeof providerIdOrSlots === 'object' && providerIdOrSlots !== null
      ? providerIdOrSlots
      : { lead: { providerId: providerIdOrSlots, modelId } }

  return Layer.succeed(ProviderState, {
    peek: (slot: string) => {
      const entry = slots[slot]
      return Effect.succeed(entry?.providerId ? { model: { providerId: entry.providerId, id: entry.modelId } } : null)
    },
  } as any)
}

function makeProviderAuth(authByProvider: Record<string, any | undefined>) {
  return Layer.succeed(ProviderAuth, {
    getAuth: (providerId: string) => Effect.succeed(authByProvider[providerId]),
  } as any)
}

function runWebSearch(
  query: string,
  providerIdOrSlots: ProviderId | SlotModels,
  authByProvider: Record<string, any | undefined> = {},
  modelId?: string,
) {
  return Effect.runPromise(
    webSearchModule.webSearch(query).pipe(
      Effect.provide(Layer.mergeAll(makeProviderState(providerIdOrSlots, modelId), makeProviderAuth(authByProvider))),
    ) as any,
  )
}

async function withPatchedFetch<T>(fetchImpl: typeof fetch, run: () => Promise<T>): Promise<T> {
  const originalFetch = globalThis.fetch
  ;(globalThis as any).fetch = fetchImpl
  try {
    return await run()
  } finally {
    ;(globalThis as any).fetch = originalFetch
  }
}

async function withEnv<T>(patch: Record<string, string | undefined>, run: () => Promise<T>): Promise<T> {
  const previous = Object.fromEntries(Object.keys(patch).map((key) => [key, process.env[key]]))
  for (const [key, value] of Object.entries(patch)) {
    if (value === undefined) delete process.env[key]
    else process.env[key] = value
  }

  try {
    return await run()
  } finally {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]
      else process.env[key] = value
    }
  }
}

describe('web search integration', () => {
  test('lead=openrouter uses the OpenRouter adapter', async () => {
    let captured: { url?: string; init?: RequestInit } = {}

    const result = await withPatchedFetch(
      (async (url: string | URL | Request, init?: RequestInit) => {
        captured = { url: String(url), init }
        return new Response(JSON.stringify({
          output: [{ type: 'message', content: [{ type: 'output_text', text: 'OpenRouter answer' }] }],
          usage: { input_tokens: 12, output_tokens: 34, server_tool_use: { web_search_requests: 1 } },
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }) as any,
      () => runWebSearch('latest news', 'openrouter', {
        openrouter: { type: 'api', key: 'stored-openrouter-key' },
      }),
    )

    expect(result.textResponse).toBe('OpenRouter answer')
    expect(captured.url).toBe('https://openrouter.ai/api/v1/responses')
    expect(captured.init?.headers).toMatchObject({
      Authorization: 'Bearer stored-openrouter-key',
    })
  })

  test('unsupported lead + worker=openrouter falls through to worker', async () => {
    let captured: { url?: string; init?: RequestInit } = {}

    const result = await withPatchedFetch(
      (async (url: string | URL | Request, init?: RequestInit) => {
        captured = { url: String(url), init }
        return new Response(JSON.stringify({
          output: [{ type: 'message', content: [{ type: 'output_text', text: 'Worker OpenRouter answer' }] }],
          usage: { input_tokens: 1, output_tokens: 2, server_tool_use: { web_search_requests: 1 } },
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }) as any,
      () => runWebSearch('latest news', {
        lead: { providerId: 'amazon-bedrock' },
        builder: { providerId: 'openrouter' },
      }, {
        openrouter: { type: 'api', key: 'worker-openrouter-key' },
      }),
    )

    expect(result.textResponse).toBe('Worker OpenRouter answer')
    expect(captured.url).toBe('https://openrouter.ai/api/v1/responses')
    expect(captured.init?.headers).toMatchObject({
      Authorization: 'Bearer worker-openrouter-key',
    })
  })

  test('lead=openai, worker=openrouter prefers lead', async () => {
    let capturedUrl = ''

    const encoder = new TextEncoder()
    const sseBody = [
      'data: {"type":"response.output_text.delta","delta":"Lead OpenAI answer"}\n',
      'data: {"type":"response.completed","response":{"output_text":"Lead OpenAI answer","output":[],"usage":{"input_tokens":1,"output_tokens":2}}}\n',
      'data: [DONE]\n',
    ].join('\n')

    const result = await withPatchedFetch(
      (async (url: string | URL | Request) => {
        capturedUrl = String(url)
        return new Response(encoder.encode(sseBody), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        })
      }) as any,
      () => runWebSearch('latest news', {
        lead: { providerId: 'openai' },
        builder: { providerId: 'openrouter' },
      }, {
        openai: { type: 'oauth', accessToken: 'lead-openai-oauth' },
        openrouter: { type: 'api', key: 'worker-openrouter-key' },
      }),
    )

    expect(result.textResponse).toBe('Lead OpenAI answer')
    expect(capturedUrl).toBe('https://chatgpt.com/backend-api/codex/responses')
  })

  test('openai oauth merges annotation citations first, then unseen web_search_call sources', async () => {
    const encoder = new TextEncoder()
    const sseBody = [
      'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","action":{"sources":[{"url":"https://example.com/source-b"},{"url":"https://example.com/source-c"}]}}}\n',
      'data: {"type":"response.output_item.done","output_index":1,"item":{"type":"message","content":[{"type":"output_text","text":"Answer with sources","annotations":[{"type":"url_citation","title":"Example A","url":"https://example.com/source-a"},{"type":"url_citation","title":"Example B","url":"https://example.com/source-b"}]}]}}\n',
      'data: {"type":"response.completed","response":{"output_text":"Answer with sources","output":[],"usage":{"input_tokens":5,"output_tokens":7}}}\n',
      'data: [DONE]\n',
    ].join('\n')

    const result = await withPatchedFetch(
      (async () =>
        new Response(encoder.encode(sseBody), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        })) as any,
      () => runWebSearch('latest news', 'openai', {
        openai: { type: 'oauth', accessToken: 'oauth-token' },
      }),
    )

    expect(result.textResponse).toBe('Answer with sources')
    expect(result.results).toEqual([
      {
        tool_use_id: 'openai-search',
        content: [
          { title: 'Example A', url: 'https://example.com/source-a' },
          { title: 'Example B', url: 'https://example.com/source-b' },
          { title: 'https://example.com/source-c', url: 'https://example.com/source-c' },
        ],
      },
    ])
    expect(result.usage).toEqual({
      input_tokens: 5,
      output_tokens: 7,
      web_search_requests: 1,
    })
  })

  test('openai oauth extracts citations from streamed message annotations when completed output is empty', async () => {
    const encoder = new TextEncoder()
    const sseBody = [
      'data: {"type":"response.output_item.done","output_index":1,"item":{"type":"message","content":[{"type":"output_text","text":"Answer with annotation-only sources","annotations":[{"type":"url_citation","title":"Example A","url":"https://example.com/source-a"},{"type":"url_citation","title":"Example B","url":"https://example.com/source-b"}]}]}}\n',
      'data: {"type":"response.completed","response":{"output_text":"Answer with annotation-only sources","output":[],"usage":{"input_tokens":9,"output_tokens":11}}}\n',
      'data: [DONE]\n',
    ].join('\n')

    const result = await withPatchedFetch(
      (async () =>
        new Response(encoder.encode(sseBody), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        })) as any,
      () => runWebSearch('latest news', 'openai', {
        openai: { type: 'oauth', accessToken: 'oauth-token' },
      }),
    )

    expect(result.textResponse).toBe('Answer with annotation-only sources')
    expect(result.results).toEqual([
      {
        tool_use_id: 'openai-search',
        content: [
          { title: 'Example A', url: 'https://example.com/source-a' },
          { title: 'Example B', url: 'https://example.com/source-b' },
        ],
      },
    ])
    expect(result.usage).toEqual({
      input_tokens: 9,
      output_tokens: 11,
      web_search_requests: 0,
    })
  })

  test('openai oauth dedupes merged sources by URL with annotation entries winning order/title', async () => {
    const encoder = new TextEncoder()
    const sseBody = [
      'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","action":{"sources":[{"url":"https://example.com/source-a"},{"url":"https://example.com/source-a"},{"url":"https://example.com/source-b"},{"url":"https://example.com/source-c"}]}}}\n',
      'data: {"type":"response.output_item.done","output_index":1,"item":{"type":"message","content":[{"type":"output_text","text":"Answer with mixed sources","annotations":[{"type":"url_citation","title":"Annotated A","url":"https://example.com/source-a"},{"type":"url_citation","title":"Annotated B","url":"https://example.com/source-b"}]}]}}\n',
      'data: {"type":"response.completed","response":{"output_text":"Answer with mixed sources","output":[],"usage":{"input_tokens":6,"output_tokens":8}}}\n',
      'data: [DONE]\n',
    ].join('\n')

    const result = await withPatchedFetch(
      (async () =>
        new Response(encoder.encode(sseBody), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        })) as any,
      () => runWebSearch('latest news', 'openai', {
        openai: { type: 'oauth', accessToken: 'oauth-token' },
      }),
    )

    expect(result.textResponse).toBe('Answer with mixed sources')
    expect(result.results).toEqual([
      {
        tool_use_id: 'openai-search',
        content: [
          { title: 'Annotated A', url: 'https://example.com/source-a' },
          { title: 'Annotated B', url: 'https://example.com/source-b' },
          { title: 'https://example.com/source-c', url: 'https://example.com/source-c' },
        ],
      },
    ])
    expect(result.usage).toEqual({
      input_tokens: 6,
      output_tokens: 8,
      web_search_requests: 1,
    })
  })

  test('openai oauth falls back to web_search_call.action.sources when annotations are absent and completed output is empty', async () => {
    const encoder = new TextEncoder()
    const sseBody = [
      'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","action":{"sources":[{"url":"https://example.com/source-a"},{"url":"https://example.com/source-b"}]}}}\n',
      'data: {"type":"response.completed","response":{"output_text":"Answer with call-only sources","output":[],"usage":{"input_tokens":6,"output_tokens":8}}}\n',
      'data: [DONE]\n',
    ].join('\n')

    const result = await withPatchedFetch(
      (async () =>
        new Response(encoder.encode(sseBody), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        })) as any,
      () => runWebSearch('latest news', 'openai', {
        openai: { type: 'oauth', accessToken: 'oauth-token' },
      }),
    )

    expect(result.textResponse).toBe('Answer with call-only sources')
    expect(result.results).toEqual([
      {
        tool_use_id: 'openai-search',
        content: [
          { title: 'https://example.com/source-a', url: 'https://example.com/source-a' },
          { title: 'https://example.com/source-b', url: 'https://example.com/source-b' },
        ],
      },
    ])
    expect(result.usage).toEqual({
      input_tokens: 6,
      output_tokens: 8,
      web_search_requests: 1,
    })
  })

  test('no search-capable lead/workers errors clearly', async () => {
    await expect(runWebSearch('latest news', {
      lead: { providerId: 'amazon-bedrock' },
      explorer: { providerId: 'google-vertex-anthropic' },
      builder: { providerId: null },
    })).rejects.toThrow('No supported web-search backend is configured on the lead or worker slots')
  })

  test('MAGNITUDE_SEARCH_PROVIDER=openrouter overrides provider detection', async () => {
    await withEnv({ MAGNITUDE_SEARCH_PROVIDER: 'openrouter' }, async () => {
      const result = await withPatchedFetch(
        (async () => new Response(JSON.stringify({
          output: [{ type: 'message', content: [{ type: 'output_text', text: 'Override answer' }] }],
          usage: { input_tokens: 1, output_tokens: 2 },
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })) as any,
        () => runWebSearch('latest news', 'google', {
          openrouter: { type: 'api', key: 'override-key' },
          google: { type: 'api', key: 'google-key' },
        }),
      )

      expect(result.textResponse).toBe('Override answer')
    })
  })

  test('invalid override value lists openrouter and github-copilot', async () => {
    await withEnv({ MAGNITUDE_SEARCH_PROVIDER: 'bad-provider' }, async () => {
      await expect(runWebSearch('latest news', 'openrouter', {
        openrouter: { type: 'api', key: 'stored-openrouter-key' },
      })).rejects.toThrow('openrouter')
      await expect(runWebSearch('latest news', 'openrouter', {
        openrouter: { type: 'api', key: 'stored-openrouter-key' },
      })).rejects.toThrow('github-copilot')
    })
  })

  test('stored auth is preferred over OPENROUTER_API_KEY', async () => {
    await withEnv({ OPENROUTER_API_KEY: 'env-openrouter-key' }, async () => {
      let capturedAuth = ''

      await withPatchedFetch(
        (async (_url: string | URL | Request, init?: RequestInit) => {
          capturedAuth = (init?.headers as Record<string, string>).Authorization
          return new Response(JSON.stringify({
            output: [{ type: 'message', content: [{ type: 'output_text', text: 'Stored auth answer' }] }],
            usage: { input_tokens: 1, output_tokens: 2, server_tool_use: { web_search_requests: 1 } },
          }), { status: 200, headers: { 'Content-Type': 'application/json' } })
        }) as any,
        () => runWebSearch('latest news', 'openrouter', {
          openrouter: { type: 'api', key: 'stored-openrouter-key' },
        }),
      )

      expect(capturedAuth).toBe('Bearer stored-openrouter-key')
    })
  })

  test('OPENROUTER_API_KEY env fallback works', async () => {
    await withEnv({ OPENROUTER_API_KEY: 'env-openrouter-key' }, async () => {
      let capturedAuth = ''

      const result = await withPatchedFetch(
        (async (_url: string | URL | Request, init?: RequestInit) => {
          capturedAuth = (init?.headers as Record<string, string>).Authorization
          return new Response(JSON.stringify({
            output: [{ type: 'message', content: [{ type: 'output_text', text: 'Env auth answer' }] }],
            usage: { input_tokens: 1, output_tokens: 2 },
          }), { status: 200, headers: { 'Content-Type': 'application/json' } })
        }) as any,
        () => runWebSearch('latest news', 'openrouter'),
      )

      expect(result.textResponse).toBe('Env auth answer')
      expect(capturedAuth).toBe('Bearer env-openrouter-key')
    })
  })

  test('missing OpenRouter auth errors clearly', async () => {
    await withEnv({ OPENROUTER_API_KEY: undefined }, async () => {
      await expect(runWebSearch('latest news', 'openrouter')).rejects.toThrow('OPENROUTER_API_KEY')
    })
  })

  test('OpenRouter response normalization merges annotation + markdown + additional sources with stable dedupe', async () => {
    let requestBody: any

    const result = await withPatchedFetch(
      (async (_url: string | URL | Request, init?: RequestInit) => {
        requestBody = JSON.parse(String(init?.body))
        return new Response(JSON.stringify({
          output: [
            {
              type: 'message',
              sources: [
                { title: 'Additional B', url: 'https://example.com/b' },
                { title: 'Duplicate C from additional', href: 'https://example.com/c' },
              ],
              content: [
                {
                  type: 'output_text',
                  text: 'Search summary with [Extra C](https://example.com/c) and [Duplicate A](https://example.com/a)',
                  annotations: [
                    { type: 'url_citation', title: 'Example', url: 'https://example.com/a' },
                    { type: 'url_citation', title: 'Duplicate', url: 'https://example.com/a' },
                  ],
                  sources: [{ title: 'Additional D', url: 'https://example.com/d' }],
                },
              ],
            },
            {
              type: 'web_search_call',
              action: {
                sources: [
                  { title: 'Additional E', url: 'https://example.com/e' },
                  'https://example.com/d',
                ],
              },
            },
          ],
          usage: {
            input_tokens: 11,
            output_tokens: 22,
          },
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }) as any,
      () => openrouterWebSearch('best editor', { type: 'api-key', value: 'key' }, {
        system: 'answer tersely',
        allowed_domains: ['example.com'],
        blocked_domains: ['blocked.com'],
        model: 'anthropic/claude-3.5-sonnet',
      }),
    )

    expect(result).toEqual({
      query: 'best editor',
      results: [{
        tool_use_id: 'openrouter-search',
        content: [
          { title: 'Example', url: 'https://example.com/a' },
          { title: 'Extra C', url: 'https://example.com/c' },
          { title: 'Additional B', url: 'https://example.com/b' },
          { title: 'Additional D', url: 'https://example.com/d' },
          { title: 'Additional E', url: 'https://example.com/e' },
        ],
      }],
      textResponse: 'Search summary with [Extra C](https://example.com/c) and [Duplicate A](https://example.com/a)',
      usage: {
        input_tokens: 11,
        output_tokens: 22,
        web_search_requests: 1,
      },
    })

    expect(requestBody).toMatchObject({
      model: 'openai/gpt-5.4',
      input: 'best editor',
      instructions: 'answer tersely',
      reasoning: { effort: 'none' },
      tools: [{
        type: 'openrouter:web_search',
        parameters: {
          allowed_domains: ['example.com'],
          excluded_domains: ['blocked.com'],
        },
      }],
    })
  })

  test('OpenRouter normalization keeps annotation-only behavior unchanged', async () => {
    const result = await withPatchedFetch(
      (async () => new Response(JSON.stringify({
        output: [
          {
            type: 'message',
            content: [{
              type: 'output_text',
              text: 'Search summary',
              annotations: [
                { type: 'url_citation', title: 'Example', url: 'https://example.com/a' },
                { type: 'url_citation', title: 'Another', url: 'https://example.com/b' },
              ],
            }],
          },
        ],
        usage: { input_tokens: 1, output_tokens: 2 },
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })) as any,
      () => openrouterWebSearch('best editor', { type: 'api-key', value: 'key' }),
    )

    expect(result.results).toEqual([{
      tool_use_id: 'openrouter-search',
      content: [
        { title: 'Example', url: 'https://example.com/a' },
        { title: 'Another', url: 'https://example.com/b' },
      ],
    }])
  })

  test('OpenRouter normalization ignores unknown and empty source buckets safely', async () => {
    const result = await withPatchedFetch(
      (async () => new Response(JSON.stringify({
        output: [
          {
            type: 'message',
            unknown_sources: [{ title: 'Ignored', url: 'https://example.com/ignored' }],
            action: { sources: [null, 42, {}, { href: 'https://example.com/f' }] },
            content: [{
              type: 'output_text',
              text: 'Search summary',
              annotations: [],
              sources: [null, { title: 'Additional G', url: 'https://example.com/g' }, { foo: 'bar' }],
            }],
          },
        ],
        usage: { input_tokens: 5, output_tokens: 6 },
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })) as any,
      () => openrouterWebSearch('best editor', { type: 'api-key', value: 'key' }),
    )

    expect(result.results).toEqual([{
      tool_use_id: 'openrouter-search',
      content: [
        { title: 'https://example.com/f', url: 'https://example.com/f' },
        { title: 'Additional G', url: 'https://example.com/g' },
      ],
    }])
  })

  test('OpenRouter adapter surfaces HTTP failures', async () => {
    await withPatchedFetch(
      (async () => new Response('bad gateway', { status: 502 })) as any,
      async () => {
        await expect(
          openrouterWebSearch('best editor', { type: 'api-key', value: 'key' }),
        ).rejects.toThrow('OpenRouter web search error 502: bad gateway')
      },
    )
  })

  test('tool contract remains compatible with normalized OpenRouter response', async () => {
    await withEnv({ OPENROUTER_API_KEY: 'env-openrouter-key' }, async () => {
      const result = await withPatchedFetch(
        (async () => new Response(JSON.stringify({
          output: [
            {
              type: 'message',
              content: [{
                type: 'output_text',
                text: 'Search summary',
                annotations: [
                  { type: 'url_citation', title: 'Example', url: 'https://example.com/a' },
                  { type: 'url_citation', title: 'Another', url: 'https://example.com/b' },
                ],
              }],
            },
          ],
          usage: {
            input_tokens: 1,
            output_tokens: 2,
            server_tool_use: { web_search_requests: 1 },
          },
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })) as any,
        () =>
          Effect.runPromise(
            webSearchTool.execute({ query: 'best editor' }, noopToolContext).pipe(
              Effect.provide(Layer.mergeAll(
                makeProviderState('openrouter'),
                makeProviderAuth({}),
              )),
            ) as any,
          ),
      )

      expect(result).toEqual({
        text: 'Search summary',
        sources: [
          { title: 'Example', url: 'https://example.com/a' },
          { title: 'Another', url: 'https://example.com/b' },
        ],
      })
    })
  })

  test('OpenRouter always uses fixed gpt-5.4 model and omits empty parameters', async () => {
    let requestBody: any
    await withPatchedFetch(
      (async (_input, init) => {
        requestBody = JSON.parse(String(init?.body ?? '{}'))
        return new Response(JSON.stringify({
          output_text: 'Search summary',
          usage: {
            input_tokens: 3,
            output_tokens: 4,
            server_tool_use: { web_search_requests: 0 },
          },
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }) as any,
      () => openrouterWebSearch('best editor', { type: 'api-key', value: 'key' }, {
        model: 'google/gemini-2.5-pro',
      }),
    )

    expect(requestBody).toMatchObject({
      model: 'openai/gpt-5.4',
      input: 'best editor',
      reasoning: { effort: 'none' },
      tools: [{ type: 'openrouter:web_search' }],
    })
    expect(requestBody.tools[0]).not.toHaveProperty('parameters')
  })

  test('normalization helpers merge buckets with first-seen title and count fallback search usage', async () => {
    const fromAnnotations = __testOnly.extractCitations([
      {
        type: 'message',
        content: [{
          type: 'output_text',
          text: 'Search summary',
          annotations: [
            { type: 'url_citation', title: 'Example', url: 'https://example.com/a' },
            { type: 'url_citation', title: 'Duplicate', url: 'https://example.com/a' },
          ],
        }],
      },
    ] as any)

    const fromMarkdown = __testOnly.extractMarkdownSources(
      'See [A markdown duplicate](https://example.com/a) and [C](https://example.com/c)',
    )
    const fromAdditional = __testOnly.extractAdditionalSources([
      {
        type: 'message',
        sources: [{ title: 'B from additional', url: 'https://example.com/b' }],
      },
    ] as any)

    const results = __testOnly.mergeSourceBuckets(fromAnnotations, fromMarkdown, fromAdditional)

    expect(results).toEqual([{
      tool_use_id: 'openrouter-search',
      content: [
        { title: 'Example', url: 'https://example.com/a' },
        { title: 'C', url: 'https://example.com/c' },
        { title: 'B from additional', url: 'https://example.com/b' },
      ],
    }])

    expect(__testOnly.extractText([
      { type: 'message', content: [{ type: 'output_text', text: 'hello ' }, { type: 'output_text', text: 'world' }] },
    ] as any)).toBe('hello world')

    expect(__testOnly.countSearchRequests({ usage: {} } as any, results)).toBe(1)
    expect(__testOnly.countSearchRequests({ usage: { server_tool_use: { web_search_requests: 3 } } } as any, results)).toBe(3)
  })
})
