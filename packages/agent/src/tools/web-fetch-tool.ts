/**
 * Web Fetch Tool
 *
 * Fetches content from a URL and returns it as cleaned markdown.
 * Uses dom-extract to convert HTML into agent-readable markdown.
 * Bun's fetch uses BoringSSL with a Chrome-like TLS fingerprint,
 * so no need for curl to avoid bot detection.
 */

import { Effect, Schedule } from 'effect'
import { Schema } from '@effect/schema'
import { createTool, ToolErrorSchema } from '@magnitudedev/tools'
import { extractHtml } from '@magnitudedev/dom-extract'

const MAX_RESPONSE_SIZE = 5 * 1024 * 1024 // 5MB
const TIMEOUT_MS = 30_000

const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
const ACCEPT = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
const ACCEPT_LANG = 'en-US,en;q=0.9'

const WebFetchError = ToolErrorSchema('WebFetchError', {})

const retryPolicy = Schedule.exponential('500 millis').pipe(
  Schedule.compose(Schedule.recurs(2)),
)

const fetchPage = (url: string) =>
  Effect.tryPromise({
    try: async () => {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS)

      try {
        const response = await fetch(url, {
          signal: controller.signal,
          headers: { 'User-Agent': USER_AGENT, 'Accept': ACCEPT, 'Accept-Language': ACCEPT_LANG },
          redirect: 'follow',
        })

        if (!response.ok) {
          throw new Error(`HTTP ${response.status} ${response.statusText}`)
        }

        const contentLength = response.headers.get('content-length')
        if (contentLength && parseInt(contentLength) > MAX_RESPONSE_SIZE) {
          throw new Error('Response too large (exceeds 5MB limit)')
        }

        const raw = await response.text()
        if (raw.length > MAX_RESPONSE_SIZE) {
          throw new Error('Response too large (exceeds 5MB limit)')
        }

        const contentType = response.headers.get('content-type') || ''
        const content = contentType.includes('text/html') ? extractHtml(raw) : raw

        return { url: response.url || url, content }
      } finally {
        clearTimeout(timeout)
      }
    },
    catch: (e) => ({
      _tag: 'WebFetchError' as const,
      message: e instanceof Error ? e.message : String(e),
    }),
  })

// =============================================================================
// Tool Definition
// =============================================================================

export const webFetchTool = createTool({
  name: 'web-fetch',
  group: 'default',
  description: 'Fetch the content of a URL. Returns the page content as cleaned markdown for you to read directly. Use this instead of running curl or wget in the shell.',

  inputSchema: Schema.Struct({
    url: Schema.String,
  }),

  outputSchema: Schema.Unknown,
  errorSchema: WebFetchError,

  argMapping: ['url'],
  bindings: {
    xmlInput: { type: 'tag', body: 'url' },
    xmlOutput: {
      type: 'tag' as const,
      childTags: [{ field: 'url', tag: 'url' }, { field: 'content', tag: 'content' }],
    },
  } as const,

  execute: ({ url }) => {
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      return Effect.fail({
        _tag: 'WebFetchError' as const,
        message: 'URL must start with http:// or https://',
      })
    }

    return fetchPage(url).pipe(Effect.retry(retryPolicy))
  },
})