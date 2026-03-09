/**
 * Web Fetch Tool
 *
 * Fetches content from a URL and returns it for the agent to read.
 * Strips <script> and <style> blocks from HTML to save tokens.
 * Uses curl as primary (better TLS fingerprint), falls back to native fetch.
 * No external dependencies.
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { createTool, ToolErrorSchema } from '@magnitudedev/tools'

const MAX_RESPONSE_SIZE = 5 * 1024 * 1024 // 5MB
const TIMEOUT_SECS = 30
const MAX_REDIRECTS = 5

const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
const ACCEPT = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
const ACCEPT_LANG = 'en-US,en;q=0.9'

const WebFetchError = ToolErrorSchema('WebFetchError', {})

// =============================================================================
// curl-based fetch (primary)
// =============================================================================

interface FetchResult {
  content: string
  url: string
  contentType: string
  status: number
}

/** Fetch via curl — handles redirects, TLS natively. */
async function curlFetch(url: string): Promise<FetchResult> {
  const proc = Bun.spawn(
    [
      'curl', '-sL',
      '-w', '\n%{http_code}\n%{content_type}\n%{url_effective}',
      '-m', String(TIMEOUT_SECS),
      '--max-redirs', String(MAX_REDIRECTS),
      '--max-filesize', String(MAX_RESPONSE_SIZE),
      '-A', USER_AGENT,
      '-H', `Accept: ${ACCEPT}`,
      '-H', `Accept-Language: ${ACCEPT_LANG}`,
      url,
    ],
    { stdout: 'pipe', stderr: 'pipe' },
  )

  const raw = await new Response(proc.stdout).text()
  const exitCode = await proc.exited

  if (exitCode !== 0) {
    throw new Error(exitCode === 63 ? 'Response too large (exceeds 5MB limit)' : `curl failed (exit ${exitCode})`)
  }

  // -w appends: \n<status>\n<content_type>\n<effective_url>
  const lines = raw.split('\n')
  const effectiveUrl = lines.pop() || url
  const contentType = lines.pop() || ''
  const status = parseInt(lines.pop() || '0', 10)
  const content = lines.join('\n')

  if (status >= 400) {
    throw new Error(`HTTP ${status}`)
  }

  return { content, url: effectiveUrl, contentType, status }
}

// =============================================================================
// Native fetch fallback
// =============================================================================

async function nativeFetch(url: string): Promise<FetchResult> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_SECS * 1000)

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

    const content = await response.text()
    if (content.length > MAX_RESPONSE_SIZE) {
      throw new Error('Response too large (exceeds 5MB limit)')
    }

    return {
      content,
      url: response.url || url,
      contentType: response.headers.get('content-type') || '',
      status: response.status,
    }
  } finally {
    clearTimeout(timeout)
  }
}

// =============================================================================
// Tool Definition
// =============================================================================

export const webFetchTool = createTool({
  name: 'webFetch',
  group: 'default',
  description: 'Fetch the content of a URL. Returns the page content for you to read directly. Use this instead of running curl or wget in the shell — it uses curl under the hood.',

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
      body: 'content',
    },
  } as const,

  execute: ({ url }) =>
    Effect.tryPromise({
      try: async () => {
        if (!url.startsWith('http://') && !url.startsWith('https://')) {
          throw new Error('URL must start with http:// or https://')
        }

        let result: FetchResult

        // Try curl first (better TLS fingerprint), fall back to native fetch
        try {
          result = await curlFetch(url)
        } catch {
          result = await nativeFetch(url)
        }

        // Strip <script> and <style> blocks from HTML to save tokens
        let { content } = result
        if (result.contentType.includes('text/html')) {
          content = content.replace(/<script[\s\S]*?<\/script>/gi, '')
          content = content.replace(/<style[\s\S]*?<\/style>/gi, '')
        }

        return { content, url: result.url, contentType: result.contentType }
      },
      catch: (e) => ({
        _tag: 'WebFetchError' as const,
        message: e instanceof Error ? e.message : String(e),
      }),
    }),
})
