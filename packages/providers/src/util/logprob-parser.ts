/**
 * Parse per-token logprobs from OpenAI-compatible SSE events or complete response bodies.
 */

export type { TokenWithLogprob, TopLogprob } from '@magnitudedev/tracing'
import type { TokenWithLogprob, TopLogprob } from '@magnitudedev/tracing'

/**
 * Extract logprobs from SSE event chunks.
 * OpenAI SSE format: data: {"choices":[{"delta":{...},"logprobs":{"content":[{"token":"...","logprob":-0.5,"top_logprobs":[...]}]}}]}
 */
export function parseLogprobsFromSSE(sseEvents: unknown[]): TokenWithLogprob[] | undefined {
  const tokens: TokenWithLogprob[] = []

  for (const event of sseEvents) {
    if (!event || typeof event !== 'object') continue

    const choices = (event as any).choices
    if (!Array.isArray(choices) || choices.length === 0) continue

    const choice = choices[0]
    const logprobs = choice?.logprobs
    if (!logprobs) continue

    const content = logprobs.content
    if (!Array.isArray(content)) continue

    for (const entry of content) {
      if (!entry || typeof entry !== 'object') continue

      const token = entry.token
      const logprob = entry.logprob
      if (typeof token !== 'string' || typeof logprob !== 'number') continue

      const topLogprobs: TopLogprob[] = []
      const rawTop = entry.top_logprobs
      if (Array.isArray(rawTop)) {
        for (const alt of rawTop) {
          if (alt && typeof alt === 'object' && typeof alt.token === 'string' && typeof alt.logprob === 'number') {
            topLogprobs.push({ token: alt.token, logprob: alt.logprob })
          }
        }
      }

      tokens.push({ token, logprob, topLogprobs })
    }
  }

  return tokens.length > 0 ? tokens : undefined
}

/**
 * Extract logprobs from a complete (non-streaming) response body.
 * OpenAI format: {"choices":[{"message":{...},"logprobs":{"content":[...]}}]}
 */
export function parseLogprobsFromCompleteBody(body: unknown): TokenWithLogprob[] | undefined {
  if (!body || typeof body !== 'object') return undefined

  const choices = (body as any).choices
  if (!Array.isArray(choices) || choices.length === 0) return undefined

  const choice = choices[0]
  const logprobs = choice?.logprobs
  if (!logprobs) return undefined

  const content = logprobs.content
  if (!Array.isArray(content)) return undefined

  const tokens: TokenWithLogprob[] = []
  for (const entry of content) {
    if (!entry || typeof entry !== 'object') continue

    const token = entry.token
    const logprob = entry.logprob
    if (typeof token !== 'string' || typeof logprob !== 'number') continue

    const topLogprobs: TopLogprob[] = []
    const rawTop = entry.top_logprobs
    if (Array.isArray(rawTop)) {
      for (const alt of rawTop) {
        if (alt && typeof alt === 'object' && typeof alt.token === 'string' && typeof alt.logprob === 'number') {
          topLogprobs.push({ token: alt.token, logprob: alt.logprob })
        }
      }
    }

    tokens.push({ token, logprob, topLogprobs })
  }

  return tokens.length > 0 ? tokens : undefined
}