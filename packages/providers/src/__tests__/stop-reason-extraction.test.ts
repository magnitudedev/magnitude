/**
 * Empirical test: Can we reliably extract stop reason from BAML collector?
 * Run with: cd packages/providers && bunx --bun vitest run src/__tests__/stop-reason-extraction.test.ts
 */
import { describe, it, expect } from 'vitest'
import { Collector } from '@magnitudedev/llm-core'
import { buildClientRegistry } from '../client-registry-builder'
import { bamlStream } from '../drivers/baml-dispatch'
import { toIncrementalStream } from '../util/incremental-stream'

const STOP_SEQUENCES = ['<yield-user/>', '<yield-invoke/>', '<yield-worker/>']

async function runWithCollector(provider: string, modelId: string, auth: any) {
  const collector = new Collector('test')
  const registry = buildClientRegistry(provider, modelId, auth, undefined, [...STOP_SEQUENCES])

  const messages = [{ role: 'user', content: ['Respond with exactly this text:\n\n<yield-user/>\n\nDo not add any other text.'] }]

  const rawStream = bamlStream('CodingAgentChat', [
    'You are a helpful assistant.',
    messages,
    '',
    false,
    true,
  ], { clientRegistry: registry, collector, signal: undefined })

  const chunks: string[] = []
  for await (const chunk of toIncrementalStream(rawStream)) {
    chunks.push(chunk)
  }

  const content = chunks.join('')
  const lastCall = collector.last?.calls.at(-1) as any

  let httpStopReason: string | null = null
  let httpStopSequence: string | null = null
  let httpFinishReason: string | null = null
  try {
    const body = lastCall?.httpResponse?.body?.json?.()
    if (body) {
      httpStopReason = body.stop_reason ?? null
      httpStopSequence = body.stop_sequence ?? null
      httpFinishReason = body.choices?.[0]?.finish_reason ?? null
    }
  } catch {}

  let sseStopReason: string | null = null
  let sseStopSequence: string | null = null
  let sseFinishReason: string | null = null
  try {
    const sseResponses = lastCall?.sseResponses?.() ?? null
    if (Array.isArray(sseResponses)) {
      for (const sse of sseResponses) {
        const d = sse.json?.()
        if (!d) continue
        if (d.delta?.stop_reason) {
          sseStopReason = d.delta.stop_reason
          sseStopSequence = d.delta.stop_sequence ?? null
        }
        if (d.choices?.[0]?.finish_reason) {
          sseFinishReason = d.choices[0].finish_reason
        }
      }
    }
  } catch {}

  return { content, httpStopReason, httpStopSequence, httpFinishReason, sseStopReason, sseStopSequence, sseFinishReason }
}

describe('stop reason extraction from collector', () => {
  it('anthropic: extracts stop_sequence from collector', async () => {
    const key = process.env.ANTHROPIC_API_KEY
    if (!key) { console.log('SKIP: no ANTHROPIC_API_KEY'); return }

    const result = await runWithCollector('anthropic', 'claude-sonnet-4-20250514', {
      type: 'api-key', apiKey: key,
    })
    console.log('Anthropic result:', JSON.stringify(result, null, 2))

    expect(result.content).not.toContain('<yield-user/>')
    const stopSeq = result.httpStopSequence ?? result.sseStopSequence
    expect(stopSeq).toBe('<yield-user/>')
  }, 30000)

  it('openai: extracts finish_reason from collector', async () => {
    const key = process.env.OPENAI_API_KEY
    if (!key) { console.log('SKIP: no OPENAI_API_KEY'); return }

    const result = await runWithCollector('openai', 'gpt-4o-mini', {
      type: 'api-key', apiKey: key,
    })
    console.log('OpenAI result:', JSON.stringify(result, null, 2))

    expect(result.content).not.toContain('<yield-user/>')
    const fr = result.httpFinishReason ?? result.sseFinishReason
    expect(fr).toBe('stop')
  }, 30000)
})
