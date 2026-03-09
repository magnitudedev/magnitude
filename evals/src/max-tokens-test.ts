/**
 * Test script to observe BAML behavior when max_tokens is hit.
 *
 * Sets max_tokens very low (100) and asks for a long response,
 * then logs exactly what happens — truncation, error, or both.
 * Also inspects getCollectorData() for stop/finish reason signals.
 *
 * Usage: bun evals/src/max-tokens-test.ts [provider:model]
 * Default: anthropic:claude-sonnet-4-6
 */

import { setModel, primary, getAuth } from '@magnitudedev/providers'
import { BamlClientFinishReasonError, BamlClientHttpError, BamlValidationError } from '@magnitudedev/llm-core'
import { getProvider } from '@magnitudedev/providers'
import type { ChatMessage } from '@magnitudedev/llm-core'

const spec = process.argv[2] ?? 'anthropic:claude-sonnet-4-6'
const [providerId, modelId] = [spec.slice(0, spec.indexOf(':')), spec.slice(spec.indexOf(':') + 1)]

console.log(`\n=== max_tokens truncation test ===`)
console.log(`Provider: ${providerId}`)
console.log(`Model:    ${modelId}\n`)

// Override the model's maxOutputTokens to something tiny
const provider = getProvider(providerId)
if (provider) {
  const model = provider.models.find(m => m.id === modelId)
  if (model) {
    console.log(`Original maxOutputTokens: ${model.maxOutputTokens}`)
    model.maxOutputTokens = 100
    console.log(`Overridden to: 100\n`)
  } else {
    console.log(`Model ${modelId} not in fallback registry, adding with maxOutputTokens=100\n`)
    provider.models.push({ id: modelId, name: modelId, supportsToolCalls: true, maxOutputTokens: 100 })
  }
}

const auth = getAuth(providerId)
setModel('primary', providerId, modelId, auth ?? null, false)

const systemPrompt = 'You are a helpful assistant.'
const messages: ChatMessage[] = [
  { role: 'user', content: ['Write a very long, detailed essay about the history of computing. Include at least 20 paragraphs covering everything from Babbage to modern AI. Do not stop early.'] },
]

console.log('--- Streaming (primary.chat) ---')
try {
  const chatStream = primary.chat(systemPrompt, messages, undefined, undefined, '<lenses>task: no</lenses>\n<comms>\n<message>Ready.</message>\n</comms>')
  let result = ''
  let chunkCount = 0
  for await (const chunk of chatStream.stream) {
    result += chunk
    chunkCount++
  }
  console.log(`Completed without error`)
  console.log(`Chunks received: ${chunkCount}`)
  console.log(`Output length: ${result.length} chars`)
  console.log(`Output tokens (approx): ~${Math.ceil(result.length / 4)}`)
  console.log(`Ends mid-sentence: ${!result.trimEnd().endsWith('.')}`)
  console.log(`Last 100 chars: ${JSON.stringify(result.slice(-100))}`)

  // Check usage
  const usage = chatStream.getUsage()
  console.log(`\n--- Usage ---`)
  console.log(JSON.stringify(usage, null, 2))

  // Check collector data for stop reason
  const collectorData = chatStream.getCollectorData()
  console.log(`\n--- Collector Data ---`)
  console.log(`rawRequestBody keys: ${collectorData.rawRequestBody ? Object.keys(collectorData.rawRequestBody as object).join(', ') : 'null'}`)

  // Look for stop_reason in response body
  const rawResp = collectorData.rawResponseBody as Record<string, unknown> | null
  if (rawResp) {
    console.log(`rawResponseBody keys: ${Object.keys(rawResp).join(', ')}`)
    if ('stop_reason' in rawResp) console.log(`stop_reason: ${rawResp.stop_reason}`)
    if ('finish_reason' in rawResp) console.log(`finish_reason: ${rawResp.finish_reason}`)
    if ('choices' in rawResp) {
      const choices = rawResp.choices as Array<Record<string, unknown>>
      for (const c of choices) {
        console.log(`choice finish_reason: ${c.finish_reason}`)
      }
    }
  } else {
    console.log(`rawResponseBody: null`)
  }

  // Look for stop reason in last SSE event
  const sseEvents = collectorData.sseEvents
  if (sseEvents && sseEvents.length > 0) {
    console.log(`\nSSE events count: ${sseEvents.length}`)
    // Check last few events for stop signals
    const lastEvents = sseEvents.slice(-3)
    for (let i = 0; i < lastEvents.length; i++) {
      const evt = lastEvents[i] as Record<string, unknown> | null
      if (!evt) continue
      const idx = sseEvents.length - 3 + i
      console.log(`\nSSE event [${idx}]:`)
      console.log(JSON.stringify(evt, null, 2))
    }
  } else {
    console.log(`SSE events: ${sseEvents === null ? 'null' : 'empty'}`)
  }

} catch (error) {
  const err = error as Error
  console.log(`Error type: ${err.constructor.name}`)
  console.log(`Message: ${err.message}`)
  if (err instanceof BamlClientFinishReasonError) {
    console.log(`Finish reason: ${err.finish_reason}`)
    console.log(`Raw output length: ${err.raw_output.length} chars`)
    console.log(`Raw output last 100: ${JSON.stringify(err.raw_output.slice(-100))}`)
  }
  if (err instanceof BamlClientHttpError) {
    console.log(`Status: ${err.status_code}`)
    console.log(`Detailed: ${err.detailed_message?.slice(0, 300)}`)
    console.log(`Raw response: ${err.raw_response?.slice(0, 300)}`)
  }
  if (err instanceof BamlValidationError) {
    console.log(`Detailed: ${err.detailed_message?.slice(0, 300)}`)
    console.log(`Raw output length: ${err.raw_output.length} chars`)
  }
}

console.log('\nDone.')
