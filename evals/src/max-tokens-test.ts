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

import { Effect, Layer, Stream } from 'effect'
import { ModelResolver, makeModelResolver, makeNoopTracer, CodingAgentChat } from '@magnitudedev/providers'
import { BamlClientFinishReasonError, BamlClientHttpError, BamlValidationError } from '@magnitudedev/llm-core'
import { getProvider } from '@magnitudedev/providers'
import type { ChatMessage } from '@magnitudedev/llm-core'
import { getEvalProviderClient } from './provider-runtime'

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

const providerClient = await getEvalProviderClient()
const auth = await providerClient.auth.getAuth(providerId)
await providerClient.state.setSelection('primary', providerId, modelId, auth ?? null, { persist: false })

const systemPrompt = 'You are a helpful assistant.'
const messages: ChatMessage[] = [
  { role: 'user', content: ['Write a very long, detailed essay about the history of computing. Include at least 20 paragraphs covering everything from Babbage to modern AI. Do not stop early.'] },
]

console.log('--- Streaming (primary.chat) ---')
try {
  const chatStream = await Effect.runPromise(
    Effect.gen(function* () {
      const runtime = yield* ModelResolver
      const model = yield* runtime.resolve('primary')
      return yield* model.invoke(
        CodingAgentChat,
        {
          systemPrompt,
          messages,
          ackTurns: [
            { role: 'user', content: '--- FEW-SHOT EXAMPLE START ---\n<system>\nRespond using the required turn format. The user reports a bug in the login redirect.\n</system>' },
            { role: 'assistant', content: '<lens name="skills">Bug report → activate the bug skill to load methodology.</lens>\n<lens name="tasks">Bug fix isn\'t one-turnable. Need to understand and delegate.</lens>\n<skill name="bug" />\n<read path="src/auth/redirect.ts" />\n<end-turn>\n<continue/>\n</end-turn>' },
            { role: 'user', content: '<turn_result>\n<tool name="skill"><content># Skill: Bug\n\nProvides methodology for diagnosing and fixing bugs.</content></tool>\n<tool name="read">export function redirectAfterLogout(req, res) {\n  res.redirect(\'/home\') // Bug: should redirect to \'/login\'\n}</tool>\n</turn_result>' },
            { role: 'assistant', content: '<lens name="skills">Skill loaded. Bug skill says: diagnose root cause first, then fix.</lens>\n<lens name="tasks">Create a bug task and spawn a debugger worker.</lens>\n<create-task id="fix-redirect" type="bug" title="Fix login redirect bug" />\n<spawn-worker id="fix-redirect">The redirect function is using \'/home\' instead of \'/login\'. Diagnose and fix.</spawn-worker>\n<message to="user">Found the bug — redirectAfterLogout sends to `/home` instead of `/login`. Worker is fixing it now.</message>\n<end-turn>\n<idle/>\n</end-turn>' },
            { role: 'user', content: '--- FEW-SHOT EXAMPLE END ---' },
          ],
        },
      )
    }).pipe(Effect.provide(Layer.merge(makeModelResolver().pipe(Layer.provide(providerClient.layer), Layer.provide(makeNoopTracer())), makeNoopTracer()))),
  )
  let chunkCount = 0
  const result = await Effect.runPromise(Stream.runFold(chatStream.stream, '', (acc, chunk) => {
    chunkCount++
    return acc + chunk
  }))
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
  const rawRequestBody = collectorData._tag === 'Baml' ? collectorData.rawRequestBody : null
  console.log(`rawRequestBody keys: ${rawRequestBody ? Object.keys(rawRequestBody as object).join(', ') : 'null'}`)

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
  const sseEvents: Array<Record<string, unknown>> = []
  if (sseEvents.length > 0) {
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
    console.log(`SSE events: empty`)
  }

} catch (error) {
  const record = (typeof error === 'object' && error !== null) ? error as Record<string, unknown> : null
  const errorType = error instanceof Error ? error.constructor.name : typeof error
  const errorMessage = error instanceof Error ? error.message : String(error)

  console.log(`Error type: ${errorType}`)
  console.log(`Message: ${errorMessage}`)

  if (error instanceof BamlClientFinishReasonError) {
    const finishReason = typeof record?.finish_reason === 'string' ? record.finish_reason : undefined
    const rawOutput = String(record?.raw_output ?? '')
    console.log(`Finish reason: ${finishReason}`)
    console.log(`Raw output length: ${rawOutput.length} chars`)
    console.log(`Raw output last 100: ${JSON.stringify(rawOutput.slice(-100))}`)
  } else if (error instanceof BamlClientHttpError) {
    const status = typeof record?.status_code === 'number' ? record.status_code : undefined
    const detailed = record?.detailed_message == null ? undefined : String(record.detailed_message)
    const rawResponse = record?.raw_response == null ? undefined : String(record.raw_response)
    console.log(`Status: ${status}`)
    console.log(`Detailed: ${detailed?.slice(0, 300)}`)
    console.log(`Raw response: ${rawResponse?.slice(0, 300)}`)
  } else if (error instanceof BamlValidationError) {
    const detailed = record?.detailed_message == null ? undefined : String(record.detailed_message)
    const rawOutput = String(record?.raw_output ?? '')
    console.log(`Detailed: ${detailed?.slice(0, 300)}`)
    console.log(`Raw output length: ${rawOutput.length} chars`)
  }
}

console.log('\nDone.')
