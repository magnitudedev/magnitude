import { Effect, Layer, Stream } from 'effect'
import type { ChatMessage } from '@magnitudedev/llm-core'
import { ModelResolver, SimpleChat, createProviderClient, makeModelResolver, makeNoopTracer } from '@magnitudedev/providers'

const MODEL_PROVIDER = 'openrouter'
const MODEL_ID = 'google/gemma-4-26b-a4b-it'

async function main() {
  const client = await createProviderClient({ slots: ['primary'] as const })
  const auth = await client.auth.getAuth(MODEL_PROVIDER)
  console.log('Auth:', auth ? `${auth.type}` : 'none')

  await client.state.setSelection('primary', MODEL_PROVIDER, MODEL_ID, auth ?? null, { persist: false })

  console.log(`\nStreaming from ${MODEL_PROVIDER}/${MODEL_ID}...\n`)

  const chatStream = await Effect.runPromise(
    Effect.gen(function* () {
      const runtime = yield* ModelResolver
      const model = yield* runtime.resolve('primary')
      console.log('Model resolved:', model.model.id, 'provider:', model.model.providerId)
      console.log('Context window:', model.model.contextWindow)
      console.log('Supports grammar:', model.model.supportsGrammar)
      console.log('Supports tool calls:', model.model.supportsToolCalls)
      return yield* model.invoke(SimpleChat, {
        systemPrompt: 'You are a helpful assistant. Respond briefly.',
        messages: [{ role: 'user', content: ['Write a haiku about programming.'] }] as ChatMessage[],
      })
    }).pipe(
      Effect.provide(
        Layer.merge(
          makeModelResolver().pipe(Layer.provide(client.layer), Layer.provide(makeNoopTracer())),
          makeNoopTracer(),
        ),
      ),
    ),
  )

  console.log('\n--- Streaming chunks ---')
  let chunkCount = 0
  let fullText = ''

  const result = await Effect.runPromise(
    Stream.runForEach(chatStream.stream, (chunk) =>
      Effect.sync(() => {
        chunkCount++
        fullText += chunk
        // Log each chunk with visible boundaries
        process.stdout.write(`[${chunkCount}]${JSON.stringify(chunk)}`)
      })
    )
  )

  console.log('\n\n--- Summary ---')
  console.log('Total chunks:', chunkCount)
  console.log('Full text length:', fullText.length)
  console.log('Full text:', fullText)

  // Now get usage
  const usage = chatStream.getUsage()
  console.log('Usage:', usage)
}

main().catch(e => {
  console.error('Error:', e)
  process.exit(1)
})
