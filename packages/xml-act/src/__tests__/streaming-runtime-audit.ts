/**
 * TEST 6: STREAMING RUNTIME
 * Feed the exact model output char-by-char (simulating LLM streaming)
 */

import { Effect, Layer, Stream } from 'effect'
import { Schema } from '@effect/schema'
import { createRuntime, type TurnEngineEvent } from '..'
import { defineTool } from '@magnitudedev/tools'

const createTaskTool = defineTool({
  name: 'create-task' as const,
  group: 'task' as const,
  description: 'Create a task.',
  inputSchema: Schema.Struct({
    id: Schema.String,
    title: Schema.String,
    parent: Schema.optional(Schema.String),
  }),
  outputSchema: Schema.Struct({ id: Schema.String }),
  execute: (input, _ctx) => Effect.succeed({ id: input.id }),
})

const spawnWorkerTool = defineTool({
  name: 'spawn-worker' as const,
  group: 'task' as const,
  description: 'Spawn a worker.',
  inputSchema: Schema.Struct({
    id: Schema.String,
    message: Schema.String,
  }),
  outputSchema: Schema.Struct({ id: Schema.String }),
  execute: (input, _ctx) => Effect.succeed({ id: input.id }),
})

const registeredTools = new Map([
  ['create-task', { tool: createTaskTool, tagName: 'create-task', groupName: 'task', meta: {}, layerProvider: () => Effect.succeed(Layer.empty) }],
  ['spawn-worker', { tool: spawnWorkerTool, tagName: 'spawn-worker', groupName: 'task', meta: {}, layerProvider: () => Effect.succeed(Layer.empty) }],
])

const modelOutput = `<|invoke:create-task>
<|parameter:id>review-staged<parameter|>
<|parameter:title>Review staged git changes<parameter|>
<|parameter:parent><parameter|>
<invoke|>

<|invoke:spawn-worker>
<|parameter:id>review-staged<parameter|>
<|parameter:message>Review all staged git changes.<parameter|>
<invoke|>

<|message:user>
Done.
<message|>

<|yield:user|>`

// Test with different chunk sizes
for (const chunkSize of [1, 2, 5, 10]) {
  Effect.runPromise(
    Effect.gen(function* () {
      const runtime = createRuntime({ tools: registeredTools, defaultProseDest: 'user' })
      
      // Create a stream that delivers the output in chunks of chunkSize
      const chunks: string[] = []
      for (let i = 0; i < modelOutput.length; i += chunkSize) {
        chunks.push(modelOutput.slice(i, i + chunkSize))
      }
      
      const textStream = Stream.fromIterable(chunks)
      const eventStream = runtime.streamWith(textStream)
      
      let errors = 0
      let toolSuccesses = 0
      let toolParseErrors = 0
      
      yield* Effect.scoped(
        eventStream.pipe(
          Stream.runForEach((event: TurnEngineEvent) =>
            Effect.sync(() => {
              if (event._tag === 'ToolExecutionEnded') {
                if (event.result._tag === 'Success') toolSuccesses++
                else errors++
              }
              if (event._tag === 'StructuralParseError' || event._tag === 'ToolParseError') {
                toolParseErrors++
                console.log(`  [chunk=${chunkSize}] PARSE ERROR: ${event.error.detail}`)
              }
            })
          ),
        )
      )
      
      const status = errors === 0 && toolParseErrors === 0 && toolSuccesses === 2 ? 'PASS' : 'FAIL'
      console.log(`chunkSize=${chunkSize}: toolSuccesses=${toolSuccesses} errors=${errors} parseErrors=${toolParseErrors} → ${status}`)
    })
  ).catch((e) => console.error(`chunkSize=${chunkSize}: CRASH`, e))
}
