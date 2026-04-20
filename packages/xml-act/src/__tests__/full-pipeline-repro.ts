/**
 * FULL PIPELINE REPRO: create-task with empty parent parameter
 */

import { Effect, Layer, Stream } from 'effect'
import { Schema } from '@effect/schema'
import { createRuntime, type RuntimeEvent } from '..'
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
  execute: (input, _ctx) =>
    Effect.gen(function* () {
      console.log(`[create-task EXECUTE] input = ${JSON.stringify(input)}`)
      console.log(`[create-task EXECUTE] input.parent = ${JSON.stringify(input.parent)}`)
      console.log(`[create-task EXECUTE] input.parent ?? null = ${JSON.stringify(input.parent ?? null)}`)
      return { id: input.id }
    }),
})

const registeredTools = new Map([
  ['create-task', {
    tool: createTaskTool,
    tagName: 'create-task',
    groupName: 'task',
    meta: { defKey: 'createTask' },
    layerProvider: () => Effect.succeed(Layer.empty),
  }],
])

const modelOutput = `<|invoke:create-task>
<|parameter:id>dummy-task<parameter|>
<|parameter:title>Dummy task for testing<parameter|>
<|parameter:parent><parameter|>
<invoke|>

<|message:user>
Created dummy task.
<message|>

<|yield:user|>`

Effect.runPromise(
  Effect.gen(function* () {
    const runtime = createRuntime({
      tools: registeredTools,
      defaultProseDest: 'user',
    })

    const textStream = Stream.succeed(modelOutput)
    const eventStream = runtime.streamWith(textStream)

    yield* Effect.scoped(
      eventStream.pipe(
        Stream.runForEach((event: RuntimeEvent) =>
          Effect.sync(() => console.log(`EVENT: ${JSON.stringify(event)}`))
        ),
      )
    )
  })
).then(() => console.log('\n=== DONE ==='))
  .catch((e) => console.error('\n=== ERROR ===', e))
