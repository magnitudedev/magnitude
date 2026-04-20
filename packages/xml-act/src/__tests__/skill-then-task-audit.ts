/**
 * TEST 7: Does skill invocation break subsequent tool execution?
 * The user's model output had skill BEFORE create-task + spawn-worker.
 */

import { Effect, Layer, Stream } from 'effect'
import { Schema } from '@effect/schema'
import { createRuntime, type RuntimeEvent } from '..'
import { defineTool } from '@magnitudedev/tools'

const skillTool = defineTool({
  name: 'skill' as const,
  group: 'task' as const,
  description: 'Activate a skill',
  inputSchema: Schema.Struct({ name: Schema.String }),
  outputSchema: Schema.Struct({ content: Schema.String }),
  execute: (input, _ctx) => Effect.succeed({ content: `Skill ${input.name} activated` }),
})

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
  ['skill', { tool: skillTool, tagName: 'skill', groupName: 'task', meta: {}, layerProvider: () => Effect.succeed(Layer.empty) }],
  ['create-task', { tool: createTaskTool, tagName: 'create-task', groupName: 'task', meta: {}, layerProvider: () => Effect.succeed(Layer.empty) }],
  ['spawn-worker', { tool: spawnWorkerTool, tagName: 'spawn-worker', groupName: 'task', meta: {}, layerProvider: () => Effect.succeed(Layer.empty) }],
])

// Exact order from user's model output: skill → create-task → spawn-worker → message → yield
const modelOutput = `<|think:alignment>
Some reasoning here.
<think|>

<|invoke:skill>
<|parameter:name>review<parameter|>
<invoke|>

<|invoke:create-task>
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

<|yield:worker|>`

Effect.runPromise(
  Effect.gen(function* () {
    const runtime = createRuntime({ tools: registeredTools, defaultProseDest: 'user' })
    const textStream = Stream.succeed(modelOutput)
    const eventStream = runtime.streamWith(textStream)

    const events: RuntimeEvent[] = []
    
    yield* Effect.scoped(
      eventStream.pipe(
        Stream.runForEach((event: RuntimeEvent) =>
          Effect.sync(() => {
            events.push(event)
            if (event._tag === 'ToolInputParseError') {
              console.log(`PARSE ERROR: ${event.error.detail}`)
            } else if (event._tag === 'ToolExecutionEnded') {
              console.log(`TOOL RESULT: ${event.toolName} → ${event.result._tag}`)
            }
          })
        ),
      )
    )

    const toolExecs = events.filter(e => e._tag === 'ToolExecutionEnded') as any[]
    const parseErrors = events.filter(e => e._tag === 'ToolInputParseError') as any[]
    
    console.log(`\nTool executions: ${toolExecs.length}`)
    for (const e of toolExecs) {
      console.log(`  ${e.toolName}: ${e.result._tag}`)
    }
    
    console.log(`Parse errors: ${parseErrors.length}`)
    for (const e of parseErrors) {
      console.log(`  ${e.tagName}: ${e.error.detail}`)
    }
    
    const success = toolExecs.length === 3 && parseErrors.length === 0 && toolExecs.every(e => e.result._tag === 'Success')
    console.log(`\nRESULT: ${success ? 'PASS' : 'FAIL'}`)
  })
).catch((e) => console.error('CRASH:', e))
