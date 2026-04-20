/**
 * TEST 5: RUNTIME ERROR HANDLING
 * 
 * What happens when input building fails? Does the error surface properly?
 * What happens with the exact user model output running through the full runtime?
 */

import { Effect, Layer, Stream } from 'effect'
import { Schema } from '@effect/schema'
import { createRuntime, type TurnEngineEvent } from '..'
import { defineTool } from '@magnitudedev/tools'

// Tool definitions matching the real agent tools
const skillTool = defineTool({
  name: 'skill' as const,
  group: 'task' as const,
  description: 'Activate a skill',
  inputSchema: Schema.Struct({ name: Schema.String }),
  outputSchema: Schema.Struct({ content: Schema.String }),
  execute: (input, _ctx) =>
    Effect.gen(function* () {
      console.log(`[skill EXECUTE] input = ${JSON.stringify(input)}`)
      return { content: `Skill ${input.name} activated` }
    }),
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
  execute: (input, _ctx) =>
    Effect.gen(function* () {
      console.log(`[create-task EXECUTE] input = ${JSON.stringify(input)}`)
      return { id: input.id }
    }),
})

const spawnWorkerTool = defineTool({
  name: 'spawn-worker' as const,
  group: 'task' as const,
  description: 'Spawn a worker for a task.',
  inputSchema: Schema.Struct({
    id: Schema.String,
    message: Schema.String,
    role: Schema.optional(Schema.String),
  }),
  outputSchema: Schema.Struct({ id: Schema.String }),
  execute: (input, _ctx) =>
    Effect.gen(function* () {
      console.log(`[spawn-worker EXECUTE] input = ${JSON.stringify(input)}`)
      return { id: input.id }
    }),
})

const registeredTools = new Map([
  ['skill', { tool: skillTool, tagName: 'skill', groupName: 'task', meta: {}, layerProvider: () => Effect.succeed(Layer.empty) }],
  ['create-task', { tool: createTaskTool, tagName: 'create-task', groupName: 'task', meta: {}, layerProvider: () => Effect.succeed(Layer.empty) }],
  ['spawn-worker', { tool: spawnWorkerTool, tagName: 'spawn-worker', groupName: 'task', meta: {}, layerProvider: () => Effect.succeed(Layer.empty) }],
])

// The EXACT model output from the user's bug report
const USER_EXACT_OUTPUT = `<|think:alignment>
User wants me to review the staged changes. Looking at the git status, there are many modified files (119+). This is not one-turnable work - I need to actually look at the diffs and understand what changed. I should create a task for this and delegate to a reviewer worker.
<think|>

<|think:tasks>
This is definitely not one-turnable. I need to:
1. Create a task for reviewing staged changes
2. Spawn a reviewer worker to do the actual review
3. The worker will need to look at git diff --staged to see what changed

I should create the task and spawn the worker.
<think|>

<|think:skills>
Review work - should check if there's a review skill that would help guide this work.
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
<|parameter:message>Review all staged git changes. Use \`git diff --staged\` to see what's changed. Look at the modified files across the codebase and provide a summary of:
1. What areas of the codebase were touched
2. The nature of the changes (features, fixes, refactors, etc.)
3. Any patterns or concerns you notice

Focus on the packages/agent/src/ files and cli/src/ files as those seem to be the main areas with changes.<parameter|>
<invoke|>

<|message:user>
Spawning a reviewer to look at the staged changes. There are quite a few files modified - I'll have them analyze what's changed and give you a summary.
<message|>

<|yield:worker|>`

Effect.runPromise(
  Effect.gen(function* () {
    const runtime = createRuntime({
      tools: registeredTools,
      defaultProseDest: 'user',
    })

    const textStream = Stream.succeed(USER_EXACT_OUTPUT)
    const eventStream = runtime.streamWith(textStream)

    console.log('=== FULL RUNTIME WITH USER EXACT OUTPUT ===\n')

    yield* Effect.scoped(
      eventStream.pipe(
        Stream.runForEach((event: TurnEngineEvent) =>
          Effect.sync(() => {
            if (event._tag === 'ToolInputStarted') {
              console.log(`EVENT: ToolInputStarted tagName=${event.tagName} toolCallId=${event.toolCallId}`)
            } else if (event._tag === 'ToolInputReady') {
              console.log(`EVENT: ToolInputReady toolCallId=${event.toolCallId} input=${JSON.stringify(event.input).slice(0, 100)}`)
            } else if (event._tag === 'ToolInputParseError') {
              console.log(`EVENT: ToolInputParseError toolCallId=${event.toolCallId} error=${event.error.detail}`)
            } else if (event._tag === 'ToolExecutionStarted') {
              console.log(`EVENT: ToolExecutionStarted toolName=${event.toolName} toolCallId=${event.toolCallId}`)
            } else if (event._tag === 'ToolExecutionEnded') {
              console.log(`EVENT: ToolExecutionEnded toolName=${event.toolName} result=${event.result._tag}`)
              if (event.result._tag === 'Error') {
                console.log(`  error: ${event.result.error}`)
              }
            } else if (event._tag === 'ToolObservation') {
              console.log(`EVENT: ToolObservation tagName=${event.tagName}`)
            } else if (event._tag === 'LensStart') {
              console.log(`EVENT: LensStart name=${event.name}`)
            } else if (event._tag === 'LensEnd') {
              console.log(`EVENT: LensEnd name=${event.name}`)
            } else if (event._tag === 'MessageStart') {
              console.log(`EVENT: MessageStart id=${event.id} to=${event.to}`)
            } else if (event._tag === 'MessageChunk') {
              console.log(`EVENT: MessageChunk text=${JSON.stringify(event.text.slice(0, 50))}`)
            } else if (event._tag === 'MessageEnd') {
              console.log(`EVENT: MessageEnd id=${event.id}`)
            } else if (event._tag === 'TurnEnd') {
              console.log(`EVENT: TurnEnd result=${event.result._tag}`)
            } else {
              console.log(`EVENT: ${event._tag}`)
            }
          })
        ),
      )
    )
  })
).then(() => console.log('\n=== DONE ==='))
  .catch((e) => console.error('\n=== ERROR ===', e))
