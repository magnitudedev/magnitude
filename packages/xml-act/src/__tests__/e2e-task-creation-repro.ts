/**
 * End-to-end reproduction of task creation failure.
 */

import { createTokenizer } from '../tokenizer'
import { createParser, type ParserEvent } from '../parser'
import { deriveParameters, type ToolSchema } from '../execution/parameter-schema'
import { buildInput, type ParsedInvoke, type ParsedParameter } from '../execution/input-builder'
import { Schema } from '@effect/schema'

// Define the create-task input schema (same as task-tools.ts)
const CreateTaskInputSchema = Schema.Struct({
  id: Schema.String,
  title: Schema.String,
  parent: Schema.optional(Schema.String),
})

// Derive parameter schema
const toolSchema: ToolSchema = deriveParameters(CreateTaskInputSchema.ast)

console.log('=== Parameter Schema ===')
for (const [name, param] of toolSchema.parameters) {
  console.log(`  ${name}: type=${JSON.stringify(param.type)}, required=${param.required}`)
}

// Parse the model output
const modelOutput = `<|think:alignment>
User wants a dummy task created.
<think|>

<|invoke:create-task>
<|parameter:id>dummy-task<parameter|>
<|parameter:title>Dummy task for testing<parameter|>
<|parameter:parent><parameter|>
<invoke|>

<|message:user>
Created dummy task.
<message|>

<|yield:user|>`

console.log('\n=== Parsing Model Output ===')
const parser = createParser()
const tokenizer = createTokenizer((token) => {
  parser.pushToken(token)
})

const allEvents: ParserEvent[] = []
for (let i = 0; i < modelOutput.length; i++) {
  tokenizer.push(modelOutput[i])
  for (const event of parser.drain()) {
    allEvents.push(event)
  }
}
tokenizer.end()
for (const event of parser.drain()) {
  allEvents.push(event)
}

// Extract invoke events
const invokeStarts = allEvents.filter(e => e._tag === 'InvokeStarted')
const paramCompletes = allEvents.filter(e => e._tag === 'ParameterComplete')
const invokeCompletes = allEvents.filter(e => e._tag === 'InvokeComplete')

console.log('\nInvoke starts:')
for (const e of invokeStarts) {
  console.log(`  ${JSON.stringify(e)}`)
}

console.log('\nParameter completes:')
for (const e of paramCompletes) {
  const pe = e as any
  console.log(`  toolCallId=${pe.toolCallId} name=${pe.parameterName} value=${JSON.stringify(pe.value)}`)
}

// Build the tool input
console.log('\n=== Building Tool Input ===')

const createTaskInvoke: ParsedInvoke = {
  tagName: 'create-task',
  toolCallId: 'call-1',
  parameters: new Map<string, ParsedParameter>(),
}

for (const e of paramCompletes) {
  const pe = e as any
  createTaskInvoke.parameters.set(pe.parameterName, {
    name: pe.parameterName,
    value: pe.value,
    isComplete: true,
  })
}

console.log('Parsed parameters:')
for (const [name, param] of createTaskInvoke.parameters) {
  console.log(`  ${name}: value=${JSON.stringify(param.value)}, isComplete=${param.isComplete}`)
}

try {
  const input = buildInput(createTaskInvoke, toolSchema.parameters)
  console.log('\nBuilt input:')
  console.log(JSON.stringify(input, null, 2))
  
  if ('parent' in input) {
    console.log(`\nparent field: ${JSON.stringify(input.parent)}`)
    console.log(`parent ?? null = ${JSON.stringify(input.parent ?? null)}`)
    console.log(`Boolean(parent) = ${Boolean(input.parent)}`)
  } else {
    console.log('\nparent field: NOT PRESENT in input')
  }
} catch (e) {
  console.log('\nBUILD INPUT FAILED:')
  console.log(e instanceof Error ? e.message : String(e))
}
