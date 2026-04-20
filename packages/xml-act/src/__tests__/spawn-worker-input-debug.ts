/**
 * Debug: what does the response-builder produce for spawn-worker?
 * And does it parse and build input correctly?
 */

import { createTokenizer } from '../tokenizer'
import { createParser, type ParserEvent } from '../parser'
import { Schema } from '@effect/schema'
import { deriveParameters, type ToolSchema } from '../execution/parameter-schema'
import { buildInput, type ParsedInvoke, type ParsedParameter } from '../execution/input-builder'

// spawn-worker schema (same as task-tools.ts)
const SpawnWorkerInputSchema = Schema.Struct({
  id: Schema.String,
  message: Schema.String,
  role: Schema.optional(Schema.String),
})

const toolSchema: ToolSchema = deriveParameters(SpawnWorkerInputSchema.ast)

console.log('=== spawn-worker Parameter Schema ===')
for (const [name, param] of toolSchema.parameters) {
  console.log(`  ${name}: type=${JSON.stringify(param.type)}, required=${param.required}`)
}

// What the response-builder produces for:
// response().spawnWorker('my-task', 'builder', 'Do the work')
const spawnWorkerMact = `<|invoke:spawn-worker>
<|parameter:id>my-task<parameter|>
<|parameter:role>builder<parameter|>
Do the work
<invoke|>`

console.log('\n=== Mact output from response-builder ===')
console.log(spawnWorkerMact)

// Parse it
console.log('\n=== Parsing ===')
const parser = createParser()
const tokenizer = createTokenizer((token) => {
  parser.pushToken(token)
})

const allEvents: ParserEvent[] = []
for (let i = 0; i < spawnWorkerMact.length; i++) {
  tokenizer.push(spawnWorkerMact[i])
  for (const event of parser.drain()) {
    allEvents.push(event)
    console.log(JSON.stringify(event))
  }
}
tokenizer.end()
for (const event of parser.drain()) {
  allEvents.push(event)
  console.log(JSON.stringify(event))
}

// Check: where does "Do the work" go?
const contentEvents = allEvents.filter(e => e._tag === 'Content')
console.log('\nContent events:')
for (const e of contentEvents) {
  console.log(`  ${(e as any).text}`)
}

// Build input
const invokeStart = allEvents.find(e => e._tag === 'InvokeStarted') as any
const paramCompletes = allEvents.filter(e => e._tag === 'ParameterComplete') as any[]
const invokeComplete = allEvents.find(e => e._tag === 'InvokeComplete') as any

if (invokeStart) {
  console.log('\n=== Building Input ===')
  const invoke: ParsedInvoke = {
    tagName: invokeStart.toolTag,
    toolCallId: invokeStart.toolCallId,
    parameters: new Map<string, ParsedParameter>(),
  }
  
  for (const pe of paramCompletes) {
    invoke.parameters.set(pe.parameterName, {
      name: pe.parameterName,
      value: pe.value,
      isComplete: true,
    })
  }
  
  console.log('Parameters:')
  for (const [name, param] of invoke.parameters) {
    console.log(`  ${name}: ${JSON.stringify(param.value)}`)
  }
  
  try {
    const input = buildInput(invoke, toolSchema.parameters)
    console.log('\nBuilt input:')
    console.log(JSON.stringify(input, null, 2))
    
    // Validate against schema
    const validated = Schema.decodeUnknownSync(SpawnWorkerInputSchema)(input)
    console.log('\nValidated input:')
    console.log(JSON.stringify(validated, null, 2))
  } catch (e) {
    console.log('\nBUILD/VALIDATE FAILED:')
    console.log(e instanceof Error ? e.message : String(e))
  }
} else {
  console.log('\nNO INVOKE START — spawn-worker was not parsed as a tool invoke!')
}
