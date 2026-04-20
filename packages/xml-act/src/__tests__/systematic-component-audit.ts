/**
 * SYSTEMATIC COMPONENT AUDIT
 * Testing every piece of the Mact migration in isolation
 * Using the EXACT model output from the user's bug report
 */

import { createTokenizer } from '../tokenizer'
import { createParser, type ParserEvent } from '../parser'
import { deriveParameters } from '../execution/parameter-schema'
import { buildInput, type ParsedInvoke, type ParsedParameter } from '../execution/input-builder'
import { Schema } from '@effect/schema'

// =====================================================================
// THE EXACT MODEL OUTPUT FROM THE USER'S BUG REPORT
// =====================================================================

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

// =====================================================================
// TEST 1: TOKENIZER
// =====================================================================

console.log('\n' + '='.repeat(60))
console.log('TEST 1: TOKENIZER')
console.log('='.repeat(60))

const tokens: any[] = []
const tokenizer = createTokenizer((token) => {
  tokens.push(token)
})

for (let i = 0; i < USER_EXACT_OUTPUT.length; i++) {
  tokenizer.push(USER_EXACT_OUTPUT[i])
}
tokenizer.end()

console.log(`Total tokens: ${tokens.length}`)

// Check for invoke tokens
const invokeTokens = tokens.filter(t => t._tag === 'Open' && t.name === 'invoke')
console.log(`Invoke open tokens: ${invokeTokens.length}`)
for (const t of invokeTokens) {
  console.log(`  variant: ${t.variant}`)
}

// Check for parameter tokens
const paramTokens = tokens.filter(t => t._tag === 'Parameter')
console.log(`Parameter open tokens: ${paramTokens.length}`)
for (const t of paramTokens) {
  console.log(`  name: ${t.name}`)
}

// Check for parameter close tokens
const paramCloseTokens = tokens.filter(t => t._tag === 'ParameterClose')
console.log(`Parameter close tokens: ${paramCloseTokens.length}`)

// Check for content tokens inside invokes
const contentTokens = tokens.filter(t => t._tag === 'Content')
console.log(`Content tokens: ${contentTokens.length}`)
for (const t of contentTokens) {
  console.log(`  text: ${JSON.stringify(t.text.slice(0, 80))}`)
}

// Check for invoke close tokens
const invokeCloseTokens = tokens.filter(t => t._tag === 'Close' && t.name === 'invoke')
console.log(`Invoke close tokens: ${invokeCloseTokens.length}`)

// =====================================================================
// TEST 2: PARSER
// =====================================================================

console.log('\n' + '='.repeat(60))
console.log('TEST 2: PARSER')
console.log('='.repeat(60))

const parser = createParser()
const tokenizer2 = createTokenizer((token) => {
  parser.pushToken(token)
})

const allEvents: ParserEvent[] = []
for (let i = 0; i < USER_EXACT_OUTPUT.length; i++) {
  tokenizer2.push(USER_EXACT_OUTPUT[i])
  for (const event of parser.drain()) {
    allEvents.push(event)
  }
}
tokenizer2.end()
for (const event of parser.drain()) {
  allEvents.push(event)
}

console.log(`Total events: ${allEvents.length}`)

// Check invoke events
const invokeStarted = allEvents.filter(e => e._tag === 'InvokeStarted') as any[]
console.log(`InvokeStarted events: ${invokeStarted.length}`)
for (const e of invokeStarted) {
  console.log(`  toolTag=${e.toolTag} toolCallId=${e.toolCallId}`)
}

// Check parameter events
const paramComplete = allEvents.filter(e => e._tag === 'ParameterComplete') as any[]
console.log(`ParameterComplete events: ${paramComplete.length}`)
for (const e of paramComplete) {
  console.log(`  toolCallId=${e.toolCallId} name=${e.parameterName} value=${JSON.stringify(e.value.slice(0, 60))}`)
}

// Check invoke complete events
const invokeComplete = allEvents.filter(e => e._tag === 'InvokeComplete') as any[]
console.log(`InvokeComplete events: ${invokeComplete.length}`)

// Check for message events
const messageStart = allEvents.filter(e => e._tag === 'MessageStart') as any[]
console.log(`MessageStart events: ${messageStart.length}`)

// Check for turn control
const turnControl = allEvents.filter(e => e._tag === 'TurnControl') as any[]
console.log(`TurnControl events: ${turnControl.length}`)

// Check for any errors or unexpected events
const unexpected = allEvents.filter(e => 
  !['LensStart', 'LensChunk', 'LensEnd', 'InvokeStarted', 'ParameterStarted', 'ParameterChunk', 
    'ParameterComplete', 'InvokeComplete', 'MessageStart', 'MessageChunk', 'MessageEnd', 
    'TurnControl', 'ProseChunk', 'ProseEnd'].includes(e._tag)
)
if (unexpected.length > 0) {
  console.log(`UNEXPECTED events: ${unexpected.length}`)
  for (const e of unexpected) {
    console.log(`  ${JSON.stringify(e).slice(0, 200)}`)
  }
}

// =====================================================================
// TEST 3: PARAMETER SCHEMA DERIVATION
// =====================================================================

console.log('\n' + '='.repeat(60))
console.log('TEST 3: PARAMETER SCHEMA DERIVATION')
console.log('='.repeat(60))

const schemas = {
  'create-task': Schema.Struct({
    id: Schema.String,
    title: Schema.String,
    parent: Schema.optional(Schema.String),
  }),
  'spawn-worker': Schema.Struct({
    id: Schema.String,
    message: Schema.String,
    role: Schema.optional(Schema.String),
  }),
  'skill': Schema.Struct({
    name: Schema.String,
  }),
}

for (const [toolName, schema] of Object.entries(schemas)) {
  const toolSchema = deriveParameters(schema.ast)
  console.log(`\n${toolName}:`)
  console.log(`  selfClosing: ${toolSchema.selfClosing}`)
  for (const [name, param] of toolSchema.parameters) {
    console.log(`  ${name}: type=${JSON.stringify(param.type)}, required=${param.required}`)
  }
}

// =====================================================================
// TEST 4: INPUT BUILDING
// =====================================================================

console.log('\n' + '='.repeat(60))
console.log('TEST 4: INPUT BUILDING')
console.log('='.repeat(60))

// Build input for create-task
const createTaskSchema = deriveParameters(schemas['create-task'].ast)
const createTaskParams = new Map<string, ParsedParameter>()
for (const e of paramComplete) {
  if (invokeStarted[1] && e.toolCallId === invokeStarted[1].toolCallId) {
    createTaskParams.set(e.parameterName, {
      name: e.parameterName,
      value: e.value,
      isComplete: true,
    })
  }
}

console.log('\ncreate-task parameters:')
for (const [name, param] of createTaskParams) {
  console.log(`  ${name}: ${JSON.stringify(param.value)}`)
}

try {
  const input = buildInput(
    { tagName: 'create-task', toolCallId: 'test-1', parameters: createTaskParams },
    createTaskSchema.parameters,
  )
  console.log('create-task built input:')
  console.log(JSON.stringify(input, null, 2))
  
  const validated = Schema.decodeUnknownSync(schemas['create-task'])(input)
  console.log('create-task validated input:')
  console.log(JSON.stringify(validated, null, 2))
} catch (e) {
  console.log('create-task BUILD/VALIDATE FAILED:')
  console.log(e instanceof Error ? e.message : String(e))
}

// Build input for spawn-worker
const spawnWorkerSchema = deriveParameters(schemas['spawn-worker'].ast)
const spawnWorkerParams = new Map<string, ParsedParameter>()
for (const e of paramComplete) {
  if (invokeStarted[2] && e.toolCallId === invokeStarted[2].toolCallId) {
    spawnWorkerParams.set(e.parameterName, {
      name: e.parameterName,
      value: e.value,
      isComplete: true,
    })
  }
}

console.log('\nspawn-worker parameters:')
for (const [name, param] of spawnWorkerParams) {
  console.log(`  ${name}: ${JSON.stringify(param.value.slice(0, 60))}`)
}

try {
  const input = buildInput(
    { tagName: 'spawn-worker', toolCallId: 'test-2', parameters: spawnWorkerParams },
    spawnWorkerSchema.parameters,
  )
  console.log('spawn-worker built input:')
  console.log(JSON.stringify(input, null, 2).slice(0, 300))
  
  const validated = Schema.decodeUnknownSync(schemas['spawn-worker'])(input)
  console.log('spawn-worker validated:')
  console.log(JSON.stringify(validated, null, 2).slice(0, 300))
} catch (e) {
  console.log('spawn-worker BUILD/VALIDATE FAILED:')
  console.log(e instanceof Error ? e.message : String(e))
}

// Build input for skill
const skillSchema = deriveParameters(schemas['skill'].ast)
const skillParams = new Map<string, ParsedParameter>()
for (const e of paramComplete) {
  if (invokeStarted[0] && e.toolCallId === invokeStarted[0].toolCallId) {
    skillParams.set(e.parameterName, {
      name: e.parameterName,
      value: e.value,
      isComplete: true,
    })
  }
}

console.log('\nskill parameters:')
for (const [name, param] of skillParams) {
  console.log(`  ${name}: ${JSON.stringify(param.value)}`)
}

try {
  const input = buildInput(
    { tagName: 'skill', toolCallId: 'test-0', parameters: skillParams },
    skillSchema.parameters,
  )
  console.log('skill built input:')
  console.log(JSON.stringify(input, null, 2))
  
  const validated = Schema.decodeUnknownSync(schemas['skill'])(input)
  console.log('skill validated:')
  console.log(JSON.stringify(validated, null, 2))
} catch (e) {
  console.log('skill BUILD/VALIDATE FAILED:')
  console.log(e instanceof Error ? e.message : String(e))
}

console.log('\n' + '='.repeat(60))
console.log('AUDIT COMPLETE')
console.log('='.repeat(60))
