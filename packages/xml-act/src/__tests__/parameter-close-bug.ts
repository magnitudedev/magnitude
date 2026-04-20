/**
 * BUG: Tokenizer emits Close name='parameter' but parser expects ParameterClose
 * 
 * This test verifies:
 * 1. Does the tokenizer ever emit ParameterClose tokens? (NO)
 * 2. What does the parser do with Close name='parameter'?
 * 3. Where do ParameterComplete events come from?
 */

import { createTokenizer } from '../tokenizer'
import { createParser, type ParserEvent } from '../parser'

// Minimal input with one parameter
const input = `<|invoke:create-task>
<|parameter:id>my-task<parameter|>
<invoke|>`

console.log('=== TOKENIZER OUTPUT ===')
const tokens: any[] = []
const tokenizer = createTokenizer((token) => {
  tokens.push({...token})
})

for (let i = 0; i < input.length; i++) {
  tokenizer.push(input[i])
}
tokenizer.end()

for (const t of tokens) {
  if (t._tag === 'Content') {
    console.log(`  ${t._tag} ${JSON.stringify(t.text.slice(0, 40))}`)
  } else {
    console.log(`  ${t._tag} ${t.name ? `name=${t.name}` : ''} ${t.variant ? `variant=${t.variant}` : ''} ${t.pipe ? `pipe=${t.pipe}` : ''}`)
  }
}

// Check: does any token have _tag=ParameterClose?
const paramCloseTokens = tokens.filter(t => t._tag === 'ParameterClose')
console.log(`\nParameterClose tokens: ${paramCloseTokens.length}`)
console.log(`Close name=parameter tokens: ${tokens.filter(t => t._tag === 'Close' && t.name === 'parameter').length}`)

console.log('\n=== PARSER OUTPUT ===')
const events: ParserEvent[] = []
const parser = createParser()
const tokenizer2 = createTokenizer((token) => {
  parser.pushToken(token)
})

// Feed char by char and capture events BEFORE end()
const beforeEndEvents: ParserEvent[] = []
for (let i = 0; i < input.length; i++) {
  tokenizer2.push(input[i])
  for (const event of parser.drain()) {
    beforeEndEvents.push(event)
    console.log(`BEFORE end(): ${event._tag}`)
  }
}

// Now call end()
console.log('\n--- Calling parser.end() ---')
tokenizer2.end()
for (const event of parser.drain()) {
  console.log(`AFTER end(): ${event._tag} ${event._tag === 'ParameterComplete' ? `name=${(event as any).parameterName} value=${JSON.stringify((event as any).value)}` : ''}`)
}

// Key question: does ParameterComplete come BEFORE or AFTER end()?
const paramCompleteBefore = beforeEndEvents.filter(e => e._tag === 'ParameterComplete')
console.log(`\nParameterComplete events BEFORE end(): ${paramCompleteBefore.length}`)
console.log(`ParameterComplete events AFTER end(): (see above)`)
