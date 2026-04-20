/**
 * Deep token audit: check ALL token types including Close tokens
 */

import { createTokenizer } from '../tokenizer'

const USER_EXACT_OUTPUT = `<|think:alignment>
User wants me to review the staged changes.
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
Spawning a reviewer.
<message|>

<|yield:worker|>`

const tokens: any[] = []
const tokenizer = createTokenizer((token) => {
  tokens.push({...token})
})

for (let i = 0; i < USER_EXACT_OUTPUT.length; i++) {
  tokenizer.push(USER_EXACT_OUTPUT[i])
}
tokenizer.end()

console.log('=== ALL TOKENS ===\n')
for (let i = 0; i < tokens.length; i++) {
  const t = tokens[i]
  if (t._tag === 'Content') {
    const preview = t.text.length > 40 ? t.text.slice(0, 40) + '...' : t.text
    console.log(`${i}: ${t._tag} ${JSON.stringify(preview)}`)
  } else if (t._tag === 'Close') {
    console.log(`${i}: ${t._tag} name=${t.name} pipe=${t.pipe ?? 'none'}`)
  } else if (t._tag === 'Parameter') {
    console.log(`${i}: ${t._tag} name=${t.name}`)
  } else if (t._tag === 'ParameterClose') {
    console.log(`${i}: ${t._tag}`)
  } else if (t._tag === 'Open') {
    console.log(`${i}: ${t._tag} name=${t.name} variant=${t.variant ?? 'none'}`)
  } else if (t._tag === 'SelfClose') {
    console.log(`${i}: ${t._tag} name=${t.name} variant=${t.variant ?? 'none'}`)
  }
}

// Count by type
const counts: Record<string, number> = {}
for (const t of tokens) {
  counts[t._tag] = (counts[t._tag] || 0) + 1
}
console.log('\n=== TOKEN COUNTS ===')
for (const [tag, count] of Object.entries(counts)) {
  console.log(`  ${tag}: ${count}`)
}

// Critical: check ParameterClose tokens
const paramCloseTokens = tokens.filter(t => t._tag === 'ParameterClose')
console.log(`\nParameterClose tokens: ${paramCloseTokens.length}`)
if (paramCloseTokens.length === 0) {
  console.log('WARNING: No ParameterClose tokens! <parameter|> is NOT being tokenized as ParameterClose!')
  
  // Check if Close tokens with name "parameter" exist
  const closeParameterTokens = tokens.filter(t => t._tag === 'Close' && t.name === 'parameter')
  console.log(`Close name=parameter tokens: ${closeParameterTokens.length}`)
  
  // Check ALL Close tokens
  const closeTokens = tokens.filter(t => t._tag === 'Close')
  console.log(`All Close tokens:`)
  for (const t of closeTokens) {
    console.log(`  name=${t.name} pipe=${t.pipe ?? 'none'}`)
  }
}
