import { createTokenizer } from '../tokenizer'
import { createParser } from '../parser'

// Simulate streaming: feed the input in small chunks like an LLM would
const fullInput = `<|think:alignment>
Some reasoning here
<think|>

<|message:user>
I'm ready and waiting for a task
<message|>

<|yield:user|>`

const tokens: any[] = []
const parser = createParser()

// Create tokenizer ONCE (like the fixed runtime)
const tokenizer = createTokenizer((token) => {
  tokens.push(token)
  parser.pushToken(token)
})

// Simulate chunk-by-chunk streaming (1-3 chars at a time)
for (let i = 0; i < fullInput.length; i += 2) {
  const chunk = fullInput.slice(i, i + 2)
  tokenizer.push(chunk)
  
  // Flush parser events after each chunk
  for (const event of parser.drain()) {
    if (event._tag === 'ProseChunk' || event._tag === 'ProseEnd') {
      if (event._tag === 'ProseEnd' && event.content.includes('<|') || event._tag === 'ProseChunk' && event.text.includes('<|')) {
        console.error(`BUG: Protocol syntax in prose! Event: ${JSON.stringify(event)}`)
      }
    }
    if (event._tag === 'TurnControl' || event._tag === 'MessageStart' || event._tag === 'LensStart') {
      console.log(`✓ Parsed: ${JSON.stringify(event)}`)
    }
  }
}

tokenizer.end()

// Flush remaining
for (const event of parser.drain()) {
  if (event._tag === 'ProseEnd' && event.content.includes('<|')) {
    console.error(`BUG: Protocol syntax in prose! Event: ${JSON.stringify(event)}`)
  }
  if (event._tag === 'TurnControl' || event._tag === 'MessageStart' || event._tag === 'LensStart') {
    console.log(`✓ Parsed: ${JSON.stringify(event)}`)
  }
}

console.log('\n=== ALL TOKENS ===')
console.log(JSON.stringify(tokens, null, 2))
