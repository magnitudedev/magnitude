import { createTokenizer } from '../tokenizer'
import { createParser } from '../parser'

// Stress test: feed input 1 character at a time
const fullInput = `<|think:alignment>
Some reasoning here
<think|>

<|message:user>
Hey Anders! I'm ready to help
<message|>

<|yield:user|>`

const errors: string[] = []
const parsed: string[] = []
const parser = createParser()
const tokenizer = createTokenizer((token) => {
  parser.pushToken(token)
})

// Feed 1 char at a time
for (let i = 0; i < fullInput.length; i++) {
  tokenizer.push(fullInput[i])
  for (const event of parser.drain()) {
    if (event._tag === 'ProseEnd' && event.content.includes('<|')) {
      errors.push(`Protocol syntax in prose: ${JSON.stringify(event)}`)
    }
    if (event._tag === 'ProseChunk' && event.text.includes('<|')) {
      errors.push(`Protocol syntax in prose chunk: ${JSON.stringify(event)}`)
    }
    if (event._tag === 'TurnControl' || event._tag === 'MessageStart' || event._tag === 'LensStart') {
      parsed.push(`✓ ${event._tag}`)
    }
  }
}

tokenizer.end()
for (const event of parser.drain()) {
  if (event._tag === 'ProseEnd' && event.content.includes('<|')) {
    errors.push(`Protocol syntax in prose: ${JSON.stringify(event)}`)
  }
  if (event._tag === 'TurnControl' || event._tag === 'MessageStart' || event._tag === 'LensStart') {
    parsed.push(`✓ ${event._tag}`)
  }
}

if (errors.length > 0) {
  console.error('ERRORS:')
  errors.forEach(e => console.error(e))
} else {
  console.log('✅ All tags parsed correctly with 1-char-at-a-time streaming')
}
console.log('Parsed:', parsed.join(', '))
