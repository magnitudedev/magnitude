import { createTokenizer } from '../tokenizer'
import { createParser } from '../parser'

const tokens: any[] = []
const parser = createParser()

const tokenizer = createTokenizer((token) => {
  tokens.push(token)
  parser.pushToken(token)
})

// Correct Mact format output (what the grammar will now generate)
const input = `<|message:user>
Hey Anders! I'm ready to help with whatever you need on the Magnitude project. What are you working on?
<message|>

<|yield:user|>`

tokenizer.push(input)
tokenizer.end()
parser.end()

const events = parser.drain()

console.log('=== TOKENS ===')
console.log(JSON.stringify(tokens, null, 2))
console.log('')
console.log('=== EVENTS ===')
console.log(JSON.stringify(events, null, 2))
