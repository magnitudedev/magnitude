import { createTokenizer } from '../tokenizer'
import { createParser } from '../parser'

const tokens: any[] = []
const parser = createParser()

const tokenizer = createTokenizer((token) => {
  tokens.push(token)
  parser.pushToken(token)
})

// Test lens/think parsing
const input = `<|think:alignment>
Some reasoning here
<think|>

<|message:user>
Hey!
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
