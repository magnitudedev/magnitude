import { createTokenizer } from '../tokenizer'
import { createParser } from '../parser'

// Simulate model output: create-task then spawn-worker in same turn
const input = [
  '<|think:tasks>',
  'Creating task and spawning worker.',
  '<think|>',
  '',
  '<|invoke:create-task>',
  '<|parameter:id>review-staged<parameter|>',
  '<|parameter:title>Review staged changes<parameter|>',
  '<|parameter:parent><parameter|>',
  '<invoke|>',
  '',
  '<|invoke:spawn-worker>',
  '<|parameter:id>review-staged<parameter|>',
  '<|parameter:message>Review all staged git changes<parameter|>',
  '<invoke|>',
  '',
  '<|message:user>',
  'Spawning a reviewer to look at the staged changes.',
  '<message|>',
  '',
  '<|yield:worker|>',
].join('\n')

console.log('Input:')
console.log(input)
console.log()

const parser = createParser()
const tokenizer = createTokenizer((token) => {
  parser.pushToken(token)
})

for (let i = 0; i < input.length; i++) {
  tokenizer.push(input[i])
  for (const event of parser.drain()) {
    console.log(JSON.stringify(event))
  }
}
tokenizer.end()
for (const event of parser.drain()) {
  console.log(JSON.stringify(event))
}
