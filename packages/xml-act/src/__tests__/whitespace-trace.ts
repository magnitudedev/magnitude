import { createTokenizer } from '../tokenizer'
import { createParser } from '../parser'

const tests = [
  {
    name: "newlines between message close and yield",
    input: `<|message:user>Hello<message|>\n\n<|yield:user|>`,
  },
  {
    name: "newlines between think close and message open",
    input: `<|think:x>reasoning<think|>\n\n<|message:user>Hello<message|>`,
  },
  {
    name: "leading whitespace before first tag",
    input: `\n\n<|message:user>Hello<message|>`,
  },
  {
    name: "trailing whitespace after yield",
    input: `<|message:user>Hello<message|><|yield:user|>\n\n`,
  },
  {
    name: "multiple newlines between blocks",
    input: `<|think:x>reasoning<think|>\n\n\n\n<|message:user>Hello<message|>\n\n<|yield:user|>`,
  },
  {
    name: "spaces and newlines mixed",
    input: `<|think:x>reasoning<think|>  \n  <|message:user>Hello<message|>`,
  },
]

for (const test of tests) {
  console.log(`\n=== ${test.name} ===`)
  console.log(`Input: ${JSON.stringify(test.input)}`)
  
  const parser = createParser()
  const tokenizer = createTokenizer((token) => {
    parser.pushToken(token)
  })
  
  for (let i = 0; i < test.input.length; i++) {
    tokenizer.push(test.input[i])
    for (const event of parser.drain()) {
      if (event._tag === 'ProseChunk' || event._tag === 'ProseEnd') {
        console.log(`  PROSE: ${event._tag} ${JSON.stringify('text' in event ? event.text : event.content)}`)
      } else if (event._tag === 'MessageChunk') {
        console.log(`  MSG:   ${event._tag} ${JSON.stringify(event.text)}`)
      } else if (event._tag === 'LensChunk') {
        console.log(`  LENS:  ${event._tag} ${JSON.stringify(event.text)}`)
      } else {
        console.log(`  EVENT: ${JSON.stringify(event)}`)
      }
    }
  }
  tokenizer.end()
  for (const event of parser.drain()) {
    if (event._tag === 'ProseChunk' || event._tag === 'ProseEnd') {
      console.log(`  PROSE: ${event._tag} ${JSON.stringify('text' in event ? event.text : event.content)}`)
    } else if (event._tag === 'MessageChunk') {
      console.log(`  MSG:   ${event._tag} ${JSON.stringify(event.text)}`)
    } else if (event._tag === 'LensChunk') {
      console.log(`  LENS:  ${event._tag} ${JSON.stringify(event.text)}`)
    } else {
      console.log(`  EVENT: ${JSON.stringify(event)}`)
    }
  }
}
