import { createTokenizer } from '../tokenizer'
import { createParser } from '../parser'

const tests = [
  {
    name: "leading newline in think",
    input: `<|think:x>\nreasoning here<think|>`,
  },
  {
    name: "leading newline in message",
    input: `<|message:user>\nHello there<message|>`,
  },
  {
    name: "trailing newline in think",
    input: `<|think:x>reasoning here\n<think|>`,
  },
  {
    name: "trailing newline in message",
    input: `<|message:user>Hello there\n<message|>`,
  },
  {
    name: "leading+trailing newlines in think",
    input: `<|think:x>\nreasoning here\n<think|>`,
  },
  {
    name: "leading+trailing newlines in message",
    input: `<|message:user>\nHello there\n<message|>`,
  },
]

for (const test of tests) {
  console.log(`\n=== ${test.name} ===`)
  const parser = createParser()
  const tokenizer = createTokenizer((token) => {
    parser.pushToken(token)
  })
  
  for (let i = 0; i < test.input.length; i++) {
    tokenizer.push(test.input[i])
    for (const event of parser.drain()) {
      if (event._tag === 'LensChunk') {
        console.log(`  LENS:  ${JSON.stringify(event.text)}`)
      } else if (event._tag === 'LensEnd') {
        console.log(`  LENS_END: content=${JSON.stringify(event.content)}`)
      } else if (event._tag === 'MessageChunk') {
        console.log(`  MSG:   ${JSON.stringify(event.text)}`)
      } else if (event._tag === 'MessageEnd') {
        // no content on end
      } else {
        console.log(`  EVENT: ${JSON.stringify(event)}`)
      }
    }
  }
  tokenizer.end()
  for (const event of parser.drain()) {
    if (event._tag === 'LensChunk') {
      console.log(`  LENS:  ${JSON.stringify(event.text)}`)
    } else if (event._tag === 'LensEnd') {
      console.log(`  LENS_END: content=${JSON.stringify(event.content)}`)
    } else if (event._tag === 'MessageChunk') {
      console.log(`  MSG:   ${JSON.stringify(event.text)}`)
    } else if (event._tag === 'ProseEnd') {
      console.log(`  PROSE_END: ${JSON.stringify(event.content)}`)
    } else {
      console.log(`  EVENT: ${JSON.stringify(event)}`)
    }
  }
}
