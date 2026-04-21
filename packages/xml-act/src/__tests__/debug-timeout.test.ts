import { describe, it } from 'vitest'
import { Schema } from '@effect/schema'
import { defineTool } from '@magnitudedev/tools'
import { Effect } from 'effect'
import { createParser } from './src/parser/index'
import { createTokenizer } from './src/tokenizer'
import type { RegisteredTool } from './src/types'

const shellTool = defineTool({
  name: 'shell', label: 'Shell', description: 'Run a shell command',
  inputSchema: Schema.Struct({ command: Schema.String, timeout: Schema.optional(Schema.Number) }),
  outputSchema: Schema.String,
  execute: (_input: any) => Effect.succeed('ok'),
})
const tools = new Map<string, RegisteredTool>([['shell', { tool: shellTool, tagName: 'shell', groupName: 'default' }]])

function parse(input: string) {
  const p = createParser({ tools })
  const tokenizer = createTokenizer((token) => p.pushToken(token), new Set(tools.keys()))
  tokenizer.push(input)
  const fromPush = p.drain()
  tokenizer.end()
  p.end()
  const fromEnd = p.drain()
  return [...fromPush, ...fromEnd]
}

describe('debug', () => {
  it('timeout', () => {
    const events = parse('<|invoke:shell>\n<|parameter:command>ls<parameter|>\n<|parameter:timeout>30<parameter|>\n<invoke|>')
    console.log(JSON.stringify(events, null, 2))
  })
})
