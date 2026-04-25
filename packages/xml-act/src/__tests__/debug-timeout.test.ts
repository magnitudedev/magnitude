import { describe, it, expect } from 'vitest'
import { Schema } from '@effect/schema'
import { defineTool } from '@magnitudedev/tools'
import { Effect } from 'effect'
import { createParser } from '../parser/index'
import { createTokenizer } from '../tokenizer'
import type { RegisteredTool } from '../types'

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
  tokenizer.push(input + '\n')
  const fromPush = p.drain()
  tokenizer.end()
  p.end()
  const fromEnd = p.drain()
  return [...fromPush, ...fromEnd]
}

describe('debug', () => {
  it('handles optional timeout parameter', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n<magnitude:parameter name="timeout">30</magnitude:parameter>\n</magnitude:invoke>')
    const ready = events.find(e => e._tag === 'ToolInputReady')
    expect(ready).toMatchObject({ _tag: 'ToolInputReady', input: { command: 'ls', timeout: 30 } })
  })
})
