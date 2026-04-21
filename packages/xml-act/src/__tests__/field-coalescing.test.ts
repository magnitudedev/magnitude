/**
 * Field coalescing tests — T13 fix verification.
 */
import { describe, it, expect } from 'vitest'
import { createParser } from '../parser/index'
import { createTokenizer } from '../tokenizer'
import type { TurnEngineEvent, RegisteredTool } from '../types'
import { Schema } from 'effect'
import { defineTool } from '@magnitudedev/tools'

const shellTool = defineTool({
  name: 'shell',
  label: 'Shell',
  group: 'fs',
  description: 'Run a shell command',
  inputSchema: Schema.Struct({ command: Schema.String }),
  outputSchema: Schema.Struct({ stdout: Schema.String }),
  execute: async () => ({ stdout: '' }),
})

function makeTools(): ReadonlyMap<string, RegisteredTool> {
  return new Map([
    ['shell', { tool: shellTool, tagName: 'shell', groupName: 'fs' }],
  ])
}

function parseIncremental(chunks: string[]): TurnEngineEvent[] {
  const allEvents: TurnEngineEvent[] = []
  const parser = createParser({ tools: makeTools() })
  const tokenizer = createTokenizer(
    (token) => parser.pushToken(token),
    new Set(['shell']),
    { toolKeyword: 'invoke' },
  )
  for (const chunk of chunks) {
    tokenizer.push(chunk)
    allEvents.push(...parser.drain())
  }
  tokenizer.end()
  parser.end()
  allEvents.push(...parser.drain())
  return allEvents
}

describe('field coalescing', () => {
  it('coalesces adjacent ToolInputFieldChunk events within same drain', () => {
    // Send multiple chars in one push — they should coalesce within that drain
    const allEvents: TurnEngineEvent[] = []
    const parser = createParser({ tools: makeTools() })
    const tokenizer = createTokenizer(
      (token) => parser.pushToken(token),
      new Set(['shell']),
      { toolKeyword: 'invoke' },
    )
    tokenizer.push('\n<|invoke:shell>\n<|parameter:command>ls -la<parameter|>\n<invoke|>')
    tokenizer.end()
    parser.end()
    allEvents.push(...parser.drain())

    const fieldChunks = allEvents.filter(e => e._tag === 'ToolInputFieldChunk')
    // All content in one push → should coalesce to 1 chunk
    expect(fieldChunks.length).toBe(1)
    expect((fieldChunks[0] as any).delta).toBe('ls -la')
  })

  it('preserves total delta across drains', () => {
    const events = parseIncremental([
      '\n<|invoke:shell>\n<|parameter:command>',
      'ls',
      ' -la',
      '<parameter|>\n<invoke|>',
    ])
    const fieldChunks = events.filter(e => e._tag === 'ToolInputFieldChunk')
    const totalDelta = fieldChunks.map(e => (e as any).delta).join('')
    expect(totalDelta).toBe('ls -la')
  })

  it('ProseChunk coalesces within same drain', () => {
    const allEvents: TurnEngineEvent[] = []
    const parser = createParser({ tools: makeTools() })
    const tokenizer = createTokenizer(
      (token) => parser.pushToken(token),
      new Set(['shell']),
      { toolKeyword: 'invoke' },
    )
    tokenizer.push('Hello world')
    tokenizer.end()
    parser.end()
    allEvents.push(...parser.drain())

    const proseChunks = allEvents.filter(e => e._tag === 'ProseChunk')
    // All in one push → should coalesce
    expect(proseChunks.length).toBe(1)
    expect((proseChunks[0] as any).text).toBe('Hello world')
  })
})
