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
    const allEvents: TurnEngineEvent[] = []
    const parser = createParser({ tools: makeTools() })
    const tokenizer = createTokenizer(
      (token) => parser.pushToken(token),
      new Set(['shell']),
    )
    tokenizer.push('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls -la</magnitude:parameter>\n</magnitude:invoke>')
    tokenizer.end()
    parser.end()
    allEvents.push(...parser.drain())

    const fieldChunks = allEvents.filter(e => e._tag === 'ToolInputFieldChunk')
    expect(fieldChunks.length).toBe(1)
    expect((fieldChunks[0] as any).delta).toBe('ls -la')
  })

  it('preserves total delta across drains', () => {
    const events = parseIncremental([
      '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">',
      'ls',
      ' -la',
      '</magnitude:parameter>\n</magnitude:invoke>',
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
    )
    tokenizer.push('Hello world')
    tokenizer.end()
    parser.end()
    allEvents.push(...parser.drain())

    const proseChunks = allEvents.filter(e => e._tag === 'ProseChunk')
    expect(proseChunks.length).toBe(1)
    expect((proseChunks[0] as any).text).toBe('Hello world')
  })
})
