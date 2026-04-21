/**
 * Context-aware tag resolution tests.
 * Verifies that tags not in the current frame's validTags become literal content.
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

function parse(input: string, tools?: ReadonlyMap<string, RegisteredTool>): TurnEngineEvent[] {
  const events: TurnEngineEvent[] = []
  const parser = createParser({ tools: tools ?? makeTools() })
  const tokenizer = createTokenizer(
    (token) => parser.pushToken(token),
    new Set(tools?.keys() ?? ['shell']),
    { toolKeyword: 'invoke' },
  )
  tokenizer.push(input)
  tokenizer.end()
  parser.end()
  return [...parser.drain()]
}

describe('context-aware tag resolution', () => {
  it('treats <div> inside message as literal content', () => {
    // Tokenizer parses <div> as Close { name: 'div' }, parser resolves as content
    // tokenRaw reconstructs as <div|> (canonical close form)
    const events = parse('\n<|message:user>\nHere is <div>some</div> text\n<message|>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    // Content preserved (in canonical token form)
    expect(text).toContain('some')
    expect(text).toContain('text')
    // No structural events for div
    expect(events.filter(e => e._tag === 'StructuralParseError' || e._tag === 'ToolParseError')).toHaveLength(0)
    // No tool events
    expect(events.filter(e => e._tag === 'ToolInputStarted')).toHaveLength(0)
  })

  it('treats <strong>bold</strong> inside message as literal content', () => {
    const events = parse('\n<|message:user>\n<strong>bold</strong>\n<message|>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('bold')
    // No structural events
    expect(events.filter(e => e._tag === 'ToolInputStarted')).toHaveLength(0)
  })

  it('treats <foo> inside message as literal content (not structural)', () => {
    const events = parse('\n<|message:user>\nSee <foo> here\n<message|>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('See')
    expect(text).toContain('here')
    // No tool or structural events
    expect(events.filter(e => e._tag === 'ToolInputStarted')).toHaveLength(0)
  })

  it('treats <|invoke:shell> inside message as literal content', () => {
    const events = parse('\n<|message:user>\nRun <|invoke:shell> to test\n<message|>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('Run')
    expect(text).toContain('to test')
    // No ToolInputStarted — invoke inside message is not structural
    expect(events.filter(e => e._tag === 'ToolInputStarted')).toHaveLength(0)
  })

  it('treats <think|> in prose (no open think) as stray close + content', () => {
    const events = parse('\n<think|>\nsome text')
    const errors = events.filter(e => e._tag === 'StructuralParseError') as any[]
    expect(errors.length).toBeGreaterThan(0)
    expect(errors[0].error._tag).toBe('StrayCloseTag')
  })

  it('treats <|think:foo> inside think frame as content', () => {
    const events = parse('\n<|think:outer>\nSome <|think:inner> text\n<think|>')
    const lensChunks = events.filter(e => e._tag === 'LensChunk')
    const text = lensChunks.map(e => (e as any).text).join('')
    expect(text).toContain('<|think:inner>')
  })

  it('treats <|parameter:x> in prose as content (no invoke frame)', () => {
    const events = parse('\n<|message:user>\n<|parameter:x>\n<message|>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('<|parameter:x>')
  })

  it('treats <parameter|> in prose as content', () => {
    const events = parse('\n<|message:user>\n<parameter|>\n<message|>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('<parameter|>')
  })
})
