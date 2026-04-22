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
  const parser = createParser({ tools: tools ?? makeTools() })
  const tokenizer = createTokenizer(
    (token) => parser.pushToken(token),
    new Set(tools?.keys() ?? ['shell']),
  )
  tokenizer.push(input + '\n')
  tokenizer.end()
  parser.end()
  return [...parser.drain()]
}

describe('context-aware tag resolution', () => {
  it('treats <div> inside message as literal content', () => {
    const events = parse('<message to="user">\nHere is <div>some</div> text\n</message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('some')
    expect(text).toContain('text')
    expect(events.filter(e => e._tag === 'StructuralParseError' || e._tag === 'ToolParseError')).toHaveLength(0)
    expect(events.filter(e => e._tag === 'ToolInputStarted')).toHaveLength(0)
  })

  it('treats <strong>bold</strong> inside message as literal content', () => {
    const events = parse('<message to="user">\n<strong>bold</strong>\n</message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('bold')
    expect(events.filter(e => e._tag === 'ToolInputStarted')).toHaveLength(0)
  })

  it('treats <foo> inside message as literal content (not structural)', () => {
    const events = parse('<message to="user">\nSee <foo> here\n</message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('See')
    expect(text).toContain('here')
    expect(events.filter(e => e._tag === 'ToolInputStarted')).toHaveLength(0)
  })

  it('treats <invoke tool="shell"> inside message as literal content', () => {
    const events = parse('<message to="user">\nRun <invoke tool="shell"> to test\n</message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('Run')
    expect(text).toContain('to test')
    expect(events.filter(e => e._tag === 'ToolInputStarted')).toHaveLength(0)
  })

  it('treats </reason> in prose (no open reason) as stray close + content', () => {
    const events = parse('</reason>\nsome text')
    const errors = events.filter(e => e._tag === 'StructuralParseError') as any[]
    expect(errors.length).toBeGreaterThan(0)
    expect(errors[0].error._tag).toBe('StrayCloseTag')
  })

  it('treats <reason about="foo"> inside reason frame as content', () => {
    const events = parse('<reason about="outer">\nSome <reason about="inner"> text\n</reason>')
    const lensChunks = events.filter(e => e._tag === 'LensChunk')
    const text = lensChunks.map(e => (e as any).text).join('')
    expect(text).toContain('<reason about="inner">')
  })

  it('treats <parameter name="x"> in message as content (no invoke frame)', () => {
    const events = parse('<message to="user">\n<parameter name="x">\n</message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('<parameter name="x">')
  })

  it('treats </parameter> in message as content', () => {
    const events = parse('<message to="user">\n</parameter>\n</message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('</parameter>')
  })
})
