/**
 * Close tag behavior spec tests (XML format).
 *
 * Close tags are structural only when their name matches the currently open frame.
 * Mismatched close tags → literal content, no transformation.
 */

import { describe, expect, it } from 'vitest'
import { createTokenizer } from '../tokenizer'
import { createParser } from '../parser/index'
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

function parse(input: string): TurnEngineEvent[] {
  const parser = createParser({ tools: makeTools() })
  const tokenizer = createTokenizer(
    (token) => parser.pushToken(token),
    new Set(['shell']),
  )
  tokenizer.push(input + '\n')
  tokenizer.end()
  parser.end()
  return [...parser.drain()]
}

describe('close tag behavior', () => {
  it('</reason> closes a reason frame', () => {
    const events = parse('<reason about="turn">\ncontent\n</reason>')
    expect(events.find(e => e._tag === 'LensEnd')).toBeDefined()
  })

  it('</message> closes a message frame', () => {
    const events = parse('<message to="user">\nhello\n</message>')
    expect(events.find(e => e._tag === 'MessageEnd')).toBeDefined()
  })

  it('</invoke> closes an invoke frame', () => {
    const events = parse('<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>')
    expect(events.find(e => e._tag === 'ToolInputReady')).toBeDefined()
  })

  it('</message> inside reason frame → literal content', () => {
    const events = parse('<reason about="turn">\ncontent</message>\n</reason>')
    const chunks = events.filter(e => e._tag === 'LensChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('</message>')
    expect(events.find(e => e._tag === 'MessageEnd')).toBeUndefined()
  })

  it('</reason> inside message frame → literal content', () => {
    const events = parse('<message to="user">\ncontent</reason>\n</message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('</reason>')
    expect(events.find(e => e._tag === 'LensEnd')).toBeUndefined()
  })

  it('</invoke> inside message frame → literal content', () => {
    const events = parse('<message to="user">\ncontent</invoke>\n</message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('</invoke>')
  })

  it('</parameter> inside message frame → literal content', () => {
    const events = parse('<message to="user">\n</parameter>\n</message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('</parameter>')
  })

  it('unknown close tag → literal content', () => {
    const events = parse('<message to="user">\n<skill-name>\n</message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('<skill-name>')
  })
})
