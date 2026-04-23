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
  it('</magnitude:reason> closes a reason frame', () => {
    const events = parse('<magnitude:reason about="turn">\ncontent\n</magnitude:reason>')
    expect(events.find(e => e._tag === 'LensEnd')).toBeDefined()
  })

  it('</magnitude:message> closes a message frame', () => {
    const events = parse('<magnitude:message to="user">\nhello\n</magnitude:message>')
    expect(events.find(e => e._tag === 'MessageEnd')).toBeDefined()
  })

  it('</magnitude:invoke> closes an invoke frame', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>')
    expect(events.find(e => e._tag === 'ToolInputReady')).toBeDefined()
  })

  it('</magnitude:message> inside reason frame → literal content', () => {
    const events = parse('<magnitude:reason about="turn">\ncontent</magnitude:message>\n</magnitude:reason>')
    const chunks = events.filter(e => e._tag === 'LensChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('</magnitude:message>')
    expect(events.find(e => e._tag === 'MessageEnd')).toBeUndefined()
  })

  it('</magnitude:reason> inside message frame → literal content', () => {
    const events = parse('<magnitude:message to="user">\ncontent</magnitude:reason>\n</magnitude:message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('</magnitude:reason>')
    expect(events.find(e => e._tag === 'LensEnd')).toBeUndefined()
  })

  it('</magnitude:invoke> inside message frame → literal content', () => {
    const events = parse('<magnitude:message to="user">\ncontent</magnitude:invoke>\n</magnitude:message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('</magnitude:invoke>')
  })

  it('</magnitude:parameter> inside message frame → literal content', () => {
    const events = parse('<magnitude:message to="user">\n</magnitude:parameter>\n</magnitude:message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('</magnitude:parameter>')
  })

  it('unknown close tag → literal content', () => {
    const events = parse('<magnitude:message to="user">\n<skill-name>\n</magnitude:message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('<skill-name>')
  })
})
