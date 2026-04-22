/**
 * No-silent-drop tests — verifies nothing silently disappears.
 */
import { describe, it, expect } from 'vitest'
import { createParser } from '../parser/index'
import { createTokenizer } from '../tokenizer'
import type { TurnEngineEvent, RegisteredTool, ToolParseErrorEvent, StructuralParseErrorEvent } from '../types'
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

const multiParamTool = defineTool({
  name: 'multi',
  label: 'Multi',
  group: 'fs',
  description: 'Tool with multiple required params',
  inputSchema: Schema.Struct({ a: Schema.String, b: Schema.String, c: Schema.String }),
  outputSchema: Schema.Struct({ result: Schema.String }),
  execute: async () => ({ result: '' }),
})

function makeTools(): ReadonlyMap<string, RegisteredTool> {
  return new Map([
    ['shell', { tool: shellTool, tagName: 'shell', groupName: 'fs' }],
    ['multi', { tool: multiParamTool, tagName: 'multi', groupName: 'fs' }],
  ])
}

function parse(input: string): TurnEngineEvent[] {
  const parser = createParser({ tools: makeTools() })
  const tokenizer = createTokenizer(
    (token) => parser.pushToken(token),
    new Set(['shell', 'multi']),
  )
  tokenizer.push(input + '\n')
  tokenizer.end()
  parser.end()
  return [...parser.drain()]
}

function getStructuralErrors(events: TurnEngineEvent[]): StructuralParseErrorEvent[] {
  return events.filter(e => e._tag === 'StructuralParseError') as StructuralParseErrorEvent[]
}

function getToolErrors(events: TurnEngineEvent[]): ToolParseErrorEvent[] {
  return events.filter(e => e._tag === 'ToolParseError') as ToolParseErrorEvent[]
}

describe('no silent drops', () => {
  it('stray close of known structural tag → error + content', () => {
    const events = parse('</message>\nsome text')
    const errors = getStructuralErrors(events)
    expect(errors.length).toBeGreaterThan(0)
    expect(errors[0].error._tag).toBe('StrayCloseTag')
    const proseChunks = events.filter(e => e._tag === 'ProseChunk')
    expect(proseChunks.length).toBeGreaterThan(0)
  })

  it('<invoke> without tool attribute → MissingToolName error', () => {
    const errors = getStructuralErrors(parse('<invoke>\n</invoke>'))
    expect(errors.length).toBeGreaterThan(0)
    expect(errors[0].error._tag).toBe('MissingToolName')
  })

  it('non-whitespace content between parameters → UnexpectedContent error', () => {
    const events = parse('<invoke tool="shell">\nstray content\n<parameter name="command">ls</parameter>\n</invoke>')
    const errors = getStructuralErrors(events)
    const unexpectedContent = errors.find(e => e.error._tag === 'UnexpectedContent')
    expect(unexpectedContent).toBeDefined()
  })

  it('orphan </parameter> in message → literal content', () => {
    const events = parse('<message to="user">\n</parameter>\n</message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    const text = chunks.map(e => (e as any).text).join('')
    expect(text).toContain('</parameter>')
  })

  it('all missing required fields reported', () => {
    const events = parse('<invoke tool="multi">\n</invoke>')
    const errors = getToolErrors(events)
    const missingFields = errors.filter(e => e.error._tag === 'MissingRequiredField')
    expect(missingFields.length).toBe(3) // a, b, c all missing
  })
})
