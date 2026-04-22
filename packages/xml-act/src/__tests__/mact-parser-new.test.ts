/**
 * Integration tests for the XML parser (parser/index.ts).
 * Verifies that the parser emits TurnEngineEvent correctly.
 */

import { describe, it, expect } from 'vitest'
import { Schema } from '@effect/schema'
import { defineTool } from '@magnitudedev/tools'
import { Effect } from 'effect'
import { createParser } from '../parser/index'
import { createTokenizer } from '../tokenizer'
import type { RegisteredTool, TurnEngineEvent } from '../types'

// ---------------------------------------------------------------------------
// Test tool setup
// ---------------------------------------------------------------------------

const shellTool = defineTool({
  name: 'shell',
  label: 'Shell',
  description: 'Run a shell command',
  inputSchema: Schema.Struct({
    command: Schema.String,
    timeout: Schema.optional(Schema.Number),
  }),
  outputSchema: Schema.String,
  execute: (_input) => Effect.succeed('ok'),
})

const shellRegistered: RegisteredTool = {
  tool: shellTool,
  tagName: 'shell',
  groupName: 'default',
}

const tools = new Map<string, RegisteredTool>([
  ['shell', shellRegistered],
])

function parse(input: string, customTools = tools) {
  const p = createParser({ tools: customTools })
  const knownToolTags = new Set(customTools.keys())
  const tokenizer = createTokenizer(
    (token) => p.pushToken(token),
    knownToolTags,
  )
  // Append trailing newline so close tags at end of input are confirmed
  tokenizer.push(input + '\n')
  const fromPush = p.drain()
  tokenizer.end()
  p.end()
  const fromEnd = p.drain()
  return [...fromPush, ...fromEnd]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('xml parser', () => {
  it('emits ToolInputStarted for known tool', () => {
    const events = parse('<invoke tool="shell">\n<parameter name="command">echo hi</parameter>\n</invoke>')
    const started = events.find(e => e._tag === 'ToolInputStarted')
    expect(started).toBeDefined()
    expect(started).toMatchObject({ _tag: 'ToolInputStarted', tagName: 'shell', toolName: 'shell' })
  })

  it('emits ToolInputFieldChunk for parameter content', () => {
    const events = parse('<invoke tool="shell">\n<parameter name="command">echo hi</parameter>\n</invoke>')
    const chunks = events.filter(e => e._tag === 'ToolInputFieldChunk')
    expect(chunks.length).toBeGreaterThan(0)
    const chunk = chunks[0]
    expect(chunk).toMatchObject({ _tag: 'ToolInputFieldChunk', field: 'command' })
  })

  it('emits ToolInputFieldComplete with coerced value', () => {
    const events = parse('<invoke tool="shell">\n<parameter name="command">echo hi</parameter>\n</invoke>')
    const complete = events.find(e => e._tag === 'ToolInputFieldComplete')
    expect(complete).toBeDefined()
    expect(complete).toMatchObject({ _tag: 'ToolInputFieldComplete', field: 'command', value: 'echo hi' })
  })

  it('emits ToolInputReady with assembled input', () => {
    const events = parse('<invoke tool="shell">\n<parameter name="command">echo hi</parameter>\n</invoke>')
    const ready = events.find(e => e._tag === 'ToolInputReady')
    expect(ready).toBeDefined()
    expect(ready).toMatchObject({ _tag: 'ToolInputReady', input: { command: 'echo hi' } })
  })

  it('emits StructuralParseError for unknown tool', () => {
    const events = parse('<invoke tool="unknown-tool">\n<parameter name="foo">bar</parameter>\n</invoke>')
    const error = events.find(e => e._tag === 'StructuralParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'StructuralParseError', error: { _tag: 'UnknownTool' } })
  })

  it('emits ToolParseError for unknown parameter', () => {
    const events = parse('<invoke tool="shell">\n<parameter name="nonexistent">value</parameter>\n</invoke>')
    const error = events.find(e => e._tag === 'ToolParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'ToolParseError', error: { _tag: 'UnknownParameter' } })
  })

  it('emits ToolParseError for missing required field', () => {
    const events = parse('<invoke tool="shell">\n</invoke>')
    const error = events.find(e => e._tag === 'ToolParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'ToolParseError', error: { _tag: 'MissingRequiredField', parameterName: 'command' } })
  })

  it('emits ToolParseError for incomplete invoke at end()', () => {
    const events = parse('<invoke tool="shell">\n<parameter name="command">echo hi</parameter>')
    const error = events.find(e => e._tag === 'ToolParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'ToolParseError', error: { _tag: 'IncompleteTool' } })
  })

  it('emits LensStart/Chunk/End for reason blocks', () => {
    const events = parse('<reason about="analyze">\nsome reasoning\n</reason>')
    expect(events.find(e => e._tag === 'LensStart')).toMatchObject({ _tag: 'LensStart', name: 'analyze' })
    expect(events.find(e => e._tag === 'LensChunk')).toBeDefined()
    expect(events.find(e => e._tag === 'LensEnd')).toMatchObject({ _tag: 'LensEnd', name: 'analyze' })
  })

  it('emits MessageStart/Chunk/End for message blocks', () => {
    const events = parse('<message to="user">\nhello world\n</message>')
    expect(events.find(e => e._tag === 'MessageStart')).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    expect(events.find(e => e._tag === 'MessageChunk')).toBeDefined()
    expect(events.find(e => e._tag === 'MessageEnd')).toBeDefined()
  })

  it('emits TurnEnd on yield', () => {
    const events = parse('some text\n<yield_user/>')
    const turnEnd = events.find(e => e._tag === 'TurnEnd')
    expect(turnEnd).toBeDefined()
    expect(turnEnd).toMatchObject({ _tag: 'TurnEnd', result: { _tag: 'Success', termination: 'natural' } })
  })

  it('handles optional number parameter', () => {
    const events = parse('<invoke tool="shell">\n<parameter name="command">ls</parameter>\n<parameter name="timeout">30</parameter>\n</invoke>')
    const ready = events.find(e => e._tag === 'ToolInputReady')
    expect(ready).toMatchObject({ _tag: 'ToolInputReady', input: { command: 'ls', timeout: 30 } })
  })

  it('ToolInputFieldChunk path is [paramName] for string fields', () => {
    const events = parse('<invoke tool="shell">\n<parameter name="command">echo</parameter>\n</invoke>')
    const chunk = events.find(e => e._tag === 'ToolInputFieldChunk')
    expect(chunk).toMatchObject({ path: ['command'] })
  })

  it('mismatched close tag is treated as content (no lenience)', () => {
    // </message> inside a reason block is not structural — treated as content
    const events = parse('<reason about="turn">\nsome reasoning</message>\n</reason>')
    expect(events.find(e => e._tag === 'LensEnd')).toBeDefined()
    // No MessageEnd since </message> was content
    expect(events.find(e => e._tag === 'MessageEnd')).toBeUndefined()
  })
})
