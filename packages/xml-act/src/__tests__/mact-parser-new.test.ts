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
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hi</magnitude:parameter>\n</magnitude:invoke>')
    const started = events.find(e => e._tag === 'ToolInputStarted')
    expect(started).toBeDefined()
    expect(started).toMatchObject({ _tag: 'ToolInputStarted', tagName: 'shell', toolName: 'shell' })
  })

  it('emits ToolInputFieldChunk for parameter content', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hi</magnitude:parameter>\n</magnitude:invoke>')
    const chunks = events.filter(e => e._tag === 'ToolInputFieldChunk')
    expect(chunks.length).toBeGreaterThan(0)
    const chunk = chunks[0]
    expect(chunk).toMatchObject({ _tag: 'ToolInputFieldChunk', field: 'command' })
  })

  it('emits ToolInputFieldComplete with coerced value', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hi</magnitude:parameter>\n</magnitude:invoke>')
    const complete = events.find(e => e._tag === 'ToolInputFieldComplete')
    expect(complete).toBeDefined()
    expect(complete).toMatchObject({ _tag: 'ToolInputFieldComplete', field: 'command', value: 'echo hi' })
  })

  it('emits ToolInputReady with assembled input', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hi</magnitude:parameter>\n</magnitude:invoke>')
    const ready = events.find(e => e._tag === 'ToolInputReady')
    expect(ready).toBeDefined()
    expect(ready).toMatchObject({ _tag: 'ToolInputReady', input: { command: 'echo hi' } })
  })

  it('emits StructuralParseError for unknown tool', () => {
    const events = parse('<magnitude:invoke tool="unknown-tool">\n<magnitude:parameter name="foo">bar</magnitude:parameter>\n</magnitude:invoke>')
    const error = events.find(e => e._tag === 'StructuralParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'StructuralParseError', error: { _tag: 'UnknownTool' } })
  })

  it('emits ToolParseError for unknown parameter', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="nonexistent">value</magnitude:parameter>\n</magnitude:invoke>')
    const error = events.find(e => e._tag === 'ToolParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'ToolParseError', error: { _tag: 'UnknownParameter' } })
  })

  it('emits ToolParseError for missing required field', () => {
    const events = parse('<magnitude:invoke tool="shell">\n</magnitude:invoke>')
    const error = events.find(e => e._tag === 'ToolParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'ToolParseError', error: { _tag: 'MissingRequiredField', parameterName: 'command' } })
  })

  it('emits ToolParseError for incomplete invoke at end()', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hi</magnitude:parameter>')
    const error = events.find(e => e._tag === 'ToolParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'ToolParseError', error: { _tag: 'IncompleteTool' } })
  })

  it('emits LensStart/Chunk/End for think blocks', () => {
    const events = parse('<magnitude:think about="analyze">\nsome reasoning\n</magnitude:think>')
    expect(events.find(e => e._tag === 'LensStart')).toMatchObject({ _tag: 'LensStart', name: 'analyze' })
    expect(events.find(e => e._tag === 'LensChunk')).toBeDefined()
    expect(events.find(e => e._tag === 'LensEnd')).toMatchObject({ _tag: 'LensEnd', name: 'analyze' })
  })

  it('emits MessageStart/Chunk/End for message blocks', () => {
    const events = parse('<magnitude:message to="user">\nhello world\n</magnitude:message>')
    expect(events.find(e => e._tag === 'MessageStart')).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    expect(events.find(e => e._tag === 'MessageChunk')).toBeDefined()
    expect(events.find(e => e._tag === 'MessageEnd')).toBeDefined()
  })

  it('emits TurnEnd on yield', () => {
    const events = parse('some text\n<magnitude:yield_user/>')
    const turnEnd = events.find(e => e._tag === 'TurnEnd')
    expect(turnEnd).toBeDefined()
    expect(turnEnd).toMatchObject({ _tag: 'TurnEnd', outcome: { _tag: 'Completed', termination: 'natural' } })
  })

  it('handles optional number parameter', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n<magnitude:parameter name="timeout">30</magnitude:parameter>\n</magnitude:invoke>')
    const ready = events.find(e => e._tag === 'ToolInputReady')
    expect(ready).toMatchObject({ _tag: 'ToolInputReady', input: { command: 'ls', timeout: 30 } })
  })

  it('ToolInputFieldChunk path is [paramName] for string fields', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo</magnitude:parameter>\n</magnitude:invoke>')
    const chunk = events.find(e => e._tag === 'ToolInputFieldChunk')
    expect(chunk).toMatchObject({ path: ['command'] })
  })

  it('mismatched close tag is treated as content (no lenience)', () => {
    // </magnitude:message> inside a think block is not structural — treated as content
    const events = parse('<magnitude:think about="turn">\nsome reasoning</magnitude:message>\n</magnitude:think>')
    expect(events.find(e => e._tag === 'LensEnd')).toBeDefined()
    // No MessageEnd since </magnitude:message> was content
    expect(events.find(e => e._tag === 'MessageEnd')).toBeUndefined()
  })
})
