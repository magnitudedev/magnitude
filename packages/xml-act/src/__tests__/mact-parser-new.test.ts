/**
 * Integration tests for the new mact parser (parser/index.ts).
 * Verifies that the parser emits TurnEngineEvent directly.
 */

import { describe, it, expect } from 'vitest'
import { Schema } from '@effect/schema'
import { defineTool } from '@magnitudedev/tools'
import { Effect } from 'effect'
import { createParser, createParserWithTokenizer } from '../parser/index'
import type { RegisteredTool } from '../types'

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
  const p = createParserWithTokenizer({ tools: customTools })
  const fromPush = p.push(input)
  const fromEnd = p.end()
  return [...fromPush, ...fromEnd]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('mact parser — new architecture', () => {
  it('emits ToolInputStarted for known tool', () => {
    const events = parse('<|invoke:shell>\n<|parameter:command>echo hi<parameter|>\n<invoke|>')
    const started = events.find(e => e._tag === 'ToolInputStarted')
    expect(started).toBeDefined()
    expect(started).toMatchObject({ _tag: 'ToolInputStarted', tagName: 'shell', toolName: 'shell' })
  })

  it('emits ToolInputFieldChunk for parameter content', () => {
    const events = parse('<|invoke:shell>\n<|parameter:command>echo hi<parameter|>\n<invoke|>')
    const chunks = events.filter(e => e._tag === 'ToolInputFieldChunk')
    expect(chunks.length).toBeGreaterThan(0)
    const chunk = chunks[0]
    expect(chunk).toMatchObject({ _tag: 'ToolInputFieldChunk', field: 'command' })
  })

  it('emits ToolInputFieldComplete with coerced value', () => {
    const events = parse('<|invoke:shell>\n<|parameter:command>echo hi<parameter|>\n<invoke|>')
    const complete = events.find(e => e._tag === 'ToolInputFieldComplete')
    expect(complete).toBeDefined()
    expect(complete).toMatchObject({ _tag: 'ToolInputFieldComplete', field: 'command', value: 'echo hi' })
  })

  it('emits ToolInputReady with assembled input', () => {
    const events = parse('<|invoke:shell>\n<|parameter:command>echo hi<parameter|>\n<invoke|>')
    const ready = events.find(e => e._tag === 'ToolInputReady')
    expect(ready).toBeDefined()
    expect(ready).toMatchObject({ _tag: 'ToolInputReady', input: { command: 'echo hi' } })
  })

  it('emits ToolInputParseError for unknown tool', () => {
    const events = parse('<|invoke:unknown-tool>\n<|parameter:foo>bar<parameter|>\n<invoke|>')
    const error = events.find(e => e._tag === 'ToolInputParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'ToolInputParseError', error: { _tag: 'UnknownTool' } })
  })

  it('emits ToolInputParseError for unknown parameter', () => {
    const events = parse('<|invoke:shell>\n<|parameter:nonexistent>value<parameter|>\n<invoke|>')
    const error = events.find(e => e._tag === 'ToolInputParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'ToolInputParseError', error: { _tag: 'UnknownParameter' } })
  })

  it('emits ToolInputParseError for missing required field', () => {
    // shell requires 'command' — invoke without it
    const events = parse('<|invoke:shell>\n<invoke|>')
    const error = events.find(e => e._tag === 'ToolInputParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'ToolInputParseError', error: { _tag: 'MissingRequiredField', parameterName: 'command' } })
  })

  it('emits ToolInputParseError for incomplete invoke at end()', () => {
    const events = parse('<|invoke:shell>\n<|parameter:command>echo hi<parameter|>')
    const error = events.find(e => e._tag === 'ToolInputParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'ToolInputParseError', error: { _tag: 'IncompleteTool' } })
  })

  it('emits LensStart/Chunk/End for think blocks', () => {
    const events = parse('<|think:analyze>\nsome reasoning\n<think|>')
    expect(events.find(e => e._tag === 'LensStart')).toMatchObject({ _tag: 'LensStart', name: 'analyze' })
    expect(events.find(e => e._tag === 'LensChunk')).toBeDefined()
    expect(events.find(e => e._tag === 'LensEnd')).toMatchObject({ _tag: 'LensEnd', name: 'analyze' })
  })

  it('emits MessageStart/Chunk/End for message blocks', () => {
    const events = parse('<|message:user>\nhello world\n<message|>')
    expect(events.find(e => e._tag === 'MessageStart')).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    expect(events.find(e => e._tag === 'MessageChunk')).toBeDefined()
    expect(events.find(e => e._tag === 'MessageEnd')).toBeDefined()
  })

  it('emits TurnEnd on yield', () => {
    const events = parse('some text\n<|yield:user|>')
    const turnEnd = events.find(e => e._tag === 'TurnEnd')
    expect(turnEnd).toBeDefined()
    expect(turnEnd).toMatchObject({ _tag: 'TurnEnd', result: { _tag: 'Success', termination: 'natural' } })
  })

  it('handles optional number parameter', () => {
    const events = parse('<|invoke:shell>\n<|parameter:command>ls<parameter|>\n<|parameter:timeout>30<parameter|>\n<invoke|>')
    const ready = events.find(e => e._tag === 'ToolInputReady')
    expect(ready).toMatchObject({ _tag: 'ToolInputReady', input: { command: 'ls', timeout: 30 } })
  })

  it('ToolInputFieldChunk path is [paramName] for string fields', () => {
    const events = parse('<|invoke:shell>\n<|parameter:command>echo<parameter|>\n<invoke|>')
    const chunk = events.find(e => e._tag === 'ToolInputFieldChunk')
    expect(chunk).toMatchObject({ path: ['command'] })
  })
})
