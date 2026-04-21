/**
 * Integration tests for the new mact parser (parser/index.ts).
 * Verifies that the parser emits TurnEngineEvent directly.
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
  tokenizer.push(input)
  const fromPush = p.drain()
  tokenizer.end()
  p.end()
  const fromEnd = p.drain()
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

  it('emits StructuralParseError for unknown tool', () => {
    const events = parse('<|invoke:unknown-tool>\n<|parameter:foo>bar<parameter|>\n<invoke|>')
    const error = events.find(e => e._tag === 'StructuralParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'StructuralParseError', error: { _tag: 'UnknownTool' } })
  })

  it('emits ToolParseError for unknown parameter', () => {
    const events = parse('<|invoke:shell>\n<|parameter:nonexistent>value<parameter|>\n<invoke|>')
    const error = events.find(e => e._tag === 'ToolParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'ToolParseError', error: { _tag: 'UnknownParameter' } })
  })

  it('emits ToolParseError for missing required field', () => {
    // shell requires 'command' — invoke without it
    const events = parse('<|invoke:shell>\n<invoke|>')
    const error = events.find(e => e._tag === 'ToolParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'ToolParseError', error: { _tag: 'MissingRequiredField', parameterName: 'command' } })
  })

  it('emits ToolParseError for incomplete invoke at end()', () => {
    const events = parse('<|invoke:shell>\n<|parameter:command>echo hi<parameter|>')
    const error = events.find(e => e._tag === 'ToolParseError')
    expect(error).toBeDefined()
    expect(error).toMatchObject({ _tag: 'ToolParseError', error: { _tag: 'IncompleteTool' } })
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

// ---------------------------------------------------------------------------
// Close-tag mismatch lenience
// ---------------------------------------------------------------------------

describe('close-tag mismatch lenience', () => {
  it('think frame closed by <message|> emits LensEnd', () => {
    const events = parse('<|think:turn>\nsome reasoning\n<message|>')
    expect(events.find(e => e._tag === 'LensStart')).toMatchObject({ _tag: 'LensStart', name: 'turn' })
    expect(events.find(e => e._tag === 'LensEnd')).toMatchObject({ _tag: 'LensEnd', name: 'turn' })
  })

  it('think frame closed by <invoke|> emits LensEnd', () => {
    const events = parse('<|think:plan>\nreasoning\n<invoke|>')
    expect(events.find(e => e._tag === 'LensEnd')).toMatchObject({ _tag: 'LensEnd', name: 'plan' })
  })

  it('message frame closed by <think|> emits MessageEnd', () => {
    const events = parse('<|message:user>\nhello\n<think|>')
    expect(events.find(e => e._tag === 'MessageStart')).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    expect(events.find(e => e._tag === 'MessageEnd')).toBeDefined()
  })

  it('invoke frame closed by <think|> finalizes tool call', () => {
    const events = parse('<|invoke:shell>\n<|parameter:command>echo hi<parameter|>\n<think|>')
    expect(events.find(e => e._tag === 'ToolInputReady')).toMatchObject({
      _tag: 'ToolInputReady',
      input: { command: 'echo hi' },
    })
  })

  it('full turn: think closed by <message|> then message closed normally', () => {
    const events = parse(
      '<|think:turn>\nreasoning\n<message|>\n<|message:user>\nhello\n<message|>\n<|yield:user|>',
    )
    expect(events.find(e => e._tag === 'LensEnd')).toBeDefined()
    expect(events.find(e => e._tag === 'MessageEnd')).toBeDefined()
    expect(events.find(e => e._tag === 'TurnEnd')).toBeDefined()
  })
})
