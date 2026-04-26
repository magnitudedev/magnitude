/**
 * Comprehensive test suite for the parser redesign (TDD).
 *
 * These tests define the EXPECTED behavior of the redesigned parser.
 * They should all pass after the redesign is complete.
 *
 * Tests are organized into:
 * 1. Crash scenarios — bugs that motivated the redesign
 * 2. Normal structural behavior — regression tests
 * 3. Content preservation — close tags in body text, unknown tags, whitespace
 * 4. Error cases — missing attrs, unknown tools, stray tags, EOF
 * 5. Event sequence verification — correct ordering of emitted events
 */

import { describe, it, expect } from 'vitest'
import { Schema } from '@effect/schema'
import { defineTool } from '@magnitudedev/tools'
import { Effect } from 'effect'
import { createParser } from '../parser/index'
import { createTokenizer } from '../tokenizer'
import type { RegisteredTool, TurnEngineEvent } from '../types'

// ---------------------------------------------------------------------------
// Tool fixtures
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

const readTool = defineTool({
  name: 'read',
  label: 'Read File',
  description: 'Read a file',
  inputSchema: Schema.Struct({
    path: Schema.String,
  }),
  outputSchema: Schema.String,
  execute: (_input) => Effect.succeed('content'),
})

const multiParamTool = defineTool({
  name: 'multi',
  label: 'Multi-param tool',
  description: 'Tool with multiple required params',
  inputSchema: Schema.Struct({
    a: Schema.String,
    b: Schema.String,
    c: Schema.optional(Schema.String),
  }),
  outputSchema: Schema.String,
  execute: (_input) => Effect.succeed('ok'),
})

const shellRegistered: RegisteredTool = {
  tool: shellTool,
  tagName: 'shell',
  groupName: 'default',
}

const readRegistered: RegisteredTool = {
  tool: readTool,
  tagName: 'read',
  groupName: 'default',
}

const multiRegistered: RegisteredTool = {
  tool: multiParamTool,
  tagName: 'multi',
  groupName: 'default',
}

const tools = new Map<string, RegisteredTool>([
  ['shell', shellRegistered],
  ['read', readRegistered],
  ['multi', multiRegistered],
])

// ---------------------------------------------------------------------------
// Parse helper
// ---------------------------------------------------------------------------

function parse(input: string, customTools = tools): TurnEngineEvent[] {
  const p = createParser({ tools: customTools })
  const knownToolTags = new Set(customTools.keys())
  const tokenizer = createTokenizer(
    (token) => p.pushToken(token),
    knownToolTags,
  )
  // Trailing newline ensures close tags at end of input are confirmed
  tokenizer.push(input + '\n')
  const fromPush = p.drain()
  tokenizer.end()
  p.end()
  const fromEnd = p.drain()
  return [...fromPush, ...fromEnd]
}

function tags(events: TurnEngineEvent[]): string[] {
  return events.map(e => e._tag)
}

// ---------------------------------------------------------------------------
// 1. CRASH SCENARIOS
// These tests document bugs in the CURRENT parser that the redesign must fix.
// ---------------------------------------------------------------------------

describe('crash scenarios (redesign must fix these)', () => {
  it('BUG: <magnitude:parameter> inside <magnitude:reason> is treated as content, not a crash', () => {
    // In the current parser, PARAMETER_VALID_TAGS includes 'parameter', so a <magnitude:parameter>
    // inside a reason frame resolves as structural and causes a cast crash.
    // After redesign: <magnitude:parameter> inside <magnitude:reason> is content.
    expect(() => {
      const events = parse('<magnitude:reason about="test">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:reason>')
      // Should complete without throwing
      expect(events.find(e => e._tag === 'LensStart')).toBeDefined()
      expect(events.find(e => e._tag === 'LensEnd')).toBeDefined()
      // No ToolInputStarted — parameter was not structural
      expect(events.find(e => e._tag === 'ToolInputStarted')).toBeUndefined()
    }).not.toThrow()
  })

  it('BUG: <magnitude:parameter> inside <magnitude:parameter> is treated as content, not a crash', () => {
    // The original crash: nested <magnitude:parameter> inside <magnitude:parameter> resolves as structural,
    // then dispatch casts ParameterFrame to InvokeFrame → seenParams is undefined → crash.
    expect(() => {
      const events = parse(
        '<magnitude:invoke tool="shell">\n' +
        '<magnitude:parameter name="command">echo <magnitude:parameter name="nested">bad</magnitude:parameter></magnitude:parameter>\n' +
        '</magnitude:invoke>'
      )
      // Should not crash. The outer parameter should complete.
      const ready = events.find(e => e._tag === 'ToolInputReady')
      expect(ready).toBeDefined()
    }).not.toThrow()
  })

  it('BUG: <magnitude:message> inside <magnitude:message> is treated as content, not a crash', () => {
    // MESSAGE_VALID_TAGS contains 'message' — allows nested messages.
    // CURRENT behavior: inner <magnitude:message> opens a second MessageFrame (2 MessageStarts).
    // EXPECTED after redesign: inner <magnitude:message> is content, only 1 MessageStart.
    // This test documents the bug — it will FAIL on the current parser (2 starts instead of 1)
    // and should PASS after the redesign.
    const events = parse('<magnitude:message to="user">\nhello <magnitude:message to="other">world</magnitude:message>\n</magnitude:message>')
    expect(events.find(e => e._tag === 'MessageStart')).toBeDefined()
    expect(events.find(e => e._tag === 'MessageEnd')).toBeDefined()
    // Only one MessageStart (not two)
    const starts = events.filter(e => e._tag === 'MessageStart')
    expect(starts).toHaveLength(1)
  })

  it('BUG: <magnitude:yield_user/> inside <magnitude:yield_user/> is treated as content, not a crash', () => {
    // INVOKE_VALID_TAGS contains 'invoke' — allows nested invokes.
    // After redesign: inner <magnitude:yield_user/> is content.
    expect(() => {
      const events = parse(
        '<magnitude:invoke tool="shell">\n' +
        '<magnitude:parameter name="command">run <magnitude:invoke tool="read">this</magnitude:invoke></magnitude:parameter>\n' +
        '</magnitude:invoke>'
      )
      const starts = events.filter(e => e._tag === 'ToolInputStarted')
      expect(starts).toHaveLength(1)
    }).not.toThrow()
  })

  it('BUG: <magnitude:filter> inside <magnitude:reason> is treated as content, not a crash', () => {
    expect(() => {
      const events = parse('<magnitude:reason about="test">\n<magnitude:filter>$.foo</magnitude:filter>\n</magnitude:reason>')
      expect(events.find(e => e._tag === 'LensStart')).toBeDefined()
      expect(events.find(e => e._tag === 'LensEnd')).toBeDefined()
      // No FilterStarted — filter was not structural inside reason
      expect(events.find(e => e._tag === 'ToolInputStarted')).toBeUndefined()
    }).not.toThrow()
  })

  it('BUG: <magnitude:parameter> in prose (top-level) is treated as content, not a crash', () => {
    expect(() => {
      const events = parse('some prose <magnitude:parameter name="foo">bar</magnitude:parameter> more prose')
      // Should not crash. No ToolInputStarted.
      expect(events.find(e => e._tag === 'ToolInputStarted')).toBeUndefined()
    }).not.toThrow()
  })
})

// ---------------------------------------------------------------------------
// 2. NORMAL STRUCTURAL BEHAVIOR (regression tests)
// ---------------------------------------------------------------------------

describe('reason blocks', () => {
  it('emits LensStart with correct name', () => {
    const events = parse('<magnitude:reason about="alignment">\nsome reasoning\n</magnitude:reason>')
    expect(events.find(e => e._tag === 'LensStart')).toMatchObject({
      _tag: 'LensStart',
      name: 'alignment',
    })
  })

  it('emits LensChunk with content', () => {
    const events = parse('<magnitude:reason about="x">\nhello world\n</magnitude:reason>')
    const chunks = events.filter(e => e._tag === 'LensChunk')
    expect(chunks.length).toBeGreaterThan(0)
    const allText = chunks.map((c: any) => c.text).join('')
    expect(allText).toContain('hello world')
  })

  it('emits LensEnd with correct name and full content', () => {
    const events = parse('<magnitude:reason about="plan">\nmy plan\n</magnitude:reason>')
    const end = events.find(e => e._tag === 'LensEnd') as any
    expect(end).toMatchObject({ _tag: 'LensEnd', name: 'plan' })
    expect(end.content).toContain('my plan')
  })

  it('emits LensStart/Chunk/End in correct order', () => {
    const events = parse('<magnitude:reason about="x">\nfoo\n</magnitude:reason>')
    const t = tags(events)
    const startIdx = t.indexOf('LensStart')
    const chunkIdx = t.indexOf('LensChunk')
    const endIdx = t.indexOf('LensEnd')
    expect(startIdx).toBeGreaterThanOrEqual(0)
    expect(chunkIdx).toBeGreaterThan(startIdx)
    expect(endIdx).toBeGreaterThan(chunkIdx)
  })

  it('handles multiple reason blocks in sequence', () => {
    const events = parse(
      '<magnitude:reason about="first">\nfoo\n</magnitude:reason>\n' +
      '<magnitude:reason about="second">\nbar\n</magnitude:reason>'
    )
    const starts = events.filter(e => e._tag === 'LensStart') as any[]
    expect(starts).toHaveLength(2)
    expect(starts[0].name).toBe('first')
    expect(starts[1].name).toBe('second')
  })
})

describe('message blocks', () => {
  it('emits MessageStart with correct recipient', () => {
    const events = parse('<magnitude:message to="user">\nhello\n</magnitude:message>')
    expect(events.find(e => e._tag === 'MessageStart')).toMatchObject({
      _tag: 'MessageStart',
      to: 'user',
    })
  })

  it('emits MessageChunk with content', () => {
    const events = parse('<magnitude:message to="user">\nhello world\n</magnitude:message>')
    const chunks = events.filter(e => e._tag === 'MessageChunk')
    expect(chunks.length).toBeGreaterThan(0)
    const allText = chunks.map((c: any) => c.text).join('')
    expect(allText).toContain('hello world')
  })

  it('emits MessageEnd', () => {
    const events = parse('<magnitude:message to="user">\nhello\n</magnitude:message>')
    expect(events.find(e => e._tag === 'MessageEnd')).toBeDefined()
  })

  it('MessageStart/Chunk/End share the same id', () => {
    const events = parse('<magnitude:message to="user">\nfoo\n</magnitude:message>')
    const start = events.find(e => e._tag === 'MessageStart') as any
    const chunk = events.find(e => e._tag === 'MessageChunk') as any
    const end = events.find(e => e._tag === 'MessageEnd') as any
    expect(start.id).toBeDefined()
    expect(chunk.id).toBe(start.id)
    expect(end.id).toBe(start.id)
  })

  it('emits MessageStart/Chunk/End in correct order', () => {
    const events = parse('<magnitude:message to="user">\nfoo\n</magnitude:message>')
    const t = tags(events)
    const startIdx = t.indexOf('MessageStart')
    const chunkIdx = t.indexOf('MessageChunk')
    const endIdx = t.indexOf('MessageEnd')
    expect(startIdx).toBeGreaterThanOrEqual(0)
    expect(chunkIdx).toBeGreaterThan(startIdx)
    expect(endIdx).toBeGreaterThan(chunkIdx)
  })
})

describe('invoke / tool calls', () => {
  it('emits ToolInputStarted with correct tool name', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>')
    expect(events.find(e => e._tag === 'ToolInputStarted')).toMatchObject({
      _tag: 'ToolInputStarted',
      toolName: 'shell',
    })
  })

  it('emits ToolInputFieldChunk for parameter content', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hi</magnitude:parameter>\n</magnitude:invoke>')
    const chunks = events.filter(e => e._tag === 'ToolInputFieldChunk')
    expect(chunks.length).toBeGreaterThan(0)
    expect(chunks[0]).toMatchObject({ _tag: 'ToolInputFieldChunk', field: 'command' })
  })

  it('emits ToolInputFieldComplete with coerced value', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hi</magnitude:parameter>\n</magnitude:invoke>')
    expect(events.find(e => e._tag === 'ToolInputFieldComplete')).toMatchObject({
      _tag: 'ToolInputFieldComplete',
      field: 'command',
      value: 'echo hi',
    })
  })

  it('emits ToolInputReady with assembled input', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hi</magnitude:parameter>\n</magnitude:invoke>')
    expect(events.find(e => e._tag === 'ToolInputReady')).toMatchObject({
      _tag: 'ToolInputReady',
      input: { command: 'echo hi' },
    })
  })

  it('handles multiple parameters', () => {
    const events = parse(
      '<magnitude:invoke tool="multi">\n' +
      '<magnitude:parameter name="a">hello</magnitude:parameter>\n' +
      '<magnitude:parameter name="b">world</magnitude:parameter>\n' +
      '</magnitude:invoke>'
    )
    const ready = events.find(e => e._tag === 'ToolInputReady') as any
    expect(ready).toMatchObject({ _tag: 'ToolInputReady', input: { a: 'hello', b: 'world' } })
  })

  it('handles optional numeric parameter', () => {
    const events = parse(
      '<magnitude:invoke tool="shell">\n' +
      '<magnitude:parameter name="command">ls</magnitude:parameter>\n' +
      '<magnitude:parameter name="timeout">30</magnitude:parameter>\n' +
      '</magnitude:invoke>'
    )
    const ready = events.find(e => e._tag === 'ToolInputReady') as any
    expect(ready).toMatchObject({ _tag: 'ToolInputReady', input: { command: 'ls', timeout: 30 } })
  })

  it('ToolInputFieldChunk path is [paramName] for top-level string fields', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo</magnitude:parameter>\n</magnitude:invoke>')
    const chunk = events.find(e => e._tag === 'ToolInputFieldChunk') as any
    expect(chunk.path).toEqual(['command'])
  })

  it('emits events in correct order: Started → FieldChunk → FieldComplete → Ready', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>')
    const t = tags(events)
    const started = t.indexOf('ToolInputStarted')
    const chunk = t.indexOf('ToolInputFieldChunk')
    const complete = t.indexOf('ToolInputFieldComplete')
    const ready = t.indexOf('ToolInputReady')
    expect(started).toBeGreaterThanOrEqual(0)
    expect(chunk).toBeGreaterThan(started)
    expect(complete).toBeGreaterThan(chunk)
    expect(ready).toBeGreaterThan(complete)
  })

  it('handles filter inside invoke', () => {
    const events = parse(
      '<magnitude:invoke tool="read">\n' +
      '<magnitude:parameter name="path">foo.json</magnitude:parameter>\n' +
      '<magnitude:filter>$.bar</magnitude:filter>\n' +
      '</magnitude:invoke>'
    )
    const ready = events.find(e => e._tag === 'ToolInputReady') as any
    expect(ready).toBeDefined()
    expect(ready.input).toMatchObject({ path: 'foo.json' })
  })
})

describe('yield', () => {
  it('emits TurnEnd on <magnitude:yield_user/>', () => {
    const events = parse('<magnitude:yield_user/>')
    const end = events.find(e => e._tag === 'TurnEnd') as any
    expect(end).toBeDefined()
    expect(end.outcome._tag).toBe('Completed')
    expect(end.outcome.termination).toBe('natural')
  })

  it('TurnEnd has correct target for yield_user', () => {
    const events = parse('<magnitude:yield_user/>')
    const end = events.find(e => e._tag === 'TurnEnd') as any
    expect(end.outcome.turnControl?.target).toBe('user')
  })

  it('TurnEnd has correct target for yield_invoke', () => {
    const events = parse('<magnitude:yield_invoke/>')
    const end = events.find(e => e._tag === 'TurnEnd') as any
    expect(end.outcome.turnControl?.target).toBe('invoke')
  })
})

describe('mixed turn sequences', () => {
  it('reason + message in sequence', () => {
    const events = parse(
      '<magnitude:reason about="plan">\nthinking\n</magnitude:reason>\n' +
      '<magnitude:message to="user">\nhello\n</magnitude:message>'
    )
    expect(events.find(e => e._tag === 'LensStart')).toBeDefined()
    expect(events.find(e => e._tag === 'LensEnd')).toBeDefined()
    expect(events.find(e => e._tag === 'MessageStart')).toBeDefined()
    expect(events.find(e => e._tag === 'MessageEnd')).toBeDefined()
  })

  it('reason + invoke in sequence', () => {
    const events = parse(
      '<magnitude:reason about="plan">\nthinking\n</magnitude:reason>\n' +
      '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>'
    )
    expect(events.find(e => e._tag === 'LensStart')).toBeDefined()
    expect(events.find(e => e._tag === 'ToolInputReady')).toBeDefined()
  })

  it('multiple invokes in sequence', () => {
    const events = parse(
      '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n' +
      '<magnitude:invoke tool="read">\n<magnitude:parameter name="path">foo.ts</magnitude:parameter>\n</magnitude:invoke>'
    )
    const readyEvents = events.filter(e => e._tag === 'ToolInputReady')
    expect(readyEvents).toHaveLength(2)
  })

  it('reason + message + invoke + yield', () => {
    const events = parse(
      '<magnitude:reason about="plan">\nthinking\n</magnitude:reason>\n' +
      '<magnitude:message to="user">\nhello\n</magnitude:message>\n' +
      '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n' +
      '<magnitude:yield_user/>'
    )
    expect(events.find(e => e._tag === 'LensStart')).toBeDefined()
    expect(events.find(e => e._tag === 'MessageStart')).toBeDefined()
    expect(events.find(e => e._tag === 'ToolInputReady')).toBeDefined()
    expect(events.find(e => e._tag === 'TurnEnd')).toBeDefined()
  })

  it('events from reason come before events from message', () => {
    const events = parse(
      '<magnitude:reason about="x">\nfoo\n</magnitude:reason>\n' +
      '<magnitude:message to="user">\nbar\n</magnitude:message>'
    )
    const t = tags(events)
    const lensStart = t.indexOf('LensStart')
    const msgStart = t.indexOf('MessageStart')
    expect(lensStart).toBeGreaterThanOrEqual(0)
    expect(msgStart).toBeGreaterThan(lensStart)
  })
})

// ---------------------------------------------------------------------------
// 3. CONTENT PRESERVATION
// ---------------------------------------------------------------------------

describe('content preservation', () => {
  it('close tag in backticks inside reason body closes immediately under first-close-wins', () => {
    const events = parse('<magnitude:reason about="x">\nthink about `</magnitude:reason>` tags\n</magnitude:reason>')
    const end = events.find(e => e._tag === 'LensEnd') as any
    expect(end).toBeDefined()
    expect(end.content).toContain('think about `')
    expect(end.content).not.toContain('</magnitude:reason>` tags')
  })

  it('close tag in quotes inside parameter body closes immediately under first-close-wins', () => {
    const events = parse(
      '<magnitude:invoke tool="shell">\n' +
      '<magnitude:parameter name="command">echo "</magnitude:parameter>"</magnitude:parameter>\n' +
      '</magnitude:invoke>'
    )
    const complete = events.find(e => e._tag === 'ToolInputFieldComplete') as any
    expect(complete).toBeDefined()
    expect(complete.value).toBe('echo "')
  })

  it('unknown tags inside reason body are content', () => {
    const events = parse('<magnitude:reason about="x">\nhello <unknown-tag>world</unknown-tag>\n</magnitude:reason>')
    const end = events.find(e => e._tag === 'LensEnd') as any
    expect(end).toBeDefined()
    expect(end.content).toContain('<unknown-tag>')
  })

  it('unknown tags inside parameter body are content', () => {
    const events = parse(
      '<magnitude:invoke tool="shell">\n' +
      '<magnitude:parameter name="command">echo <unknown>hi</unknown></magnitude:parameter>\n' +
      '</magnitude:invoke>'
    )
    const complete = events.find(e => e._tag === 'ToolInputFieldComplete') as any
    expect(complete).toBeDefined()
    expect(complete.value).toContain('<unknown>')
  })

  it('unknown tags inside message body are content', () => {
    const events = parse('<magnitude:message to="user">\nhello <em>world</em>\n</magnitude:message>')
    const end = events.find(e => e._tag === 'MessageEnd')
    expect(end).toBeDefined()
    const chunks = events.filter(e => e._tag === 'MessageChunk') as any[]
    const allText = chunks.map(c => c.text).join('')
    expect(allText).toContain('<em>')
  })

  it('mismatched close tag is treated as content (no lenience)', () => {
    // </magnitude:message> inside a reason block is content, not structural
    const events = parse('<magnitude:reason about="turn">\nsome reasoning</magnitude:message>\n</magnitude:reason>')
    expect(events.find(e => e._tag === 'LensEnd')).toBeDefined()
    // No MessageEnd — </magnitude:message> was treated as content
    expect(events.find(e => e._tag === 'MessageEnd')).toBeUndefined()
  })

  it('prose before structural tags emits ProseChunk', () => {
    const events = parse('some prose text\n<magnitude:message to="user">\nhello\n</magnitude:message>')
    expect(events.find(e => e._tag === 'ProseChunk')).toBeDefined()
  })

  it('whitespace between tags does not generate spurious events', () => {
    const events = parse(
      '<magnitude:reason about="x">\nfoo\n</magnitude:reason>\n\n\n<magnitude:message to="user">\nbar\n</magnitude:message>'
    )
    const starts = events.filter(e => e._tag === 'LensStart')
    const msgStarts = events.filter(e => e._tag === 'MessageStart')
    expect(starts).toHaveLength(1)
    expect(msgStarts).toHaveLength(1)
  })
})

// ---------------------------------------------------------------------------
// 4. ERROR CASES
// ---------------------------------------------------------------------------

describe('error cases', () => {
  it('emits StructuralParseError for unknown tool', () => {
    const events = parse('<magnitude:invoke tool="nonexistent">\n<magnitude:parameter name="foo">bar</magnitude:parameter>\n</magnitude:invoke>')
    const error = events.find(e => e._tag === 'StructuralParseError') as any
    expect(error).toBeDefined()
    expect(error.error._tag).toBe('UnknownTool')
  })

  it('emits ToolParseError for unknown parameter', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="nonexistent">value</magnitude:parameter>\n</magnitude:invoke>')
    const error = events.find(e => e._tag === 'ToolParseError') as any
    expect(error).toBeDefined()
    expect(error.error._tag).toBe('UnknownParameter')
  })

  it('emits ToolParseError for missing required field', () => {
    const events = parse('<magnitude:invoke tool="shell">\n</magnitude:invoke>')
    const error = events.find(e => e._tag === 'ToolParseError') as any
    expect(error).toBeDefined()
    expect(error.error._tag).toBe('MissingRequiredField')
    expect(error.error.parameterName).toBe('command')
  })

  it('emits ToolParseError for incomplete invoke at EOF', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hi</magnitude:parameter>')
    const error = events.find(e => e._tag === 'ToolParseError') as any
    expect(error).toBeDefined()
    expect(error.error._tag).toBe('IncompleteTool')
  })

  it('emits StructuralParseError for <magnitude:invoke> without tool attribute', () => {
    const events = parse('<magnitude:invoke>\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>')
    const error = events.find(e => e._tag === 'StructuralParseError') as any
    expect(error).toBeDefined()
    expect(error.error._tag).toBe('MissingToolName')
  })

  it('stray </magnitude:reason> with no open reason is treated as content or error', () => {
    // A stray close tag with no matching open should not crash
    expect(() => {
      const events = parse('some prose </magnitude:reason> more prose')
      // Should not crash. No LensEnd event.
      expect(events.find(e => e._tag === 'LensEnd')).toBeUndefined()
    }).not.toThrow()
  })

  it('stray </magnitude:invoke> with no open invoke is treated as content or error', () => {
    expect(() => {
      const events = parse('some prose </magnitude:invoke> more prose')
      expect(events.find(e => e._tag === 'ToolInputReady')).toBeUndefined()
    }).not.toThrow()
  })

  it('EOF with unclosed reason frame — flushes without crash', () => {
    expect(() => {
      const events = parse('<magnitude:reason about="x">\nsome content without close')
      // Should not crash
      expect(events.find(e => e._tag === 'LensStart')).toBeDefined()
    }).not.toThrow()
  })

  it('EOF with unclosed invoke frame — emits IncompleteTool', () => {
    const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>')
    const error = events.find(e => e._tag === 'ToolParseError') as any
    expect(error).toBeDefined()
    expect(error.error._tag).toBe('IncompleteTool')
  })

  it('EOF with unclosed message frame — flushes without crash', () => {
    expect(() => {
      const events = parse('<magnitude:message to="user">\nhello without close')
      expect(events.find(e => e._tag === 'MessageStart')).toBeDefined()
    }).not.toThrow()
  })

  it('multiple missing required fields — all reported', () => {
    const events = parse('<magnitude:invoke tool="multi">\n</magnitude:invoke>')
    const errors = events.filter(e => e._tag === 'ToolParseError') as any[]
    expect(errors.length).toBeGreaterThanOrEqual(2)
    const missingFields = errors
      .filter((e: any) => e.error._tag === 'MissingRequiredField')
      .map((e: any) => e.error.parameterName)
    expect(missingFields).toContain('a')
    expect(missingFields).toContain('b')
  })
})

// ---------------------------------------------------------------------------
// 5. EVENT SEQUENCE VERIFICATION
// ---------------------------------------------------------------------------

describe('event sequence verification', () => {
  it('full invoke sequence has no gaps or out-of-order events', () => {
    const events = parse(
      '<magnitude:invoke tool="shell">\n' +
      '<magnitude:parameter name="command">ls -la</magnitude:parameter>\n' +
      '</magnitude:invoke>'
    )
    const t = tags(events)
    // ToolInputStarted before any field events
    const started = t.indexOf('ToolInputStarted')
    const firstChunk = t.indexOf('ToolInputFieldChunk')
    const firstComplete = t.indexOf('ToolInputFieldComplete')
    const ready = t.indexOf('ToolInputReady')
    expect(started).toBeGreaterThanOrEqual(0)
    expect(firstChunk).toBeGreaterThan(started)
    expect(firstComplete).toBeGreaterThan(firstChunk)
    expect(ready).toBeGreaterThan(firstComplete)
  })

  it('two invokes produce interleaved events in document order', () => {
    const events = parse(
      '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n' +
      '<magnitude:invoke tool="read">\n<magnitude:parameter name="path">foo.ts</magnitude:parameter>\n</magnitude:invoke>'
    )
    const t = tags(events)
    const starts = events.filter(e => e._tag === 'ToolInputStarted') as any[]
    const readyEvents = events.filter(e => e._tag === 'ToolInputReady') as any[]
    expect(starts).toHaveLength(2)
    expect(readyEvents).toHaveLength(2)
    // First ToolInputStarted comes before second ToolInputStarted
    const firstStartIdx = t.indexOf('ToolInputStarted')
    const secondStartIdx = t.lastIndexOf('ToolInputStarted')
    expect(secondStartIdx).toBeGreaterThan(firstStartIdx)
    // First ToolInputReady before second ToolInputReady
    const firstReadyIdx = t.indexOf('ToolInputReady')
    const secondReadyIdx = t.lastIndexOf('ToolInputReady')
    expect(secondReadyIdx).toBeGreaterThan(firstReadyIdx)
  })

  it('TurnEnd is always the last event', () => {
    const events = parse(
      '<magnitude:reason about="x">\nfoo\n</magnitude:reason>\n' +
      '<magnitude:message to="user">\nbar\n</magnitude:message>\n' +
      '<magnitude:yield_user/>'
    )
    const t = tags(events)
    const turnEndIdx = t.lastIndexOf('TurnEnd')
    expect(turnEndIdx).toBe(t.length - 1)
  })

  it('LensStart/LensEnd pairs are balanced across multiple reasons', () => {
    const events = parse(
      '<magnitude:reason about="a">\nfoo\n</magnitude:reason>\n' +
      '<magnitude:reason about="b">\nbar\n</magnitude:reason>\n' +
      '<magnitude:reason about="c">\nbaz\n</magnitude:reason>'
    )
    const starts = events.filter(e => e._tag === 'LensStart')
    const ends = events.filter(e => e._tag === 'LensEnd')
    expect(starts).toHaveLength(3)
    expect(ends).toHaveLength(3)
  })

  it('parameter events are scoped to their toolCallId', () => {
    const events = parse(
      '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n' +
      '<magnitude:invoke tool="read">\n<magnitude:parameter name="path">foo.ts</magnitude:parameter>\n</magnitude:invoke>'
    )
    const starts = events.filter(e => e._tag === 'ToolInputStarted') as any[]
    const readyEvents = events.filter(e => e._tag === 'ToolInputReady') as any[]
    const id1 = starts[0].toolCallId
    const id2 = starts[1].toolCallId
    expect(id1).not.toBe(id2)
    expect(readyEvents[0].toolCallId).toBe(id1)
    expect(readyEvents[1].toolCallId).toBe(id2)
  })

  it('no events emitted after TurnEnd', () => {
    const events = parse('<magnitude:yield_user/>\nsome trailing text')
    const t = tags(events)
    const turnEndIdx = t.indexOf('TurnEnd')
    expect(turnEndIdx).toBeGreaterThanOrEqual(0)
    // Nothing after TurnEnd
    expect(t.slice(turnEndIdx + 1)).toHaveLength(0)
  })
})
