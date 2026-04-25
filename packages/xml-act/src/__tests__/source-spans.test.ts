import { describe, it, expect } from 'vitest'
import { Schema } from '@effect/schema'
import { Effect } from 'effect'
import { defineTool } from '@magnitudedev/tools'
import { createParser } from '../parser/index'
import { createTokenizer } from '../tokenizer'
import { renderParseError } from '../presentation/error-render'
import type {
  Token,
  SourceSpan,
  RegisteredTool,
  ToolParseErrorEvent,
  StructuralParseErrorEvent,
  TurnEngineEvent,
} from '../types'

// =============================================================================
// Test tools
// =============================================================================

const shellTool = defineTool({
  name: 'shell',
  label: () => 'Shell',
  group: 'globals',
  description: 'Run a shell command',
  inputSchema: Schema.Struct({
    command: Schema.String,
    timeout: Schema.optional(Schema.Number),
  }),
  outputSchema: Schema.Struct({ stdout: Schema.String }),
  execute: () => Effect.succeed({ stdout: '' }),
})

const readTool = defineTool({
  name: 'read',
  label: () => 'Read',
  group: 'fs',
  description: 'Read a file',
  inputSchema: Schema.Struct({
    path: Schema.String,
    offset: Schema.optional(Schema.Number),
  }),
  outputSchema: Schema.Struct({ content: Schema.String }),
  execute: () => Effect.succeed({ content: '' }),
})

function makeTools(...tools: Array<{ def: any; tag: string; group: string }>): ReadonlyMap<string, RegisteredTool> {
  return new Map(tools.map(t => [t.tag, { tool: t.def, tagName: t.tag, groupName: t.group }]))
}

const defaultTools = makeTools(
  { def: shellTool, tag: 'shell', group: 'globals' },
  { def: readTool, tag: 'read', group: 'fs' },
)

// =============================================================================
// Helpers
// =============================================================================

function tokenize(input: string, knownToolTags?: ReadonlySet<string>): Token[] {
  const tokens: Token[] = []
  const tokenizer = createTokenizer((t) => tokens.push(t), knownToolTags)
  tokenizer.push(input)
  tokenizer.end()
  return tokens
}

function tokenizeChunks(chunks: string[], knownToolTags?: ReadonlySet<string>): Token[] {
  const tokens: Token[] = []
  const tokenizer = createTokenizer((t) => tokens.push(t), knownToolTags)
  for (const chunk of chunks) tokenizer.push(chunk)
  tokenizer.end()
  return tokens
}

function parse(input: string, tools: ReadonlyMap<string, RegisteredTool> = defaultTools): TurnEngineEvent[] {
  const parser = createParser({ tools })
  const tokenizer = createTokenizer(
    (token) => parser.pushToken(token),
    new Set(tools.keys()),
  )
  tokenizer.push(input + '\n')
  tokenizer.end()
  parser.end()
  return [...parser.drain()]
}

function findToolError(events: TurnEngineEvent[], tag: string): ToolParseErrorEvent | undefined {
  return events.find((e): e is ToolParseErrorEvent =>
    e._tag === 'ToolParseError' && e.error._tag === tag
  ) as ToolParseErrorEvent | undefined
}

function findStructuralError(events: TurnEngineEvent[], tag: string): StructuralParseErrorEvent | undefined {
  return events.find((e): e is StructuralParseErrorEvent =>
    e._tag === 'StructuralParseError' && e.error._tag === tag
  ) as StructuralParseErrorEvent | undefined
}

// =============================================================================
// Tokenizer span tests
// =============================================================================

describe('tokenizer spans', () => {
  describe('basic positions', () => {
    it('single open tag spans from start to end', () => {
      const tokens = tokenize('<magnitude:invoke tool="shell">')
      expect(tokens).toHaveLength(1)
      const span = tokens[0].span
      expect(span.start).toEqual({ offset: 0, line: 1, col: 1 })
      expect(span.end).toEqual({ offset: 31, line: 1, col: 32 })
    })

    it('single close tag', () => {
      const tokens = tokenize('</magnitude:invoke>')
      expect(tokens).toHaveLength(1)
      const span = tokens[0].span
      expect(span.start).toEqual({ offset: 0, line: 1, col: 1 })
      expect(span.end).toEqual({ offset: 19, line: 1, col: 20 })
    })

    it('single self-closing tag', () => {
      const tokens = tokenize('<magnitude:yield_user/>')
      expect(tokens).toHaveLength(1)
      const span = tokens[0].span
      expect(span.start).toEqual({ offset: 0, line: 1, col: 1 })
      expect(span.end).toEqual({ offset: 23, line: 1, col: 24 })
    })

    it('plain content', () => {
      const tokens = tokenize('hello world')
      expect(tokens).toHaveLength(1)
      expect(tokens[0]._tag).toBe('Content')
      const span = tokens[0].span
      expect(span.start).toEqual({ offset: 0, line: 1, col: 1 })
      expect(span.end).toEqual({ offset: 11, line: 1, col: 12 })
    })

    it('empty input produces no tokens', () => {
      const tokens = tokenize('')
      expect(tokens).toHaveLength(0)
    })
  })

  describe('multiline', () => {
    it('content spanning multiple lines', () => {
      const tokens = tokenize('line1\nline2\nline3')
      expect(tokens).toHaveLength(1)
      expect(tokens[0]._tag).toBe('Content')
      expect(tokens[0].span.start).toEqual({ offset: 0, line: 1, col: 1 })
      expect(tokens[0].span.end).toEqual({ offset: 17, line: 3, col: 6 })
    })

    it('tag on second line', () => {
      const tokens = tokenize('hello\n<magnitude:invoke tool="shell">')
      expect(tokens).toHaveLength(2)
      // Content: "hello\n"
      expect(tokens[0]._tag).toBe('Content')
      expect(tokens[0].span.start).toEqual({ offset: 0, line: 1, col: 1 })
      expect(tokens[0].span.end).toEqual({ offset: 6, line: 2, col: 1 })
      // Open tag starts at line 2
      expect(tokens[1]._tag).toBe('Open')
      expect(tokens[1].span.start).toEqual({ offset: 6, line: 2, col: 1 })
      expect(tokens[1].span.end).toEqual({ offset: 37, line: 2, col: 32 })
    })

    it('multiple tags across lines', () => {
      const input = '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>'
      const tokens = tokenize(input, new Set(['shell']))
      const opens = tokens.filter(t => t._tag === 'Open')
      const closes = tokens.filter(t => t._tag === 'Close')

      // First open tag: line 1
      expect(opens[0].span.start.line).toBe(1)
      // Parameter open: line 2
      expect(opens[1].span.start.line).toBe(2)
      // Parameter close: line 2
      expect(closes[0].span.start.line).toBe(2)
      // Invoke close: line 3
      expect(closes[1].span.start.line).toBe(3)
    })
  })

  describe('chunk boundaries', () => {
    it('tag split across chunks', () => {
      const tokens = tokenizeChunks(['<magnitude:in', 'voke tool="shell">'])
      expect(tokens).toHaveLength(1)
      expect(tokens[0]._tag).toBe('Open')
      expect(tokens[0].span.start).toEqual({ offset: 0, line: 1, col: 1 })
      expect(tokens[0].span.end).toEqual({ offset: 31, line: 1, col: 32 })
    })

    it('content then tag across chunks', () => {
      const tokens = tokenizeChunks(['hello', ' world<magnitude:invoke tool="shell">'])
      // Content should be flushed, then tag
      const content = tokens.find(t => t._tag === 'Content')
      const open = tokens.find(t => t._tag === 'Open')
      expect(content).toBeDefined()
      expect(open).toBeDefined()
      expect(open!.span.start).toEqual({ offset: 11, line: 1, col: 12 })
    })

    it('pendingLt across chunk boundary resolves to tag', () => {
      // '<' at end of first chunk, tag name at start of second
      const tokens = tokenizeChunks(['hello<', 'magnitude:invoke tool="shell">'])
      const open = tokens.find(t => t._tag === 'Open')
      expect(open).toBeDefined()
      expect(open!.span.start).toEqual({ offset: 5, line: 1, col: 6 })
    })

    it('pendingLt across chunk boundary resolves to content', () => {
      // '<' at end of first chunk, non-tag char at start of second
      const tokens = tokenizeChunks(['hello<', '3 is less'])
      expect(tokens).toHaveLength(1)
      expect(tokens[0]._tag).toBe('Content')
      expect(tokens[0].span.start).toEqual({ offset: 0, line: 1, col: 1 })
    })
  })

  describe('malformed tags', () => {
    it('unknown open tag still emits as Open with correct span', () => {
      const tokens = tokenize('<notaknowntag stuff>')
      expect(tokens).toHaveLength(1)
      expect(tokens[0]._tag).toBe('Open')
      expect(tokens[0].span.start).toEqual({ offset: 0, line: 1, col: 1 })
      expect(tokens[0].span.end).toEqual({ offset: 20, line: 1, col: 21 })
    })

    it('known tool tag in malformed mode still emits with span', () => {
      const tokens = tokenize('<magnitude:invoke tool="shell" bad attr>', new Set(['shell']))
      // Malformed but known tool tag — emitted as Open
      const open = tokens.find(t => t._tag === 'Open')
      expect(open).toBeDefined()
      expect(open!.span.start).toEqual({ offset: 0, line: 1, col: 1 })
    })
  })

  describe('EOF handling', () => {
    it('unclosed tag at EOF emits with span', () => {
      const tokens = tokenize('<magnitude:invoke tool="shell"', new Set(['shell']))
      // Should emit partial open tag at EOF
      const open = tokens.find(t => t._tag === 'Open')
      expect(open).toBeDefined()
      expect(open!.span.start).toEqual({ offset: 0, line: 1, col: 1 })
    })

    it('content at EOF gets correct span', () => {
      const tokens = tokenize('trailing content')
      expect(tokens).toHaveLength(1)
      expect(tokens[0].span.end).toEqual({ offset: 16, line: 1, col: 17 })
    })
  })
})

// =============================================================================
// Parser error span tests
// =============================================================================

describe('parser error spans', () => {
  describe('MissingRequiredField', () => {
    it('points at the invoke open tag', () => {
      const input = '<magnitude:invoke tool="shell">\n</magnitude:invoke>'
      const events = parse(input)
      const err = findToolError(events, 'MissingRequiredField')
      expect(err).toBeDefined()
      expect(err!.error.primarySpan).toBeDefined()
      expect(err!.error.primarySpan!.start.line).toBe(1)
      expect(err!.error.primarySpan!.start.col).toBe(1)
    })

    it('repeated invocations: each error points at its own invoke', () => {
      const input = [
        '<magnitude:invoke tool="shell">',
        '<magnitude:parameter name="command">ls</magnitude:parameter>',
        '</magnitude:invoke>',
        '<magnitude:invoke tool="shell">',
        '</magnitude:invoke>',
      ].join('\n')
      const events = parse(input)
      const missingErrors = events.filter((e): e is ToolParseErrorEvent =>
        e._tag === 'ToolParseError' && e.error._tag === 'MissingRequiredField'
      )
      // Only the second invocation should have MissingRequiredField (missing 'command')
      expect(missingErrors.length).toBeGreaterThanOrEqual(1)
      const err = missingErrors[0]
      // Should point at line 4 (the second invoke), not line 1
      expect(err.error.primarySpan).toBeDefined()
      expect(err.error.primarySpan!.start.line).toBe(4)
    })

    it('three identical invocations: error on third points at line of third', () => {
      const input = [
        '<magnitude:invoke tool="shell">',           // line 1
        '<magnitude:parameter name="command">a</magnitude:parameter>',
        '</magnitude:invoke>',
        '<magnitude:invoke tool="shell">',           // line 4
        '<magnitude:parameter name="command">b</magnitude:parameter>',
        '</magnitude:invoke>',
        '<magnitude:invoke tool="shell">',           // line 7
        '</magnitude:invoke>',
      ].join('\n')
      const events = parse(input)
      const missingErrors = events.filter((e): e is ToolParseErrorEvent =>
        e._tag === 'ToolParseError' && e.error._tag === 'MissingRequiredField'
      )
      expect(missingErrors.length).toBeGreaterThanOrEqual(1)
      expect(missingErrors[0].error.primarySpan!.start.line).toBe(7)
    })
  })

  describe('UnknownParameter', () => {
    it('points at the parameter open tag, with invoke as related span', () => {
      const input = [
        '<magnitude:invoke tool="shell">',
        '<magnitude:parameter name="nonexistent">val</magnitude:parameter>',
        '</magnitude:invoke>',
      ].join('\n')
      const events = parse(input)
      const err = findToolError(events, 'UnknownParameter')
      expect(err).toBeDefined()
      // primarySpan should point at the parameter tag (line 2)
      expect(err!.error.primarySpan).toBeDefined()
      expect(err!.error.primarySpan!.start.line).toBe(2)
      // relatedSpans should include the invoke open (line 1)
      expect(err!.error.relatedSpans).toBeDefined()
      expect(err!.error.relatedSpans!.length).toBeGreaterThanOrEqual(1)
      expect(err!.error.relatedSpans![0].start.line).toBe(1)
    })
  })

  describe('DuplicateParameter', () => {
    it('points at the duplicate parameter tag', () => {
      const input = [
        '<magnitude:invoke tool="shell">',
        '<magnitude:parameter name="command">first</magnitude:parameter>',
        '<magnitude:parameter name="command">second</magnitude:parameter>',
        '</magnitude:invoke>',
      ].join('\n')
      const events = parse(input)
      const err = findToolError(events, 'DuplicateParameter')
      expect(err).toBeDefined()
      // primarySpan should point at the second (duplicate) parameter tag (line 3)
      expect(err!.error.primarySpan).toBeDefined()
      expect(err!.error.primarySpan!.start.line).toBe(3)
    })
  })

  describe('IncompleteTool', () => {
    it('points at the invoke open tag when tool is never closed', () => {
      const input = '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>'
      const events = parse(input)
      const err = findToolError(events, 'IncompleteTool')
      expect(err).toBeDefined()
      expect(err!.error.primarySpan).toBeDefined()
      expect(err!.error.primarySpan!.start.line).toBe(1)
    })

    it('repeated unclosed invocations: each points at its own open', () => {
      // Two unclosed invocations — but parser handles them sequentially
      const input = '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">a</magnitude:parameter>'
      const events = parse(input)
      const incompletes = events.filter((e): e is ToolParseErrorEvent =>
        e._tag === 'ToolParseError' && e.error._tag === 'IncompleteTool'
      )
      expect(incompletes.length).toBeGreaterThanOrEqual(1)
      expect(incompletes[0].error.primarySpan!.start.line).toBe(1)
    })
  })

  describe('UnknownTool', () => {
    it('points at the invoke open tag', () => {
      const input = '<magnitude:invoke tool="nonexistent_tool">\n</magnitude:invoke>'
      const events = parse(input)
      const err = findStructuralError(events, 'UnknownTool')
      expect(err).toBeDefined()
      expect(err!.error.primarySpan).toBeDefined()
      expect(err!.error.primarySpan!.start.line).toBe(1)
    })
  })

  describe('UnclosedThink', () => {
    it('points at the reason open tag', () => {
      const input = '<magnitude:reason about="test">some thinking'
      const events = parse(input)
      const err = findStructuralError(events, 'UnclosedThink')
      expect(err).toBeDefined()
      expect(err!.error.primarySpan).toBeDefined()
      expect(err!.error.primarySpan!.start.line).toBe(1)
      expect(err!.error.primarySpan!.start.col).toBe(1)
    })

    it('reason on later line points at correct line', () => {
      const input = 'some prose\n<magnitude:reason about="test">unclosed thinking'
      const events = parse(input)
      const err = findStructuralError(events, 'UnclosedThink')
      expect(err).toBeDefined()
      expect(err!.error.primarySpan!.start.line).toBe(2)
    })
  })

  describe('StrayCloseTag', () => {
    it('points at the stray close tag', () => {
      const input = '</magnitude:invoke>'
      const events = parse(input)
      const err = findStructuralError(events, 'StrayCloseTag')
      expect(err).toBeDefined()
      expect(err!.error.primarySpan).toBeDefined()
      expect(err!.error.primarySpan!.start.line).toBe(1)
    })
  })

  describe('InvalidMagnitudeOpen', () => {
    it('points at the invalid tag', () => {
      const input = '<magnitude:invoke tool="shell">\n<magnitude:bogus>content</magnitude:bogus>\n</magnitude:invoke>'
      const events = parse(input)
      const err = findStructuralError(events, 'InvalidMagnitudeOpen')
      expect(err).toBeDefined()
      expect(err!.error.primarySpan).toBeDefined()
      expect(err!.error.primarySpan!.start.line).toBe(2)
    })
  })

  describe('MissingToolName', () => {
    it('points at the invoke tag without tool attribute', () => {
      const input = '<magnitude:invoke>\n</magnitude:invoke>'
      const events = parse(input)
      const err = findStructuralError(events, 'MissingToolName')
      expect(err).toBeDefined()
      expect(err!.error.primarySpan).toBeDefined()
      expect(err!.error.primarySpan!.start.line).toBe(1)
    })
  })
})

// =============================================================================
// End-to-end presentation tests
// =============================================================================

describe('presentation with spans', () => {
  const createTaskTool = defineTool({
    name: 'create_task',
    label: () => 'Create Task',
    group: 'tasks',
    description: 'Create a task',
    inputSchema: Schema.Struct({
      id: Schema.String,
      title: Schema.String,
      parent: Schema.optional(Schema.String),
    }),
    outputSchema: Schema.Struct({ id: Schema.String }),
    execute: (input) => Effect.succeed({ id: input.id }),
  })

  const taskTools = makeTools({ def: createTaskTool, tag: 'create_task', group: 'tasks' })

  it('repeated invocations: error snippet shows the failing invocation', () => {
    const input = [
      '<magnitude:invoke tool="create_task">',
      '<magnitude:parameter name="id">p1</magnitude:parameter>',
      '<magnitude:parameter name="title">Title 1</magnitude:parameter>',
      '</magnitude:invoke>',
      '<magnitude:invoke tool="create_task">',
      '<magnitude:parameter name="id">p2</magnitude:parameter>',
      '<magnitude:parameter name="title">Title 2</magnitude:parameter>',
      '</magnitude:invoke>',
      '<magnitude:invoke tool="create_task">',
      '<magnitude:parameter name="id">p3</magnitude:parameter>',
      '</magnitude:invoke>',
    ].join('\n')

    const events = parse(input, taskTools)
    const missingTitle = events.find((e): e is ToolParseErrorEvent =>
      e._tag === 'ToolParseError' && e.error._tag === 'MissingRequiredField' && e.error.parameterName === 'title'
    )
    expect(missingTitle).toBeDefined()

    const rendered = renderParseError(missingTitle!, input)
    // Should cite the 3rd invocation (line 9), not the 1st (line 1)
    expect(rendered).toContain('9|<magnitude:invoke tool="create_task">')
    expect(rendered).toContain('10|<magnitude:parameter name="id">p3</magnitude:parameter>')
    expect(rendered).not.toContain('\n1|<magnitude:invoke tool="create_task">')
  })

  it('unknown parameter snippet shows the parameter line, not the invoke line', () => {
    const input = [
      '<magnitude:invoke tool="shell">',
      '<magnitude:parameter name="nonexistent">val</magnitude:parameter>',
      '</magnitude:invoke>',
    ].join('\n')

    const events = parse(input)
    const unknownParam = events.find((e): e is ToolParseErrorEvent =>
      e._tag === 'ToolParseError' && e.error._tag === 'UnknownParameter'
    )
    expect(unknownParam).toBeDefined()

    const rendered = renderParseError(unknownParam!, input)
    // Should cite line 2 (the unknown parameter), not line 1 (the invoke)
    expect(rendered).toContain('2|<magnitude:parameter name="nonexistent">val</magnitude:parameter>')
  })

  it('five identical invocations: error on 5th cites line of 5th', () => {
    const lines: string[] = []
    for (let i = 1; i <= 4; i++) {
      lines.push('<magnitude:invoke tool="create_task">')
      lines.push(`<magnitude:parameter name="id">p${i}</magnitude:parameter>`)
      lines.push(`<magnitude:parameter name="title">T${i}</magnitude:parameter>`)
      lines.push('</magnitude:invoke>')
    }
    // 5th invocation missing title
    lines.push('<magnitude:invoke tool="create_task">')
    lines.push('<magnitude:parameter name="id">p5</magnitude:parameter>')
    lines.push('</magnitude:invoke>')

    const input = lines.join('\n')
    const events = parse(input, taskTools)
    const missingTitle = events.find((e): e is ToolParseErrorEvent =>
      e._tag === 'ToolParseError' && e.error._tag === 'MissingRequiredField' && e.error.parameterName === 'title'
    )
    expect(missingTitle).toBeDefined()

    const rendered = renderParseError(missingTitle!, input)
    // 5th invocation starts at line 17 (4*4 + 1)
    expect(rendered).toContain('17|<magnitude:invoke tool="create_task">')
    expect(rendered).not.toContain('\n1|<magnitude:invoke tool="create_task">')
  })

  it('structural error with span renders correct location', () => {
    const fakeEvent: StructuralParseErrorEvent = {
      _tag: 'StructuralParseError',
      error: {
        _tag: 'UnknownTool',
        tagName: 'nonexistent',
        detail: 'Unknown tool',
        primarySpan: {
          start: { offset: 0, line: 1, col: 1 },
          end: { offset: 20, line: 1, col: 21 },
        },
      },
    }
    const rendered = renderParseError(fakeEvent, 'some response text\nsecond line')
    expect(rendered).toContain('Unknown tool')
    expect(rendered).toContain('1|some response text')
  })
})
