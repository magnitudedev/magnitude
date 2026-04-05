/**
 * End-to-end tests for xml-act runtime.
 *
 * Each test feeds realistic LLM XML output through createXmlRuntime and
 * asserts on the full event stream that comes out the other end.
 */
import { describe, test, expect } from 'bun:test'
import { Effect, Stream, Layer } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool } from '@magnitudedev/tools'
import {
  createXmlRuntime,
  createStreamingXmlParser,
  initialReactorState,
  foldReactorState,
  ToolInterceptorTag,
  buildOutputTree,
  type OutputNode,
  type XmlRuntimeConfig,
  type XmlRuntimeEvent,
  type RegisteredTool,
  type XmlTagBinding,
  type ToolInterceptor,
  type ReactorState,

} from './index'

const defineToolUnsafe: any = defineTool

const ACTIONS_TAG_OPEN = '<task id="t1">'
const ACTIONS_TAG_CLOSE = '</task>'
import type { ParseEvent } from './format/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Collect all events from a runtime stream */
function runStream(
  config: XmlRuntimeConfig,
  xml: string,
  opts?: { layers?: Layer.Layer<never>; initialState?: ReactorState },
): Promise<XmlRuntimeEvent[]> {
  const runtime = createXmlRuntime(config)
  const stream = runtime.streamWith(Stream.make(xml), {
    initialState: opts?.initialState,
  })
  const collected = Stream.runCollect(stream)
  const withLayers = opts?.layers ? collected.pipe(Effect.provide(opts.layers)) : collected
  return Effect.runPromise(withLayers).then(c => Array.from(c))
}

/** Collect events with streaming — xml split into individual characters */
function runStreamCharByChar(
  config: XmlRuntimeConfig,
  xml: string,
  layers?: Layer.Layer<never>,
): Promise<XmlRuntimeEvent[]> {
  const runtime = createXmlRuntime(config)
  const stream = runtime.streamWith(Stream.fromIterable([...xml]))
  const collected = Stream.runCollect(stream)
  const withLayers = layers ? collected.pipe(Effect.provide(layers)) : collected
  return Effect.runPromise(withLayers).then(c => Array.from(c))
}

/** Collect events from a runtime stream with explicit chunk boundaries */
function runStreamChunked(
  config: XmlRuntimeConfig,
  chunks: string[],
): Promise<XmlRuntimeEvent[]> {
  const runtime = createXmlRuntime(config)
  const stream = runtime.streamWith(Stream.fromIterable(chunks))
  return Effect.runPromise(Stream.runCollect(stream)).then(c => Array.from(c))
}

/**
 * Run XML through the parser directly with explicit chunk boundaries.
 * Returns all parse events. Does NOT go through the runtime.
 */
function parseChunked(chunks: string[], knownTags: string[] = []): ParseEvent[] {
  const parser = createStreamingXmlParser(
    new Set(knownTags),
    new Map(),
  )
  const events: ParseEvent[] = []
  for (const chunk of chunks) {
    events.push(...parser.processChunk(chunk))
  }
  events.push(...parser.flush())
  return events
}

function parseEvents<T extends ParseEvent['_tag']>(events: ParseEvent[], tag: T): Extract<ParseEvent, { _tag: T }>[] {
  return events.filter(e => e._tag === tag) as Extract<ParseEvent, { _tag: T }>[]
}

/** Build a RegisteredTool from a createTool result + binding */
function registered(
  tool: any,
  tagName: string,
  binding: any,
  opts?: { groupName?: string },
): RegisteredTool {
  return {
    tool,
    tagName,
    groupName: opts?.groupName ?? 'default',
    binding,
  }
}

/** Build a config from registered tools */
function config(
  tools: RegisteredTool[],
): XmlRuntimeConfig {
  return {
    tools: new Map(tools.map(t => [t.tagName, t])),
  }
}

/** Filter events to a specific tag */
function eventsOfType<T extends XmlRuntimeEvent['_tag']>(
  events: XmlRuntimeEvent[],
  tag: T,
): Extract<XmlRuntimeEvent, { _tag: T }>[] {
  return events.filter(e => e._tag === tag) as Extract<XmlRuntimeEvent, { _tag: T }>[]
}

// ---------------------------------------------------------------------------
// Mock tools
// ---------------------------------------------------------------------------

const readTool = defineToolUnsafe({
  name: 'read',
  description: 'Read a file',
  inputSchema: Schema.Struct({
    path: Schema.String,
  }),
  outputSchema: Schema.Struct({
    content: Schema.String,
    lines: Schema.Number,
  }),
  bindings: {
    xmlInput: { type: 'tag' },
    xmlOutput: { type: 'tag', childTags: [{ field: 'content', tag: 'content' }, { field: 'lines', tag: 'lines' }] },
  } as const,
  execute: ({ path }: any) => Effect.succeed({ content: `contents of ${path}`, lines: 42 }),
})

const writeTool = defineToolUnsafe({
  name: 'write',
  description: 'Write a file',
  inputSchema: Schema.Struct({
    path: Schema.String,
    content: Schema.String,
  }),
  outputSchema: Schema.String,
  bindings: {
    xmlInput: { type: 'tag' },
    xmlOutput: { type: 'tag' },
  } as const,
  execute: ({ path }: any) => Effect.succeed(`wrote ${path}`),
})

const shellTool = defineToolUnsafe({
  name: 'shell',
  description: 'Run a shell command',
  inputSchema: Schema.Struct({
    command: Schema.String,
  }),
  outputSchema: Schema.Struct({
    stdout: Schema.String,
    exitCode: Schema.Number,
  }),
  bindings: {
    xmlInput: { type: 'tag' },
    xmlOutput: { type: 'tag', childTags: [{ field: 'stdout', tag: 'stdout' }, { field: 'exitCode', tag: 'exitCode' }] },
  } as const,
  execute: ({ command }: any) => Effect.succeed({ stdout: `output of: ${command}`, exitCode: 0 }),
})

const editTool = defineToolUnsafe({
  name: 'edit',
  description: 'Edit a file',
  inputSchema: Schema.Struct({
    path: Schema.String,
    edits: Schema.Array(Schema.Struct({
      old: Schema.String,
      new: Schema.String,
    })),
  }),
  outputSchema: Schema.String,
  bindings: {
    xmlInput: { type: 'tag' },
    xmlOutput: { type: 'tag' },
  } as const,
  execute: ({ path, edits }: any) => Effect.succeed(`edited ${path}: ${edits.length} changes`),
})

const addTool = defineToolUnsafe({
  name: 'add',
  description: 'Add two numbers',
  inputSchema: Schema.Struct({
    a: Schema.Number,
    b: Schema.Number,
  }),
  outputSchema: Schema.Number,
  bindings: {
    xmlInput: { type: 'tag' },
    xmlOutput: { type: 'tag' },
  } as const,
  execute: ({ a, b }: any) => Effect.succeed(a + b),
})

const failingTool = defineToolUnsafe({
  name: 'fail',
  description: 'Always fails',
  inputSchema: Schema.Struct({
    reason: Schema.String,
  }),
  outputSchema: Schema.String,
  errorSchema: Schema.String,
  bindings: {
    xmlInput: { type: 'tag' },
    xmlOutput: { type: 'tag' },
  } as const,
  execute: ({ reason }: any) => Effect.fail(reason),
})

const kvTool = defineToolUnsafe({
  name: 'set_env',
  description: 'Set env vars',
  inputSchema: Schema.Struct({
    vars: Schema.Record({ key: Schema.String, value: Schema.String }),
  }),
  outputSchema: Schema.String,
  bindings: {
    xmlInput: { type: 'tag' },
    xmlOutput: { type: 'tag' },
  } as const,
  execute: ({ vars }: any) => Effect.succeed(`set ${Object.keys(vars).length} vars`),
})

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

const readBinding: any = { attributes: [{ field: 'path', attr: 'path' }] }
const writeBinding: any = { attributes: [{ field: 'path', attr: 'path' }], body: 'content' }
const shellBinding: any = { body: 'command' }
const editBinding: any = {
  attributes: [{ field: 'path', attr: 'path' }],
  children: [{ field: 'edits', tag: 'edit', attributes: [{ field: 'old', attr: 'old' }], body: 'new' }],
}
const addBinding: any = {
  attributes: [{ field: 'a', attr: 'a' }, { field: 'b', attr: 'b' }],
  selfClosing: true,
}
const failBinding: any = { attributes: [{ field: 'reason', attr: 'reason' }] }
const kvBinding: any = {
  childRecord: { field: 'vars', tag: 'var', keyAttr: 'name' },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('xml-act end-to-end', () => {

  // =========================================================================
  // Basic tool execution
  // =========================================================================

  test('self-closing tool with attributes', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<read id="r1" path="src/index.ts"/>${ACTIONS_TAG_CLOSE}`)

    const started = eventsOfType(events, 'ToolInputStarted')
    expect(started).toHaveLength(1)
    expect(started[0].toolName).toBe('read')
    expect(started[0].tagName).toBe('read')

    const fieldValues = eventsOfType(events, 'ToolInputFieldValue')
    expect(fieldValues).toHaveLength(1)
    expect(fieldValues[0].field).toBe('path')
    expect(fieldValues[0].value).toBe('src/index.ts')

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    expect(ready[0].input).toEqual({ path: 'src/index.ts' })

    const execStarted = eventsOfType(events, 'ToolExecutionStarted')
    expect(execStarted).toHaveLength(1)

    const execEnded = eventsOfType(events, 'ToolExecutionEnded')
    expect(execEnded).toHaveLength(1)
    expect(execEnded[0].result._tag).toBe('Success')
    if (execEnded[0].result._tag === 'Success') {
      expect(execEnded[0].result.output).toEqual({ content: 'contents of src/index.ts', lines: 42 })
    }

    const end = eventsOfType(events, 'TurnEnd')
    expect(end).toHaveLength(1)
    expect(end[0].result._tag).toBe('Success')
  })

  test('tool with body content', async () => {
    const cfg = config([registered(writeTool, 'write', writeBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<write id="r1" path="out.txt">hello world</write>${ACTIONS_TAG_CLOSE}`)

    const bodyChunks = eventsOfType(events, 'ToolInputBodyChunk')
    expect(bodyChunks.length).toBeGreaterThan(0)
    expect(bodyChunks[0].path).toEqual(['content'])

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready[0].input).toEqual({ path: 'out.txt', content: 'hello world' })

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended[0].result._tag).toBe('Success')
    if (ended[0].result._tag === 'Success') {
      expect(ended[0].result.output).toBe('wrote out.txt')
    }
  })

  test('tool with dotted-path attribute and body bindings', async () => {
    const nestedTool = defineToolUnsafe({
      name: 'nested',
      description: 'Nested binding test',
      inputSchema: Schema.Struct({
        options: Schema.Struct({
          type: Schema.String,
          message: Schema.String,
        }),
      }),
      outputSchema: Schema.Struct({
        type: Schema.String,
        message: Schema.String,
      }),
      bindings: {
        xmlInput: { type: 'tag' },
        xmlOutput: { type: 'tag', childTags: [{ field: 'type', tag: 'type' }, { field: 'message', tag: 'message' }] },
      } as const,
      execute: ({ options }: any) => Effect.succeed({ type: options.type, message: options.message }),
    })

    const nestedBinding: any = {
      attributes: [{ field: 'options.type', attr: 'type' }],
      body: 'options.message',
    }

    const cfg = config([registered(nestedTool, 'nested', nestedBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<nested id="r1" type="planner">build a plan</nested>${ACTIONS_TAG_CLOSE}`)

    const fieldValues = eventsOfType(events, 'ToolInputFieldValue')
    expect(fieldValues).toHaveLength(1)
    expect(fieldValues[0].field).toBe('options.type')
    expect(fieldValues[0].value).toBe('planner')

    const bodyChunks = eventsOfType(events, 'ToolInputBodyChunk')
    expect(bodyChunks.length).toBeGreaterThan(0)
    expect(bodyChunks[0].path).toEqual(['options.message'])

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready[0].input).toEqual({ options: { type: 'planner', message: 'build a plan' } })
  })

  test('tool with body-only binding (no attributes)', async () => {
    const cfg = config([registered(shellTool, 'shell', shellBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<shell id="r1">ls -la</shell>${ACTIONS_TAG_CLOSE}`)

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready[0].input).toEqual({ command: 'ls -la' })

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended[0].result._tag).toBe('Success')
    if (ended[0].result._tag === 'Success') {
      expect(ended[0].result.output).toEqual({ stdout: 'output of: ls -la', exitCode: 0 })
    }
  })

  // =========================================================================
  // Multiple sequential tools
  // =========================================================================

  test('multiple tools in sequence', async () => {
    const cfg = config([
      registered(readTool, 'read', readBinding),
      registered(writeTool, 'write', writeBinding),
    ])

    const xml = `${ACTIONS_TAG_OPEN}<read id="r1" path="a.ts"/><write id="r2" path="b.ts">new content</write>${ACTIONS_TAG_CLOSE}`
    const events = await runStream(cfg, xml)

    const execEnded = eventsOfType(events, 'ToolExecutionEnded')
    expect(execEnded).toHaveLength(2)
    expect(execEnded[0].toolName).toBe('read')
    expect(execEnded[1].toolName).toBe('write')
  })

  // =========================================================================
  // Child elements (array fields)
  // =========================================================================

  test('tool with child elements', async () => {
    const cfg = config([registered(editTool, 'edit', editBinding)])

    const xml = `${ACTIONS_TAG_OPEN}<edit id="r1" path="foo.ts">
  <edit old="const x = 1">const x = 2</edit>
  <edit old="const y = 3">const y = 4</edit>
</edit>${ACTIONS_TAG_CLOSE}`

    const events = await runStream(cfg, xml)

    const childStarted = eventsOfType(events, 'ToolInputChildStarted')
    expect(childStarted.length).toBeGreaterThan(0)
    expect(childStarted[0].field).toBe('edits')
    expect(childStarted[0].index).toBe(0)
    expect(childStarted[0].attributes).toEqual({ old: 'const x = 1' })

    const childComplete = eventsOfType(events, 'ToolInputChildComplete')
    expect(childComplete.length).toBeGreaterThan(0)
    expect(childComplete[0].value).toEqual({ old: 'const x = 1', new: 'const x = 2' })

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready[0].input).toEqual({
      path: 'foo.ts',
      edits: [
        { old: 'const x = 1', new: 'const x = 2' },
        { old: 'const y = 3', new: 'const y = 4' },
      ],
    })

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended[0].result._tag).toBe('Success')
  })

  // =========================================================================
  // Numeric coercion from attributes
  // =========================================================================

  test('coerces number attributes from strings', async () => {
    const cfg = config([registered(addTool, 'add', addBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<add id="r1" a="3" b="7"/>${ACTIONS_TAG_CLOSE}`)

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready[0].input).toEqual({ a: 3, b: 7 })

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended[0].result._tag).toBe('Success')
    if (ended[0].result._tag === 'Success') {
      expect(ended[0].result.output).toBe(10)
    }
  })

  // =========================================================================
  // Prose handling
  // =========================================================================

  test('bare text becomes message events for user prose', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])
    const events = await runStream(cfg, `Let me read the file.\n${ACTIONS_TAG_OPEN}<read id="r1" path="x.ts"/>${ACTIONS_TAG_CLOSE}`)

    const starts = eventsOfType(events, 'MessageStart')
    expect(starts.length).toBeGreaterThan(0)
    expect(starts[0]).toBeDefined()
    expect(starts[0]?.to).toBeNull()

    const chunks = eventsOfType(events, 'MessageChunk')
    expect(chunks.length).toBeGreaterThan(0)

    const ends = eventsOfType(events, 'MessageEnd')
    expect(ends.length).toBeGreaterThanOrEqual(1)
  })

  test('think tags emit prose events', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])
    const events = await runStream(cfg, `<think>analyzing...\n</think>\n${ACTIONS_TAG_OPEN}\n<read id="r1" path="x.ts"/>\n${ACTIONS_TAG_CLOSE}`)

    const proseChunks = eventsOfType(events, 'ProseChunk')
    const thinkChunks = proseChunks.filter(p => p.patternId === 'think')
    expect(thinkChunks.length).toBeGreaterThan(0)

    const proseEnds = eventsOfType(events, 'ProseEnd')
    const thinkEnds = proseEnds.filter(p => p.patternId === 'think')
    expect(thinkEnds).toHaveLength(1)
    expect(thinkEnds[0].content).toBe('analyzing...\n')
  })

  // =========================================================================
  // Streaming — chunk boundaries don't matter
  // =========================================================================

  test('char-by-char streaming produces same tool results as single chunk', async () => {
    const cfg = config([registered(writeTool, 'write', writeBinding)])
    const xml = `${ACTIONS_TAG_OPEN}<write id="r1" path="test.txt">some content here</write>${ACTIONS_TAG_CLOSE}`

    const singleChunk = await runStream(cfg, xml)
    const charByChar = await runStreamCharByChar(cfg, xml)

    const singleEnded = eventsOfType(singleChunk, 'ToolExecutionEnded')
    const charEnded = eventsOfType(charByChar, 'ToolExecutionEnded')
    expect(singleEnded).toHaveLength(1)
    expect(charEnded).toHaveLength(1)
    expect(singleEnded[0].result).toEqual(charEnded[0].result)

    const singleReady = eventsOfType(singleChunk, 'ToolInputReady')
    const charReady = eventsOfType(charByChar, 'ToolInputReady')
    expect(singleReady[0].input).toEqual(charReady[0].input)
  })

  // =========================================================================
  // Tool execution errors
  // =========================================================================

  test('tool that fails returns Error result', async () => {
    const cfg = config([registered(failingTool, 'fail', failBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<fail id="r1" reason="something broke"/>${ACTIONS_TAG_CLOSE}`)

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended).toHaveLength(1)
    expect(ended[0].result._tag).toBe('Error')
    if (ended[0].result._tag === 'Error') {
      expect(ended[0].result.error).toBe('something broke')
    }

    const end = eventsOfType(events, 'TurnEnd')
    expect(end[0].result._tag).toBe('Success')
  })

  test('missing required field produces ToolInputParseError', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<read id="r1"/>${ACTIONS_TAG_CLOSE}`)

    const execStarted = eventsOfType(events, 'ToolExecutionStarted')
    expect(execStarted).toHaveLength(0)
    const execEnded = eventsOfType(events, 'ToolExecutionEnded')
    expect(execEnded).toHaveLength(0)

    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('MissingRequiredFields')
    if (parseErrors[0].error._tag === 'MissingRequiredFields') {
      expect(parseErrors[0].error.fields).toContain('path')
      expect(parseErrors[0].error.detail).toContain('path')
      expect(parseErrors[0].error.detail).toContain('read')
    }

    const end = eventsOfType(events, 'TurnEnd')
    expect(end[0].result._tag).toBe('Success')
  })

  // =========================================================================
  // Interceptor (permission gating)
  // =========================================================================

  test('interceptor can reject a tool call', async () => {
    const cfg = config([registered(shellTool, 'shell', shellBinding)])

    const interceptor: ToolInterceptor = {
      beforeExecute: (ctx) => Effect.succeed({
        _tag: 'Reject' as const,
        rejection: { reason: 'not allowed', command: ctx.input },
      }),
    }
    const layer = Layer.succeed(ToolInterceptorTag, interceptor)

    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<shell id="r1">rm -rf /</shell>${ACTIONS_TAG_CLOSE}`, { layers: layer })

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended).toHaveLength(1)
    expect(ended[0].result._tag).toBe('Rejected')

    const end = eventsOfType(events, 'TurnEnd')
    expect(end).toHaveLength(1)
    expect(end[0].result._tag).toBe('GateRejected')
  })

  test('interceptor can modify input', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])

    const interceptor: ToolInterceptor = {
      beforeExecute: () => Effect.succeed({
        _tag: 'Proceed' as const,
        modifiedInput: { path: '/overridden/path.ts' },
      }),
    }
    const layer = Layer.succeed(ToolInterceptorTag, interceptor)

    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<read id="r1" path="original.ts"/>${ACTIONS_TAG_CLOSE}`, { layers: layer })

    const execStarted = eventsOfType(events, 'ToolExecutionStarted')
    expect(execStarted[0].input).toEqual({ path: '/overridden/path.ts' })

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended[0].result._tag).toBe('Success')
    if (ended[0].result._tag === 'Success') {
      expect(ended[0].result.output).toEqual({
        content: 'contents of /overridden/path.ts',
        lines: 42,
      })
    }
  })

  test('afterExecute interceptor can reject', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])

    const interceptor: ToolInterceptor = {
      beforeExecute: () => Effect.succeed({ _tag: 'Proceed' as const }),
      afterExecute: () => Effect.succeed({
        _tag: 'Reject' as const,
        rejection: 'post-execution rejection',
      }),
    }
    const layer = Layer.succeed(ToolInterceptorTag, interceptor)

    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<read id="r1" path="secret.ts"/>${ACTIONS_TAG_CLOSE}`, { layers: layer })

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended[0].result._tag).toBe('Rejected')
    if (ended[0].result._tag === 'Rejected') {
      expect(ended[0].result.rejection).toBe('post-execution rejection')
    }
  })

  // =========================================================================
  // Actions wrapper
  // =========================================================================

  test('tools inside <task id="t1"> block execute normally', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])
    const xml = `${ACTIONS_TAG_OPEN}<read id="r1" path="a.ts"/><read id="r2" path="b.ts"/>${ACTIONS_TAG_CLOSE}`
    const events = await runStream(cfg, xml)

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended).toHaveLength(2)
    expect(ended[0].result._tag).toBe('Success')
    expect(ended[1].result._tag).toBe('Success')
  })

  // =========================================================================
  // Prose before/between/after tools
  // =========================================================================

  test('prose interleaved with tools emits messages', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])

    const xml = `I'll check the file.\n${ACTIONS_TAG_OPEN}<read id="r1" path="a.ts"/>${ACTIONS_TAG_CLOSE}\nLooks good.`
    const events = await runStream(cfg, xml)

    const messageEnds = eventsOfType(events, 'MessageEnd')
    expect(messageEnds.length).toBeGreaterThanOrEqual(1)

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended).toHaveLength(1)
  })

  // =========================================================================
  // childRecord binding
  // =========================================================================

  test('childRecord binding maps keyed children to record', async () => {
    const cfg = config([registered(kvTool, 'set_env', kvBinding)])

    const xml = `${ACTIONS_TAG_OPEN}<set_env id="r1">
  <var name="FOO">bar</var>
  <var name="BAZ">qux</var>
</set_env>${ACTIONS_TAG_CLOSE}`

    const events = await runStream(cfg, xml)

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    expect(ready[0].input).toEqual({
      vars: { FOO: 'bar', BAZ: 'qux' },
    })
  })

  // =========================================================================
  // Unknown tags treated as prose
  // =========================================================================

  test('unknown tags are reconstructed as user messages', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])
    const xml = `<unknown attr="val">body</unknown>\n${ACTIONS_TAG_OPEN}<read id="r1" path="x.ts"/>${ACTIONS_TAG_CLOSE}`
    const events = await runStream(cfg, xml)

    const toolStarts = eventsOfType(events, 'ToolInputStarted')
    expect(toolStarts).toHaveLength(1)
    expect(toolStarts[0].toolName).toBe('read')

    const messageEnds = eventsOfType(events, 'MessageEnd')
    expect(messageEnds.length).toBeGreaterThan(0)
  })

  // =========================================================================
  // Code fences stripped from prose
  // =========================================================================

  test('code fences in prose are stripped', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])
    const xml = `\`\`\`xml\n${ACTIONS_TAG_OPEN}<read id="r1" path="a.ts"/>${ACTIONS_TAG_CLOSE}\n\`\`\``
    const events = await runStream(cfg, xml)

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended).toHaveLength(1)
    expect(ended[0].result._tag).toBe('Success')
  })

  // =========================================================================
  // Rejection stops further tools from executing
  // =========================================================================

  test('rejection stops all subsequent tool execution', async () => {
    const cfg = config([
      registered(shellTool, 'shell', shellBinding),
      registered(readTool, 'read', readBinding),
    ])

    const interceptor: ToolInterceptor = {
      beforeExecute: (ctx) => {
        if (ctx.toolName === 'shell') {
          return Effect.succeed({ _tag: 'Reject' as const, rejection: 'denied' })
        }
        return Effect.succeed({ _tag: 'Proceed' as const })
      },
    }
    const layer = Layer.succeed(ToolInterceptorTag, interceptor)

    const xml = `${ACTIONS_TAG_OPEN}<shell id="r1">danger</shell><read id="r2" path="safe.ts"/>${ACTIONS_TAG_CLOSE}`
    const events = await runStream(cfg, xml, { layers: layer })

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended).toHaveLength(1)
    expect(ended[0].toolName).toBe('shell')
    expect(ended[0].result._tag).toBe('Rejected')

    const starts = eventsOfType(events, 'ToolInputStarted')
    expect(starts).toHaveLength(1)

    const end = eventsOfType(events, 'TurnEnd')
    expect(end[0].result._tag).toBe('GateRejected')
  })

  // =========================================================================
  // Empty stream
  // =========================================================================

  test('empty stream produces only TurnEnd', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])
    const events = await runStream(cfg, '')

    expect(events).toHaveLength(1)
    expect(events[0]._tag).toBe('TurnEnd')
    expect((events[0] as Extract<XmlRuntimeEvent, { _tag: 'TurnEnd' }>).result._tag).toBe('Success')
  })

  // =========================================================================
  // Multiline body content
  // =========================================================================

  test('multiline body content preserved correctly', async () => {
    const cfg = config([registered(writeTool, 'write', writeBinding)])

    const xml = `${ACTIONS_TAG_OPEN}<write id="r1" path="script.sh">#!/bin/bash
echo "hello"
echo "world"</write>${ACTIONS_TAG_CLOSE}`

    const events = await runStream(cfg, xml)

    const ready = eventsOfType(events, 'ToolInputReady')
    const input = ready[0].input as { path: string; content: string }
    expect(input.content).toContain('#!/bin/bash')
    expect(input.content).toContain('echo "hello"')
    expect(input.content).toContain('echo "world"')
  })

  // =========================================================================
  // think/thinking variants
  // =========================================================================

  test('thinking tag works same as think', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])
    const events = await runStream(cfg, `<thinking>hmm...\n</thinking>\n${ACTIONS_TAG_OPEN}\n<read id="r1" path="x.ts"/>\n${ACTIONS_TAG_CLOSE}`)

    const proseEnds = eventsOfType(events, 'ProseEnd')
    const thinkEnds = proseEnds.filter(p => p.patternId === 'think')
    expect(thinkEnds).toHaveLength(1)
    expect(thinkEnds[0].content).toBe('hmm...\n')
  })

  // =========================================================================
  // Mixed realistic scenario
  // =========================================================================

  test('realistic agent turn: think, read, edit, write', async () => {
    const cfg = config([
      registered(readTool, 'read', readBinding),
      registered(editTool, 'edit', editBinding),
      registered(writeTool, 'write', writeBinding),
    ])

    const xml = `<think>I need to fix the bug in utils.ts by changing the return type.
</think>
${ACTIONS_TAG_OPEN}
<read id="r1" path="src/utils.ts"/>
<edit id="r2" path="src/utils.ts">
  <edit old="return null">return undefined</edit>
</edit>
<write id="r3" path="src/utils.test.ts">test('returns undefined', () => {
  expect(fn()).toBeUndefined()
})</write>
${ACTIONS_TAG_CLOSE}`

    const events = await runStream(cfg, xml)

    const thinkEnds = eventsOfType(events, 'ProseEnd').filter(p => p.patternId === 'think')
    expect(thinkEnds).toHaveLength(1)
    expect(thinkEnds[0].content).toContain('fix the bug')

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended).toHaveLength(3)
    expect(ended[0].toolName).toBe('read')
    expect(ended[1].toolName).toBe('edit')
    expect(ended[2].toolName).toBe('write')
    expect(ended.every(e => e.result._tag === 'Success')).toBe(true)

    const end = eventsOfType(events, 'TurnEnd')
    expect(end[0].result._tag).toBe('Success')
  })

  // =========================================================================
  // Inline coercion
  // =========================================================================

  test('ToolInputFieldValue carries coerced number values', async () => {
    const cfg = config([registered(addTool, 'add', addBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<add id="r1" a="3" b="7"/>${ACTIONS_TAG_CLOSE}`)

    const fieldValues = eventsOfType(events, 'ToolInputFieldValue')
    const aField = fieldValues.find(f => f.field === 'a')
    const bField = fieldValues.find(f => f.field === 'b')
    expect(aField?.value).toBe(3)
    expect(typeof aField?.value).toBe('number')
    expect(bField?.value).toBe(7)
    expect(typeof bField?.value).toBe('number')
  })

  test('unquoted number attributes coerce correctly', async () => {
    const cfg = config([registered(addTool, 'add', addBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<add id="r1" a=3 b=7/>${ACTIONS_TAG_CLOSE}`)

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready[0].input).toEqual({ a: 3, b: 7 })

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended[0].result._tag).toBe('Success')
  })

  // =========================================================================
  // Boolean coercion
  // =========================================================================

  test('boolean attribute coercion — various representations', async () => {
    const boolTool = defineToolUnsafe({
      name: 'flag',
      description: 'Tool with boolean',
      inputSchema: Schema.Struct({ enabled: Schema.Boolean }),
      outputSchema: Schema.Boolean,
      bindings: { xmlInput: { type: 'tag' }, xmlOutput: { type: 'tag' } } as const,
      execute: ({ enabled }: any) => Effect.succeed(enabled),
    })
    const boolBinding: any = { attributes: [{ field: 'enabled', attr: 'enabled' }], selfClosing: true }

    for (const [input, expected] of [
      ['true', true], ['false', false],
      ['True', true], ['False', false],
      ['TRUE', true], ['FALSE', false],
      ['1', true], ['0', false],
      ['yes', true], ['no', false],
      ['Yes', true], ['No', false],
      ['YES', true], ['NO', false],
    ] as const) {
      const cfg = config([registered(boolTool, 'flag', boolBinding)])
      const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<flag id="r1" enabled="${input}"/>${ACTIONS_TAG_CLOSE}`)
      const ended = eventsOfType(events, 'ToolExecutionEnded')
      expect(ended).toHaveLength(1)
      expect(ended[0].result._tag).toBe('Success')
      if (ended[0].result._tag === 'Success') {
        expect(ended[0].result.output).toBe(expected)
      }
    }
  })

  // =========================================================================
  // Inline validation errors (ToolInputParseError)
  // =========================================================================

  test('unknown attribute produces ToolInputParseError', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<read id="r1" path="x.ts" verbose="true"/>${ACTIONS_TAG_CLOSE}`)

    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('UnknownAttribute')
    if (parseErrors[0].error._tag === 'UnknownAttribute') {
      expect(parseErrors[0].error.attribute).toBe('verbose')
    }

    const execEnded = eventsOfType(events, 'ToolExecutionEnded')
    expect(execEnded).toHaveLength(0)
  })

  test('invalid attribute value produces ToolInputParseError', async () => {
    const cfg = config([registered(addTool, 'add', addBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<add id="r1" a="abc" b="7"/>${ACTIONS_TAG_CLOSE}`)

    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('InvalidAttributeValue')
    if (parseErrors[0].error._tag === 'InvalidAttributeValue') {
      expect(parseErrors[0].error.attribute).toBe('a')
      expect(parseErrors[0].error.expected).toBe('number')
      expect(parseErrors[0].error.received).toBe('abc')
    }

    const execEnded = eventsOfType(events, 'ToolExecutionEnded')
    expect(execEnded).toHaveLength(0)
  })

  test('unexpected body on bodyless tool produces ToolInputParseError', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<read id="r1" path="x.ts">some body</read>${ACTIONS_TAG_CLOSE}`)

    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('UnexpectedBody')

    const execEnded = eventsOfType(events, 'ToolExecutionEnded')
    expect(execEnded).toHaveLength(0)

    const started = eventsOfType(events, 'ToolInputStarted')
    expect(started).toHaveLength(1)
  })

  test('invalid boolean value produces ToolInputParseError', async () => {
    const boolTool = defineToolUnsafe({
      name: 'flag',
      description: 'Tool with boolean',
      inputSchema: Schema.Struct({ enabled: Schema.Boolean }),
      outputSchema: Schema.Boolean,
      bindings: { xmlInput: { type: 'tag' }, xmlOutput: { type: 'tag' } } as const,
      execute: ({ enabled }: any) => Effect.succeed(enabled),
    })
    const boolBinding: any = { attributes: [{ field: 'enabled', attr: 'enabled' }], selfClosing: true }
    const cfg = config([registered(boolTool, 'flag', boolBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<flag id="r1" enabled="maybe"/>${ACTIONS_TAG_CLOSE}`)

    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('InvalidAttributeValue')
  })

  // =========================================================================
  // Dead tool call suppression
  // =========================================================================

  test('dead tool call suppresses further events', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<read id="r1" path="x.ts" bad="yes">body text</read>${ACTIONS_TAG_CLOSE}`)

    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)

    const bodyChunks = eventsOfType(events, 'ToolInputBodyChunk')
    expect(bodyChunks).toHaveLength(0)

    const execEnded = eventsOfType(events, 'ToolExecutionEnded')
    expect(execEnded).toHaveLength(0)
  })

  // =========================================================================
  // Pairing guarantee
  // =========================================================================

  test('every ToolInputStarted has exactly one terminal event', async () => {
    const cfg = config([
      registered(readTool, 'read', readBinding),
      registered(addTool, 'add', addBinding),
    ])

    const xml = `${ACTIONS_TAG_OPEN}<read id="r1" path="a.ts"/><read id="r2"/><add id="r3" a="abc" b="1"/>${ACTIONS_TAG_CLOSE}`
    const events = await runStream(cfg, xml)

    const started = eventsOfType(events, 'ToolInputStarted')
    expect(started).toHaveLength(3)

    for (const start of started) {
      const execEnded = events.filter(
        e => e._tag === 'ToolExecutionEnded' && e.toolCallId === start.toolCallId
      )
      const parseError = events.filter(
        e => e._tag === 'ToolInputParseError' && e.toolCallId === start.toolCallId
      )
      expect(execEnded.length + parseError.length).toBe(1)
    }
  })

  // =========================================================================
  // Binding validation at registration time
  // =========================================================================

  test('binding validation rejects array field as attribute', () => {
    const badTool = defineToolUnsafe({
      name: 'bad',
      description: 'Bad binding',
      inputSchema: Schema.Struct({
        items: Schema.Array(Schema.String),
      }),
      outputSchema: Schema.String,
      bindings: { xmlInput: { type: 'tag' }, xmlOutput: { type: 'tag' } } as const,
      execute: () => Effect.succeed('ok'),
    })
    const badBinding: any = { attributes: [{ field: 'items', attr: 'items' }] }

    expect(() => {
      createXmlRuntime(config([registered(badTool, 'bad', badBinding)]))
    }).toThrow(/attributes must be scalar/)
  })

  test('binding validation rejects non-string body field', () => {
    const badTool = defineToolUnsafe({
      name: 'bad',
      description: 'Bad binding',
      inputSchema: Schema.Struct({
        count: Schema.Number,
      }),
      outputSchema: Schema.String,
      bindings: { xmlInput: { type: 'tag' }, xmlOutput: { type: 'tag' } } as const,
      execute: () => Effect.succeed('ok'),
    })
    const badBinding: any = { body: 'count' }

    expect(() => {
      createXmlRuntime(config([registered(badTool, 'bad', badBinding)]))
    }).toThrow(/body must be a string/)
  })

  test('binding validation rejects nonexistent attribute field', () => {
    const badTool = defineToolUnsafe({
      name: 'bad',
      description: 'Bad binding',
      inputSchema: Schema.Struct({
        path: Schema.String,
      }),
      outputSchema: Schema.String,
      bindings: { xmlInput: { type: 'tag' }, xmlOutput: { type: 'tag' } } as const,
      execute: () => Effect.succeed('ok'),
    })
    const badBinding: any = {
      attributes: [{ field: 'path', attr: 'path' }, { field: 'nonexistent', attr: 'nonexistent' }],
    }

    expect(() => {
      createXmlRuntime(config([registered(badTool, 'bad', badBinding)]))
    }).toThrow(/does not exist in the schema/)
  })

  // =========================================================================
  // Incomplete tag on flush
  // =========================================================================

  test('incomplete tag emits ToolInputParseError on flush', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])
    const events = await runStream(cfg, `${ACTIONS_TAG_OPEN}<read id="r1" path="x.ts"`)

    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)

    const execEnded = eventsOfType(events, 'ToolExecutionEnded')
    expect(execEnded).toHaveLength(0)
  })
})

// =============================================================================
// Unknown tag immediate prose emission
// =============================================================================

describe('unknown tag prose emission', () => {
  const cfg = config([registered(readTool, 'read', readBinding)])
  const messageText = (events: any[]) => eventsOfType(events, 'MessageChunk').map(c => c.text).join('')

  test('unknown tag emitted as prose immediately, not buffered', async () => {
    const events = await runStream(cfg, '<unknown>body text here</unknown>')
    const prose = eventsOfType(events, 'MessageEnd')
    expect(prose.length).toBeGreaterThanOrEqual(1)
    const text = messageText(events)
    expect(text).toContain('<unknown>')
    expect(text).toContain('body text here')
    expect(text).toContain('</unknown>')
    expect(eventsOfType(events, 'ToolInputStarted')).toHaveLength(0)
  })

  test('unknown tag with attributes preserves raw characters', async () => {
    const events = await runStream(cfg, '<foo  bar="baz"  qux=123>content</foo>')
    const prose = eventsOfType(events, 'MessageEnd')
    expect(prose.length).toBeGreaterThanOrEqual(1)
    const text = messageText(events)
    expect(text).toContain('<foo  bar="baz"  qux=123>')
    expect(text).toContain('content')
    expect(text).toContain('</foo>')
  })

  test('unknown self-closing tag emitted as prose with raw characters', async () => {
    const events = await runStream(cfg, '<unknown  attr="val" />')
    const prose = eventsOfType(events, 'MessageEnd')
    expect(prose.length).toBeGreaterThanOrEqual(1)
    expect(messageText(events)).toContain('<unknown  attr="val" />')
  })

  test('unknown tag between known tool calls does not interfere', async () => {
    const xml = `${ACTIONS_TAG_OPEN}<read id="r1" path="a.ts"/><read id="r2" path="b.ts"/>${ACTIONS_TAG_CLOSE}`
    const events = await runStream(cfg, xml)

    const execEnded = eventsOfType(events, 'ToolExecutionEnded')
    expect(execEnded).toHaveLength(2)
    expect(execEnded[0].result._tag).toBe('Success')
    expect(execEnded[1].result._tag).toBe('Success')
  })

  test('unknown tag body streams as message char-by-char', async () => {
    const events = await runStreamCharByChar(cfg, '<unk>body</unk>')
    const chunks = eventsOfType(events, 'MessageChunk')
    expect(chunks.length).toBeGreaterThan(0)
    const prose = eventsOfType(events, 'MessageEnd')
    expect(prose).toHaveLength(1)
  })

  test('unknown tag with long body does not buffer', async () => {
    const longBody = 'x'.repeat(1000)
    const events = await runStream(cfg, `<unk>${longBody}</unk>`)
    const prose = eventsOfType(events, 'MessageEnd')
    expect(prose.length).toBeGreaterThanOrEqual(1)
    expect(messageText(events)).toContain(longBody)
    expect(eventsOfType(events, 'ToolInputStarted')).toHaveLength(0)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
  })

  test('orphan close tag emitted as prose', async () => {
    const events = await runStream(cfg, 'text before</orphan>text after')
    const prose = eventsOfType(events, 'MessageEnd')
    expect(prose.length).toBeGreaterThanOrEqual(1)
    expect(messageText(events)).toContain('</orphan>')
  })

  test('unknown tag with nested tags inside body', async () => {
    const xml = '<unknown>body with <nested>inner</nested> tags</unknown>'
    const events = await runStream(cfg, xml)
    const prose = eventsOfType(events, 'MessageEnd')
    expect(prose.length).toBeGreaterThanOrEqual(1)
    const text = messageText(events)
    expect(text).toContain('<unknown>')
    expect(text).toContain('<nested>')
    expect(text).toContain('inner')
    expect(text).toContain('</nested>')
    expect(text).toContain('</unknown>')
  })

  test('unknown tag incomplete on flush emits as prose', async () => {
    const events = await runStream(cfg, '<unknown>partial body')
    const prose = eventsOfType(events, 'MessageEnd')
    expect(prose.length).toBeGreaterThanOrEqual(1)
    const text = messageText(events)
    expect(text).toContain('<unknown>')
    expect(text).toContain('partial body')
    expect(eventsOfType(events, 'ToolInputParseError')).toHaveLength(0)
  })

  test('prose before and after unknown tag preserved', async () => {
    const events = await runStream(cfg, 'hello <unknown>x</unknown> world')
    const prose = eventsOfType(events, 'MessageEnd')
    expect(prose.length).toBeGreaterThanOrEqual(1)
    const text = messageText(events)
    expect(text).toContain('hello')
    expect(text).toContain('<unknown>')
    expect(text).toContain('world')
  })

  test('multiple unknown tags in sequence', async () => {
    const xml = '<a>1</a><b>2</b><c>3</c>'
    const events = await runStream(cfg, xml)
    const prose = eventsOfType(events, 'MessageEnd')
    expect(prose.length).toBeGreaterThanOrEqual(1)
    const text = messageText(events)
    expect(text).toContain('<a>')
    expect(text).toContain('1')
    expect(text).toContain('</a>')
    expect(text).toContain('<b>')
    expect(text).toContain('2')
    expect(text).toContain('</b>')
  })
})

// =============================================================================
// Reactor state fold
// =============================================================================

describe('foldReactorState', () => {
  test('ToolInputStarted adds to toolCallMap', () => {
    const state = initialReactorState()
    const next = foldReactorState(state, {
      _tag: 'ToolInputStarted',
      toolCallId: 'tc_1',
      taskId: 't1',
      tagName: 'read',
      toolName: 'read',
      group: 'default',
    })
    expect(next.toolCallMap.get('tc_1')).toBe('read')
  })

  test('ToolInputParseError adds to deadToolCalls and toolOutcomes', () => {
    const state = initialReactorState()
    const next = foldReactorState(state, {
      _tag: 'ToolInputParseError',
      toolCallId: 'tc_1',
      tagName: 'read',
      toolName: 'read',
      group: 'default',
      error: { _tag: 'MissingRequiredFields', id: 'tc_1', tagName: 'read', fields: ['path'], detail: 'missing' },
    })
    expect(next.deadToolCalls.has('tc_1')).toBe(true)
    expect(next.toolOutcomes.get('tc_1')).toEqual({ _tag: 'ParseError' })
  })

  test('ToolExecutionEnded adds Completed outcome', () => {
    const state = initialReactorState()
    const outputTree: OutputNode = { tag: 'element', name: 'read', attrs: {}, children: [] }
    const result = { _tag: 'Success' as const, output: { data: 1 }, outputTree: { tag: 'read', tree: outputTree }, query: '.' }
    const next = foldReactorState(state, {
      _tag: 'ToolExecutionEnded',
      toolCallId: 'tc_1',
      group: 'default',
      toolName: 'read',
      result,
    })
    expect(next.toolOutcomes.get('tc_1')).toEqual({ _tag: 'Completed', result })
  })

  test('ToolExecutionEnded Success does not update outputTrees', () => {
    const state = initialReactorState()
    const outputTree: OutputNode = { tag: 'element', name: 'read', attrs: {}, children: [
      { tag: 'element', name: 'content', attrs: {}, children: [{ tag: 'text', value: 'hello' }] },
    ] }
    const next = foldReactorState(state, {
      _tag: 'ToolExecutionEnded',
      toolCallId: 'tc_1',
      group: 'default',
      toolName: 'read',
      result: { _tag: 'Success', output: { content: 'hello' }, outputTree: { tag: 'read', tree: outputTree }, query: '.' },
    })
    expect(next.outputTrees.size).toBe(0)
  })

  test('ToolExecutionEnded Error does not add to outputTrees', () => {
    const state = initialReactorState()
    const next = foldReactorState(state, {
      _tag: 'ToolExecutionEnded',
      toolCallId: 'tc_1',
      group: 'default',
      toolName: 'read',
      result: { _tag: 'Error', error: 'something went wrong' },
    })
    expect(next.outputTrees.size).toBe(0)
  })

  test('TurnEnd sets stopped', () => {
    const state = initialReactorState()
    const next = foldReactorState(state, {
      _tag: 'TurnEnd',
      result: { _tag: 'Success', turnControl: null },
    })
    expect(next.stopped).toBe(true)
  })

  test('other events do not change state', () => {
    const state = initialReactorState()
    const same = foldReactorState(state, {
      _tag: 'ProseChunk',
      patternId: 'message',
      text: 'hello',
    })
    expect(same).toBe(state) // reference equality — no change
  })
})

// =============================================================================
// Replay tests
// =============================================================================

describe('replay (initialState with toolOutcomes)', () => {
  test('completed tools are fully suppressed on replay', async () => {
    const cfg = config([
      registered(readTool, 'read', readBinding),
      registered(writeTool, 'write', writeBinding),
    ])

    // Build initial state: tc_1 completed
    let state = initialReactorState()
    state = foldReactorState(state, {
      _tag: 'ToolInputStarted', toolCallId: 'tc_1', taskId: 't1', tagName: 'read', toolName: 'read', group: 'default',
    })
    state = foldReactorState(state, {
      _tag: 'ToolExecutionEnded', toolCallId: 'tc_1', group: 'default', toolName: 'read',
      result: { _tag: 'Success', output: { content: 'cached', lines: 1 }, outputTree: { tag: 'read', tree: { tag: 'element' as const, name: 'read', attrs: {}, children: [] } }, query: '.' },
    })

    // Replay same XML — first tool should be suppressed, second should execute fresh
    const xml = `${ACTIONS_TAG_OPEN}<read id="r1" path="a.ts"/><write id="r2" path="b.ts">content</write>${ACTIONS_TAG_CLOSE}`
    const events = await runStream(cfg, xml, { initialState: state })

    // No ToolInputStarted for first tool (suppressed)
    const started = eventsOfType(events, 'ToolInputStarted')
    expect(started).toHaveLength(1)
    expect(typeof started[0].toolCallId).toBe('string')
    expect(started[0].toolCallId.length).toBeGreaterThan(0)

    // Only second tool executed
    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended).toHaveLength(1)
    expect(ended[0].toolName).toBe('write')
  })

  test('in-flight tool suppresses input but dispatches on TagClosed', async () => {
    const cfg = config([registered(readTool, 'read', readBinding)])

    // Build initial state: tc_1 started but no outcome (in-flight)
    let state = initialReactorState()
    state = foldReactorState(state, {
      _tag: 'ToolInputStarted', toolCallId: 'tc_1', taskId: 't1', tagName: 'read', toolName: 'read', group: 'default',
    })

    const xml = `${ACTIONS_TAG_OPEN}<read id="r1" path="a.ts"/>${ACTIONS_TAG_CLOSE}`
    const events = await runStream(cfg, xml, { initialState: state })

    // No ToolInputStarted emitted (already in toolCallMap)
    const started = eventsOfType(events, 'ToolInputStarted')
    expect(started).toHaveLength(0)

    // No ToolInputFieldValue emitted
    const fieldValues = eventsOfType(events, 'ToolInputFieldValue')
    expect(fieldValues).toHaveLength(0)

    // But tool was dispatched and executed
    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended).toHaveLength(1)
    expect(ended[0].result._tag).toBe('Success')
  })

  test('full replay: 3 tools, first 2 completed, third fresh', async () => {
    const cfg = config([
      registered(readTool, 'read', readBinding),
      registered(writeTool, 'write', writeBinding),
      registered(shellTool, 'shell', shellBinding),
    ])

    // Build initial state: tc_1 and tc_2 completed
    let state = initialReactorState()
    state = foldReactorState(state, {
      _tag: 'ToolInputStarted', toolCallId: 'tc_1', taskId: 't1', tagName: 'read', toolName: 'read', group: 'default',
    })
    state = foldReactorState(state, {
      _tag: 'ToolExecutionEnded', toolCallId: 'tc_1', group: 'default', toolName: 'read',
      result: { _tag: 'Success', output: { content: 'cached', lines: 1 }, outputTree: { tag: 'read', tree: { tag: 'element' as const, name: 'read', attrs: {}, children: [] } }, query: '.' },
    })
    state = foldReactorState(state, {
      _tag: 'ToolInputStarted', toolCallId: 'tc_2', taskId: 't1', tagName: 'write', toolName: 'write', group: 'default',
    })
    state = foldReactorState(state, {
      _tag: 'ToolExecutionEnded', toolCallId: 'tc_2', group: 'default', toolName: 'write',
      result: { _tag: 'Success', output: 'wrote b.ts', outputTree: { tag: 'write', tree: { tag: 'element' as const, name: 'write', attrs: {}, children: [] } }, query: '.' },
    })

    const xml = `${ACTIONS_TAG_OPEN}<read id="r1" path="a.ts"/><write id="r2" path="b.ts">content</write><shell id="r3">echo hi</shell>${ACTIONS_TAG_CLOSE}`
    const events = await runStream(cfg, xml, { initialState: state })

    // Only shell gets full event cycle
    const started = eventsOfType(events, 'ToolInputStarted')
    expect(started).toHaveLength(1)
    expect(typeof started[0].toolCallId).toBe('string')
    expect(started[0].toolCallId.length).toBeGreaterThan(0)

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended).toHaveLength(1)
    expect(ended[0].toolName).toBe('shell')
    expect(ended[0].result._tag).toBe('Success')
  })


})

// =============================================================================
// Prose streaming tests (parser-level)
// =============================================================================

describe('prose streaming (parser-level)', () => {
  test('single line of prose emits ProseChunk on flush', () => {
    const events = parseChunked(['Hello world'])
    const chunks = parseEvents(events, 'ProseChunk')
    expect(chunks).toHaveLength(1)
    expect(chunks[0].text).toBe('Hello world')
    expect(chunks[0].patternId).toBe('prose')
    const ends = parseEvents(events, 'ProseEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].content).toBe('Hello world')
  })

  test('prose with newline emits chunk per line', () => {
    const events = parseChunked(['Hello\nWorld'])
    const chunks = parseEvents(events, 'ProseChunk')
    // "Hello" emitted on newline, deferred \n emitted before "World", "World" on flush
    expect(chunks).toHaveLength(3)
    expect(chunks[0].text).toBe('Hello')
    expect(chunks[1].text).toBe('\n')
    expect(chunks[2].text).toBe('World')
  })

  test('prose split across chunks', () => {
    const events = parseChunked(['Hel', 'lo w', 'orld'])
    const chunks = parseEvents(events, 'ProseChunk')
    // Each input chunk produces its own ProseChunk
    expect(chunks).toHaveLength(3)
    expect(chunks[0].text).toBe('Hel')
    expect(chunks[1].text).toBe('lo w')
    expect(chunks[2].text).toBe('orld')
  })

  test('prose split across chunks with newline in middle', () => {
    const events = parseChunked(['Hello\nWo', 'rld'])
    const chunks = parseEvents(events, 'ProseChunk')
    // "Hello" emitted at newline boundary, deferred \n emitted before next content,
    // "Wo" is the rest of chunk 1, "rld" is chunk 2
    expect(chunks).toHaveLength(4)
    expect(chunks[0].text).toBe('Hello')
    expect(chunks[1].text).toBe('\n')
    expect(chunks[2].text).toBe('Wo')
    expect(chunks[3].text).toBe('rld')
  })

  test('newline at chunk boundary', () => {
    const events = parseChunked(['Hello', '\nWorld'])
    const chunks = parseEvents(events, 'ProseChunk')
    // "Hello" emitted on newline, deferred \n emitted before "World", "World" on flush
    expect(chunks).toHaveLength(3)
    expect(chunks[0].text).toBe('Hello')
    expect(chunks[1].text).toBe('\n')
    expect(chunks[2].text).toBe('World')
  })

  test('prose then actions tag', () => {
    const events = parseChunked([`Hello world\n${ACTIONS_TAG_OPEN}\n${ACTIONS_TAG_CLOSE}`], ['read'])
    const chunks = parseEvents(events, 'ProseChunk')
    expect(chunks).toHaveLength(1)
    expect(chunks[0].text).toBe('Hello world')
    const ends = parseEvents(events, 'ProseEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].content).toBe('Hello world')
  })

  test('prose then actions split at tag boundary', () => {
    const events = parseChunked(['Hello world\n', `${ACTIONS_TAG_OPEN}\n${ACTIONS_TAG_CLOSE}`], ['read'])
    const chunks = parseEvents(events, 'ProseChunk')
    // "Hello world" flushed when < starts tag
    expect(chunks).toHaveLength(1)
    expect(chunks[0].text).toBe('Hello world')
  })

  test('prose with < split across chunk boundary', () => {
    // "Hello\n<" in one chunk, "actions>\n</task>" in next — the < flushes lineBuffer
    const events = parseChunked(['Hello\n<', `actions>\n${ACTIONS_TAG_CLOSE}`], ['read'])
    const chunks = parseEvents(events, 'ProseChunk')
    expect(chunks.length).toBeGreaterThan(0)
    expect(chunks.map(c => c.text).join('')).toContain('Hello')
    const ends = parseEvents(events, 'ProseEnd')
    expect(ends).toHaveLength(1)
    expect(ends[0].content).toContain('Hello')
  })

  test('multiline prose before actions', () => {
    const events = parseChunked([`Line one\nLine two\n${ACTIONS_TAG_OPEN}\n${ACTIONS_TAG_CLOSE}`], ['read'])
    const chunks = parseEvents(events, 'ProseChunk')
    // "Line one" on first \n, deferred \n emitted before "Line two",
    // "Line two" on second \n. Trailing \n before <task id="t1"> is dropped.
    expect(chunks).toHaveLength(3)
    expect(chunks[0].text).toBe('Line one')
    expect(chunks[1].text).toBe('\n')
    expect(chunks[2].text).toBe('Line two')
  })

  test('code fence lines are stripped', () => {
    const events = parseChunked([`\`\`\`xml\n${ACTIONS_TAG_OPEN}\n${ACTIONS_TAG_CLOSE}\n\`\`\``])
    const chunks = parseEvents(events, 'ProseChunk')
    // Both code fence lines should be stripped (they match ```xml and ```)
    // No prose chunks expected
    expect(chunks).toHaveLength(0)
  })

  test('code fence with content around it', () => {
    const events = parseChunked([`Hello\n\`\`\`xml\n${ACTIONS_TAG_OPEN}\n${ACTIONS_TAG_CLOSE}\n\`\`\`\nBye`])
    const chunks = parseEvents(events, 'ProseChunk')
    const allText = chunks.map(c => c.text).join('')
    // "Hello" and "Bye" should appear
    expect(allText).toContain('Hello')
    expect(allText).toContain('Bye')
    // Code fences should NOT appear
    expect(allText).not.toContain('```')
  })

  test('innocuous markdown code fences are preserved', () => {
    // Non-xml fences (```json, ```ts, bare closing ```) inside prose should NOT be stripped.
    // This reproduces the bug where closing ``` fences get eaten because they match the
    // fence pattern, even though they're regular markdown — not wrapping <task id="t1">.
    const prose = [
      'The clean way to have both: **subpath exports**.',
      '',
      '```json',
      '// packages/tools/package.json',
      '{',
      '  "exports": {',
      '    ".": "./src/index.ts"',
      '  }',
      '}',
      '```',
      '',
      'Then colocation is real:',
      '',
      '```ts',
      '// packages/tools/src/shell.ts',
      'export const shellTool = defineToolUnsafe({ name: "shell" })',
      '```',
    ].join('\n')

    const events = parseChunked([prose])
    const chunks = parseEvents(events, 'ProseChunk')
    const allText = chunks.map(c => c.text).join('')

    // All code fences must survive — they're regular markdown, not xml-act wrappers
    expect(allText).toContain('```json')
    expect(allText).toContain('```ts')
    // The closing ``` fences must also survive (this is the core bug)
    const backtickOnlyLines = allText.split('\n').filter(l => l.trim() === '```')
    expect(backtickOnlyLines).toHaveLength(2) // one closing fence per code block
  })

  test('code fence split across chunks', () => {
    const events = parseChunked(['Hello\n``', `\`xml\n${ACTIONS_TAG_OPEN}${ACTIONS_TAG_CLOSE}`])
    const chunks = parseEvents(events, 'ProseChunk')
    expect(chunks[0].text).toBe('Hello')
    // No chunk should contain the code fence
    expect(chunks.map(c => c.text).join('')).not.toContain('```xml')
  })

  test('think tag emits think chunks, not prose chunks', () => {
    const events = parseChunked(['<think>reasoning\n</think>Hello'])
    const thinkChunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'think')
    const proseChunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    expect(thinkChunks.length).toBeGreaterThan(0)
    expect(thinkChunks.map(c => c.text).join('')).toBe('reasoning\n')
    // "Hello" should be prose
    expect(proseChunks.length).toBeGreaterThan(0)
    expect(proseChunks.map(c => c.text).join('')).toBe('Hello')
  })

  test('think then prose then actions', () => {
    const events = parseChunked([`<think>thought\n</think>Message\n${ACTIONS_TAG_OPEN}\n${ACTIONS_TAG_CLOSE}`])
    const thinkEnds = parseEvents(events, 'ProseEnd').filter(e => e.patternId === 'think')
    expect(thinkEnds).toHaveLength(1)
    expect(thinkEnds[0].content).toBe('thought\n')
    const proseChunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    // Trailing \n before <task id="t1"> is dropped
    expect(proseChunks.map(c => c.text).join('')).toBe('Message')
  })

  test('non-matching close tag inside think block is treated as think body text', () => {
    const events = parseChunked(['<think>before</foo>after\n</think>Hello'])
    const thinkEnds = parseEvents(events, 'ProseEnd').filter(e => e.patternId === 'think')
    expect(thinkEnds).toHaveLength(1)
    // The entire content including </foo> should be part of the think body
    expect(thinkEnds[0].content).toBe('before</foo>after\n')
    // "Hello" should be regular prose after the think block
    const proseChunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    expect(proseChunks.map(c => c.text).join('')).toBe('Hello')
  })

  test('non-matching close tag inside think block — character by character', () => {
    const events = parseChunked([...'<think>text</div>more\n</think>Done'])
    const thinkEnds = parseEvents(events, 'ProseEnd').filter(e => e.patternId === 'think')
    expect(thinkEnds).toHaveLength(1)
    expect(thinkEnds[0].content).toBe('text</div>more\n')
    const proseChunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    expect(proseChunks.map(c => c.text).join('')).toBe('Done')
  })

  test('nested think tags inside think block use depth counting', () => {
    const events = parseChunked(['<think>outer\n<think>inner\n</think>still outer\n</think>Done'])
    const thinkEnds = parseEvents(events, 'ProseEnd').filter(e => e.patternId === 'think')
    expect(thinkEnds).toHaveLength(1)
    expect(thinkEnds[0].content).toBe('outer\n<think>inner\n</think>still outer\n')
    const proseChunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    expect(proseChunks.map(c => c.text).join('')).toBe('Done')
  })

  test('nested think tags — character by character', () => {
    const events = parseChunked([...'<think>a\n<think>b\n</think>c\n</think>D'])
    const thinkEnds = parseEvents(events, 'ProseEnd').filter(e => e.patternId === 'think')
    expect(thinkEnds).toHaveLength(1)
    expect(thinkEnds[0].content).toBe('a\n<think>b\n</think>c\n')
    const proseChunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    expect(proseChunks.map(c => c.text).join('')).toBe('D')
  })

  test('deeply nested think tags', () => {
    const events = parseChunked(['<think>1\n<think>2\n<think>3\n</think>2\n</think>1\n</think>X'])
    const thinkEnds = parseEvents(events, 'ProseEnd').filter(e => e.patternId === 'think')
    expect(thinkEnds).toHaveLength(1)
    expect(thinkEnds[0].content).toBe('1\n<think>2\n<think>3\n</think>2\n</think>1\n')
    const proseChunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    expect(proseChunks.map(c => c.text).join('')).toBe('X')
  })



  test('think then actions with self-closing tool — character by character', () => {
    const xml = `<think>thought\n</think>\n${ACTIONS_TAG_OPEN}\n<read id="r1" path="x.ts" />\n${ACTIONS_TAG_CLOSE}`
    const events = parseChunked([...xml], ['read'])

    const thinkEnds = parseEvents(events, 'ProseEnd').filter(e => e.patternId === 'think')
    expect(thinkEnds).toHaveLength(1)

    const tagOpened = parseEvents(events, 'TagOpened')
    expect(tagOpened).toHaveLength(1)
    expect(tagOpened[0].tagName).toBe('read')
    expect(tagOpened[0].attributes.get('path')).toBe('x.ts')

    const tagClosed = parseEvents(events, 'TagClosed').filter(e => e.tagName === 'read')
    expect(tagClosed).toHaveLength(1)
  })

  test('unknown tag becomes prose', () => {
    const events = parseChunked(['<unknown>stuff</unknown>'])
    const chunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    const allText = chunks.map(c => c.text).join('')
    expect(allText).toContain('<unknown>')
    expect(allText).toContain('stuff')
    expect(allText).toContain('</unknown>')
  })

  test('unknown tag split across chunks', () => {
    const events = parseChunked(['<unk', 'nown>stuff</un', 'known>'])
    const chunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    const allText = chunks.map(c => c.text).join('')
    expect(allText).toContain('<unknown>')
    expect(allText).toContain('stuff')
    expect(allText).toContain('</unknown>')
  })

  test('prose between tool tags inside actions', () => {
    const events = parseChunked([`${ACTIONS_TAG_OPEN}<read id="r1" path="a"/>`, 'some text', `<read id="r2" path="b"/>${ACTIONS_TAG_CLOSE}`], ['read'])
    const tagOpened = parseEvents(events, 'TagOpened')
    expect(tagOpened).toHaveLength(2)
    expect(tagOpened[0].tagName).toBe('read')
    expect(tagOpened[1].tagName).toBe('read')
  })

  test('empty prose emits nothing', () => {
    const events = parseChunked([`${ACTIONS_TAG_OPEN}\n${ACTIONS_TAG_CLOSE}`])
    const chunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    expect(chunks).toHaveLength(0)
    const ends = parseEvents(events, 'ProseEnd').filter(e => e.patternId === 'prose')
    expect(ends).toHaveLength(0)
  })

  test('whitespace-only prose before tag produces no ProseEnd', () => {
    const events = parseChunked([`   \n  \n${ACTIONS_TAG_OPEN}\n${ACTIONS_TAG_CLOSE}`])
    // Leading spaces may be emitted as chunks (before newline triggers buffering),
    // but whitespace after newlines is buffered and dropped before the tag.
    // ProseEnd is not emitted since trimmed content is empty.
    const ends = parseEvents(events, 'ProseEnd').filter(e => e.patternId === 'prose')
    expect(ends).toHaveLength(0)
  })

  test('character-by-character delivery', () => {
    const xml = `Hi\n${ACTIONS_TAG_OPEN}\n${ACTIONS_TAG_CLOSE}`
    const events = parseChunked([...xml])
    const chunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    // Each char is a separate input chunk. Trailing \n before <task id="t1"> is dropped.
    expect(chunks).toHaveLength(2)
    expect(chunks[0].text).toBe('H')
    expect(chunks[1].text).toBe('i')
  })

  test('ProseEnd emitted before TaskOpen', () => {
    const events = parseChunked([`Hello\n${ACTIONS_TAG_OPEN}\n${ACTIONS_TAG_CLOSE}`])
    const proseEndIdx = events.findIndex(e => e._tag === 'ProseEnd' && e.patternId === 'prose')
    const actionsIdx = events.findIndex(e => e._tag === 'TaskOpen')
    expect(proseEndIdx).not.toBe(-1)
    expect(actionsIdx).not.toBe(-1)
    expect(proseEndIdx).toBeLessThan(actionsIdx)
  })

  test('multiple prose blocks separated by structural tags', () => {
    const events = parseChunked([`First\n<think>t\n</think>Second\n${ACTIONS_TAG_OPEN}\n${ACTIONS_TAG_CLOSE}`])
    const proseEnds = parseEvents(events, 'ProseEnd').filter(e => e.patternId === 'prose')
    expect(proseEnds).toHaveLength(2)
    expect(proseEnds[0].content).toBe('First')
    expect(proseEnds[1].content).toBe('Second')
  })

  test('trailing whitespace before tags is dropped, not emitted as prose', () => {
    // Prose followed by blank lines before <task id="t1"> — whitespace should be dropped
    const events = parseChunked([`some prose\n\n${ACTIONS_TAG_OPEN}\n${ACTIONS_TAG_CLOSE}`], ['read'])
    const chunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    const allText = chunks.map(c => c.text).join('')
    // Only "some prose" should appear — no trailing newlines
    expect(allText).toBe('some prose')
    const ends = parseEvents(events, 'ProseEnd').filter(e => e.patternId === 'prose')
    expect(ends).toHaveLength(1)
    expect(ends[0].content).toBe('some prose')
  })

  test('trailing whitespace before known tool tag is dropped', () => {
    const events = parseChunked([`message\n\n\n${ACTIONS_TAG_OPEN}\n<read id="r1" path="x"/>\n${ACTIONS_TAG_CLOSE}`], ['read'])
    const chunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    const allText = chunks.map(c => c.text).join('')
    expect(allText).toBe('message')
  })

  test('whitespace between prose lines is preserved', () => {
    const events = parseChunked([`Line one\n\nLine two\n${ACTIONS_TAG_OPEN}\n${ACTIONS_TAG_CLOSE}`], ['read'])
    const chunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    const allText = chunks.map(c => c.text).join('')
    // Newlines between prose lines preserved, trailing newline before tag dropped
    expect(allText).toBe('Line one\n\nLine two')
  })

  test('whitespace inside actions block does not produce prose', () => {
    const events = parseChunked([`prose\n${ACTIONS_TAG_OPEN}\n  \n${ACTIONS_TAG_CLOSE}`], ['read'])
    const chunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    const allText = chunks.map(c => c.text).join('')
    expect(allText).toBe('prose')
  })
  test('known tool tag outside actions block is treated as prose, not a tool call', () => {
    // A known tag appearing in prose (outside any actions block) should NOT be
    // parsed as a tool invocation — it should pass through as prose text.
    const xml = `Use <read path="x.ts"/> to read files.\n${ACTIONS_TAG_OPEN}<read id="r1" path="a.ts"/>${ACTIONS_TAG_CLOSE}`
    const events = parseChunked([xml], ['read'])

    // Only the read tag inside the actions block should be recognized as a tool tag
    const tagOpened = parseEvents(events, 'TagOpened')
    expect(tagOpened).toHaveLength(1)

    // The prose read tag should appear as literal text in prose events
    const proseChunks = parseEvents(events, 'ProseChunk').filter(c => c.patternId === 'prose')
    const allText = proseChunks.map(c => c.text).join('')
    expect(allText).toContain('<read')
  })
})

// =============================================================================
// Prose streaming tests (runtime-level, full pipeline)
// =============================================================================

describe('prose streaming (runtime-level)', () => {

  const cfg = config([registered(readTool, 'read', readBinding)])

  test('prose emits MessageChunk', async () => {
    const events = await runStreamChunked(cfg, [`Hello world\n${ACTIONS_TAG_OPEN}<read id="r1" path="x"/>${ACTIONS_TAG_CLOSE}`])
    const chunks = eventsOfType(events, 'MessageChunk')
    expect(chunks.length).toBeGreaterThan(0)
    const allText = chunks.map(c => c.text).join('')
    expect(allText).toContain('Hello world')
  })

  test('prose emits MessageEnd', async () => {
    const events = await runStreamChunked(cfg, [`Hello\n${ACTIONS_TAG_OPEN}\n<read id="r1" path="x"/>\n${ACTIONS_TAG_CLOSE}`])
    const ends = eventsOfType(events, 'MessageEnd')
    expect(ends).toHaveLength(1)
  })

  test('think emits ProseChunk with patternId think', async () => {
    const events = await runStreamChunked(cfg, [`<think>reasoning\n</think>\n${ACTIONS_TAG_OPEN}\n<read id="r1" path="x"/>\n${ACTIONS_TAG_CLOSE}`])
    const thinkChunks = eventsOfType(events, 'ProseChunk').filter(c => c.patternId === 'think')
    expect(thinkChunks.length).toBeGreaterThan(0)
    const allText = thinkChunks.map(c => c.text).join('')
    expect(allText).toBe('reasoning\n')
  })

  test('prose streams incrementally across chunks', async () => {
    const events = await runStreamChunked(cfg, [
      'Hel', 'lo ', 'wor', 'ld\n',
      `${ACTIONS_TAG_OPEN}<read id="r1" path="x"/>${ACTIONS_TAG_CLOSE}`,
    ])
    const chunks = eventsOfType(events, 'MessageChunk')
    expect(chunks.length).toBeGreaterThan(0)
    const allText = chunks.map(c => c.text).join('')
    expect(allText).toContain('Hello world')
  })

  test('code fences stripped in runtime pipeline', async () => {
    const events = await runStreamChunked(cfg, [
      `\`\`\`xml\n${ACTIONS_TAG_OPEN}<read id="r1" path="x"/>${ACTIONS_TAG_CLOSE}\n\`\`\``,
    ])
    const chunks = eventsOfType(events, 'MessageChunk')
    const allText = chunks.map(c => c.text).join('')
    expect(allText).not.toContain('```')
  })

  test('prose before and after think', async () => {
    const events = await runStreamChunked(cfg, [
      `Before\n<think>thought\n</think>After\n${ACTIONS_TAG_OPEN}\n<read id="r1" path="x"/>\n${ACTIONS_TAG_CLOSE}`,
    ])
    const proseChunks = eventsOfType(events, 'MessageChunk')
    const thinkChunks = eventsOfType(events, 'ProseChunk').filter(c => c.patternId === 'think')
    const proseText = proseChunks.map(c => c.text).join('')
    expect(proseText).toContain('Before')
    expect(proseText).toContain('After')
    expect(thinkChunks.map(c => c.text).join('')).toBe('thought\n')
  })

  test('no prose events when only actions', async () => {
    const events = await runStreamChunked(cfg, [`${ACTIONS_TAG_OPEN}\n<read id="r1" path="x"/>\n${ACTIONS_TAG_CLOSE}`])
    const chunks = eventsOfType(events, 'MessageChunk')
    expect(chunks).toHaveLength(0)
  })
})

describe('tool body containing XML-like closing tags', () => {
  // Regression: when a shell command body contains HTML/Svelte/JSX closing tags
  // like </script>, </div>, etc., the parser should NOT interpret them as
  // closing the <shell> tool tag. Only </shell> should close the shell tool.

  test('shell body with </script> tag inside heredoc is parsed correctly', async () => {
    const cfg = config([registered(shellTool, 'shell', shellBinding)])
    const xml = `${ACTIONS_TAG_OPEN}<shell id="s1">cat > app.svelte <<'EOF'
<script lang="ts">
  let count = 0;
</script>
<button>{count}</button>
EOF</shell>${ACTIONS_TAG_CLOSE}`

    const events = await runStream(cfg, xml)

    // Should have parsed as a tool call, not prose
    const started = eventsOfType(events, 'ToolInputStarted')
    expect(started).toHaveLength(1)
    expect(started[0].toolName).toBe('shell')

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    expect((ready[0].input as { command: string }).command).toContain('</script>')
    expect((ready[0].input as { command: string }).command).toContain('<button>{count}</button>')

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended).toHaveLength(1)
    expect(ended[0].result._tag).toBe('Success')
  })

  test('shell body with multiple HTML closing tags', async () => {
    const cfg = config([registered(shellTool, 'shell', shellBinding)])
    const xml = `${ACTIONS_TAG_OPEN}<shell id="s1">cat > index.html <<'EOF'
<html>
<body>
  <div class="app">
    <h1>Hello</h1>
  </div>
</body>
</html>
EOF</shell>${ACTIONS_TAG_CLOSE}`

    const events = await runStream(cfg, xml)

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    expect((ready[0].input as { command: string }).command).toContain('</html>')
    expect((ready[0].input as { command: string }).command).toContain('</body>')
    expect((ready[0].input as { command: string }).command).toContain('</div>')
  })

  test('write body with closing tags does not break parsing', async () => {
    const cfg = config([registered(writeTool, 'write', writeBinding)])
    const xml = `${ACTIONS_TAG_OPEN}<write id="w1" path="Component.svelte"><script>
  import X from './X.svelte';
</script>
<main>
  <X />
</main></write>${ACTIONS_TAG_CLOSE}`

    const events = await runStream(cfg, xml)

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    expect((ready[0].input as { content: string }).content).toContain('</script>')
    expect((ready[0].input as { content: string }).content).toContain('</main>')
  })

  test('closing tags in body are preserved exactly as streamed', async () => {
    const cfg = config([registered(shellTool, 'shell', shellBinding)])
    const command = `echo '</ScRiPt>' && echo '</DIV  >'`
    const xml = `${ACTIONS_TAG_OPEN}<shell id="s1">${command}</shell>${ACTIONS_TAG_CLOSE}`

    const events = await runStream(cfg, xml)

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    expect((ready[0].input as { command: string }).command).toBe(command)
  })

  test('streamed char-by-char with closing tags in body', async () => {
    const cfg = config([registered(shellTool, 'shell', shellBinding)])
    const xml = `${ACTIONS_TAG_OPEN}<shell id="s1">cat <<'EOF'
<div>hello</div>
</body>
EOF</shell>${ACTIONS_TAG_CLOSE}`

    const events = await runStreamCharByChar(cfg, xml)

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    expect((ready[0].input as { command: string }).command).toContain('</div>')
    expect((ready[0].input as { command: string }).command).toContain('</body>')

    const ended = eventsOfType(events, 'ToolExecutionEnded')
    expect(ended).toHaveLength(1)
    expect(ended[0].result._tag).toBe('Success')
  })
})

// ---------------------------------------------------------------------------
// Structural tag auto-close
// ---------------------------------------------------------------------------
// When the parser encounters a structural tag that comes later in the
// sequence (lenses → comms → actions → next/yield), it should auto-close
// the currently open earlier structural block.

describe('structural tag auto-close', () => {
  // --- lenses → comms ---
  test('unclosed lenses auto-closes when comms opens', () => {
    const events = parseChunked([
      '<lenses>\n<lens name="foo">thinking</lens>\n<task id="t2">\n<message>hi</message>\n</task>',
    ])
    expect(parseEvents(events, 'TaskOpen')).toHaveLength(1)
    expect(parseEvents(events, 'TaskClose')).toHaveLength(1)
    expect(parseEvents(events, 'MessageStart')).toHaveLength(1)
    expect(parseEvents(events, 'MessageEnd')).toHaveLength(1)
    expect(parseEvents(events, 'ParseError').length).toBeLessThanOrEqual(1)
  })

  // --- lenses → actions ---
  test('unclosed lenses auto-closes when actions opens', () => {
    const events = parseChunked([
      '<lenses>\n<lens name="foo">thinking</lens>\n<task id="t1">\n</task>',
    ], ['read'])
    expect(parseEvents(events, 'TaskOpen')).toHaveLength(1)
    expect(parseEvents(events, 'TaskClose')).toHaveLength(1)
    expect(parseEvents(events, 'ParseError').length).toBeLessThanOrEqual(1)
  })

  // --- lenses → next ---
  test('unclosed lenses auto-closes when next is encountered', () => {
    const events = parseChunked([
      '<lenses>\n<lens name="foo">thinking</lens>\n<next/>',
    ])
    const tc = parseEvents(events, 'TurnControl')
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
    expect(parseEvents(events, 'ParseError')).toHaveLength(0)
  })

  // --- lenses → yield ---
  test('unclosed lenses auto-closes when yield is encountered', () => {
    const events = parseChunked([
      '<lenses>\n<lens name="foo">thinking</lens>\n<yield/>',
    ])
    const tc = parseEvents(events, 'TurnControl')
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('yield')
    expect(parseEvents(events, 'ParseError')).toHaveLength(0)
  })

  // --- comms → actions ---
  test('unclosed comms auto-closes when actions opens', () => {
    const events = parseChunked([
      '<task id="t2">\n<message>hi</message>\n<task id="t1">\n</task>',
    ], ['read'])
    expect(parseEvents(events, 'TaskOpen')).toHaveLength(2)
    expect(parseEvents(events, 'TaskClose').length).toBeGreaterThanOrEqual(1)
    expect(parseEvents(events, 'ParseError').length).toBeLessThanOrEqual(1)
  })

  // --- comms → next ---
  test('unclosed comms keeps next as prose content (no auto-close, no turn control)', () => {
    const events = parseChunked([
      '<task id="t2">\n<message>hi</message>\n<next/>',
    ])
    expect(parseEvents(events, 'TaskOpen')).toHaveLength(1)
    expect(parseEvents(events, 'TaskClose')).toHaveLength(0)
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)
    const prose = parseEvents(events, 'ProseChunk').map(e => e.text).join('')
    expect(prose).toContain('<next/>')
    expect(parseEvents(events, 'ParseError').some(e => e.error._tag === 'UnclosedTask')).toBe(true)
  })

  // --- comms → yield ---
  test('unclosed comms keeps yield as prose content (no auto-close, no turn control)', () => {
    const events = parseChunked([
      '<task id="t2">\n<message>hi</message>\n<yield/>',
    ])
    expect(parseEvents(events, 'TaskOpen')).toHaveLength(1)
    expect(parseEvents(events, 'TaskClose')).toHaveLength(0)
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)
    const prose = parseEvents(events, 'ProseChunk').map(e => e.text).join('')
    expect(prose).toContain('<yield/>')
    expect(parseEvents(events, 'ParseError').some(e => e.error._tag === 'UnclosedTask')).toBe(true)
  })

  // --- actions → next ---
  test('unclosed actions keeps next as prose content (no auto-close, no turn control)', () => {
    const events = parseChunked([
      '<task id="t1">\n<next/>',
    ], ['read'])
    expect(parseEvents(events, 'TaskOpen')).toHaveLength(1)
    expect(parseEvents(events, 'TaskClose')).toHaveLength(0)
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)
    const prose = parseEvents(events, 'ProseChunk').map(e => e.text).join('')
    expect(prose).toContain('<next/>')
    expect(parseEvents(events, 'ParseError').some(e => e.error._tag === 'UnclosedTask')).toBe(true)
  })

  // --- actions → yield ---
  test('unclosed actions keeps yield as prose content (no auto-close, no turn control)', () => {
    const events = parseChunked([
      '<task id="t1">\n<yield/>',
    ], ['read'])
    expect(parseEvents(events, 'TaskOpen')).toHaveLength(1)
    expect(parseEvents(events, 'TaskClose')).toHaveLength(0)
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)
    const prose = parseEvents(events, 'ProseChunk').map(e => e.text).join('')
    expect(prose).toContain('<yield/>')
    expect(parseEvents(events, 'ParseError').some(e => e.error._tag === 'UnclosedTask')).toBe(true)
  })

  test('turn-control inside unclosed task is passthrough, not recognized as top-level turn control', () => {
    const events = parseChunked([
      '<task id="t1">\n<next/>',
    ], ['read'])

    expect(parseEvents(events, 'TaskOpen')).toHaveLength(1)
    expect(parseEvents(events, 'TaskClose')).toHaveLength(0)
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)

    const prose = parseEvents(events, 'ProseChunk').map((e) => e.text).join('')
    expect(prose).toContain('<next/>')
  })
})

describe('turn-control scoping across parser contexts', () => {
  const proseText = (events: ParseEvent[]) =>
    parseEvents(events, 'ProseChunk').map(e => e.text).join('')

  const messageText = (events: ParseEvent[]) =>
    parseEvents(events, 'MessageChunk').map(e => e.text).join('')

  const lensText = (events: ParseEvent[]) =>
    parseEvents(events, 'LensChunk').map(e => e.text).join('')

  test('yield inside unclosed task is passthrough, not TurnControl', () => {
    const events = parseChunked(['<task id="t1">\n<yield/>'], ['read'])
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)
    expect(parseEvents(events, 'TaskClose')).toHaveLength(0)
    expect(proseText(events)).toContain('<yield/>')
  })

  test('next inside nested unclosed tasks is passthrough', () => {
    const events = parseChunked(['<task id="t1"><task id="t2"><next/>'], ['read'])
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)
    expect(parseEvents(events, 'TaskOpen')).toHaveLength(2)
    expect(parseEvents(events, 'TaskClose')).toHaveLength(0)
    expect(proseText(events)).toContain('<next/>')
  })

  test('yield inside nested unclosed tasks is passthrough', () => {
    const events = parseChunked(['<task id="t1"><task id="t2"><yield/>'], ['read'])
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)
    expect(parseEvents(events, 'TaskOpen')).toHaveLength(2)
    expect(parseEvents(events, 'TaskClose')).toHaveLength(0)
    expect(proseText(events)).toContain('<yield/>')
  })

  test('turn control after properly closed task is recognized at top-level', () => {
    const events = parseChunked(['<task id="t1"></task><next/>'], ['read'])
    const tc = parseEvents(events, 'TurnControl')
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
  })

  test('turn control inside unclosed task with prior content remains passthrough', () => {
    const events = parseChunked(['<task id="t1">before\n<next/>after'], ['read'])
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)
    const raw = proseText(events)
    expect(raw).toContain('<next/>')
  })

  test('turn control inside unclosed message block within task is message content', () => {
    const events = parseChunked(['<task id="t1"><message to="parent">hi <next/>'], ['read'])
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)
    expect(messageText(events)).toContain('<next/>')
  })

  test('multiple turn-control tags inside unclosed task are passthrough', () => {
    const events = parseChunked(['<task id="t1"><next/><yield/>'], ['read'])
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)
    const raw = proseText(events)
    expect(raw).toContain('<next/>')
    expect(raw).toContain('<yield/>')
  })

  test('turn control inside unclosed lenses/think block is not TurnControl', () => {
    const events = parseChunked(['<lenses><lens name="q">thinking <next/>'])
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)
    expect(lensText(events)).toContain('<next/>')
  })

  test('turn control immediately after task open tag is passthrough', () => {
    const events = parseChunked(['<task id="t1"><next/>'], ['read'])
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)
    expect(proseText(events)).toContain('<next/>')
  })

  test('mix of closed and unclosed tasks: only turn control in unclosed task is passthrough', () => {
    const events = parseChunked(['<task id="t1"></task><task id="t2"><yield/>'], ['read'])
    expect(parseEvents(events, 'TaskClose')).toHaveLength(1)
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)
    expect(proseText(events)).toContain('<yield/>')
  })

  test('turn control inside deeply nested unclosed tasks (3+ levels) is passthrough', () => {
    const events = parseChunked(['<task id="a"><task id="b"><task id="c"><next/>'], ['read'])
    expect(parseEvents(events, 'TaskOpen')).toHaveLength(3)
    expect(parseEvents(events, 'TaskClose')).toHaveLength(0)
    expect(parseEvents(events, 'TurnControl')).toHaveLength(0)
    expect(proseText(events)).toContain('<next/>')
  })

  test('both next and yield remain recognized when fully top-level', () => {
    const nextEvents = parseChunked(['<next/>'])
    const yieldEvents = parseChunked(['<yield/>'])

    const nextTc = parseEvents(nextEvents, 'TurnControl')
    expect(nextTc).toHaveLength(1)
    expect(nextTc[0].decision).toBe('continue')

    const yieldTc = parseEvents(yieldEvents, 'TurnControl')
    expect(yieldTc).toHaveLength(1)
    expect(yieldTc[0].decision).toBe('yield')
  })
})
