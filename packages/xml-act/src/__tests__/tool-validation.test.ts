import { describe, expect, test } from 'vitest'
import { Effect, Stream } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, type ToolDefinition } from '@magnitudedev/tools'
import {
  createXmlRuntime,
  type RegisteredTool,
  type XmlRuntimeConfig,
  type XmlRuntimeEvent,
  type XmlTagBinding,
} from '../index'

function runStream(config: XmlRuntimeConfig, xml: string): Promise<XmlRuntimeEvent[]> {
  const runtime = createXmlRuntime(config)
  const stream = runtime.streamWith(Stream.make(xml))
  return Effect.runPromise(Stream.runCollect(stream)).then(c => Array.from(c))
}

function registered(
  tool: ToolDefinition,
  tagName: string,
  binding: XmlTagBinding,
  opts?: { groupName?: string },
): RegisteredTool {
  return {
    tool,
    tagName,
    groupName: opts?.groupName ?? 'default',
    binding,
  }
}

function config(tools: RegisteredTool[]): XmlRuntimeConfig {
  return {
    tools: new Map(tools.map(t => [t.tagName, t])),
  }
}

function eventsOfType<T extends XmlRuntimeEvent['_tag']>(
  events: XmlRuntimeEvent[],
  tag: T,
): Extract<XmlRuntimeEvent, { _tag: T }>[] {
  return events.filter(e => e._tag === tag) as Extract<XmlRuntimeEvent, { _tag: T }>[]
}

function responseWithActions(actionsXml: string): string {
  return `<lens name="turn">planning</lens>\n${actionsXml}<idle/>`
}

// 1) Tool with both attrs and childTags
const agentCreateLikeTool = defineTool({
  name: 'agent_create_like',
  description: 'Create an agent-like request',
  inputSchema: Schema.Struct({
    id: Schema.String,
    type: Schema.String,
    title: Schema.String,
    message: Schema.String,
  }),
  outputSchema: Schema.Struct({
    ok: Schema.Boolean,
  }),
  execute: () => Effect.succeed({ ok: true }),
  label: ({ id, type }) => `agent_create_like:${id}:${type}`,
})

const agentCreateLikeBinding: XmlTagBinding = {
  tag: 'agent_create_like',
  attributes: [{ field: 'id', attr: 'id' }],
  childTags: [
    { field: 'type', tag: 'type' },
    { field: 'title', tag: 'title' },
    { field: 'message', tag: 'message' },
  ],
}

// 2) Tool with only attrs
const readLikeTool = defineTool({
  name: 'read_like',
  description: 'Read a path-like string',
  inputSchema: Schema.Struct({
    path: Schema.String,
  }),
  outputSchema: Schema.Struct({
    content: Schema.String,
  }),
  execute: ({ path }) => Effect.succeed({ content: `contents:${path}` }),
  label: ({ path }) => `read_like:${path}`,
})

const readLikeBinding: XmlTagBinding = {
  tag: 'read_like',
  attributes: [{ field: 'path', attr: 'path' }],
}

// 3) Tool with body + optional attr
const shellLikeTool = defineTool({
  name: 'shell_like',
  description: 'Run shell-like command',
  inputSchema: Schema.Struct({
    command: Schema.String,
    timeout: Schema.optional(Schema.Number),
    recursive: Schema.optional(Schema.Boolean),
  }),
  outputSchema: Schema.Struct({
    stdout: Schema.String,
    exitCode: Schema.Number,
  }),
  execute: ({ command }) => Effect.succeed({ stdout: `ran:${command}`, exitCode: 0 }),
  label: ({ command }) => `shell_like:${command}`,
})

const shellLikeBinding: XmlTagBinding = {
  tag: 'shell_like',
  body: 'command',
  attributes: [
    { field: 'timeout', attr: 'timeout' },
    { field: 'recursive', attr: 'recursive' },
  ],
}

const grepLikeTool = defineTool({
  name: 'grep_like',
  description: 'attribute-only grep-like tool',
  inputSchema: Schema.Struct({
    pattern: Schema.String,
    path: Schema.String,
  }),
  outputSchema: Schema.Struct({
    ok: Schema.Boolean,
  }),
  execute: () => Effect.succeed({ ok: true }),
  label: ({ pattern, path }) => `grep_like:${pattern}:${path}`,
})

const grepLikeBinding: XmlTagBinding = {
  tag: 'grep_like',
  attributes: [
    { field: 'pattern', attr: 'pattern' },
    { field: 'path', attr: 'path' },
  ],
}

const strictTitleTool = defineTool({
  name: 'strict_title',
  description: 'Validate title min length',
  inputSchema: Schema.Struct({
    id: Schema.String,
    title: Schema.String.pipe(Schema.minLength(5)),
  }),
  outputSchema: Schema.Struct({
    ok: Schema.Boolean,
  }),
  execute: () => Effect.succeed({ ok: true }),
  label: ({ id }) => `strict_title:${id}`,
})

const strictTitleBinding: XmlTagBinding = {
  tag: 'strict_title',
  attributes: [{ field: 'id', attr: 'id' }],
  childTags: [{ field: 'title', tag: 'title' }],
}

describe('tool validation scenarios (strict attr/body placement)', () => {
  test('1) valid usage — all fields in canonical positions', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const xml = responseWithActions(
      `<agent_create_like id="a1"><type>explorer</type><title>Hello</title><message>Do work</message></agent_create_like>`,
    )
    const events = await runStream(cfg, xml)

    expect(eventsOfType(events, 'ToolInputParseError')).toHaveLength(0)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)
    expect(eventsOfType(events, 'ToolExecutionEnded')).toHaveLength(1)

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    expect(ready[0].input).toMatchObject({
      id: 'a1',
      type: 'explorer',
      title: 'Hello',
      message: 'Do work',
    })
  })

  test('2) child-form attr target is not normalized', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const xml = responseWithActions(
      `<agent_create_like><id>a1</id><type>explorer</type><title>Hello</title><message>Do work</message></agent_create_like>`,
    )
    const events = await runStream(cfg, xml)

    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('MissingRequiredFields')
  })

  test('3) attr-form childTag target is not normalized', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const xml = responseWithActions(
      `<agent_create_like id="a1" type="explorer"><title>Hello</title><message>Do work</message></agent_create_like>`,
    )
    const events = await runStream(cfg, xml)

    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('UnknownAttribute')
  })

  test('4) canonical attribute still wins even if matching child appears in body', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const xml = responseWithActions(
      `<agent_create_like id="foo"><id>bar</id><type>explorer</type><title>Hello</title><message>Do work</message></agent_create_like>`,
    )
    const events = await runStream(cfg, xml)

    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)

    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    expect(ready[0].input).toMatchObject({ id: 'foo' })
  })

  test('5) multiple child tags for attr target remain invalid', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const xml = responseWithActions(
      `<agent_create_like><id>a1</id><id>a2</id><type>explorer</type><title>Hello</title><message>Do work</message></agent_create_like>`,
    )
    const events = await runStream(cfg, xml)

    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('MissingRequiredFields')
  })

  test('6) child with attributes is not eligible for attr binding', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const xml = responseWithActions(
      `<agent_create_like><id source="x">a1</id><type>explorer</type><title>Hello</title><message>Do work</message></agent_create_like>`,
    )
    const events = await runStream(cfg, xml)

    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('MissingRequiredFields')
  })

  test('7) unknown attribute still errors', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const xml = responseWithActions(`<shell_like foo="bar">echo hi</shell_like>`)
    const events = await runStream(cfg, xml)

    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('UnknownAttribute')
  })

  test('8) missing required fields', async () => {
    const cfg = config([registered(readLikeTool, 'read_like', readLikeBinding)])
    const events = await runStream(cfg, responseWithActions(`<read_like />`))

    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('MissingRequiredFields')
  })

  test('9) schema validation failure → ToolValidationFailed', async () => {
    const cfg = config([registered(strictTitleTool, 'strict_title', strictTitleBinding)])
    const xml = responseWithActions(`<strict_title id="a1"><title>bad</title></strict_title>`)
    const events = await runStream(cfg, xml)

    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('ToolValidationFailed')
  })

  test('10) dead-call gating — parse error blocks execution', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const xml = responseWithActions(`<shell_like timeout="wat">echo hi</shell_like>`)
    const events = await runStream(cfg, xml)

    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors.length).toBeGreaterThan(0)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
    expect(eventsOfType(events, 'ToolExecutionEnded')).toHaveLength(0)
  })

  test('11) child-form number attr is preserved as literal body text for body tools', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const xml = responseWithActions(`<shell_like><timeout>5</timeout>echo hi</shell_like>`)
    const events = await runStream(cfg, xml)

    expect(eventsOfType(events, 'ToolInputParseError')).toHaveLength(0)
    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    expect(ready[0].input).toMatchObject({ command: '<timeout>5</timeout>echo hi' })
    expect((ready[0].input as { timeout?: unknown }).timeout).toBeUndefined()
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)
  })

  test('12) invalid child-form number attr is likewise preserved as literal body text', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const xml = responseWithActions(`<shell_like><timeout>abc</timeout>echo hi</shell_like>`)
    const events = await runStream(cfg, xml)

    expect(eventsOfType(events, 'ToolInputParseError')).toHaveLength(0)
    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    expect(ready[0].input).toMatchObject({ command: '<timeout>abc</timeout>echo hi' })
    expect((ready[0].input as { timeout?: unknown }).timeout).toBeUndefined()
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)
  })

  test('13) child-form boolean attr is preserved as literal body text for body tools', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const xml = responseWithActions(`<shell_like><recursive>true</recursive>echo hi</shell_like>`)
    const events = await runStream(cfg, xml)

    expect(eventsOfType(events, 'ToolInputParseError')).toHaveLength(0)
    const ready = eventsOfType(events, 'ToolInputReady')
    expect(ready).toHaveLength(1)
    expect(ready[0].input).toMatchObject({ command: '<recursive>true</recursive>echo hi' })
    expect((ready[0].input as { recursive?: unknown }).recursive).toBeUndefined()
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)
  })

  test('14) attr-only tools do not accept child-form attrs', async () => {
    const cfg = config([registered(grepLikeTool, 'grep_like', grepLikeBinding)])
    const xml = responseWithActions(`<grep_like><pattern>foo</pattern></grep_like>`)
    const events = await runStream(cfg, xml)

    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('UnexpectedBody')
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
  })

  test('15) malformed top-level known tool syntax should surface ToolInputParseError (not prose fallback)', async () => {
    const cfg = config([registered(grepLikeTool, 'grep_like', grepLikeBinding)])
    const xml = responseWithActions(`<grep_like pattern="from ['"]foo['"]" path="src/**/*.{ts,tsx}"/>`)
    const events = await runStream(cfg, xml)

    expect(eventsOfType(events, 'ToolInputParseError').length).toBeGreaterThan(0)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)

    const prose = eventsOfType(events, 'ProseChunk').map(e => e.text).join('')
    const messages = eventsOfType(events, 'MessageChunk').map(e => e.text).join('')
    expect(prose).not.toContain('<grep_like')
    expect(messages).not.toContain('<grep_like')
  })

  test('16) malformed known tool remains a tool-parse error when streamed across chunks', async () => {
    const cfg = config([registered(grepLikeTool, 'grep_like', grepLikeBinding)])
    const xml = `<lens name="turn">planning</lens>\n<grep_like pattern="from ['"]foo['"]" path="src"\n/><idle/>`
    const events = await runStream(cfg, xml)

    expect(eventsOfType(events, 'ToolInputParseError').length).toBeGreaterThan(0)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)

    const prose = eventsOfType(events, 'ProseChunk').map(e => e.text).join('')
    expect(prose).not.toContain('<grep_like')
  })
})
