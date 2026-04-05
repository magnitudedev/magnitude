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
  return Effect.runPromise(Stream.runCollect(runtime.streamWith(Stream.make(xml)))).then(c => Array.from(c))
}

function runStreamChunks(config: XmlRuntimeConfig, chunks: string[]): Promise<XmlRuntimeEvent[]> {
  const runtime = createXmlRuntime(config)
  return Effect.runPromise(Stream.runCollect(runtime.streamWith(Stream.fromIterable(chunks)))).then(c => Array.from(c))
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
  return { tools: new Map(tools.map(t => [t.tagName, t])) }
}

function responseWithActions(actionsXml: string): string {
  return `<lens name="turn">planning</lens>\n${actionsXml}<idle/>`
}

function toolEvents(events: XmlRuntimeEvent[]): XmlRuntimeEvent[] {
  return events.filter(e => e._tag.startsWith('ToolInput') || e._tag.startsWith('ToolExecution'))
}

function eventTags(events: XmlRuntimeEvent[]): string[] {
  return toolEvents(events).map(e => e._tag)
}

function eventsOfType<T extends XmlRuntimeEvent['_tag']>(
  events: XmlRuntimeEvent[],
  tag: T,
): Extract<XmlRuntimeEvent, { _tag: T }>[] {
  return events.filter(e => e._tag === tag) as Extract<XmlRuntimeEvent, { _tag: T }>[]
}

function callEvents(events: XmlRuntimeEvent[], toolCallId: string): XmlRuntimeEvent[] {
  return events.filter(e => 'toolCallId' in e && e.toolCallId === toolCallId)
}

const agentCreateLikeTool = defineTool({
  name: 'agent_create_like',
  description: 'agent-create like',
  inputSchema: Schema.Struct({
    id: Schema.String,
    type: Schema.String,
    title: Schema.String,
    message: Schema.String,
  }),
  outputSchema: Schema.Struct({ ok: Schema.Boolean }),
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

const pathModeTool = defineTool({
  name: 'path_mode',
  description: 'attrs-only tool',
  inputSchema: Schema.Struct({
    path: Schema.String,
    mode: Schema.String,
  }),
  outputSchema: Schema.Struct({ ok: Schema.Boolean }),
  execute: () => Effect.succeed({ ok: true }),
  label: ({ path, mode }) => `path_mode:${path}:${mode}`,
})

const pathModeBinding: XmlTagBinding = {
  tag: 'path_mode',
  attributes: [
    { field: 'path', attr: 'path' },
    { field: 'mode', attr: 'mode' },
  ],
}

const shellLikeTool = defineTool({
  name: 'shell_like',
  description: 'body tool',
  inputSchema: Schema.Struct({
    command: Schema.String,
    timeout: Schema.optional(Schema.Number),
    recursive: Schema.optional(Schema.Boolean),
  }),
  outputSchema: Schema.Struct({ stdout: Schema.String, exitCode: Schema.Number }),
  execute: ({ command }) => Effect.succeed({ stdout: command, exitCode: 0 }),
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

const strictTitleTool = defineTool({
  name: 'strict_title',
  description: 'schema failure test',
  inputSchema: Schema.Struct({
    id: Schema.String,
    title: Schema.String.pipe(Schema.minLength(5)),
  }),
  outputSchema: Schema.Struct({ ok: Schema.Boolean }),
  execute: () => Effect.succeed({ ok: true }),
  label: ({ id }) => `strict_title:${id}`,
})

const strictTitleBinding: XmlTagBinding = {
  tag: 'strict_title',
  attributes: [{ field: 'id', attr: 'id' }],
  childTags: [{ field: 'title', tag: 'title' }],
}

describe('streaming normalization (attr ↔ childTag) event sequences', () => {
  test('A1 canonical positions emit canonical streaming sequence', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const xml = responseWithActions(
      `<agent_create_like id="a1"><type>explorer</type><title>Hello</title><message>Do work</message></agent_create_like>`,
    )
    const events = await runStream(cfg, xml)

    expect(eventTags(events)).toEqual([
      'ToolInputStarted',
      'ToolInputFieldValue',
      'ToolInputChildStarted',
      'ToolInputBodyChunk',
      'ToolInputChildComplete',
      'ToolInputChildStarted',
      'ToolInputBodyChunk',
      'ToolInputChildComplete',
      'ToolInputChildStarted',
      'ToolInputBodyChunk',
      'ToolInputChildComplete',
      'ToolInputReady',
      'ToolExecutionStarted',
      'ToolExecutionEnded',
    ])
  })

  test('A2 body tool canonical usage emits ToolInputBodyChunk', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const events = await runStream(cfg, responseWithActions(`<shell_like timeout="5">echo hi</shell_like>`))
    expect(eventTags(events)).toEqual([
      'ToolInputStarted',
      'ToolInputFieldValue',
      'ToolInputBodyChunk',
      'ToolInputReady',
      'ToolExecutionStarted',
      'ToolExecutionEnded',
    ])
  })

  test('B3 child->attr single swap emits field value (no child events for swapped field)', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(
        `<agent_create_like><id>a1</id><type>explorer</type><title>Hello</title><message>Do work</message></agent_create_like>`,
      ),
    )

    const started = eventsOfType(events, 'ToolInputStarted')[0]
    const byCall = callEvents(events, started.toolCallId)
    const idChildEvents = byCall.filter(
      e => (e._tag === 'ToolInputChildStarted' || e._tag === 'ToolInputChildComplete') && e.field === 'id',
    )
    const idFieldValues = byCall.filter(e => e._tag === 'ToolInputFieldValue' && e.field === 'id')
    expect(idChildEvents).toHaveLength(0)
    expect(idFieldValues).toHaveLength(1)
    const idField = idFieldValues[0]
    expect(idField?._tag).toBe('ToolInputFieldValue')
    if (!idField || idField._tag !== 'ToolInputFieldValue') throw new Error('Expected ToolInputFieldValue')
    expect(idField.value).toBe('a1')

    const ready = eventsOfType(events, 'ToolInputReady')[0]
    expect((ready.input as { id: string }).id).toBe('a1')
  })

  test('B4 child->attr all attrs swapped as child tags each emit ToolInputFieldValue', async () => {
    const cfg = config([registered(pathModeTool, 'path_mode', pathModeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<path_mode><path>/tmp/a</path><mode>r</mode></path_mode>`),
    )
    expect(eventTags(events)).toEqual([
      'ToolInputStarted',
      'ToolInputFieldValue',
      'ToolInputFieldValue',
      'ToolInputReady',
      'ToolExecutionStarted',
      'ToolExecutionEnded',
    ])
  })

  test('B5 mixed canonical attrs + swapped children: timing split is preserved', async () => {
    const cfg = config([registered(pathModeTool, 'path_mode', pathModeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<path_mode path="/tmp/a"><mode>r</mode></path_mode>`),
    )

    const fields = eventsOfType(events, 'ToolInputFieldValue')
    expect(fields).toHaveLength(2)
    expect(fields[0].field).toBe('path')
    expect(fields[1].field).toBe('mode')
  })

  test('C6 attr->child single swap emits synthetic child start/complete and no field value', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(
        `<agent_create_like id="a1" type="explorer"><title>Hello</title><message>Do work</message></agent_create_like>`,
      ),
    )
    const started = eventsOfType(events, 'ToolInputStarted')[0]
    const byCall = callEvents(events, started.toolCallId)

    const typeFieldValues = byCall.filter(e => e._tag === 'ToolInputFieldValue' && e.field === 'type')
    const typeChildStarts = byCall.filter(e => e._tag === 'ToolInputChildStarted' && e.field === 'type')
    const typeChildCompletes = byCall.filter(e => e._tag === 'ToolInputChildComplete' && e.field === 'type')

    expect(typeFieldValues).toHaveLength(0)
    expect(typeChildStarts).toHaveLength(1)
    expect(typeChildCompletes).toHaveLength(1)

    const ready = eventsOfType(events, 'ToolInputReady')[0]
    expect((ready.input as { type: string }).type).toBe('explorer')
  })

  test('C7 attr->child multiple swapped fields each emit synthetic child events', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(
        `<agent_create_like id="a1" type="explorer" title="Hello" message="Do work"></agent_create_like>`,
      ),
    )
    expect(eventTags(events)).toEqual([
      'ToolInputStarted',
      'ToolInputFieldValue',
      'ToolInputChildStarted',
      'ToolInputChildComplete',
      'ToolInputChildStarted',
      'ToolInputChildComplete',
      'ToolInputChildStarted',
      'ToolInputChildComplete',
      'ToolInputReady',
      'ToolExecutionStarted',
      'ToolExecutionEnded',
    ])
  })

  test('C8 self-closing tag with swapped childTag attrs emits canonical child events', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<agent_create_like id="a1" type="explorer" title="Hello" message="Do work" />`),
    )
    expect(eventsOfType(events, 'ToolInputChildStarted')).toHaveLength(3)
    expect(eventsOfType(events, 'ToolInputChildComplete')).toHaveLength(3)
    expect(eventsOfType(events, 'ToolInputReady')).toHaveLength(1)
  })

  test('D9 mixed both directions yields canonical streaming sequence', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(
        `<agent_create_like type="explorer"><id>a1</id><title>Hello</title><message>Do work</message></agent_create_like>`,
      ),
    )
    expect(eventTags(events)).toEqual([
      'ToolInputStarted',
      'ToolInputChildStarted',
      'ToolInputChildComplete',
      'ToolInputFieldValue',
      'ToolInputChildStarted',
      'ToolInputBodyChunk',
      'ToolInputChildComplete',
      'ToolInputChildStarted',
      'ToolInputBodyChunk',
      'ToolInputChildComplete',
      'ToolInputReady',
      'ToolExecutionStarted',
      'ToolExecutionEnded',
    ])
  })

  test('E10 canonical attr + swapped child duplicate: canonical attr wins, duplicate suppressed', async () => {
    const cfg = config([registered(pathModeTool, 'path_mode', pathModeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<path_mode path="/tmp/a" mode="r"><path>/tmp/b</path></path_mode>`),
    )
    const pathFieldValues = eventsOfType(events, 'ToolInputFieldValue').filter(e => e.field === 'path')
    expect(pathFieldValues).toHaveLength(1)
    const pathField = pathFieldValues[0]
    expect(pathField?._tag).toBe('ToolInputFieldValue')
    if (!pathField || pathField._tag !== 'ToolInputFieldValue') throw new Error('Expected ToolInputFieldValue')
    expect(pathField.value).toBe('/tmp/a')
  })

  test('E11 swapped attr + canonical child duplicate: duplicate child is ignored after first canonical emission', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(
        `<agent_create_like id="a1" type="explorer"><type>explorer</type><title>Hello</title><message>Do work</message></agent_create_like>`,
      ),
    )

    const typeChildStarted = eventsOfType(events, 'ToolInputChildStarted').filter(e => e.field === 'type')
    const typeChildComplete = eventsOfType(events, 'ToolInputChildComplete').filter(e => e.field === 'type')
    expect(typeChildStarted).toHaveLength(1)
    expect(typeChildComplete).toHaveLength(1)
    expect((typeChildComplete[0].value as { type: string }).type).toBe('explorer')

    const ready = eventsOfType(events, 'ToolInputReady')[0]
    expect((ready.input as { type: string }).type).toBe('explorer')
  })

  test('F12 child with attributes is not eligible for child->attr normalization', async () => {
    const cfg = config([registered(pathModeTool, 'path_mode', pathModeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<path_mode><path kind="x">/tmp/a</path><mode>r</mode></path_mode>`),
    )
    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('MissingRequiredFields')
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
  })

  test('F13 multiple children for attr target are not normalized (MissingRequiredFields)', async () => {
    const cfg = config([registered(pathModeTool, 'path_mode', pathModeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<path_mode><path>/a</path><path>/b</path><mode>r</mode></path_mode>`),
    )
    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('MissingRequiredFields')
  })

  test('F14 empty child body normalizes to empty string for attr target', async () => {
    const cfg = config([registered(pathModeTool, 'path_mode', pathModeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<path_mode><path></path><mode>r</mode></path_mode>`),
    )
    const ready = eventsOfType(events, 'ToolInputReady')[0]
    expect((ready.input as { path: string }).path).toBe('')
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)
  })

  test('F15 chunk-split streaming yields same canonical event tag sequence as single chunk', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const chunks = [
      `<lens name="turn">planning</lens>\n<agent_create_like id="a1" type="explorer">`,
      `<title>Hello</title>`,
      `<message>Do work</message></agent_create_like>`,
      `<idle/>`,
    ]
    const single = await runStream(cfg, chunks.join(''))
    const split = await runStreamChunks(cfg, chunks)

    expect(eventTags(split)).toEqual(eventTags(single))
  })

  test('G16 child->attr normalization coerces streaming field value to number', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const events = await runStream(cfg, responseWithActions(`<shell_like><timeout>5</timeout>echo hi</shell_like>`))

    const timeoutField = eventsOfType(events, 'ToolInputFieldValue').find(e => e.field === 'timeout')
    expect(timeoutField?.value).toBe(5)

    const ready = eventsOfType(events, 'ToolInputReady')[0]
    expect((ready.input as { timeout?: unknown }).timeout).toBe(5)
  })

  test('G17 child->attr normalization coercion failure emits parse error and suppresses field', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const events = await runStream(cfg, responseWithActions(`<shell_like><timeout>wat</timeout>echo hi</shell_like>`))

    const timeoutField = eventsOfType(events, 'ToolInputFieldValue').find(e => e.field === 'timeout')
    expect(timeoutField).toBeUndefined()

    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('InvalidAttributeValue')
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
  })

  test('G18 child->attr normalization coerces streaming field value to boolean', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<shell_like><recursive>true</recursive>echo hi</shell_like>`),
    )

    const recursiveField = eventsOfType(events, 'ToolInputFieldValue').find(e => e.field === 'recursive')
    expect(recursiveField?.value).toBe(true)

    const ready = eventsOfType(events, 'ToolInputReady')[0]
    expect((ready.input as { recursive?: unknown }).recursive).toBe(true)
  })

  test('G19 child->attr normalization coerces "0" to number 0', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<shell_like><timeout>0</timeout>echo hi</shell_like>`),
    )

    const timeoutField = eventsOfType(events, 'ToolInputFieldValue').find(e => e.field === 'timeout')
    expect(timeoutField?.value).toBe(0)

    const ready = eventsOfType(events, 'ToolInputReady')[0]
    expect((ready.input as { timeout?: unknown }).timeout).toBe(0)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)
  })

  test('G20 child->attr normalization coerces "false" to boolean false', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<shell_like><recursive>false</recursive>echo hi</shell_like>`),
    )

    const recursiveField = eventsOfType(events, 'ToolInputFieldValue').find(e => e.field === 'recursive')
    expect(recursiveField?.value).toBe(false)

    const ready = eventsOfType(events, 'ToolInputReady')[0]
    expect((ready.input as { recursive?: unknown }).recursive).toBe(false)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)
  })

  test('H19 validation: unknown attribute + missing required + schema validation + dead-call gating', async () => {
    const cfgUnknown = config([registered(pathModeTool, 'path_mode', pathModeBinding)])
    const unknown = await runStream(cfgUnknown, responseWithActions(`<path_mode nope="x" path="/a" mode="r" />`))
    expect(eventsOfType(unknown, 'ToolInputParseError')[0].error._tag).toBe('UnknownAttribute')
    expect(eventsOfType(unknown, 'ToolExecutionStarted')).toHaveLength(0)

    const missing = await runStream(cfgUnknown, responseWithActions(`<path_mode path="/a" />`))
    expect(eventsOfType(missing, 'ToolInputParseError')[0].error._tag).toBe('MissingRequiredFields')
    expect(eventsOfType(missing, 'ToolExecutionStarted')).toHaveLength(0)

    const cfgSchema = config([registered(strictTitleTool, 'strict_title', strictTitleBinding)])
    const schemaBad = await runStream(
      cfgSchema,
      responseWithActions(`<strict_title id="a1"><title>bad</title></strict_title>`),
    )
    expect(eventsOfType(schemaBad, 'ToolInputParseError')[0].error._tag).toBe('ToolValidationFailed')
    expect(eventsOfType(schemaBad, 'ToolExecutionStarted')).toHaveLength(0)

    const dead = await runStream(cfgUnknown, responseWithActions(`<path_mode path="/a" mode="r" extra="1"></path_mode>`))
    expect(eventsOfType(dead, 'ToolInputParseError').length).toBeGreaterThan(0)
    expect(eventsOfType(dead, 'ToolInputReady')).toHaveLength(0)
    expect(eventsOfType(dead, 'ToolExecutionStarted')).toHaveLength(0)
    expect(eventsOfType(dead, 'ToolExecutionEnded')).toHaveLength(0)
  })
})
