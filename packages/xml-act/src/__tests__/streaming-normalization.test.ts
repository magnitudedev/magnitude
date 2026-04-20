import { describe, expect, test } from 'vitest'
import { Effect, Stream } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, type ToolDefinition } from '@magnitudedev/tools'
import {
  createXmlRuntime,
  type RegisteredTool,
  type XmlRuntimeConfig,
  type XmlTurnEngineEvent,
  type XmlTagBinding,
} from '../index'

function runStream(config: XmlRuntimeConfig, xml: string): Promise<XmlTurnEngineEvent[]> {
  const runtime = createXmlRuntime(config)
  return Effect.runPromise(Stream.runCollect(runtime.streamWith(Stream.make(xml)))).then(c => Array.from(c))
}

function runStreamChunks(config: XmlRuntimeConfig, chunks: string[]): Promise<XmlTurnEngineEvent[]> {
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

function toolEvents(events: XmlTurnEngineEvent[]): XmlTurnEngineEvent[] {
  return events.filter(e => e._tag.startsWith('ToolInput') || e._tag.startsWith('ToolExecution'))
}

function eventTags(events: XmlTurnEngineEvent[]): string[] {
  return toolEvents(events).map(e => e._tag)
}

function eventsOfType<T extends XmlTurnEngineEvent['_tag']>(
  events: XmlTurnEngineEvent[],
  tag: T,
): Extract<XmlTurnEngineEvent, { _tag: T }>[] {
  return events.filter(e => e._tag === tag) as Extract<XmlTurnEngineEvent, { _tag: T }>[]
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

describe('streaming tool input behavior without attr/child normalization', () => {
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

  test('B3 child-form attr causes missing required fields after canonical children stream normally', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(
        `<agent_create_like><id>a1</id><type>explorer</type><title>Hello</title><message>Do work</message></agent_create_like>`,
      ),
    )

    expect(eventTags(events)).toEqual([
      'ToolInputStarted',
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
      'ToolInputParseError',
    ])
    expect(eventsOfType(events, 'ToolInputReady')).toHaveLength(1)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
    expect(eventsOfType(events, 'ToolInputParseError')[0].error._tag).toBe('MissingRequiredFields')
  })

  test('B4 attr-only tool with child-form attrs yields UnexpectedBody', async () => {
    const cfg = config([registered(pathModeTool, 'path_mode', pathModeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<path_mode><mode>r</mode></path_mode>`),
    )
    expect(eventTags(events)).toEqual([
      'ToolInputStarted',
      'ToolInputParseError',
    ])
    expect(eventsOfType(events, 'ToolInputParseError')[0].error._tag).toBe('UnexpectedBody')
  })

  test('B5 mixed canonical attr + child-form attr still fails strictly', async () => {
    const cfg = config([registered(pathModeTool, 'path_mode', pathModeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<path_mode path="/tmp/a"><mode>r</mode></path_mode>`),
    )

    expect(eventTags(events)).toEqual([
      'ToolInputStarted',
      'ToolInputFieldValue',
      'ToolInputParseError',
    ])
    expect(eventsOfType(events, 'ToolInputParseError')[0].error._tag).toBe('UnexpectedBody')
  })

  test('C6 attr-form childTag target emits UnknownAttribute before tool input starts', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(
        `<agent_create_like id="a1" type="explorer"><title>Hello</title><message>Do work</message></agent_create_like>`,
      ),
    )

    expect(eventTags(events)).toEqual([
      'ToolInputParseError',
      'ToolInputStarted',
      'ToolInputFieldValue',
    ])
    expect(eventsOfType(events, 'ToolInputParseError')[0].error._tag).toBe('UnknownAttribute')
  })

  test('C7 multiple childTag fields in attr position collapse to one UnknownAttribute before start', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(
        `<agent_create_like id="a1" type="explorer" title="Hello" message="Do work"></agent_create_like>`,
      ),
    )
    expect(eventTags(events)).toEqual([
      'ToolInputParseError',
      'ToolInputStarted',
      'ToolInputFieldValue',
    ])
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
  })

  test('C8 self-closing tag with childTag attrs is rejected with one UnknownAttribute', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<agent_create_like id="a1" type="explorer" title="Hello" message="Do work" />`),
    )
    expect(eventsOfType(events, 'ToolInputChildStarted')).toHaveLength(0)
    expect(eventsOfType(events, 'ToolInputChildComplete')).toHaveLength(0)
    expect(eventsOfType(events, 'ToolInputReady')).toHaveLength(0)
    expect(eventsOfType(events, 'ToolInputParseError')).toHaveLength(1)
    expect(eventsOfType(events, 'ToolInputParseError')[0].error._tag).toBe('UnknownAttribute')
  })

  test('D9 mixed both directions now fails without normalization', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(
        `<agent_create_like type="explorer"><id>a1</id><title>Hello</title><message>Do work</message></agent_create_like>`,
      ),
    )
    expect(eventsOfType(events, 'ToolInputParseError').length).toBeGreaterThan(0)
    expect(eventsOfType(events, 'ToolInputReady')).toHaveLength(0)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
  })

  test('E10 canonical attrs in canonical form still execute cleanly', async () => {
    const cfg = config([registered(pathModeTool, 'path_mode', pathModeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<path_mode path="/tmp/a" mode="r"></path_mode>`),
    )
    const pathFieldValues = eventsOfType(events, 'ToolInputFieldValue').filter(e => e.field === 'path')
    expect(pathFieldValues).toHaveLength(1)
    const pathField = pathFieldValues[0]
    expect(pathField?._tag).toBe('ToolInputFieldValue')
    if (!pathField || pathField._tag !== 'ToolInputFieldValue') throw new Error('Expected ToolInputFieldValue')
    expect(pathField.value).toBe('/tmp/a')
    expect(eventsOfType(events, 'ToolInputParseError')).toHaveLength(0)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)
  })

  test('E11 canonical child is preserved; attr-form duplicate child is rejected separately', async () => {
    const cfg = config([registered(agentCreateLikeTool, 'agent_create_like', agentCreateLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(
        `<agent_create_like id="a1" type="explorer"><type>explorer</type><title>Hello</title><message>Do work</message></agent_create_like>`,
      ),
    )

    expect(eventsOfType(events, 'ToolInputParseError')).toHaveLength(1)
    expect(eventsOfType(events, 'ToolInputReady')).toHaveLength(0)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
  })

  test('F12 child with attributes on attr-only tool yields UnexpectedBody', async () => {
    const cfg = config([registered(pathModeTool, 'path_mode', pathModeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<path_mode><mode>r</mode></path_mode>`),
    )
    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('UnexpectedBody')
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
  })

  test('F13 multiple child-form attrs on attr-only tool still yield UnexpectedBody', async () => {
    const cfg = config([registered(pathModeTool, 'path_mode', pathModeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<path_mode><mode>r</mode></path_mode>`),
    )
    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(1)
    expect(parseErrors[0].error._tag).toBe('UnexpectedBody')
  })

  test('F14 empty child body is not normalized for attr-only tools', async () => {
    const cfg = config([registered(pathModeTool, 'path_mode', pathModeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<path_mode><mode>r</mode></path_mode>`),
    )
    expect(eventsOfType(events, 'ToolInputReady')).toHaveLength(0)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(0)
    expect(eventsOfType(events, 'ToolInputParseError')[0].error._tag).toBe('UnexpectedBody')
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

  test('G16 child-form number attr does not emit streaming field values but command still executes', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const events = await runStream(cfg, responseWithActions(`<shell_like><timeout>5</timeout>echo hi</shell_like>`))

    const timeoutField = eventsOfType(events, 'ToolInputFieldValue').find(e => e.field === 'timeout')
    expect(timeoutField).toBeUndefined()
    expect(eventsOfType(events, 'ToolInputReady')).toHaveLength(1)
    expect(eventsOfType(events, 'ToolInputParseError')).toHaveLength(0)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)
  })

  test('G17 invalid child-form number attr is also ignored without coercion', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const events = await runStream(cfg, responseWithActions(`<shell_like><timeout>wat</timeout>echo hi</shell_like>`))

    const timeoutField = eventsOfType(events, 'ToolInputFieldValue').find(e => e.field === 'timeout')
    expect(timeoutField).toBeUndefined()

    const parseErrors = eventsOfType(events, 'ToolInputParseError')
    expect(parseErrors).toHaveLength(0)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)
  })

  test('G18 child-form boolean attr does not emit streaming field values but command still executes', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<shell_like><recursive>true</recursive>echo hi</shell_like>`),
    )

    const recursiveField = eventsOfType(events, 'ToolInputFieldValue').find(e => e.field === 'recursive')
    expect(recursiveField).toBeUndefined()
    expect(eventsOfType(events, 'ToolInputReady')).toHaveLength(1)
    expect(eventsOfType(events, 'ToolInputParseError')).toHaveLength(0)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)
  })

  test('G19 child-form zero number attr is ignored without normalization', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<shell_like><timeout>0</timeout>echo hi</shell_like>`),
    )

    const timeoutField = eventsOfType(events, 'ToolInputFieldValue').find(e => e.field === 'timeout')
    expect(timeoutField).toBeUndefined()

    expect(eventsOfType(events, 'ToolInputReady')).toHaveLength(1)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)
    expect(eventsOfType(events, 'ToolInputParseError')).toHaveLength(0)
  })

  test('G20 child-form false boolean attr is ignored without normalization', async () => {
    const cfg = config([registered(shellLikeTool, 'shell_like', shellLikeBinding)])
    const events = await runStream(
      cfg,
      responseWithActions(`<shell_like><recursive>false</recursive>echo hi</shell_like>`),
    )

    const recursiveField = eventsOfType(events, 'ToolInputFieldValue').find(e => e.field === 'recursive')
    expect(recursiveField).toBeUndefined()

    expect(eventsOfType(events, 'ToolInputReady')).toHaveLength(1)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)
    expect(eventsOfType(events, 'ToolInputParseError')).toHaveLength(0)
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
