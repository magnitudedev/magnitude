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

const ACTIONS_CLOSE = '</' + 'actions>'
const LENSES_CLOSE = '</' + 'lenses>'
const COMMS_CLOSE = '</' + 'comms>'

function runStream(config: XmlRuntimeConfig, xml: string): Promise<XmlRuntimeEvent[]> {
  const runtime = createXmlRuntime(config)
  const stream = runtime.streamWith(Stream.make(xml))
  return Effect.runPromise(Stream.runCollect(stream)).then(c => Array.from(c))
}

function registered(tool: ToolDefinition, tagName: string, binding: XmlTagBinding): RegisteredTool {
  return { tool, tagName, groupName: 'default', binding }
}

function config(tools: RegisteredTool[]): XmlRuntimeConfig {
  return { tools: new Map(tools.map(t => [t.tagName, t])) }
}

function responseWithActions(actionsXml: string): string {
  return `<lenses>${LENSES_CLOSE}<comms>${COMMS_CLOSE}<actions>${actionsXml}${ACTIONS_CLOSE}<yield/>`
}

function turnFailure(events: XmlRuntimeEvent[]): string {
  const turnEnd = [...events].reverse().find((e) => e._tag === 'TurnEnd')
  if (!turnEnd || turnEnd.result._tag !== 'Failure') {
    throw new Error('Expected TurnEnd failure')
  }
  return turnEnd.result.error
}

describe('repro: tool defect message formatting', () => {
  test('plain object defect with _tag/name/message includes structured fields', async () => {
    const defect = { _tag: 'Abort', name: 'BamlError', message: 'stream closed' }
    const tool = defineTool({
      name: 'defect_tool',
      description: 'dies with plain object defect',
      inputSchema: Schema.Struct({}),
      outputSchema: Schema.Struct({ ok: Schema.Boolean }),
      execute: () => Effect.die(defect),
      label: () => 'defect_tool',
    })

    const toolClose = '</' + 'defect_tool>'
    const events = await runStream(
      config([registered(tool, 'defect_tool', { tag: 'defect_tool' })]),
      responseWithActions(`<defect_tool>${toolClose}`),
    )

    const message = turnFailure(events)
    expect(message).toContain('Tool defect')
    expect(message).toContain('[Abort]')
    expect(message).toContain('BamlError')
    expect(message).toContain('stream closed')
    expect(message).not.toContain('[object Object]')
  })

  test('plain object defect without message/name falls back to JSON', async () => {
    const defect = { foo: 'bar' }
    const tool = defineTool({
      name: 'defect_tool_json',
      description: 'dies with unstructured plain object defect',
      inputSchema: Schema.Struct({}),
      outputSchema: Schema.Struct({ ok: Schema.Boolean }),
      execute: () => Effect.die(defect),
      label: () => 'defect_tool_json',
    })

    const toolClose = '</' + 'defect_tool_json>'
    const events = await runStream(
      config([registered(tool, 'defect_tool_json', { tag: 'defect_tool_json' })]),
      responseWithActions(`<defect_tool_json>${toolClose}`),
    )

    const message = turnFailure(events)
    expect(message).toContain('Tool defect')
    expect(message).toContain('{"foo":"bar"}')
    expect(message).not.toContain('[object Object]')
  })
})