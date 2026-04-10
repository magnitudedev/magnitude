import { describe, it, expect } from 'vitest'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'

function parse(xml: string, knownTags = new Set(['create-task', 'update-task', 'spawn-worker', 'kill-worker'])): ParseEvent[] {
  const parser = createStreamingXmlParser(
    knownTags,
    new Map(),
    new Map([
      ['create-task', { acceptsBody: false, attributes: new Map([['id', { type: 'string', required: true }]]), children: new Map() }],
      ['update-task', { acceptsBody: false, attributes: new Map([['id', { type: 'string', required: true }], ['status', { type: 'string', required: true }]]), children: new Map() }],
      ['spawn-worker', { acceptsBody: false, attributes: new Map([['id', { type: 'string', required: true }], ['role', { type: 'string', required: true }]]), children: new Map() }],
      ['kill-worker', { acceptsBody: false, attributes: new Map([['id', { type: 'string', required: true }]]), children: new Map() }],
    ]),
  )
  return [...parser.processChunk(xml), ...parser.flush()]
}

describe('flat task/worker tool parsing', () => {
  it('treats <task> blocks as unknown (no task events)', () => {
    const events = parse('<task id="t1"><message>nope</message></task><idle/>')
    expect(events.some((e) => e._tag === 'TagOpened' && e.tagName === 'task')).toBe(false)
    expect(events.some((e) => e._tag === 'ParseError')).toBe(false)
  })

  it('parses flat create/update/spawn/kill tags as regular tools', () => {
    const events = parse('<create-task id="t1" />\n<update-task id="t1" status="working" />\n<spawn-worker id="t1" role="explorer" />\n<kill-worker id="t1" />\n<idle/>')
    const opened = events.filter((e): e is Extract<ParseEvent, { _tag: 'TagOpened' }> => e._tag === 'TagOpened').map((e) => e.tagName)
    expect(opened).toEqual(['create-task', 'update-task', 'spawn-worker', 'kill-worker'])
  })

  it('spawn-worker with body is still parsed as a regular tool tag at format layer', () => {
    const events = parse('<spawn-worker id="t1" role="explorer">body</spawn-worker>\n<idle/>')
    expect(events.some((e) => e._tag === 'TagOpened' && e.tagName === 'spawn-worker')).toBe(true)
    expect(events.some((e) => e._tag === 'TagClosed' && e.tagName === 'spawn-worker')).toBe(true)
  })
})
