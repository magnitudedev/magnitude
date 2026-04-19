import { describe, it, expect } from 'vitest'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'
import { YIELD_USER, YIELD_TOOL, YIELD_WORKER, YIELD_PARENT, SUBAGENT_YIELD_TAGS } from '../constants'

const knownTags = new Set(['shell', 'read', 'message'])
const childTagMap = new Map<string, Set<string>>()

function parse(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  return [...parser.processChunk(xml), ...parser.flush()]
}

function turnControls(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'TurnControl' }> => e._tag === 'TurnControl')
}

describe('yield parsing behavior', () => {
  it('yield-user is parsed structurally', () => {
    const events = parse(YIELD_USER)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
  })

  it('yield-tool is parsed structurally', () => {
    const events = parse(YIELD_TOOL)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('tool')
  })

  it('yield-worker is parsed structurally', () => {
    const events = parse(YIELD_WORKER)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('worker')
  })

  it('yield-parent is parsed structurally (with subagent yield tags)', () => {
    const parser = createStreamingXmlParser(knownTags, childTagMap, undefined, undefined, undefined, SUBAGENT_YIELD_TAGS)
    const events = [...parser.processChunk(YIELD_PARENT), ...parser.flush()]
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('parent')
  })

  it('observing mode is entered after yield-user — trailing content is not parsed', () => {
    const input = YIELD_USER + '\n<message to="user">trailing</message>'
    const events = parse(input)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
    // No MessageStart should appear — trailing content is observed, not parsed
    const messages = events.filter(e => e._tag === 'MessageStart')
    expect(messages).toHaveLength(0)
  })

  it('observing mode is entered after yield-tool — trailing content is not parsed', () => {
    const input = YIELD_TOOL + '\n<message to="user">trailing</message>'
    const events = parse(input)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('tool')
    // No MessageStart should appear
    const messages = events.filter(e => e._tag === 'MessageStart')
    expect(messages).toHaveLength(0)
  })

  it('observing mode is entered after yield-worker — trailing content is not parsed', () => {
    const input = YIELD_WORKER + '\n<message to="user">trailing</message>'
    const events = parse(input)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('worker')
    // No MessageStart should appear
    const messages = events.filter(e => e._tag === 'MessageStart')
    expect(messages).toHaveLength(0)
  })

  it('duplicate yield tags — first wins', () => {
    // When the model emits a second yield tag, the first one wins
    const input = [
      YIELD_USER,
      '<message to="user">hello world</message>',
      YIELD_TOOL,
    ].join('\n')

    const events = parse(input)
    const tc = turnControls(events)

    // The first yield-user wins
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
  })
})
