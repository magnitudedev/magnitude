import { describe, it, expect } from 'bun:test'
import { createStreamingXmlParser } from '../parser/streaming-xml-parser'
import type { ParseEvent } from '../parser/types'

const knownTags = new Set(['shell', 'fs-read'])
const childTagMap = new Map<string, Set<string>>()

function parse(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  return [...parser.processChunk(xml), ...parser.flush()]
}

function eventTags(events: ParseEvent[]): string[] {
  return events.map(e => e._tag)
}

function findTag(events: ParseEvent[], tag: ParseEvent['_tag']): number {
  return events.findIndex(e => e._tag === tag)
}

describe('lenient structural parsing variations', () => {
  it('1) comms inside actions emits expected block and message events', () => {
    const events = parse('<think>\nplan\n</think>\n<actions>\n<shell>ls</shell>\n<comms>\n<message to="user">done</message>\n</comms>\n</actions>')
    const tags = eventTags(events)

    expect(tags).toContain('ActionsOpen')
    expect(tags).toContain('TagOpened')
    expect(tags).toContain('TagClosed')
    expect(tags).toContain('CommsOpen')
    expect(tags).toContain('MessageTagOpen')
    expect(tags).toContain('MessageBodyChunk')
    expect(tags).toContain('MessageTagClose')
    expect(tags).toContain('CommsClose')
    expect(tags).toContain('ActionsClose')

    expect(findTag(events, 'ActionsOpen')).toBeLessThan(findTag(events, 'CommsOpen'))
    expect(findTag(events, 'CommsClose')).toBeLessThan(findTag(events, 'ActionsClose'))
  })

  it('2) wrong block order: comms before think', () => {
    const events = parse('<comms>\n<message>hello</message>\n</comms>\n<think>\nplan\n</think>')
    expect(findTag(events, 'CommsOpen')).toBeGreaterThanOrEqual(0)
    expect(findTag(events, 'CommsClose')).toBeGreaterThan(findTag(events, 'CommsOpen'))
    expect(findTag(events, 'ProseEnd')).toBeGreaterThan(findTag(events, 'CommsClose'))
  })

  it('3) wrong block order: actions before comms', () => {
    const events = parse('<actions>\n<shell>ls</shell>\n</actions>\n<comms>\n<message>done</message>\n</comms>')
    expect(findTag(events, 'ActionsOpen')).toBeGreaterThanOrEqual(0)
    expect(findTag(events, 'ActionsClose')).toBeGreaterThan(findTag(events, 'ActionsOpen'))
    expect(findTag(events, 'CommsOpen')).toBeGreaterThan(findTag(events, 'ActionsClose'))
    expect(findTag(events, 'CommsClose')).toBeGreaterThan(findTag(events, 'CommsOpen'))
  })

  it('4) bare prose between think and actions emits prose events', () => {
    const events = parse('<think>\nplan\n</think>\nHere are results\n<actions>\n<shell>ls</shell>\n</actions>')
    const proseEnds = events.filter((e): e is Extract<ParseEvent, { _tag: 'ProseEnd' }> => e._tag === 'ProseEnd')
    expect(proseEnds.some(e => e.patternId === 'think')).toBe(true)
    expect(proseEnds.some(e => e.patternId === 'prose' && e.content.includes('Here are results'))).toBe(true)
    expect(findTag(events, 'ActionsOpen')).toBeGreaterThan(findTag(events, 'ProseChunk'))
  })

  it('5) prose after actions close emits prose chunk', () => {
    const events = parse('<actions>\n<shell>ls</shell>\n</actions>\nLet me know')
    expect(findTag(events, 'ActionsClose')).toBeGreaterThanOrEqual(0)
    const proseChunkIdx = findTag(events, 'ProseChunk')
    expect(proseChunkIdx).toBeGreaterThan(findTag(events, 'ActionsClose'))
  })

  it('6) multiple think blocks emit multiple think ProseEnd events', () => {
    const events = parse('<think>\none\n</think>\n<think>\ntwo\n</think>')
    const thinkEnds = events.filter((e): e is Extract<ParseEvent, { _tag: 'ProseEnd' }> => e._tag === 'ProseEnd' && e.patternId === 'think')
    expect(thinkEnds).toHaveLength(2)
  })

  it('7) multiple comms blocks emit two comms open/close pairs', () => {
    const events = parse('<comms>\n<message>a</message>\n</comms>\n<comms>\n<message>b</message>\n</comms>')
    expect(events.filter(e => e._tag === 'CommsOpen')).toHaveLength(2)
    expect(events.filter(e => e._tag === 'CommsClose')).toHaveLength(2)
  })

  it('8) multiple actions blocks emit two actions open/close pairs', () => {
    const events = parse('<actions>\n<shell>one</shell>\n</actions>\n<actions>\n<shell>two</shell>\n</actions>')
    expect(events.filter(e => e._tag === 'ActionsOpen')).toHaveLength(2)
    expect(events.filter(e => e._tag === 'ActionsClose')).toHaveLength(2)
  })

  it('9) think inside actions emits think events while in actions', () => {
    const events = parse('<actions>\n<shell>ls</shell>\n<think>\nreconsider\n</think>\n</actions>')
    expect(findTag(events, 'ActionsOpen')).toBeGreaterThanOrEqual(0)
    const thinkEnd = events.find((e): e is Extract<ParseEvent, { _tag: 'ProseEnd' }> => e._tag === 'ProseEnd' && e.patternId === 'think')
    expect(thinkEnd?.content).toContain('reconsider')
    expect(findTag(events, 'ActionsClose')).toBeGreaterThan(findTag(events, 'ProseEnd'))
  })

  it('10) prose inside comms but outside message emits prose events', () => {
    const events = parse('<comms>\nhello from comms\n</comms>')
    expect(findTag(events, 'CommsOpen')).toBeGreaterThanOrEqual(0)
    expect(events.some(e => e._tag === 'ProseChunk')).toBe(true)
  })

  it('11) no think block, starts with prose', () => {
    const events = parse('hello world')
    expect(events.some(e => e._tag === 'ProseChunk')).toBe(true)
    expect(events.some(e => e._tag === 'ProseEnd')).toBe(true)
  })

  it('12) no think block, starts with comms', () => {
    const events = parse('<comms>\n<message>hi</message>\n</comms>')
    expect(findTag(events, 'CommsOpen')).toBeGreaterThanOrEqual(0)
  })

  it('13) no think block, starts with actions', () => {
    const events = parse('<actions>\n<shell>ls</shell>\n</actions>')
    expect(findTag(events, 'ActionsOpen')).toBeGreaterThanOrEqual(0)
  })

  it('14) bare message at top level emits message events', () => {
    const events = parse('<message>Hello</message>')
    expect(events.some(e => e._tag === 'MessageTagOpen')).toBe(true)
    expect(events.some(e => e._tag === 'MessageBodyChunk')).toBe(true)
    expect(events.some(e => e._tag === 'MessageTagClose')).toBe(true)
    expect(events.some(e => e._tag === 'ProseEnd')).toBe(false)
  })

  it('15) message inside actions without comms emits message events', () => {
    const events = parse('<actions>\n<shell>ls</shell>\n<message>done</message>\n</actions>')
    expect(events.some(e => e._tag === 'MessageTagOpen')).toBe(true)
    expect(events.some(e => e._tag === 'MessageTagClose')).toBe(true)
  })

  it('16) tool call outside actions emits tag open/close', () => {
    const events = parse('<shell>echo hi</shell>')
    const opened = events.filter((e): e is Extract<ParseEvent, { _tag: 'TagOpened' }> => e._tag === 'TagOpened')
    const closed = events.filter((e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> => e._tag === 'TagClosed')
    expect(opened).toHaveLength(1)
    expect(closed).toHaveLength(1)
    expect(closed[0].element.body).toBe('echo hi')
  })

  it('17) prose-only response emits only prose events', () => {
    const events = parse('Hey! What can I help you with?')
    const tags = new Set(eventTags(events))
    expect(tags).toEqual(new Set(['ProseChunk', 'ProseEnd']))
  })

  it('18) message with artifacts attribute emits artifactsRaw on open', () => {
    const events = parse('<message artifacts="a,b">hi</message>')
    const open = events.find((e): e is Extract<ParseEvent, { _tag: 'MessageTagOpen' }> => e._tag === 'MessageTagOpen')
    expect(open?.artifactsRaw).toBe('a,b')
  })

  it('19) message without to attribute defaults dest to user', () => {
    const events = parse('<message>hi</message>')
    const open = events.find((e): e is Extract<ParseEvent, { _tag: 'MessageTagOpen' }> => e._tag === 'MessageTagOpen')
    expect(open?.dest).toBe('user')
  })

  it('20) comms with multiple messages emits multiple message sequences', () => {
    const events = parse('<comms>\n<message>a</message>\n<message>b</message>\n</comms>')
    expect(events.filter(e => e._tag === 'MessageTagOpen')).toHaveLength(2)
    expect(events.filter(e => e._tag === 'MessageTagClose')).toHaveLength(2)
  })

  it('21) self-closing message at top level emits open + close', () => {
    const events = parse('<message />')
    const opens = events.filter((e): e is Extract<ParseEvent, { _tag: 'MessageTagOpen' }> => e._tag === 'MessageTagOpen')
    const closes = events.filter((e): e is Extract<ParseEvent, { _tag: 'MessageTagClose' }> => e._tag === 'MessageTagClose')
    expect(opens).toHaveLength(1)
    expect(closes).toHaveLength(1)
    expect(closes[0].id).toBe(opens[0].id)
  })

  it('22) self-closing tool at top level emits tag open + close', () => {
    const events = parse('<shell />')
    const opened = events.filter((e): e is Extract<ParseEvent, { _tag: 'TagOpened' }> => e._tag === 'TagOpened')
    const closed = events.filter((e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> => e._tag === 'TagClosed')
    expect(opened).toHaveLength(1)
    expect(closed).toHaveLength(1)
    expect(closed[0].element.body).toBe('')
  })
})