import { describe, it, expect } from 'bun:test'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'

const knownTags = new Set(['shell'])
const childTagMap = new Map<string, Set<string>>()

function parse(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  return [...parser.processChunk(xml), ...parser.flush()]
}

function parseCharByChar(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  for (const ch of xml) parser.push(ch)
  parser.flush()
  return [...parser.events]
}

function parseByChunks(chunks: string[]): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  const events: ParseEvent[] = []
  for (const chunk of chunks) events.push(...parser.processChunk(chunk))
  events.push(...parser.flush())
  return events
}

function lensStarts(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'LensStart' }> => e._tag === 'LensStart')
}
function lensChunks(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'LensChunk' }> => e._tag === 'LensChunk')
}
function lensEnds(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'LensEnd' }> => e._tag === 'LensEnd')
}
function parseErrors(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'ParseError' }> => e._tag === 'ParseError')
}

describe('lenses parsing', () => {
  it('1) self-closing lens emits LensStart + LensEnd (no chunks)', () => {
    const events = parse('<lenses>\n<lens name="task"/>\n</lenses>\n')
    expect(lensStarts(events)).toEqual([{ _tag: 'LensStart', name: 'task' }])
    expect(lensChunks(events)).toHaveLength(0)
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'task', content: '' }])
  })

  it('2) lens with content emits LensStart + LensChunk(s) + LensEnd', () => {
    const events = parse('<lenses>\n<lens name="task">some reasoning</lens>\n</lenses>\n')
    expect(lensStarts(events).map(e => e.name)).toEqual(['task'])
    expect(lensChunks(events).length).toBeGreaterThan(0)
    expect(lensChunks(events).map(e => e.text).join('')).toBe('some reasoning')
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'task', content: 'some reasoning' }])
  })

  it('3) mixed content and self-closing lenses emit correct events', () => {
    const events = parse('<lenses>\n<lens name="task">alpha</lens>\n<lens name="turn"/>\n</lenses>\n')
    expect(lensStarts(events).map(e => e.name)).toEqual(['task', 'turn'])
    expect(lensEnds(events)).toEqual([
      { _tag: 'LensEnd', name: 'task', content: 'alpha' },
      { _tag: 'LensEnd', name: 'turn', content: '' },
    ])
  })

  it('4) empty lenses block emits no lens events', () => {
    const events = parse('<lenses>\n</lenses>\n')
    expect(lensStarts(events)).toHaveLength(0)
    expect(lensChunks(events)).toHaveLength(0)
    expect(lensEnds(events)).toHaveLength(0)
  })

  it('5) multiple content lenses each get start/chunks/end', () => {
    const events = parse('<lenses>\n<lens name="task">first</lens>\n<lens name="turn">second</lens>\n</lenses>\n')
    expect(lensStarts(events).map(e => e.name)).toEqual(['task', 'turn'])
    expect(lensEnds(events)).toEqual([
      { _tag: 'LensEnd', name: 'task', content: 'first' },
      { _tag: 'LensEnd', name: 'turn', content: 'second' },
    ])
  })

  it('6) lens content supports special chars (newlines, angle brackets)', () => {
    const content = 'line 1\nline 2 <notlens> and < still text'
    const events = parse(`<lenses>\n<lens name="task">${content}</lens>\n</lenses>\n`)
    expect(lensStarts(events).map(e => e.name)).toEqual(['task'])
    expect(lensChunks(events).map(e => e.text).join('')).toBe(content)
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'task', content }])
  })

  it('7) char-by-char streaming yields same lens events as single chunk', () => {
    const xml = '<lenses>\n<lens name="task">alpha</lens>\n<lens name="turn"/>\n</lenses>\n'
    const one = parse(xml)
    const byChar = parseCharByChar(xml)
    expect(lensStarts(byChar)).toEqual(lensStarts(one))
    expect(lensChunks(byChar)).toEqual(lensChunks(one))
    expect(lensEnds(byChar)).toEqual(lensEnds(one))
  })

  it('8) chunk boundary splits (mid-tag, mid-attr, mid-content) still parse correctly', () => {
    const events = parseByChunks([
      '<len',
      'ses>\n<le',
      'ns na',
      'me="ta',
      'sk">hel',
      'lo wo',
      'rld</le',
      'ns>\n</len',
      'ses>\n',
    ])
    expect(lensStarts(events)).toEqual([{ _tag: 'LensStart', name: 'task' }])
    expect(lensChunks(events).map(e => e.text).join('')).toBe('hello world')
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'task', content: 'hello world' }])
  })

  it('9) full turn: lenses then comms then actions', () => {
    const events = parse(
      [
        '<lenses>',
        '<lens name="task">Reason briefly.</lens>',
        '</lenses>',
        '<comms>',
        '<message to="parent">hello</message>',
        '</comms>',
        '<actions>',
        '<shell>echo hi</shell>',
        '</actions>',
      ].join('\n') + '\n',
    )

    expect(lensStarts(events).map(e => e.name)).toEqual(['task'])
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'task', content: 'Reason briefly.' }])
    expect(events.some(e => e._tag === 'ContainerOpen')).toBe(true)
    expect(events.some(e => e._tag === 'ContainerClose')).toBe(true)
    expect(events.some(e => e._tag === 'ContainerOpen')).toBe(true)
    expect(events.some(e => e._tag === 'ContainerClose')).toBe(true)
    expect(events.some(e => e._tag === 'TagClosed' && e.tagName === 'shell')).toBe(true)
  })

  it('10) unclosed lenses block emits ParseError on flush', () => {
    const events = parse('<lenses>\n<lens name="task">content')
    expect(parseErrors(events).some(e => e.error._tag === 'UnclosedThink')).toBe(true)
    expect(lensStarts(events)).toEqual([{ _tag: 'LensStart', name: 'task' }])
    expect(lensEnds(events)).toHaveLength(0)
  })

  it('11) self-closing lens with extra whitespace parses', () => {
    const events = parse('<lenses>\n<lens   name="task"   />\n</lenses>\n')
    expect(lensStarts(events)).toEqual([{ _tag: 'LensStart', name: 'task' }])
    expect(lensChunks(events)).toHaveLength(0)
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'task', content: '' }])
  })
})