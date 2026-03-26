import { describe, it, expect } from 'bun:test'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'
import {
  ACTIONS_OPEN,
  ACTIONS_CLOSE,
  COMMS_OPEN,
  COMMS_CLOSE,
  LENSES_OPEN,
  LENSES_CLOSE,
  TURN_CONTROL_NEXT,
  TURN_CONTROL_YIELD,
} from '../constants'

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
function proseEnds(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'ProseEnd' }> => e._tag === 'ProseEnd')
}

const thinkOpen = () => '<think>'
const thinkClose = () => '</think>'
const messageOpen = (to: string) => `<message to="${to}">`
const messageClose = () => '</message>'
const shellOpen = () => '<shell>'
const shellClose = () => '</shell>'
const lensOpen = (name: string) => `<lens name="${name}">`
const lensClose = () => '</lens>'
const finishOpen = () => '<finish>'
const finishClose = () => '</finish>'
const thinkingOpen = () => '<thinking>'
const thinkingClose = () => '</thinking>'
const reasonOpen = () => '<reason>'
const reasonClose = () => '</reason>'
const tooluseOpen = () => '<tooluse>'
const tooluseClose = () => '</tooluse>'
const respondOpen = () => '<respond>'
const respondClose = () => '</respond>'

function assertNoEvents(events: ParseEvent[], tags: string[]) {
  for (const tag of tags) {
    expect(events.filter(e => e._tag === tag)).toHaveLength(0)
  }
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

describe('tags inside lenses are passthrough', () => {
  it('next inside lens is passthrough', () => {
    const events = parse(`${LENSES_OPEN}\n${lensOpen('t')}${TURN_CONTROL_NEXT}${lensClose()}\n${LENSES_CLOSE}\n`)
    expect(lensEnds(events)[0].content).toContain(TURN_CONTROL_NEXT)
    assertNoEvents(events, ['TurnControl'])
  })

  it('yield inside lens is passthrough', () => {
    const events = parse(`${LENSES_OPEN}\n${lensOpen('t')}${TURN_CONTROL_YIELD}${lensClose()}\n${LENSES_CLOSE}\n`)
    expect(lensEnds(events)[0].content).toContain(TURN_CONTROL_YIELD)
    assertNoEvents(events, ['TurnControl'])
  })

  it('finish with content inside lens is passthrough', () => {
    const inner = `${finishOpen()}evidence here${finishClose()}`
    const events = parse(`${LENSES_OPEN}\n${lensOpen('t')}${inner}${lensClose()}\n${LENSES_CLOSE}\n`)
    expect(lensEnds(events)[0].content).toContain(inner)
    assertNoEvents(events, ['TurnControl', 'FinishCollecting', 'FinishComplete'])
  })

  it('actions inside lens is passthrough', () => {
    const inner = `${ACTIONS_OPEN}\nsome text\n${ACTIONS_CLOSE}`
    const events = parse(`${LENSES_OPEN}\n${lensOpen('t')}${inner}${lensClose()}\n${LENSES_CLOSE}\n`)
    expect(lensEnds(events)[0].content).toContain(ACTIONS_OPEN)
    assertNoEvents(events, ['ContainerOpen', 'ContainerClose'])
  })

  it('comms inside lens is passthrough', () => {
    const inner = `${COMMS_OPEN}\ntext\n${COMMS_CLOSE}`
    const events = parse(`${LENSES_OPEN}\n${lensOpen('t')}${inner}${lensClose()}\n${LENSES_CLOSE}\n`)
    expect(lensEnds(events)[0].content).toContain(COMMS_OPEN)
    assertNoEvents(events, ['ContainerOpen', 'ContainerClose'])
  })

  it('nested think inside lens is passthrough', () => {
    const inner = `${thinkOpen()}\ninner thought\n${thinkClose()}`
    const events = parse(`${LENSES_OPEN}\n${lensOpen('t')}${inner}${lensClose()}\n${LENSES_CLOSE}\n`)
    expect(lensEnds(events)[0].content).toContain(thinkOpen())
    expect(lensEnds(events)[0].content).toContain(thinkClose())
    assertNoEvents(events, ['ThinkStart', 'ThinkEnd'])
  })

  it('message inside lens is passthrough', () => {
    const inner = `${messageOpen('user')}hello${messageClose()}`
    const events = parse(`${LENSES_OPEN}\n${lensOpen('t')}${inner}${lensClose()}\n${LENSES_CLOSE}\n`)
    expect(lensEnds(events)[0].content).toContain(inner)
    assertNoEvents(events, ['MessageStart', 'MessageEnd', 'MessageChunk'])
  })

  it('thinking alias inside lens is passthrough', () => {
    const inner = `${thinkingOpen()}stuff${thinkingClose()}`
    const events = parse(`${LENSES_OPEN}\n${lensOpen('t')}${inner}${lensClose()}\n${LENSES_CLOSE}\n`)
    expect(lensEnds(events)[0].content).toContain(inner)
  })

  it('reason alias inside lens is passthrough', () => {
    const inner = `${reasonOpen()}stuff${reasonClose()}`
    const events = parse(`${LENSES_OPEN}\n${lensOpen('t')}${inner}${lensClose()}\n${LENSES_CLOSE}\n`)
    expect(lensEnds(events)[0].content).toContain(inner)
  })

  it('tooluse alias inside lens is passthrough', () => {
    const inner = `${tooluseOpen()}stuff${tooluseClose()}`
    const events = parse(`${LENSES_OPEN}\n${lensOpen('t')}${inner}${lensClose()}\n${LENSES_CLOSE}\n`)
    expect(lensEnds(events)[0].content).toContain(inner)
    assertNoEvents(events, ['ContainerOpen', 'ContainerClose'])
  })

  it('respond alias inside lens is passthrough', () => {
    const inner = `${respondOpen()}stuff${respondClose()}`
    const events = parse(`${LENSES_OPEN}\n${lensOpen('t')}${inner}${lensClose()}\n${LENSES_CLOSE}\n`)
    expect(lensEnds(events)[0].content).toContain(inner)
    assertNoEvents(events, ['ContainerOpen', 'ContainerClose'])
  })

  it('shell tool tag inside lens is passthrough', () => {
    const inner = `${shellOpen()}echo hi${shellClose()}`
    const events = parse(`${LENSES_OPEN}\n${lensOpen('t')}${inner}${lensClose()}\n${LENSES_CLOSE}\n`)
    expect(lensEnds(events)[0].content).toContain(inner)
    assertNoEvents(events, ['TagOpened', 'TagClosed', 'ToolStart', 'ToolEnd'])
  })
})

describe('tags inside plain think are passthrough', () => {
  it('next inside plain think is passthrough', () => {
    const events = parse(`${thinkOpen()}\n${TURN_CONTROL_NEXT}\n${thinkClose()}\n`)
    const allContent = proseEnds(events).map(e => e.content).join('')
    expect(allContent).toContain(TURN_CONTROL_NEXT)
    assertNoEvents(events, ['TurnControl'])
  })

  it('yield inside plain think is passthrough', () => {
    const events = parse(`${thinkOpen()}\n${TURN_CONTROL_YIELD}\n${thinkClose()}\n`)
    const allContent = proseEnds(events).map(e => e.content).join('')
    expect(allContent).toContain(TURN_CONTROL_YIELD)
    assertNoEvents(events, ['TurnControl'])
  })

  it('actions inside plain think is passthrough', () => {
    const events = parse(`${thinkOpen()}\n${ACTIONS_OPEN}\nstuff\n${ACTIONS_CLOSE}\n${thinkClose()}\n`)
    const allContent = proseEnds(events).map(e => e.content).join('')
    expect(allContent).toContain(ACTIONS_OPEN)
    assertNoEvents(events, ['ContainerOpen', 'ContainerClose'])
  })

  it('shell inside plain think is passthrough', () => {
    const events = parse(`${thinkOpen()}\n${shellOpen()}echo hi${shellClose()}\n${thinkClose()}\n`)
    const allContent = proseEnds(events).map(e => e.content).join('')
    expect(allContent).toContain(shellOpen())
    assertNoEvents(events, ['TagOpened', 'TagClosed'])
  })
})

describe('char-by-char passthrough parity', () => {
  it('char-by-char: tags inside lenses remain passthrough', () => {
    const xml = `${LENSES_OPEN}\n${lensOpen('t')}${TURN_CONTROL_NEXT} ${ACTIONS_OPEN}x${ACTIONS_CLOSE} ${shellOpen()}y${shellClose()}${lensClose()}\n${LENSES_CLOSE}\n`
    const bulk = parse(xml)
    const charByChar = parseCharByChar(xml)
    expect(lensEnds(charByChar)).toEqual(lensEnds(bulk))
    assertNoEvents(bulk, ['TurnControl', 'ContainerOpen', 'TagOpened'])
    assertNoEvents(charByChar, ['TurnControl', 'ContainerOpen', 'TagOpened'])
  })
})
