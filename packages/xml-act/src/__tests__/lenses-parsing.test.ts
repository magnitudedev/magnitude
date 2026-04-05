import { describe, it, expect } from 'vitest'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'
import { TURN_CONTROL_IDLE } from '../constants'

const TASK_A_OPEN = '<task id="t1">'
const TASK_A_CLOSE = '</task>'
const TASK_B_OPEN = '<task id="t2">'
const TASK_B_CLOSE = '</task>'

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
const messageOpen = () => `<message>`
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
const legacyToolOpen = () => '<legacy-tool>'
const legacyToolClose = () => '</legacy-tool>'
const respondOpen = () => '<respond>'
const respondClose = () => '</respond>'

function assertNoEvents(events: ParseEvent[], tags: string[]) {
  for (const tag of tags) {
    expect(events.filter(e => e._tag === tag)).toHaveLength(0)
  }
}

describe('lenses parsing', () => {
  it('1) self-closing lens emits LensStart + LensEnd (no chunks)', () => {
    const events = parse('<lens name="task"/>\n')
    expect(lensStarts(events)).toEqual([{ _tag: 'LensStart', name: 'task' }])
    expect(lensChunks(events)).toHaveLength(0)
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'task', content: '' }])
  })

  it('2) lens with content emits LensStart + LensChunk(s) + LensEnd', () => {
    const events = parse('<lens name="task">some reasoning</lens>\n')
    expect(lensStarts(events).map(e => e.name)).toEqual(['task'])
    expect(lensChunks(events).length).toBeGreaterThan(0)
    expect(lensChunks(events).map(e => e.text).join('')).toBe('some reasoning')
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'task', content: 'some reasoning' }])
  })

  it('3) mixed content and self-closing lenses emit correct events', () => {
    const events = parse('<lens name="task">alpha</lens>\n<lens name="turn"/>\n')
    expect(lensStarts(events).map(e => e.name)).toEqual(['task', 'turn'])
    expect(lensEnds(events)).toEqual([
      { _tag: 'LensEnd', name: 'task', content: 'alpha' },
      { _tag: 'LensEnd', name: 'turn', content: '' },
    ])
  })

  it('4) empty input emits no lens events', () => {
    const events = parse('\n')
    expect(lensStarts(events)).toHaveLength(0)
    expect(lensChunks(events)).toHaveLength(0)
    expect(lensEnds(events)).toHaveLength(0)
  })

  it('5) multiple content lenses each get start/chunks/end', () => {
    const events = parse('<lens name="task">first</lens>\n<lens name="turn">second</lens>\n')
    expect(lensStarts(events).map(e => e.name)).toEqual(['task', 'turn'])
    expect(lensEnds(events)).toEqual([
      { _tag: 'LensEnd', name: 'task', content: 'first' },
      { _tag: 'LensEnd', name: 'turn', content: 'second' },
    ])
  })

  it('6) lens content supports special chars (newlines, angle brackets)', () => {
    const content = 'line 1\nline 2 <notlens> and < still text'
    const events = parse(`<lens name="task">${content}</lens>\n`)
    expect(lensStarts(events).map(e => e.name)).toEqual(['task'])
    expect(lensChunks(events).map(e => e.text).join('')).toBe(content)
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'task', content }])
  })

  it('7) char-by-char streaming yields same lens events as single chunk', () => {
    const xml = '<lens name="task">alpha</lens>\n<lens name="turn"/>\n'
    const one = parse(xml)
    const byChar = parseCharByChar(xml)
    expect(lensStarts(byChar)).toEqual(lensStarts(one))
    expect(lensChunks(byChar)).toEqual(lensChunks(one))
    expect(lensEnds(byChar)).toEqual(lensEnds(one))
  })

  it('8) chunk boundary splits (mid-tag, mid-attr, mid-content) still parse correctly', () => {
    const events = parseByChunks([
      '<le',
      'ns na',
      'me="ta',
      'sk">hel',
      'lo wo',
      'rld</le',
      'ns>\n',
    ])
    expect(lensStarts(events)).toEqual([{ _tag: 'LensStart', name: 'task' }])
    expect(lensChunks(events).map(e => e.text).join('')).toBe('hello world')
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'task', content: 'hello world' }])
  })

  it('9) full turn: lens then task tags', () => {
    const events = parse(
      [
        '<lens name="task">Reason briefly.</lens>',
        '<task id="t2">',
        '<message>hello</message>',
        '</task>',
        '<task id="t1">',
        '<shell>echo hi</shell>',
        '</task>',
      ].join('\n') + '\n',
    )

    expect(lensStarts(events).map(e => e.name)).toEqual(['task'])
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'task', content: 'Reason briefly.' }])
    expect(events.some(e => e._tag === 'TagOpened')).toBe(true)
    expect(events.some(e => e._tag === 'TagClosed')).toBe(true)
    expect(events.some(e => e._tag === 'TagOpened')).toBe(true)
    expect(events.some(e => e._tag === 'TagClosed')).toBe(true)
    expect(events.some(e => e._tag === 'TagClosed' && e.tagName === 'shell')).toBe(true)
  })

  it('10) unclosed lens emits ParseError on flush', () => {
    const events = parse('<lens name="task">content')
    expect(parseErrors(events).some(e => e.error._tag === 'UnclosedThink')).toBe(true)
    expect(lensStarts(events)).toEqual([{ _tag: 'LensStart', name: 'task' }])
    expect(lensEnds(events)).toHaveLength(0)
  })

  it('11) self-closing lens with extra whitespace parses', () => {
    const events = parse('<lens   name="task"   />\n')
    expect(lensStarts(events)).toEqual([{ _tag: 'LensStart', name: 'task' }])
    expect(lensChunks(events)).toHaveLength(0)
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'task', content: '' }])
  })
})

describe('tags inside lenses are passthrough', () => {
  it('idle inside lens is passthrough', () => {
    const events = parse(`${lensOpen('t')}${TURN_CONTROL_IDLE}${lensClose()}\n`)
    expect(lensEnds(events)[0].content).toContain(TURN_CONTROL_IDLE)
    assertNoEvents(events, ['TurnControl'])
  })

  it('idle inside lens is passthrough (duplicate case)', () => {
    const events = parse(`${lensOpen('t')}${TURN_CONTROL_IDLE}${lensClose()}\n`)
    expect(lensEnds(events)[0].content).toContain(TURN_CONTROL_IDLE)
    assertNoEvents(events, ['TurnControl'])
  })

  it('finish with content inside lens is passthrough', () => {
    const inner = `${finishOpen()}evidence here${finishClose()}`
    const events = parse(`${lensOpen('t')}${inner}${lensClose()}\n`)
    expect(lensEnds(events)[0].content).toContain(inner)
    assertNoEvents(events, ['TurnControl', 'FinishCollecting', 'FinishComplete'])
  })

  it('task block text inside lens is passthrough', () => {
    const inner = `${TASK_A_OPEN}\nsome text\n${TASK_A_CLOSE}`
    const events = parse(`${lensOpen('t')}${inner}${lensClose()}\n`)
    expect(lensEnds(events)[0].content).toContain(TASK_A_OPEN)
    assertNoEvents(events, ['TagOpened', 'TagClosed'])
  })

  it('task block text (alternate id) inside lens is passthrough', () => {
    const inner = `${TASK_B_OPEN}\ntext\n${TASK_B_CLOSE}`
    const events = parse(`${lensOpen('t')}${inner}${lensClose()}\n`)
    expect(lensEnds(events)[0].content).toContain(TASK_B_OPEN)
    assertNoEvents(events, ['TagOpened', 'TagClosed'])
  })

  it('nested think inside lens is passthrough', () => {
    const inner = `${thinkOpen()}\ninner thought\n${thinkClose()}`
    const events = parse(`${lensOpen('t')}${inner}${lensClose()}\n`)
    expect(lensEnds(events)[0].content).toContain(thinkOpen())
    expect(lensEnds(events)[0].content).toContain(thinkClose())
    assertNoEvents(events, ['ThinkStart', 'ThinkEnd'])
  })

  it('message inside lens is passthrough', () => {
    const inner = `${messageOpen()}hello${messageClose()}`
    const events = parse(`${lensOpen('t')}${inner}${lensClose()}\n`)
    expect(lensEnds(events)[0].content).toContain(inner)
    assertNoEvents(events, ['MessageStart', 'MessageEnd', 'MessageChunk'])
  })

  it('thinking alias inside lens is passthrough', () => {
    const inner = `${thinkingOpen()}stuff${thinkingClose()}`
    const events = parse(`${lensOpen('t')}${inner}${lensClose()}\n`)
    expect(lensEnds(events)[0].content).toContain(inner)
  })

  it('reason-like tag text inside lens is passthrough', () => {
    const inner = `${reasonOpen()}stuff${reasonClose()}`
    const events = parse(`${lensOpen('t')}${inner}${lensClose()}\n`)
    expect(lensEnds(events)[0].content).toContain(inner)
  })

  it('legacy tool-like tag text inside lens is passthrough', () => {
    const inner = `${legacyToolOpen()}stuff${legacyToolClose()}`
    const events = parse(`${lensOpen('t')}${inner}${lensClose()}\n`)
    expect(lensEnds(events)[0].content).toContain(inner)
    assertNoEvents(events, ['TagOpened', 'TagClosed'])
  })

  it('respond-like tag text inside lens is passthrough', () => {
    const inner = `${respondOpen()}stuff${respondClose()}`
    const events = parse(`${lensOpen('t')}${inner}${lensClose()}\n`)
    expect(lensEnds(events)[0].content).toContain(inner)
    assertNoEvents(events, ['TagOpened', 'TagClosed'])
  })

  it('shell tool tag inside lens is passthrough', () => {
    const inner = `${shellOpen()}echo hi${shellClose()}`
    const events = parse(`${lensOpen('t')}${inner}${lensClose()}\n`)
    expect(lensEnds(events)[0].content).toContain(inner)
    assertNoEvents(events, ['TagOpened', 'TagClosed', 'ToolStart', 'ToolEnd'])
  })
})

describe('tags inside plain think are passthrough', () => {
  it('idle inside plain think is passthrough', () => {
    const events = parse(`${thinkOpen()}\n${TURN_CONTROL_IDLE}\n${thinkClose()}\n`)
    const allContent = proseEnds(events).map(e => e.content).join('')
    expect(allContent).toContain(TURN_CONTROL_IDLE)
    assertNoEvents(events, ['TurnControl'])
  })

  it('idle inside plain think is passthrough (duplicate case)', () => {
    const events = parse(`${thinkOpen()}\n${TURN_CONTROL_IDLE}\n${thinkClose()}\n`)
    const allContent = proseEnds(events).map(e => e.content).join('')
    expect(allContent).toContain(TURN_CONTROL_IDLE)
    assertNoEvents(events, ['TurnControl'])
  })

  it('task block text inside plain think is passthrough', () => {
    const events = parse(`${thinkOpen()}\n${TASK_A_OPEN}\nstuff\n${TASK_A_CLOSE}\n${thinkClose()}\n`)
    const allContent = proseEnds(events).map(e => e.content).join('')
    expect(allContent).toContain(TASK_A_OPEN)
    assertNoEvents(events, ['TagOpened', 'TagClosed'])
  })

  it('shell inside plain think is passthrough', () => {
    const events = parse(`${thinkOpen()}\n${shellOpen()}echo hi${shellClose()}\n${thinkClose()}\n`)
    const allContent = proseEnds(events).map(e => e.content).join('')
    expect(allContent).toContain(shellOpen())
    assertNoEvents(events, ['TagOpened', 'TagClosed'])
  })
})

function proseChunks(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'ProseChunk' }> => e._tag === 'ProseChunk')
}
function thinkProseChunks(events: ParseEvent[]) {
  return proseChunks(events).filter(e => e.patternId === 'think')
}

describe('lenses whitespace suppression', () => {
  it('no think prose chunks emitted for inter-lens whitespace', () => {
    const events = parse('<lens name="a">content a</lens>\n<lens name="b">content b</lens>\n')
    // Whitespace between </lens> and <lens> should NOT produce ProseChunk events with patternId 'think'
    expect(thinkProseChunks(events)).toHaveLength(0)
  })

  it('no think prose chunks for whitespace-only input', () => {
    const events = parse('\n\n\n')
    expect(thinkProseChunks(events)).toHaveLength(0)
  })

  it('no think prose chunks for whitespace around lens (char-by-char)', () => {
    const xml = '<lens name="a">hello</lens>\n'
    const events = parseCharByChar(xml)
    expect(thinkProseChunks(events)).toHaveLength(0)
  })
})

describe('lens content trimming', () => {
  it('lens content trims leading/trailing spaces and tabs', () => {
    const events = parse('<lens name="a">  \tcontent here\t  </lens>\n')
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'a', content: 'content here' }])
  })

  it('lens content trims leading/trailing newlines', () => {
    const events = parse('<lens name="a">\n\ncontent here\n\n</lens>\n')
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'a', content: 'content here' }])
  })

  it('lens content trims mixed whitespace at boundaries', () => {
    const events = parse('<lens name="a">\n  \tcontent here\t  \n</lens>\n')
    expect(lensEnds(events)).toEqual([{ _tag: 'LensEnd', name: 'a', content: 'content here' }])
  })

  it('unclosed lens has no LensEnd', () => {
    const events = parse('<lens name="a">\n  content here  \n')
    expect(lensEnds(events)).toHaveLength(0)
  })
})

describe('char-by-char passthrough parity', () => {
  it('char-by-char: tags inside lenses remain passthrough', () => {
    const xml = `${lensOpen('t')}${TURN_CONTROL_IDLE} ${TASK_A_OPEN}x${TASK_A_CLOSE} ${shellOpen()}y${shellClose()}${lensClose()}\n`
    const bulk = parse(xml)
    const charByChar = parseCharByChar(xml)
    expect(lensEnds(charByChar)).toEqual(lensEnds(bulk))
    assertNoEvents(bulk, ['TurnControl', 'TagOpened', 'TagOpened'])
    assertNoEvents(charByChar, ['TurnControl', 'TagOpened', 'TagOpened'])
  })
})
