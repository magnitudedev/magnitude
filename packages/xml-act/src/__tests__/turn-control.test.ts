import { describe, it, expect } from 'bun:test'
import { createStreamingXmlParser } from '../parser/streaming-xml-parser'
import { TURN_CONTROL_NEXT, TURN_CONTROL_YIELD, actionsTagOpen, actionsTagClose, thinkTagOpen, thinkTagClose, commsTagOpen, commsTagClose } from '../constants'
import type { ParseEvent } from '../parser/types'

const knownTags = new Set(['shell', 'fs-read'])
const childTagMap = new Map<string, Set<string>>()

function parse(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  return [...parser.processChunk(xml), ...parser.flush()]
}

function parseCharByChar(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  const events: ParseEvent[] = []
  for (const ch of xml) events.push(...parser.processChunk(ch))
  events.push(...parser.flush())
  return events
}

function turnControls(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'TurnControl' }> => e._tag === 'TurnControl')
}

function parseErrors(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'ParseError' }> => e._tag === 'ParseError')
}

// =============================================================================
// Basic turn control parsing
// =============================================================================

describe('basic turn control', () => {
  it('<next/> emits continue decision', () => {
    const events = parse(`<${TURN_CONTROL_NEXT}/>`)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
  })

  it('<yield/> emits yield decision', () => {
    const events = parse(`<${TURN_CONTROL_YIELD}/>`)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('yield')
  })

  it('<next/> char-by-char', () => {
    const events = parseCharByChar(`<${TURN_CONTROL_NEXT}/>`)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
  })

  it('<yield/> char-by-char', () => {
    const events = parseCharByChar(`<${TURN_CONTROL_YIELD}/>`)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('yield')
  })
})

// =============================================================================
// Turn control after other blocks
// =============================================================================

describe('turn control after content blocks', () => {
  it('after think block', () => {
    const xml = `${thinkTagOpen()}planning\n${thinkTagClose()}\n<${TURN_CONTROL_YIELD}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('yield')
  })

  it('after actions block', () => {
    const xml = `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}\n<${TURN_CONTROL_NEXT}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
  })

  it('after comms block', () => {
    const xml = `${commsTagOpen()}\n<message to="user">hello</message>\n${commsTagClose()}\n<${TURN_CONTROL_YIELD}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('yield')
  })

  it('after think + comms + actions', () => {
    const xml = [
      `${thinkTagOpen()}plan\n${thinkTagClose()}`,
      `${commsTagOpen()}\n<message to="user">hi</message>\n${commsTagClose()}`,
      `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}`,
      `<${TURN_CONTROL_NEXT}/>`,
    ].join('\n')
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
  })

  it('after think + comms + actions (char-by-char)', () => {
    const xml = [
      `${thinkTagOpen()}plan\n${thinkTagClose()}`,
      `${commsTagOpen()}\n<message to="user">hi</message>\n${commsTagClose()}`,
      `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}`,
      `<${TURN_CONTROL_NEXT}/>`,
    ].join('\n')
    const events = parseCharByChar(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
  })
})

// =============================================================================
// Duplicate tags — first wins, rest silently ignored
// =============================================================================

describe('duplicate turn control tags', () => {
  it('duplicate <next/> — only one TurnControl event', () => {
    const xml = `<${TURN_CONTROL_NEXT}/>\n<${TURN_CONTROL_NEXT}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('duplicate <yield/> — only one TurnControl event', () => {
    const xml = `<${TURN_CONTROL_YIELD}/>\n<${TURN_CONTROL_YIELD}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('yield')
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('<next/> then <yield/> — first wins', () => {
    const xml = `<${TURN_CONTROL_NEXT}/>\n<${TURN_CONTROL_YIELD}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('<yield/> then <next/> — first wins', () => {
    const xml = `<${TURN_CONTROL_YIELD}/>\n<${TURN_CONTROL_NEXT}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('yield')
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('three turn control tags — first wins', () => {
    const xml = `<${TURN_CONTROL_NEXT}/>\n<${TURN_CONTROL_YIELD}/>\n<${TURN_CONTROL_NEXT}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
    expect(parseErrors(events)).toHaveLength(0)
  })
})

// =============================================================================
// Turn control inside structural blocks (should NOT be recognized)
// =============================================================================

describe('turn control inside blocks is not recognized', () => {
  it('inside actions block — not recognized as turn control', () => {
    const xml = `${actionsTagOpen()}\n<${TURN_CONTROL_NEXT}/>\n${actionsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(0)
  })

  it('inside comms block — not recognized as turn control', () => {
    const xml = `${commsTagOpen()}\n<${TURN_CONTROL_YIELD}/>\n${commsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(0)
  })
})

// =============================================================================
// Content after turn control is ignored
// =============================================================================

describe('content after turn control is dropped', () => {
  it('tool calls after <yield/> are not parsed', () => {
    const xml = `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}\n<${TURN_CONTROL_YIELD}/>\n${actionsTagOpen()}\n<shell>rm -rf /</shell>\n${actionsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    const tagCloseds = events.filter(e => e._tag === 'TagClosed')
    expect(tagCloseds).toHaveLength(1) // only the first shell
  })

  it('tool calls after <yield/> are not parsed (char-by-char)', () => {
    const xml = `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}\n<${TURN_CONTROL_YIELD}/>\n${actionsTagOpen()}\n<shell>rm -rf /</shell>\n${actionsTagClose()}`
    const events = parseCharByChar(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    const tagCloseds = events.filter(e => e._tag === 'TagClosed')
    expect(tagCloseds).toHaveLength(1)
  })

  it('prose after <next/> is not emitted', () => {
    const xml = `<${TURN_CONTROL_NEXT}/>\nsome trailing text`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    const prose = events.filter(e => e._tag === 'Prose')
    expect(prose).toHaveLength(0)
  })

  it('no unclosed errors from content after turn control', () => {
    const xml = `<${TURN_CONTROL_YIELD}/>\n${actionsTagOpen()}\n<shell>orphaned`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(parseErrors(events)).toHaveLength(0)
  })
})

// =============================================================================
// No turn control tag
// =============================================================================

describe('missing turn control', () => {
  it('no turn control tag emits no TurnControl event', () => {
    const xml = `${thinkTagOpen()}plan\n${thinkTagClose()}\n${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(0)
  })
})