import { describe, it, expect } from 'bun:test'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'

const TURN_CONTROL_NEXT = 'next'
const TURN_CONTROL_YIELD = 'yield'
const TURN_CONTROL_FINISH = 'finish'
const actionsTagOpen = () => '<actions>'
const actionsTagClose = () => '</actions>'
const thinkTagOpen = () => '<think>'
const thinkTagClose = () => '</think>'
const commsTagOpen = () => '<comms>'
const commsTagClose = () => '</comms>'

const knownTags = new Set(['shell', 'read'])
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

  it('<finish/> without evidence emits ParseError', () => {
    const events = parse(`<${TURN_CONTROL_FINISH}/>`)
    const tc = turnControls(events)
    expect(tc).toHaveLength(0)
    const errors = parseErrors(events)
    expect(errors).toHaveLength(1)
    expect(errors[0].error._tag).toBe('FinishWithoutEvidence')
  })

  it('<finish>evidence</finish> emits finish with evidence', () => {
    const events = parse(`<${TURN_CONTROL_FINISH}>task verified</${TURN_CONTROL_FINISH}>`)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('finish')
    expect((tc[0] as any).evidence).toBe('task verified')
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

  it('<finish>evidence</finish> char-by-char', () => {
    const events = parseCharByChar(`<${TURN_CONTROL_FINISH}>all tests pass</${TURN_CONTROL_FINISH}>`)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('finish')
    expect((tc[0] as any).evidence).toBe('all tests pass')
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

  it('after actions block with <finish>evidence</finish>', () => {
    const xml = `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}\n<${TURN_CONTROL_FINISH}>verified</${TURN_CONTROL_FINISH}>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('finish')
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

  it('duplicate <finish/> — emits ParseError, no TurnControl', () => {
    const xml = `<${TURN_CONTROL_FINISH}/>\n<${TURN_CONTROL_FINISH}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(0)
    const errors = parseErrors(events)
    expect(errors.some(e => e.error._tag === 'FinishWithoutEvidence')).toBe(true)
  })

  it('<next/> then <yield/> — first wins', () => {
    const xml = `<${TURN_CONTROL_NEXT}/>\n<${TURN_CONTROL_YIELD}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('<finish/> then <yield/> — finish errors, no turn control', () => {
    const xml = `<${TURN_CONTROL_FINISH}/>\n<${TURN_CONTROL_YIELD}/>`
    const events = parse(xml)
    const errors = parseErrors(events)
    expect(errors.some(e => e.error._tag === 'FinishWithoutEvidence')).toBe(true)
  })

  it('<next/> then <finish/> — first wins', () => {
    const xml = `<${TURN_CONTROL_NEXT}/>\n<${TURN_CONTROL_FINISH}/>`
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
// Turn control inside structural blocks auto-closes the block
// =============================================================================

describe('turn control inside blocks auto-closes and is recognized', () => {
  it('inside actions block — auto-closes actions and emits turn control', () => {
    const xml = `${actionsTagOpen()}\n<${TURN_CONTROL_NEXT}/>\n${actionsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
  })

  it('inside comms block — auto-closes comms and emits turn control', () => {
    const xml = `${commsTagOpen()}\n<${TURN_CONTROL_YIELD}/>\n${commsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('yield')
  })

  it('inside actions block with <finish>evidence</finish> — auto-closes and emits finish', () => {
    const xml = `${actionsTagOpen()}\n<${TURN_CONTROL_FINISH}>verified</${TURN_CONTROL_FINISH}>\n${actionsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('finish')
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
    const prose = events.filter(e => e._tag === 'ProseChunk' || e._tag === 'ProseEnd')
    expect(prose).toHaveLength(0)
  })

  it('content after <finish>evidence</finish> is dropped', () => {
    const xml = `<${TURN_CONTROL_FINISH}>done</${TURN_CONTROL_FINISH}>\n${actionsTagOpen()}\n<shell>rm -rf /</shell>\n${actionsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('finish')
    const tagCloseds = events.filter(e => e._tag === 'TagClosed')
    expect(tagCloseds).toHaveLength(0)
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