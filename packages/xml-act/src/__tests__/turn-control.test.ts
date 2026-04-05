import { describe, it, expect } from 'bun:test'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'

const TURN_CONTROL_IDLE = 'idle'
const TURN_CONTROL_FINISH = 'finish'
const actionsTagOpen = () => '<task id="t1">'
const actionsTagClose = () => '</task>'
const thinkTagOpen = () => '<think>'
const thinkTagClose = () => '</think>'
const commsTagOpen = () => '<task id="t2">'
const commsTagClose = () => '</task>'

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
  it('<idle/> emits idle decision', () => {
    const events = parse(`<${TURN_CONTROL_IDLE}/>`)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
  })

  it('<idle/> emits idle decision (duplicate case)', () => {
    const events = parse(`<${TURN_CONTROL_IDLE}/>`)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
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

  it('<idle/> char-by-char', () => {
    const events = parseCharByChar(`<${TURN_CONTROL_IDLE}/>`)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
  })

  it('<idle/> char-by-char', () => {
    const events = parseCharByChar(`<${TURN_CONTROL_IDLE}/>`)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
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
    const xml = `${thinkTagOpen()}planning\n${thinkTagClose()}\n<${TURN_CONTROL_IDLE}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
  })

  it('after actions block', () => {
    const xml = `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}\n<${TURN_CONTROL_IDLE}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
  })

  it('after actions block with <finish>evidence</finish>', () => {
    const xml = `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}\n<${TURN_CONTROL_FINISH}>verified</${TURN_CONTROL_FINISH}>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('finish')
  })

  it('after comms block', () => {
    const xml = `${commsTagOpen()}\n<message>hello</message>\n${commsTagClose()}\n<${TURN_CONTROL_IDLE}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
  })

  it('after think + comms + actions', () => {
    const xml = [
      `${thinkTagOpen()}plan\n${thinkTagClose()}`,
      `${commsTagOpen()}\n<message>hi</message>\n${commsTagClose()}`,
      `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}`,
      `<${TURN_CONTROL_IDLE}/>`,
    ].join('\n')
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
  })

  it('after think + comms + actions (char-by-char)', () => {
    const xml = [
      `${thinkTagOpen()}plan\n${thinkTagClose()}`,
      `${commsTagOpen()}\n<message>hi</message>\n${commsTagClose()}`,
      `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}`,
      `<${TURN_CONTROL_IDLE}/>`,
    ].join('\n')
    const events = parseCharByChar(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
  })
})

// =============================================================================
// Duplicate tags — first wins, rest silently ignored
// =============================================================================

describe('duplicate turn control tags', () => {
  it('duplicate <idle/> — only one TurnControl event', () => {
    const xml = `<${TURN_CONTROL_IDLE}/>\n<${TURN_CONTROL_IDLE}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('duplicate <idle/> — only one TurnControl event', () => {
    const xml = `<${TURN_CONTROL_IDLE}/>\n<${TURN_CONTROL_IDLE}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
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

  it('<idle/> then <idle/> — first wins', () => {
    const xml = `<${TURN_CONTROL_IDLE}/>\n<${TURN_CONTROL_IDLE}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('<finish/> then <idle/> — finish errors, no turn control', () => {
    const xml = `<${TURN_CONTROL_FINISH}/>\n<${TURN_CONTROL_IDLE}/>`
    const events = parse(xml)
    const errors = parseErrors(events)
    expect(errors.some(e => e.error._tag === 'FinishWithoutEvidence')).toBe(true)
  })

  it('<idle/> then <finish/> — first wins', () => {
    const xml = `<${TURN_CONTROL_IDLE}/>\n<${TURN_CONTROL_FINISH}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('<idle/> then <idle/> — first wins', () => {
    const xml = `<${TURN_CONTROL_IDLE}/>\n<${TURN_CONTROL_IDLE}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('three turn control tags — first wins', () => {
    const xml = `<${TURN_CONTROL_IDLE}/>\n<${TURN_CONTROL_IDLE}/>\n<${TURN_CONTROL_IDLE}/>`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
    expect(parseErrors(events)).toHaveLength(0)
  })
})

// =============================================================================
// Turn control inside structural blocks auto-closes the block
// =============================================================================

describe('turn control inside blocks is passthrough and not recognized', () => {
  it('inside actions block — no turn control is emitted', () => {
    const xml = `${actionsTagOpen()}\n<${TURN_CONTROL_IDLE}/>\n${actionsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(0)
    const prose = events
      .filter((e): e is Extract<ParseEvent, { _tag: 'ProseChunk' }> => e._tag === 'ProseChunk')
      .map(e => e.text)
      .join('')
    expect(prose).toContain(`<${TURN_CONTROL_IDLE}/>`)
  })

  it('inside comms block — no turn control is emitted', () => {
    const xml = `${commsTagOpen()}\n<${TURN_CONTROL_IDLE}/>\n${commsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(0)
    const prose = events
      .filter((e): e is Extract<ParseEvent, { _tag: 'ProseChunk' }> => e._tag === 'ProseChunk')
      .map(e => e.text)
      .join('')
    expect(prose).toContain(`<${TURN_CONTROL_IDLE}/>`)
  })

  it('inside actions block with <finish>evidence</finish> is ignored in task scope', () => {
    const xml = `${actionsTagOpen()}\n<${TURN_CONTROL_FINISH}>verified</${TURN_CONTROL_FINISH}>\n${actionsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(0)
    expect(parseErrors(events)).toHaveLength(0)
  })
})

// =============================================================================
// Content after turn control is ignored
// =============================================================================

describe('content after turn control is dropped', () => {
  it('tool calls after <idle/> are not parsed', () => {
    const xml = `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}\n<${TURN_CONTROL_IDLE}/>\n${actionsTagOpen()}\n<shell>rm -rf /</shell>\n${actionsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    const tagCloseds = events.filter(e => e._tag === 'TagClosed')
    expect(tagCloseds).toHaveLength(1) // only the first shell
  })

  it('tool calls after <idle/> are not parsed (char-by-char)', () => {
    const xml = `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}\n<${TURN_CONTROL_IDLE}/>\n${actionsTagOpen()}\n<shell>rm -rf /</shell>\n${actionsTagClose()}`
    const events = parseCharByChar(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    const tagCloseds = events.filter(e => e._tag === 'TagClosed')
    expect(tagCloseds).toHaveLength(1)
  })

  it('prose after <idle/> is not emitted', () => {
    const xml = `<${TURN_CONTROL_IDLE}/>\nsome trailing text`
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
    const xml = `<${TURN_CONTROL_IDLE}/>\n${actionsTagOpen()}\n<shell>orphaned`
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