import { describe, it, expect } from 'vitest'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'
import { YIELD_USER, YIELD_TOOL, YIELD_WORKER, YIELD_PARENT, SUBAGENT_YIELD_TAGS } from '../constants'

const actionsTagOpen = () => ''
const actionsTagClose = () => ''
const thinkTagOpen = () => '<think name="test">'
const thinkTagClose = () => '</think>'
const commsTagOpen = () => ''
const commsTagClose = () => ''

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
// Basic yield tag parsing
// =============================================================================

describe('basic yield control', () => {
  it('yield-user emits target user', () => {
    const events = parse(YIELD_USER)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
  })

  it('yield-tool emits target tool', () => {
    const events = parse(YIELD_TOOL)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('tool')
  })

  it('yield-worker emits target worker', () => {
    const events = parse(YIELD_WORKER)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('worker')
  })

  it('yield-parent emits target parent (with subagent yield tags)', () => {
    const parser = createStreamingXmlParser(knownTags, childTagMap, undefined, undefined, undefined, SUBAGENT_YIELD_TAGS)
    const events = [...parser.processChunk(YIELD_PARENT), ...parser.flush()]
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('parent')
  })

  it('yield-user char-by-char', () => {
    const events = parseCharByChar(YIELD_USER)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
  })

  it('yield-tool char-by-char', () => {
    const events = parseCharByChar(YIELD_TOOL)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('tool')
  })

  it('yield-worker char-by-char', () => {
    const events = parseCharByChar(YIELD_WORKER)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('worker')
  })

  it('yield-parent char-by-char (with subagent yield tags)', () => {
    const parser = createStreamingXmlParser(knownTags, childTagMap, undefined, undefined, undefined, SUBAGENT_YIELD_TAGS)
    const events: ParseEvent[] = []
    for (const ch of YIELD_PARENT) events.push(...parser.processChunk(ch))
    events.push(...parser.flush())
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('parent')
  })
})

// =============================================================================
// Yield control after other blocks
// =============================================================================

describe('yield control after content blocks', () => {
  it('after think block', () => {
    const xml = `${thinkTagOpen()}planning\n${thinkTagClose()}\n${YIELD_USER}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
  })

  it('after actions block', () => {
    const xml = `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}\n${YIELD_USER}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
  })

  it('after actions block with yield-tool', () => {
    const xml = `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}\n${YIELD_TOOL}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('tool')
  })

  it('after comms block', () => {
    const xml = `${commsTagOpen()}\n<message>hello</message>\n${commsTagClose()}\n${YIELD_USER}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
  })

  it('after think + comms + actions', () => {
    const xml = [
      `${thinkTagOpen()}plan\n${thinkTagClose()}`,
      `${commsTagOpen()}\n<message>hi</message>\n${commsTagClose()}`,
      `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}`,
      YIELD_USER,
    ].join('\n')
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
  })

  it('after think + comms + actions (char-by-char)', () => {
    const xml = [
      `${thinkTagOpen()}plan\n${thinkTagClose()}`,
      `${commsTagOpen()}\n<message>hi</message>\n${commsTagClose()}`,
      `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}`,
      YIELD_USER,
    ].join('\n')
    const events = parseCharByChar(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
  })
})

// =============================================================================
// Duplicate tags — first wins, rest silently ignored
// =============================================================================

describe('duplicate yield control tags', () => {
  it('duplicate yield-user — only one TurnControl event', () => {
    const xml = `${YIELD_USER}\n${YIELD_USER}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('duplicate yield-tool — only one TurnControl event', () => {
    const xml = `${YIELD_TOOL}\n${YIELD_TOOL}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('tool')
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('yield-user then yield-tool — first wins', () => {
    const xml = `${YIELD_USER}\n${YIELD_TOOL}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('yield-tool then yield-user — first wins', () => {
    const xml = `${YIELD_TOOL}\n${YIELD_USER}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('tool')
    expect(parseErrors(events)).toHaveLength(0)
  })

  it('three yield-user tags — first wins', () => {
    const xml = `${YIELD_USER}\n${YIELD_USER}\n${YIELD_USER}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('user')
    expect(parseErrors(events)).toHaveLength(0)
  })
})

// =============================================================================
// Content after yield control is ignored
// =============================================================================

describe('content after yield control is dropped', () => {
  it('tool calls after yield-user are not parsed', () => {
    const xml = `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}\n${YIELD_USER}\n${actionsTagOpen()}\n<shell>rm -rf /</shell>\n${actionsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    const tagCloseds = events.filter(e => e._tag === 'TagClosed')
    expect(tagCloseds).toHaveLength(1) // only the first shell
  })

  it('tool calls after yield-user are not parsed (char-by-char)', () => {
    const xml = `${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}\n${YIELD_USER}\n${actionsTagOpen()}\n<shell>rm -rf /</shell>\n${actionsTagClose()}`
    const events = parseCharByChar(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    const tagCloseds = events.filter(e => e._tag === 'TagClosed')
    expect(tagCloseds).toHaveLength(1)
  })

  it('prose after yield-user is not emitted', () => {
    const xml = `${YIELD_USER}\nsome trailing text`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    const prose = events.filter(e => e._tag === 'ProseChunk' || e._tag === 'ProseEnd')
    expect(prose).toHaveLength(0)
  })

  it('content after yield-tool is dropped', () => {
    const xml = `${YIELD_TOOL}\n${actionsTagOpen()}\n<shell>rm -rf /</shell>\n${actionsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].target).toBe('tool')
    const tagCloseds = events.filter(e => e._tag === 'TagClosed')
    expect(tagCloseds).toHaveLength(0)
  })

  it('no unclosed errors from content after yield control', () => {
    const xml = `${YIELD_USER}\n${actionsTagOpen()}\n<shell>orphaned`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(parseErrors(events)).toHaveLength(0)
  })
})

// =============================================================================
// No yield control tag
// =============================================================================

describe('missing yield control', () => {
  it('no yield control tag emits no TurnControl event', () => {
    const xml = `${thinkTagOpen()}plan\n${thinkTagClose()}\n${actionsTagOpen()}\n<shell>ls</shell>\n${actionsTagClose()}`
    const events = parse(xml)
    const tc = turnControls(events)
    expect(tc).toHaveLength(0)
  })
})
