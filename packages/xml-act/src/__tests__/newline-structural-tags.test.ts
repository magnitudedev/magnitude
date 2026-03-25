import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'

const actionsTagOpen = () => '<actions>'
const actionsTagClose = () => '</actions>'
const thinkTagOpen = () => '<think>'
const thinkTagClose = () => '</think>'


const knownTags = new Set(['fs-search', 'shell', 'write'])
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

function tagCloseds(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> => e._tag === 'TagClosed')
}

function thinkEnds(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'ProseEnd' }> => e._tag === 'ProseEnd' && e.patternId === 'think')
}

function proseEnds(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'ProseEnd' }> => e._tag === 'ProseEnd' && e.patternId === 'prose')
}

// =============================================================================
// Think open: valid at start of input or after \n
// =============================================================================

describe('think open tag', () => {
  // VALID: \n<think>...\n</think>  (start of input counts as \n)
  it('at start of input: valid', () => {
    const events = parse(`${thinkTagOpen()}text\n${thinkTagClose()}`)
    expect(thinkEnds(events)).toHaveLength(1)
    expect(thinkEnds(events)[0].content).toBe('text\n')
  })

  it('at start of input: valid (char-by-char)', () => {
    const events = parseCharByChar(`${thinkTagOpen()}text\n${thinkTagClose()}`)
    expect(thinkEnds(events)).toHaveLength(1)
    expect(thinkEnds(events)[0].content).toBe('text\n')
  })

  // VALID: \n<think>...
  it('after newline: valid', () => {
    const events = parse(`hello\n${thinkTagOpen()}text\n${thinkTagClose()}`)
    expect(thinkEnds(events)).toHaveLength(1)
  })

  // NOT VALID: foo<think>...
  it('inline (no preceding newline): NOT valid', () => {
    const events = parse(`foo${thinkTagOpen()}text\n${thinkTagClose()}`)
    expect(thinkEnds(events)).toHaveLength(0)
    expect(proseEnds(events)).toHaveLength(1)
    expect(proseEnds(events)[0].content).toContain(thinkTagOpen())
  })

  it('inline (no preceding newline): NOT valid (char-by-char)', () => {
    const events = parseCharByChar(`foo${thinkTagOpen()}text\n${thinkTagClose()}`)
    expect(thinkEnds(events)).toHaveLength(0)
  })
})

// =============================================================================
// Think close: requires \n before </think>
// =============================================================================

describe('think close tag', () => {
  // VALID: <think>text\n</think>
  it('after newline: valid', () => {
    const events = parse(`${thinkTagOpen()}text\n${thinkTagClose()}`)
    expect(thinkEnds(events)).toHaveLength(1)
    expect(thinkEnds(events)[0].content).toBe('text\n')
  })

  // VALID: <think>\ntext\n</think>
  it('multiline with close after newline: valid', () => {
    const events = parse(`${thinkTagOpen()}\ntext\n${thinkTagClose()}`)
    expect(thinkEnds(events)).toHaveLength(1)
    expect(thinkEnds(events)[0].content).toBe('\ntext\n')
  })

  // Same-line close is accepted at stream end (flush resolves PendingThinkClose at depth 0)
  it('same-line close: valid at stream end', () => {
    const events = parse(`${thinkTagOpen()}text${thinkTagClose()}`)
    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toBe('text')
  })

  it('same-line close: valid at stream end (char-by-char)', () => {
    const events = parseCharByChar(`${thinkTagOpen()}text${thinkTagClose()}`)
    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toBe('text')
  })
})

// =============================================================================
// Actions open: valid at start of input or after \n
// =============================================================================

describe('actions open tag', () => {
  // VALID: <actions> at start of input
  it('at start of input: valid', () => {
    const events = parse(`${actionsTagOpen()}\n<shell>echo hi</shell>\n${actionsTagClose()}`)
    expect(events.some(e => e._tag === 'ContainerOpen')).toBe(true)
    expect(tagCloseds(events)).toHaveLength(1)
  })

  // VALID: \n<actions>
  it('after newline: valid', () => {
    const events = parse(`hello\n${actionsTagOpen()}\n<shell>echo hi</shell>\n${actionsTagClose()}`)
    expect(events.some(e => e._tag === 'ContainerOpen')).toBe(true)
  })

  // NOT VALID: foo<actions>
  it('inline (no preceding newline): NOT valid', () => {
    const events = parse(`foo${actionsTagOpen()}\n<shell>echo hi</shell>\n${actionsTagClose()}`)
    expect(events.some(e => e._tag === 'ContainerOpen')).toBe(false)
  })

  it('inline (no preceding newline): NOT valid (char-by-char)', () => {
    const events = parseCharByChar(`foo${actionsTagOpen()}\n<shell>echo hi</shell>\n${actionsTagClose()}`)
    expect(events.some(e => e._tag === 'ContainerOpen')).toBe(false)
  })
})

// =============================================================================
// Actions close: valid ONLY after \n (not start of input)
// =============================================================================

describe('actions close tag', () => {
  // VALID: \n</actions>
  it('after newline: valid', () => {
    const events = parse(`${actionsTagOpen()}\n<shell>hi</shell>\n${actionsTagClose()}`)
    expect(events.some(e => e._tag === 'ContainerClose')).toBe(true)
  })

  // NOT VALID: <actions>...</actions>  (close not after \n)
  it('same-line close (no newline before close): NOT valid', () => {
    const events = parse(`${actionsTagOpen()}${actionsTagClose()}`)
    expect(events.some(e => e._tag === 'ContainerOpen')).toBe(true)
    expect(events.some(e => e._tag === 'ContainerClose')).toBe(false)
  })

  it('same-line close: NOT valid (char-by-char)', () => {
    const events = parseCharByChar(`${actionsTagOpen()}${actionsTagClose()}`)
    expect(events.some(e => e._tag === 'ContainerOpen')).toBe(true)
    expect(events.some(e => e._tag === 'ContainerClose')).toBe(false)
  })

  // NOT VALID: inline </actions>
  it('inline close in prose: NOT valid', () => {
    const events = parse(`and ${actionsTagClose()} closes it`)
    expect(events.some(e => e._tag === 'ContainerClose')).toBe(false)
  })
})

// =============================================================================
// Think depth: only increment on \n<tagname>
// =============================================================================

describe('think depth only increments on newline-prefixed opens', () => {
  it('inline think tag inside think body does NOT increment depth', () => {
    const kw = { think: 'think', actions: 'actions' }
    const events = parse([
      thinkTagOpen(),
      `So searching for \`${thinkTagOpen()}\` returns 0 results.`,
      thinkTagClose(),
    ].join('\n'))

    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).not.toContain(thinkTagClose())
    expect(thinks[0].content).toContain(thinkTagOpen())
  })

  it('inline think tag inside think body does NOT increment depth (char-by-char)', () => {
    const events = parseCharByChar([
      thinkTagOpen(),
      `So searching for \`${thinkTagOpen()}\` returns 0 results.`,
      thinkTagClose(),
    ].join('\n'))

    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).not.toContain(thinkTagClose())
  })

  it('newline-prefixed think tag inside think body DOES increment depth', () => {
    const events = parse([
      thinkTagOpen(),
      'outer',
      thinkTagOpen(),
      'inner',
      thinkTagClose(),
      'still outer',
      thinkTagClose(),
    ].join('\n'))

    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toContain('inner')
    expect(thinks[0].content).toContain('still outer')
  })

  it('newline-prefixed think tag inside think body DOES increment depth (char-by-char)', () => {
    const events = parseCharByChar([
      thinkTagOpen(),
      'outer',
      thinkTagOpen(),
      'inner',
      thinkTagClose(),
      'still outer',
      thinkTagClose(),
    ].join('\n'))

    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toContain('inner')
    expect(thinks[0].content).toContain('still outer')
  })
})

// =============================================================================
// Inline think open+close in think body is literal (not depth-tracked)
// =============================================================================

describe('inline think tag pair in think body is literal', () => {
  it('with newline before outer close', () => {
    const events = parse(`${thinkTagOpen()}hm the think tag works like this: ${thinkTagOpen()}text${thinkTagClose()} interesting\n${thinkTagClose()}`)
    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toBe(`hm the think tag works like this: ${thinkTagOpen()}text${thinkTagClose()} interesting\n`)
  })

  it('with newline before outer close (char-by-char)', () => {
    const events = parseCharByChar(`${thinkTagOpen()}hm the think tag works like this: ${thinkTagOpen()}text${thinkTagClose()} interesting\n${thinkTagClose()}`)
    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toBe(`hm the think tag works like this: ${thinkTagOpen()}text${thinkTagClose()} interesting\n`)
  })

  it('multiline body with inline tags', () => {
    const events = parse(`${thinkTagOpen()}\nhm the think tag works like this: ${thinkTagOpen()}text${thinkTagClose()} interesting\n${thinkTagClose()}`)
    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toBe(`\nhm the think tag works like this: ${thinkTagOpen()}text${thinkTagClose()} interesting\n`)
  })

  it('multiline body with inline tags (char-by-char)', () => {
    const events = parseCharByChar(`${thinkTagOpen()}\nhm the think tag works like this: ${thinkTagOpen()}text${thinkTagClose()} interesting\n${thinkTagClose()}`)
    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toBe(`\nhm the think tag works like this: ${thinkTagOpen()}text${thinkTagClose()} interesting\n`)
  })
})

// =============================================================================
// Session repro: inline think tag in think body eats tool calls
// =============================================================================

describe('session repro: inline think tag in think body eats tool calls', () => {
  it('tool calls after think block with inline tag mention are preserved', () => {
    const events = parse([
      thinkTagOpen(),
      `So searching for \`${thinkTagOpen()}\` returns 0 results.`,
      thinkTagClose(),
      '',
      'Let me check if the pattern is being eaten.',
      '',
      actionsTagOpen(),
      '<fs-search id="s2" pattern="test" />',
      '<shell id="s4">grep foo</shell>',
      '<inspect>',
      '<ref tool="s2" />',
      '<ref tool="s4" />',
      '</inspect>',
      actionsTagClose(),
    ].join('\n'))

    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).not.toContain(actionsTagOpen())
    expect(thinks[0].content).not.toContain('<shell')

    expect(events.some(e => e._tag === 'ContainerOpen')).toBe(true)

    const tools = tagCloseds(events)
    expect(tools).toHaveLength(2)
    expect(tools[0].tagName).toBe('fs-search')
    expect(tools[0].element.attributes.get('id')).toBe('s2')
    expect(tools[1].tagName).toBe('shell')
    expect(tools[1].element.attributes.get('id')).toBe('s4')
  })

  it('same repro char-by-char', () => {
    const events = parseCharByChar([
      thinkTagOpen(),
      `So searching for \`${thinkTagOpen()}\` returns 0 results.`,
      thinkTagClose(),
      '',
      'Let me check.',
      '',
      actionsTagOpen(),
      '<shell id="s5">grep foo</shell>',
      actionsTagClose(),
    ].join('\n'))

    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).not.toContain(actionsTagOpen())

    expect(events.some(e => e._tag === 'ContainerOpen')).toBe(true)
    const tools = tagCloseds(events)
    expect(tools).toHaveLength(1)
    expect(tools[0].element.attributes.get('id')).toBe('s5')
  })
})

// =============================================================================
// Structural tags inside tool body / attribute values — always literal
// =============================================================================

describe('structural tags inside tool body remain literal', () => {
  it('think tag in tool body is literal', () => {
    const events = parse(`${actionsTagOpen()}\n<shell>echo "${thinkTagOpen()}test${thinkTagClose()}"</shell>\n${actionsTagClose()}`)
    const tools = tagCloseds(events)
    expect(tools).toHaveLength(1)
    expect(tools[0].element.body).toContain(thinkTagOpen())
    expect(tools[0].element.body).toContain(thinkTagClose())
  })
})

describe('structural tags inside attribute values remain literal', () => {
  it('think tag in attribute value is literal', () => {
    const kw = { think: 'think', actions: 'actions' }
    const events = parse(`${actionsTagOpen()}\n<fs-search id="s2" pattern="${thinkTagOpen()}" />\n${actionsTagClose()}`)
    const tools = tagCloseds(events)
    expect(tools).toHaveLength(1)
    expect(tools[0].element.attributes.get('id')).toBe('s2')
    expect(tools[0].element.attributes.get('pattern')).toBe(thinkTagOpen())
  })

  it('think tag in attribute value is literal (char-by-char)', () => {
    const events = parseCharByChar(`${actionsTagOpen()}\n<fs-search id="s2" pattern="${thinkTagOpen()}" />\n${actionsTagClose()}`)
    const tools = tagCloseds(events)
    expect(tools).toHaveLength(1)
    expect(tools[0].element.attributes.get('id')).toBe('s2')
    expect(tools[0].element.attributes.get('pattern')).toBe(thinkTagOpen())
  })
})

// =============================================================================
// "Followed by newline" rule: open/close tags valid if followed by \n
// (even without preceding \n) — handles inline OpenAI-style think tags
// =============================================================================

describe('think close tag followed by newline (no preceding newline): valid', () => {
  // The OpenAI inline case: <reason>text</reason>\n
  it('inline close followed by newline: valid', () => {
    const events = parse(`${thinkTagOpen()}text${thinkTagClose()}\n`)
    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toBe('text')
  })

  it('inline close followed by newline: valid (char-by-char)', () => {
    const events = parseCharByChar(`${thinkTagOpen()}text${thinkTagClose()}\n`)
    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toBe('text')
  })

  // Full realistic inline case: open at start, inline close, then prose after
  it('full inline think then prose: valid', () => {
    const events = parse(`${thinkTagOpen()}User sent a minimal input. Offer help${thinkTagClose()}\nSure, I can help!`)
    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toBe('User sent a minimal input. Offer help')
    const prose = proseEnds(events)
    expect(prose).toHaveLength(1)
    expect(prose[0].content).toContain('Sure, I can help!')
  })

  it('full inline think then prose: valid (char-by-char)', () => {
    const events = parseCharByChar(`${thinkTagOpen()}User sent a minimal input. Offer help${thinkTagClose()}\nSure, I can help!`)
    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toBe('User sent a minimal input. Offer help')
  })

  // Close tag NOT followed by newline and no preceding newline: still NOT valid
  it('inline close NOT followed by newline: NOT valid', () => {
    const events = parse(`${thinkTagOpen()}text${thinkTagClose()}more prose`)
    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toContain(thinkTagClose())
    expect(thinks[0].content).toContain('more prose')
  })

  it('inline close NOT followed by newline: NOT valid (char-by-char)', () => {
    const events = parseCharByChar(`${thinkTagOpen()}text${thinkTagClose()}more prose`)
    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toContain(thinkTagClose())
  })
})

describe('think open tag followed by newline (no preceding newline): valid', () => {
  // Inline open followed immediately by newline should open the think block
  it('inline open followed by newline: valid', () => {
    const events = parse(`foo${thinkTagOpen()}\ntext\n${thinkTagClose()}`)
    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toContain('text')
  })

  it('inline open followed by newline: valid (char-by-char)', () => {
    const events = parseCharByChar(`foo${thinkTagOpen()}\ntext\n${thinkTagClose()}`)
    const thinks = thinkEnds(events)
    expect(thinks).toHaveLength(1)
    expect(thinks[0].content).toContain('text')
  })

  // Inline open NOT followed by newline: still NOT valid
  it('inline open NOT followed by newline: NOT valid', () => {
    const events = parse(`foo${thinkTagOpen()}text\n${thinkTagClose()}`)
    expect(thinkEnds(events)).toHaveLength(0)
  })

  it('inline open NOT followed by newline: NOT valid (char-by-char)', () => {
    const events = parseCharByChar(`foo${thinkTagOpen()}text\n${thinkTagClose()}`)
    expect(thinkEnds(events)).toHaveLength(0)
  })
})

describe('actions close tag followed by newline (no preceding newline): valid', () => {
  it('inline actions close followed by newline: valid', () => {
    const events = parse(`${actionsTagOpen()}\n<shell>hi</shell>${actionsTagClose()}\n`)
    expect(events.some(e => e._tag === 'ContainerClose')).toBe(true)
  })

  it('inline actions close followed by newline: valid (char-by-char)', () => {
    const events = parseCharByChar(`${actionsTagOpen()}\n<shell>hi</shell>${actionsTagClose()}\n`)
    expect(events.some(e => e._tag === 'ContainerClose')).toBe(true)
  })

  it('inline actions close NOT followed by newline: NOT valid', () => {
    const events = parse(`${actionsTagOpen()}\n<shell>hi</shell>${actionsTagClose()}more`)
    expect(events.some(e => e._tag === 'ContainerClose')).toBe(false)
  })
})
