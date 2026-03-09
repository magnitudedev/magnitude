import { describe, it, expect, afterAll } from 'bun:test'
import { createStreamingXmlParser } from '../parser/streaming-xml-parser'
import { useAltKeywords, useDefaultKeywords } from '../constants'
import type { ParseEvent } from '../parser/types'

afterAll(() => useDefaultKeywords())

/**
 * Repro: <reason> (think tag with alt keywords) inside think body text
 * increments depth counter, causing the real </reason> close tag to be
 * swallowed. Everything after — including <tooluse> and all tool calls —
 * gets eaten into the think block body.
 *
 * From a real session where the LLM output:
 *   <reason>
 *   So searching for `<reason>` returns 0 results...
 *   </reason>
 *
 *   <tooluse>
 *   <shell id="s5">...</shell>
 *   </tooluse>
 *
 * The `<reason>` inside backticks increments think depth. The real
 * </reason> only decrements back to 0 but doesn't close. The entire
 * <tooluse> block is swallowed into the think body → 0 tool calls.
 */
describe('repro: think tag name in think body causes depth mismatch', () => {
  it('reason mentioned in think body eats all subsequent tool calls', () => {
    useAltKeywords()

    const knownTags = new Set(['fs-search', 'shell'])
    const parser = createStreamingXmlParser(knownTags, new Map())

    const xml = [
      '<reason>',
      'So searching for `<reason>` returns 0 results.',
      '</reason>',
      '',
      'Let me check if the pattern is being eaten by the XML protocol.',
      '',
      '<tooluse>',
      '<shell id="s5">grep -rn \'reason>\' packages/xml-act/src/ 2>&1 | head -10</shell>',
      '<inspect>',
      '<ref tool="s5" />',
      '</inspect>',
      '</tooluse>',
    ].join('\n')

    const events = [...parser.processChunk(xml), ...parser.flush()]

    // The think block should close at the first </reason>
    const thinkEnd = events.find(
      (e): e is Extract<ParseEvent, { _tag: 'ProseEnd' }> =>
        e._tag === 'ProseEnd' && e.patternId === 'think',
    )
    expect(thinkEnd).toBeDefined()
    expect(thinkEnd!.content).not.toContain('<tooluse>')
    expect(thinkEnd!.content).not.toContain('<shell')

    // The tooluse block should be parsed as actions
    const actionsOpen = events.find(e => e._tag === 'ActionsOpen')
    expect(actionsOpen).toBeDefined()

    // The shell tool should be parsed with its attributes
    const shellClosed = events.find(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'shell',
    )
    expect(shellClosed).toBeDefined()
    expect(shellClosed!.element.attributes.get('id')).toBe('s5')
  })
})
