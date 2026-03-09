import { describe, it, expect } from 'bun:test'
import { createStreamingXmlParser } from '../parser/streaming-xml-parser'
import type { ParseEvent } from '../parser/types'

/**
 * Repro: <fs-write> body contains embedded structural tags like <inspect>,
 * </inspect>, <results>, </results>, <think>, <actions>, <ref> etc.
 *
 * The parser should treat ALL content inside the fs-write body as literal
 * text — not as structural open/close events. But the embedded </inspect>
 * and </results> (or other structural-looking close tags) may cause the
 * parser to break out of the tool body prematurely.
 *
 * From a real scenario where the LLM writes a file containing XML-like
 * content that references structural tags.
 */

const knownTags = new Set(['fs-write', 'fs-read', 'fs-search', 'agent-create', 'shell'])
const childTagMap = new Map<string, Set<string>>()

function parse(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  return [...parser.processChunk(xml), ...parser.flush()]
}

function parseCharByChar(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  const events: ParseEvent[] = []
  for (const ch of xml) {
    events.push(...parser.processChunk(ch))
  }
  events.push(...parser.flush())
  return events
}

// The actual content from the breaking scenario — fs-write body contains
// embedded structural tags like <think>, <actions>, <inspect>, <ref>, </inspect>, </results>
const FS_WRITE_BODY = `import type { Scenario } from '../../types'
import { hasThinkBlock } from './checks'
import { actionsTagOpen, actionsTagClose } from '@magnitudedev/xml-act'

const AO = actionsTagOpen()
const AC = actionsTagClose()

export const ALL_SCENARIOS = [
  {
    id: 'tenet1/communicate-before-building',
    messages: [
      { role: 'user', content: ['<session_context>\\nGit branch: main\\n</session_context>'] },
      { role: 'user', content: ['<user mode="text">add rate limiting</user>'] },
      {
        role: 'assistant',
        content: [
          '<think>I should scout the codebase first.</think>\\n' +
          AO + '\\n' +
          '<agent-create agentId="scout-1">\\n' +
          '<type>scout</type>\\n' +
          '<title>Scout codebase</title>\\n' +
          '<prompt>Map the API routes.</prompt>\\n' +
          '</agent-create>\\n' +
          AC,
        ],
      },
      {
        role: 'user',
        content: [
          '<results>\\n</results>\\n' +
          '<agent_response from="scout-1">\\n' +
          'Found 3 route files.\\n' +
          '</agent_response>\\n' +
          '<agents_status>\\n' +
          '- scout-1 (scout): idle\\n' +
          '</agents_status>',
        ],
      },
    ],
  },
  {
    id: 'tenet2/surface-color-assumption',
    messages: [
      {
        role: 'assistant',
        content: [
          '<think>I need to find the header.</think>\\n' +
          AO + '\\n' +
          '<fs-read path="src/styles/tokens.ts" />\\n' +
          '<fs-search pattern="header" path="src/" />\\n' +
          '<inspect>\\n' +
          '<ref tool="fs-read" />\\n' +
          '<ref tool="fs-search" />\\n' +
          '</inspect>\\n' +
          AC,
        ],
      },
      {
        role: 'user',
        content: [
          '<results>\\n' +
          '<inspect>\\n' +
          '<ref tool="fs-read">export const colors = {}</ref>\\n' +
          '<ref tool="fs-search">\\n' +
          '<item file="src/app.ts">12|  header</item>\\n' +
          '</ref>\\n' +
          '</inspect>\\n' +
          '</results>',
        ],
      },
    ],
  },
]
`

const FULL_XML = `<actions>
<fs-write path="evals/src/evals/a5/scenarios.ts">${FS_WRITE_BODY}</fs-write>
<inspect>
<ref tool="fs-write" />
</inspect>
</actions>`

describe('isolation: ref is the sole issue', () => {
  it('fs-write body with <ref> gets ref consumed as child, corrupting body', () => {
    const xml = '<actions>\n<fs-write path="x.ts">before <ref tool="foo" /> after</fs-write>\n</actions>'
    const events = parse(xml)
    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'fs-write',
    )
    expect(closed).toHaveLength(1)
    // This is the bug: <ref> gets eaten as a child instead of literal body text
    expect(closed[0].element.body).toBe('before <ref tool="foo" /> after')
  })

  it('fs-write body with non-ref unknown tags like <foo> is fine (treated as literal)', () => {
    const xml = '<actions>\n<fs-write path="x.ts">before <foo bar="1">baz</foo> after</fs-write>\n</actions>'
    const events = parse(xml)
    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'fs-write',
    )
    expect(closed).toHaveLength(1)
    expect(closed[0].element.body).toContain('before')
    expect(closed[0].element.body).toContain('after')
  })

  it('fs-write body with known tag <fs-read> is fine (not valid child, flushed back)', () => {
    const xml = '<actions>\n<fs-write path="x.ts">before <fs-read path="y" /> after</fs-write>\n</actions>'
    const events = parse(xml)
    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'fs-write',
    )
    expect(closed).toHaveLength(1)
    expect(closed[0].element.body).toBe('before <fs-read path="y" /> after')
  })
})

describe('repro: fs-write body with embedded structural tags', () => {
  it('parses fs-write with embedded <inspect>, </inspect>, <results>, </results> in body', () => {
    const events = parse(FULL_XML)

    // Should have actions open/close
    expect(events.some(e => e._tag === 'ActionsOpen')).toBe(true)
    expect(events.some(e => e._tag === 'ActionsClose')).toBe(true)

    // fs-write should be parsed as a complete tool
    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'fs-write',
    )
    expect(closed).toHaveLength(1)
    expect(closed[0].element.attributes.get('path')).toBe('evals/src/evals/a5/scenarios.ts')

    // The body should contain the full file content
    expect(closed[0].element.body).toBe(FS_WRITE_BODY)

    // Inspect block after the fs-write should still be parsed
    const inspectOpen = events.filter(e => e._tag === 'InspectOpen')
    const inspectClose = events.filter(e => e._tag === 'InspectClose')
    expect(inspectOpen).toHaveLength(1)
    expect(inspectClose).toHaveLength(1)

    // No parse errors
    const errors = events.filter(e => e._tag === 'ParseError')
    expect(errors).toHaveLength(0)
  })

  it('same result when streamed char-by-char', () => {
    const events = parseCharByChar(FULL_XML)

    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'fs-write',
    )
    expect(closed).toHaveLength(1)
    expect(closed[0].element.body).toBe(FS_WRITE_BODY)
    expect(closed[0].element.attributes.get('path')).toBe('evals/src/evals/a5/scenarios.ts')

    expect(events.some(e => e._tag === 'ActionsOpen')).toBe(true)
    expect(events.some(e => e._tag === 'ActionsClose')).toBe(true)
    expect(events.filter(e => e._tag === 'ParseError')).toHaveLength(0)
  })

  it('embedded </inspect> inside fs-write body does not corrupt the body', () => {
    const xml = `<actions>
<fs-write path="test.ts">some code with </inspect> in it and </results> too</fs-write>
</actions>`

    const events = parse(xml)

    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'fs-write',
    )
    expect(closed).toHaveLength(1)
    expect(closed[0].element.body).toBe('some code with </inspect> in it and </results> too')
    expect(events.some(e => e._tag === 'ActionsOpen')).toBe(true)
    expect(events.some(e => e._tag === 'ActionsClose')).toBe(true)
    expect(events.filter(e => e._tag === 'ParseError')).toHaveLength(0)
  })

  it('embedded <think> and </think> inside fs-write body are treated as literal text', () => {
    const xml = `<actions>
<fs-write path="test.ts">const x = '<think>hello</think>'</fs-write>
</actions>`

    const events = parse(xml)

    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'fs-write',
    )
    expect(closed).toHaveLength(1)
    expect(closed[0].element.body).toBe("const x = '<think>hello</think>'")

    // Should NOT have a ProseEnd for the embedded think
    const thinkEnds = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'ProseEnd' }> =>
        e._tag === 'ProseEnd' && e.patternId === 'think',
    )
    expect(thinkEnds).toHaveLength(0)
  })

  it('embedded <actions> inside fs-write body does not create nested actions block', () => {
    const xml = `<actions>
<fs-write path="test.ts">code with <actions> and </actions> inside</fs-write>
</actions>`

    const events = parse(xml)

    // Should have exactly ONE ActionsOpen (the outer one)
    const actionsOpens = events.filter(e => e._tag === 'ActionsOpen')
    expect(actionsOpens).toHaveLength(1)

    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'fs-write',
    )
    expect(closed).toHaveLength(1)
    expect(closed[0].element.body).toBe('code with <actions> and </actions> inside')
  })
})
