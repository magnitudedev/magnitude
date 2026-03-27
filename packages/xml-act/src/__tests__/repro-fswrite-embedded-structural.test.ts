import { describe, it, expect } from 'bun:test'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'

/**
 * Repro: <write> body contains embedded structural tags like <results>,
 * </results>, <think>, <actions>, and tool tags with observe attrs.
 *
 * The parser should treat ALL content inside the write body as literal
 * text — not as structural open/close events. But embedded structural-looking
 * close tags may cause the parser to break out of the tool body prematurely.
 *
 * From a real scenario where the LLM writes a file containing XML-like
 * content that references protocol tags.
 */

const knownTags = new Set(['write', 'read', 'grep', 'agent-create', 'shell'])
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

// The actual content from the breaking scenario — write body contains
// embedded structural tags like <think>, <actions>, tool tags, and </results>
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
          '<think>I should explore the codebase first.</think>\\n' +
          AO + '\\n' +
          '<agent-create agentId="explorer-1">\\n' +
          '<type>explorer</type>\\n' +
          '<title>Explore codebase</title>\\n' +
          '<prompt>Map the API routes.</prompt>\\n' +
          '</agent-create>\\n' +
          AC,
        ],
      },
      {
        role: 'user',
        content: [
          '<results>\\n</results>\\n' +
          '<agent_response from="explorer-1">\\n' +
          'Found 3 route files.\\n' +
          '</agent_response>\\n' +
          '<agents_status>\\n' +
          '- explorer-1 (explorer): idle\\n' +
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
          '<read path="src/styles/tokens.ts" observe="." />\\n' +
          '<grep pattern="header" path="src/" observe="//item" />\\n' +
          AC,
        ],
      },
      {
        role: 'user',
        content: [
          '<results>\\n' +
          '<read observe=".">export const colors = {}</read>\\n' +
          '<grep observe="//item">\\n' +
          '<item file="src/app.ts">12|  header</item>\\n' +
          '</grep>\\n' +
          '</results>',
        ],
      },
    ],
  },
]
`

const FULL_XML = `<actions>
<write path="evals/src/evals/a5/scenarios.ts">${FS_WRITE_BODY}</write>
<read path="evals/src/evals/a5/scenarios.ts" observe="content" />
</actions>`

describe('isolation: embedded tool-like tags are treated as literal text', () => {
  it('write body with an observe-bearing tool tag stays literal', () => {
    const xml = '<actions>\n<write path="x.ts">before <read path="foo" observe="." /> after</write>\n</actions>'
    const events = parse(xml)
    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'write',
    )
    expect(closed).toHaveLength(1)
    expect(closed[0].element.body).toBe('before <read path="foo" observe="." /> after')
  })

  it('write body with unknown tags like <foo> is fine (treated as literal)', () => {
    const xml = '<actions>\n<write path="x.ts">before <foo bar="1">baz</foo> after</write>\n</actions>'
    const events = parse(xml)
    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'write',
    )
    expect(closed).toHaveLength(1)
    expect(closed[0].element.body).toContain('before')
    expect(closed[0].element.body).toContain('after')
  })

  it('write body with known tag <read> is fine (not valid child, flushed back)', () => {
    const xml = '<actions>\n<write path="x.ts">before <read path="y" /> after</write>\n</actions>'
    const events = parse(xml)
    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'write',
    )
    expect(closed).toHaveLength(1)
    expect(closed[0].element.body).toBe('before <read path="y" /> after')
  })
})

describe('repro: write body with embedded structural tags', () => {
  it('parses write with embedded tool and structural tags in body', () => {
    const events = parse(FULL_XML)

    // Should have actions open/close
    expect(events.some(e => e._tag === 'ContainerOpen')).toBe(true)
    expect(events.some(e => e._tag === 'ContainerClose')).toBe(true)

    // write should be parsed as a complete tool
    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'write',
    )
    expect(closed).toHaveLength(1)
    expect(closed[0].element.attributes.get('path')).toBe('evals/src/evals/a5/scenarios.ts')

    // The body should contain the full file content
    expect(closed[0].element.body).toBe(FS_WRITE_BODY)

    // No parse errors
    const errors = events.filter(e => e._tag === 'ParseError')
    expect(errors).toHaveLength(0)
  })

  it('same result when streamed char-by-char', () => {
    const events = parseCharByChar(FULL_XML)

    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'write',
    )
    expect(closed).toHaveLength(1)
    expect(closed[0].element.body).toBe(FS_WRITE_BODY)
    expect(closed[0].element.attributes.get('path')).toBe('evals/src/evals/a5/scenarios.ts')

    expect(events.some(e => e._tag === 'ContainerOpen')).toBe(true)
    expect(events.some(e => e._tag === 'ContainerClose')).toBe(true)
    expect(events.filter(e => e._tag === 'ParseError')).toHaveLength(0)
  })

  it('embedded </results> inside write body does not corrupt the body', () => {
    const xml = `<actions>
<write path="test.ts">some code with </results> in it and observe="." too</write>
</actions>`

    const events = parse(xml)

    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'write',
    )
    expect(closed).toHaveLength(1)
    expect(closed[0].element.body).toBe('some code with </results> in it and observe="." too')
    expect(events.some(e => e._tag === 'ContainerOpen')).toBe(true)
    expect(events.some(e => e._tag === 'ContainerClose')).toBe(true)
    expect(events.filter(e => e._tag === 'ParseError')).toHaveLength(0)
  })

  it('embedded <think> and </think> inside write body are treated as literal text', () => {
    const xml = `<actions>
<write path="test.ts">const x = '<think>hello</think>'</write>
</actions>`

    const events = parse(xml)

    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'write',
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

  it('embedded <actions> inside write body does not create nested actions block', () => {
    const xml = `<actions>
<write path="test.ts">code with <actions> and </actions> inside</write>
</actions>`

    const events = parse(xml)

    // Should have exactly ONE ContainerOpen (the outer one)
    const actionsOpens = events.filter(e => e._tag === 'ContainerOpen')
    expect(actionsOpens).toHaveLength(1)

    const closed = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'write',
    )
    expect(closed).toHaveLength(1)
    expect(closed[0].element.body).toBe('code with <actions> and </actions> inside')
  })
})
