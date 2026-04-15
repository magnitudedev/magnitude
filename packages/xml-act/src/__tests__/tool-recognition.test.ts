import { describe, expect, test } from 'vitest'
import { createStreamingXmlParser } from '../parser'

function parse(xml: string) {
  const knownTags = new Set(['skill', 'read', 'edit', 'shell', 'create-task', 'spawn-worker', 'message'])
  const parser = createStreamingXmlParser(knownTags)
  parser.push(xml)
  parser.end()
  return parser.events
}

function eventTags(events: ReturnType<typeof parse>): string[] {
  return events.map(e => e._tag)
}

function tagNames(events: ReturnType<typeof parse>): string[] {
  return events
    .filter(e => e._tag === 'TagOpened')
    .map(e => (e as any).tagName)
}

function toolExecutedNames(events: ReturnType<typeof parse>): string[] {
  return events
    .filter(e => e._tag === 'ToolExecuted')
    .map(e => (e as any).tagName ?? (e as any).element?.tagName)
}

function proseTexts(events: ReturnType<typeof parse>): string[] {
  return events
    .filter(e => e._tag === 'ProseChunk')
    .map(e => (e as any).text)
}

function messageTexts(events: ReturnType<typeof parse>): string[] {
  return events
    .filter(e => e._tag === 'MessageChunk' || e._tag === 'MessageStart' || e._tag === 'MessageEnd')
    .map(e => (e as any).text ?? '')
}

describe('Tool recognition', () => {
  // =========================================================================
  // RULE: Tools are always recognized outside of messages, regardless of
  // newline position or surrounding prose.
  // =========================================================================
  describe('outside message: tools always recognized', () => {
    test('tool on its own line', () => {
      const xml = `<skill name="bug" />
`
      const events = parse(xml)
      expect(tagNames(events)).toContain('skill')
    })

    test('tool immediately after closing lens tag, no newline', () => {
      const xml = `<lens name="skills">thinking</lens><skill name="bug" />
`
      const events = parse(xml)
      expect(tagNames(events)).toContain('skill')
    })

    test('tool after closing lens tag with newline', () => {
      const xml = `<lens name="skills">thinking</lens>
<skill name="bug" />
`
      const events = parse(xml)
      expect(tagNames(events)).toContain('skill')
    })

    test('tool after prose text on same line', () => {
      const xml = `Now let me fix this: <edit observe="." path="src/file.ts"><old>old</old><new>new</new></edit>
`
      const events = parse(xml)
      expect(tagNames(events)).toContain('edit')
    })

    test('tool after prose text with newline', () => {
      const xml = `Now let me fix this:
<edit observe="." path="src/file.ts"><old>old</old><new>new</new></edit>
`
      const events = parse(xml)
      expect(tagNames(events)).toContain('edit')
    })

    test('multiple tools on same line', () => {
      const xml = `<skill name="bug" /><read path="src/file.ts" />
`
      const events = parse(xml)
      expect(tagNames(events)).toContain('skill')
      expect(tagNames(events)).toContain('read')
    })

    test('tool between prose chunks on same line', () => {
      const xml = `Let me read the file: <read path="src/file.ts" /> and check the result
`
      const events = parse(xml)
      expect(tagNames(events)).toContain('read')
    })

    test('tool after closing tag of another tool, same line', () => {
      const xml = `<read path="a.ts" /><edit observe="." path="b.ts"><old>x</old><new>y</new></edit>
`
      const events = parse(xml)
      expect(tagNames(events)).toContain('read')
      expect(tagNames(events)).toContain('edit')
    })
  })

  // =========================================================================
  // RULE: Inside a <message> tag, everything is prose. No tools recognized.
  // Angle brackets in message content are literal text.
  // =========================================================================
  describe('inside message: tools NOT recognized', () => {
    test('known tool tag inside message is prose, not a tool', () => {
      const xml = `<message to="user">Use the <skill name="bug" /> tool</message>
`
      const events = parse(xml)
      expect(tagNames(events)).not.toContain('skill')
      // The message should contain the literal text
      const msgChunks = events
        .filter(e => e._tag === 'MessageChunk')
        .map(e => (e as any).text)
      const fullMessage = msgChunks.join('')
      expect(fullMessage).toContain('skill')
    })

    test('known tool tag with body inside message is prose', () => {
      const xml = `<message to="user">Run <edit observe="." path="file.ts"><old>old</old><new>new</new></edit> to fix it</message>
`
      const events = parse(xml)
      expect(tagNames(events)).not.toContain('edit')
    })

    test('read tag inside message is prose', () => {
      const xml = `<message to="user">Check <read path="src/file.ts" /> for details</message>
`
      const events = parse(xml)
      expect(tagNames(events)).not.toContain('read')
    })

    test('nested message with tool inside is prose', () => {
      const xml = `<message to="user">Nesting: <message to="worker"><read path="file.ts" /></message></message>
`
      const events = parse(xml)
      // Inner read should be prose, not a tool
      expect(tagNames(events)).not.toContain('read')
    })
  })

  // =========================================================================
  // RULE: Message content is preserved as literal text, including angle
  // brackets that would normally be tool tags.
  // =========================================================================
  describe('message content preserved as literal text', () => {
    test('angle brackets in message content', () => {
      const xml = `<message to="user">Use the <Redirect> component from react-router</message>
`
      const events = parse(xml)
      const msgChunks = events
        .filter(e => e._tag === 'MessageChunk')
        .map(e => (e as any).text)
      const fullMessage = msgChunks.join('')
      expect(fullMessage).toContain('<Redirect>')
    })

    test('HTML-like content in message', () => {
      const xml = `<message to="user">The element <div class="foo">bar</div> should be styled</message>
`
      const events = parse(xml)
      const msgChunks = events
        .filter(e => e._tag === 'MessageChunk')
        .map(e => (e as any).text)
      const fullMessage = msgChunks.join('')
      expect(fullMessage).toContain('<div class="foo">bar</div>')
    })
  })

  // =========================================================================
  // EDGE CASES
  // =========================================================================
  describe('edge cases', () => {
    test('tool after closed message on same line', () => {
      const xml = `<message to="user">Done.</message><skill name="bug" />
`
      const events = parse(xml)
      expect(tagNames(events)).toContain('skill')
    })

    test('tool on new line after message', () => {
      const xml = `<message to="user">Done.</message>
<skill name="bug" />
`
      const events = parse(xml)
      expect(tagNames(events)).toContain('skill')
    })

    test('lens + message + tool sequence', () => {
      const xml = `<lens name="skills">Activate skill.</lens><skill name="bug" /><message to="user">Found a bug.</message><read path="file.ts" />
`
      const events = parse(xml)
      expect(tagNames(events)).toContain('skill')
      expect(tagNames(events)).toContain('read')
    })
  })
})
