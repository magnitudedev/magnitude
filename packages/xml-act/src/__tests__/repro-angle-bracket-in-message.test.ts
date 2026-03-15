/**
 * Repro tests for the generic tag matching bug.
 *
 * In several parser states, `<` followed by `[a-zA-Z0-9_-]` enters a tag-parsing
 * state that accumulates characters until it sees `>`. If no `>` ever arrives
 * (e.g. `<20%`, `<expensive for users`), the parser gets stuck in that state and
 * swallows the rest of the stream — including real structural close tags.
 *
 * Affected sites:
 *   message.ts  — MessageBodyOpenTag for any char, should only match `message`
 *   think.ts    — LensTagAttrs / LensTagName for `<lens` inside lens body
 *   tool-body.ts — ChildTagName/ChildAttrs for valid child tag names
 */
import { describe, it, expect } from 'bun:test'
import { createStreamingXmlParser } from '../parser/streaming-xml-parser'
import { TURN_CONTROL_YIELD, commsTagOpen, commsTagClose, actionsTagOpen, actionsTagClose } from '../constants'
import type { ParseEvent } from '../parser/types'

const knownTags = new Set(['shell', 'fs-read'])
const childTagMap = new Map<string, ReadonlySet<string>>([
  ['shell', new Set(['stdin'])],
])

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

function messageBody(events: ParseEvent[]): string {
  return events
    .filter((e): e is Extract<ParseEvent, { _tag: 'MessageBodyChunk' }> => e._tag === 'MessageBodyChunk')
    .map(e => e.text)
    .join('')
}

function turnControls(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'TurnControl' }> => e._tag === 'TurnControl')
}

// =============================================================================
// message.ts — MessageBodyOpenTag enters for ANY [a-zA-Z0-9_-] after <
// Should only match `m` (prefix of `message`)
// =============================================================================

describe('message.ts: generic tag matching in message body', () => {
  it('<20% in message body swallows rest of stream including </message>', () => {
    const xml = [
      `${commsTagOpen()}`,
      `<message to="user">The difference is **<20%** between them.</message>`,
      `${commsTagClose()}`,
      `<${TURN_CONTROL_YIELD}/>`,
    ].join('\n')

    const events = parse(xml)
    const body = messageBody(events)

    // BUG: </message>, </comms>, <yield/> all leak into message body
    expect(body).not.toContain('</message>')
    expect(events.filter(e => e._tag === 'MessageTagClose')).toHaveLength(1)
    expect(turnControls(events)).toHaveLength(1)
  })

  it('<20% in message body (char-by-char streaming)', () => {
    const xml = [
      `${commsTagOpen()}`,
      `<message to="user">The difference is **<20%** between them.</message>`,
      `${commsTagClose()}`,
      `<${TURN_CONTROL_YIELD}/>`,
    ].join('\n')

    const events = parseCharByChar(xml)
    const body = messageBody(events)

    expect(body).not.toContain('</message>')
    expect(events.filter(e => e._tag === 'MessageTagClose')).toHaveLength(1)
    expect(turnControls(events)).toHaveLength(1)
  })

  it('exact production repro: pricing comparison with <20%', () => {
    const xml = [
      `<lenses>`,
      `<lens name="intent">User wants a direct pricing comparison.</lens>`,
      `</lenses>`,
      `${commsTagOpen()}`,
      `<message to="user">It's close but roughly:`,
      `- **L4 / small GPUs:** ~$1-2/hr range per GPU.`,
      `- **A100:** Similar pricing across both.`,
      `The pricing difference is usually **<20%** and shifts by region.`,
      `Don't pick your cloud based on GPU list price.</message>`,
      `${commsTagClose()}`,
      `<${TURN_CONTROL_YIELD}/>`,
    ].join('\n')

    const events = parse(xml)
    const body = messageBody(events)

    expect(body).not.toContain('</message>')
    expect(body).not.toContain('</comms>')
    expect(body).not.toContain('<yield/>')
    expect(body).toContain('<20%')
    expect(events.filter(e => e._tag === 'MessageTagOpen')).toHaveLength(1)
    expect(events.filter(e => e._tag === 'MessageTagClose')).toHaveLength(1)
    expect(events.filter(e => e._tag === 'CommsClose')).toHaveLength(1)
    expect(turnControls(events)).toHaveLength(1)
  })

  it('<expensive (no >) in message body eats everything until end of stream', () => {
    const xml = [
      `${commsTagOpen()}`,
      `<message to="user">Price is <expensive for most users. Buy now!</message>`,
      `${commsTagClose()}`,
      `<${TURN_CONTROL_YIELD}/>`,
    ].join('\n')

    const events = parse(xml)
    const body = messageBody(events)

    expect(body).not.toContain('</message>')
    expect(events.filter(e => e._tag === 'MessageTagClose')).toHaveLength(1)
    expect(turnControls(events)).toHaveLength(1)
  })
})

// =============================================================================
// think.ts — LensTagName/LensTagAttrs: <lens inside lens body without >
// enters LensTagAttrs which accumulates until > that never comes
// =============================================================================

describe('think.ts: <lens inside lens body without closing >', () => {
  it('<lens (no >) inside lens body prevents lens from closing', () => {
    const xml = [
      `<lenses>`,
      `<lens name="a">content has <lens without closing bracket rest of content.</lens>`,
      `</lenses>`,
      `${commsTagOpen()}`,
      `<message to="user">done</message>`,
      `${commsTagClose()}`,
    ].join('\n')

    const events = parse(xml)
    const lensEnds = events.filter((e): e is Extract<ParseEvent, { _tag: 'LensEnd' }> => e._tag === 'LensEnd')
    const msgClose = events.filter(e => e._tag === 'MessageTagClose')

    // BUG: <lens enters LensTagAttrs which eats everything, lens never closes
    expect(lensEnds).toHaveLength(1)
    expect(msgClose).toHaveLength(1)
  })

  it('<lens name=" (unclosed attr value) inside lens body eats entire stream', () => {
    const xml = [
      `<lenses>`,
      `<lens name="a">content has <lens name="inner without closing stuff.</lens>`,
      `</lenses>`,
      `${commsTagOpen()}`,
      `<message to="user">done</message>`,
      `${commsTagClose()}`,
    ].join('\n')

    const events = parse(xml)
    const lensEnds = events.filter((e): e is Extract<ParseEvent, { _tag: 'LensEnd' }> => e._tag === 'LensEnd')
    const msgClose = events.filter(e => e._tag === 'MessageTagClose')

    // BUG: unclosed " in attr value eats entire rest of stream
    expect(lensEnds).toHaveLength(1)
    expect(msgClose).toHaveLength(1)
  })
})

// =============================================================================
// tool-body.ts — ChildTagName/ChildAttrs: valid child tag name without >
// enters ChildAttrs which accumulates until > that never comes
// =============================================================================

describe('tool-body.ts: valid child tag name without closing >', () => {
  it('<stdin (no >) in tool body enters ChildAttrs and swallows </shell>', () => {
    const xml = [
      `${actionsTagOpen()}`,
      `<shell>echo <stdin without closing bracket</shell>`,
      `${actionsTagClose()}`,
    ].join('\n')

    const events = parse(xml)
    const shells = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> => e._tag === 'TagClosed' && e.tagName === 'shell',
    )

    // BUG: <stdin is a valid child tag for shell, enters ChildAttrs,
    // never sees >, swallows </shell> and </actions>
    expect(shells).toHaveLength(1)
    expect(shells[0].element.body).toContain('<stdin')
  })

  it('<stdin with attrs (no >) in tool body eats entire stream', () => {
    const xml = [
      `${actionsTagOpen()}`,
      `<shell>echo <stdin without ever closing the angle bracket and the rest</shell>`,
      `${actionsTagClose()}`,
      `<${TURN_CONTROL_YIELD}/>`,
    ].join('\n')

    const events = parse(xml)
    const shells = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> => e._tag === 'TagClosed' && e.tagName === 'shell',
    )

    expect(shells).toHaveLength(1)
    expect(turnControls(events)).toHaveLength(1)
  })
})
