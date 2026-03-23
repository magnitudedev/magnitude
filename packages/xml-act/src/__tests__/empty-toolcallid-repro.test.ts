import { describe, it, expect } from 'bun:test'
import { createStreamingXmlParser } from '../parser/streaming-xml-parser'
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

function getTagOpened(events: ParseEvent[], tagName: string): Extract<ParseEvent, { _tag: 'TagOpened' }> {
  const event = events.find(
    (e): e is Extract<ParseEvent, { _tag: 'TagOpened' }> =>
      e._tag === 'TagOpened' && e.tagName === tagName,
  )
  expect(event).toBeDefined()
  return event!
}

describe('empty toolCallId repro', () => {
  it('repro: no-attributes tool tag should still have non-empty toolCallId (currently fails)', () => {
    const events = parse('<actions>\n<shell>echo hello</shell>\n</actions>\n<next/>')
    const opened = getTagOpened(events, 'shell')
    expect(opened.toolCallId).toBeTruthy()
  })

  it('with attributes, tool tag gets non-empty toolCallId (working path)', () => {
    const events = parse('<actions>\n<shell timeout="10">echo hello</shell>\n</actions>\n<next/>')
    const opened = getTagOpened(events, 'shell')
    expect(opened.toolCallId).toBeTruthy()
  })

  it('self-closing fs-read with attribute and no space before /> gets non-empty toolCallId', () => {
    const events = parse('<actions>\n<fs-read path="x.ts"/>\n</actions>\n<next/>')
    const opened = getTagOpened(events, 'fs-read')
    expect(opened.toolCallId).toBeTruthy()
  })

  it('self-closing fs-read with attribute and space before /> gets non-empty toolCallId', () => {
    const events = parse('<actions>\n<fs-read path="x.ts" />\n</actions>\n<next/>')
    const opened = getTagOpened(events, 'fs-read')
    expect(opened.toolCallId).toBeTruthy()
  })

  it('char-by-char repro: no-attributes tool tag should still have non-empty toolCallId (currently fails)', () => {
    const events = parseCharByChar('<actions>\n<shell>echo hello</shell>\n</actions>\n<next/>')
    const opened = getTagOpened(events, 'shell')
    expect(opened.toolCallId).toBeTruthy()
  })
})
