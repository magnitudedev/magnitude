import { describe, expect, it } from 'bun:test'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'

const knownTags = new Set(['shell', 'edit'])
const childTagMap = new Map<string, Set<string>>([
  ['edit', new Set(['old', 'new'])],
])

const CDATA_OPEN = '<!' + '[CDATA['
const CDATA_CLOSE = ']]>'

function cdata(text: string): string {
  return `${CDATA_OPEN}${text}${CDATA_CLOSE}`
}

function parse(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  return [...parser.processChunk(xml), ...parser.flush()]
}

function parseChunks(chunks: string[]): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  const events: ParseEvent[] = []
  for (const chunk of chunks) events.push(...parser.processChunk(chunk))
  events.push(...parser.flush())
  return events
}

function byTag<T extends ParseEvent['_tag']>(
  events: ParseEvent[],
  tag: T,
): Extract<ParseEvent, { _tag: T }>[] {
  return events.filter((e): e is Extract<ParseEvent, { _tag: T }> => e._tag === tag)
}

describe('CDATA support', () => {
  it('1) CDATA content passes through as plain text in prose', () => {
    const events = parse(`Hello ${cdata('<world> & friends')} done`)
    const prose = byTag(events, 'ProseChunk').map(e => e.text).join('')
    expect(prose).toBe('Hello <world> & friends done')
  })

  it('2) CDATA content inside a tool body preserves raw text including angle brackets', () => {
    const events = parse(`<actions><shell>${cdata('echo "<hello>" && cat <file>')}</shell></actions>`)
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('echo "<hello>" && cat <file>')
  })

  it('3) CDATA content inside a message preserves raw text', () => {
    const events = parse(`<comms><message to="user">${cdata('Use <actions> for tools')}</message></comms>`)
    const msg = byTag(events, 'MessageChunk').map(e => e.text).join('')
    expect(msg).toBe('Use <actions> for tools')
  })

  it('4) CDATA split across multiple chunks is assembled correctly', () => {
    const events = parseChunks([
      '<actions><shell><!',
      '[CDATA[some <content>',
      ' here]]></shell></actions>',
    ])
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('some <content> here')
  })

  it('5) multiple CDATA sections in one tool body are both captured', () => {
    const events = parse(`<actions><shell>${cdata('first')}${cdata(' second')}</shell></actions>`)
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('first second')
  })

  it('6) empty CDATA still allows tool to close properly', () => {
    const events = parse(`<actions><shell>${cdata('')}</shell></actions>`)
    const closed = byTag(events, 'TagClosed').find(e => e.tagName === 'shell')
    expect(closed).toBeDefined()
  })

  it('7) CDATA preserves XML-like content without parsing nested tags', () => {
    const raw = '<actions><shell>nested</shell></actions>'
    const events = parse(`<actions><shell>${cdata(raw)}</shell></actions>`)

    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe(raw)

    const shellOpened = byTag(events, 'TagOpened').filter(e => e.tagName === 'shell')
    expect(shellOpened).toHaveLength(1)
  })

  it('8) unclosed CDATA at EOF is handled gracefully (content emitted as prose)', () => {
    const events = parse(`before ${CDATA_OPEN}unterminated <x>`)
    const prose = byTag(events, 'ProseChunk').map(e => e.text).join('')
    expect(prose).toBe(`before ${CDATA_OPEN}unterminated <x>`)
  })
})
