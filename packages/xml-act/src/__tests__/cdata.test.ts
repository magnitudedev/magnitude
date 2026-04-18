import { describe, expect, it } from 'vitest'
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
    const events = parse(`<shell>${cdata('echo "<hello>" && cat <file>')}</shell>`)
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('echo "<hello>" && cat <file>')
  })

  it('3) CDATA content inside a message preserves raw text', () => {
    const events = parse(`<task id="t2"><message>${cdata('Use  for tools')}</message>`)
    const msg = byTag(events, 'MessageChunk').map(e => e.text).join('')
    expect(msg).toBe('Use  for tools')
  })

  it('4) CDATA split across multiple chunks is assembled correctly', () => {
    const events = parseChunks([
      '<shell><!' + '[CDATA[some <content>',
      ' here' + ']]></shell>',
    ])
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('some <content> here')
  })

  it('5) multiple CDATA sections in one tool body are both captured', () => {
    const events = parse(`<shell>${cdata('first')}${cdata(' second')}</shell>`)
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('first second')
  })

  it('6) empty CDATA still allows tool to close properly', () => {
    const events = parse(`<shell>${cdata('')}</shell>`)
    const closed = byTag(events, 'TagClosed').find(e => e.tagName === 'shell')
    expect(closed).toBeDefined()
  })

  it('7) CDATA preserves XML-like content without parsing nested tags', () => {
    const raw = '<shell>nested</shell>'
    const events = parse(`<shell>${cdata(raw)}</shell>`)

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

  it('9) CDATA close split as ]] + > across chunks', () => {
    const events = parseChunks([
      '<shell><!' + '[CDATA[hello world]' + ']',
      '></shell>',
    ])
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('hello world')
  })

  it('10) CDATA close split as ] + ]> across chunks', () => {
    const events = parseChunks([
      '<shell><!' + '[CDATA[hello world]',
      ']></shell>',
    ])
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('hello world')
  })

  it('11) CDATA close split across three chunks as ] + ] + >', () => {
    const events = parseChunks([
      '<shell><!' + '[CDATA[hello world]',
      ']',
      '></shell>',
    ])
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('hello world')
  })

  it('12) content containing ] before the real close is preserved', () => {
    const events = parseChunks([
      '<shell><!' + '[CDATA[data]more',
      ']]></shell>',
    ])
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('data]more')
  })

  it('13) content containing ]] that is not a close is preserved', () => {
    const events = parseChunks([
      '<shell><!' + '[CDATA[arr[0]]stuff',
      ' more]]></shell>',
    ])
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('arr[0]]stuff more')
  })

  it('14) multiple CDATA sections still work when one close is split across chunks', () => {
    const events = parseChunks([
      '<shell><!' + '[CDATA[first]]><!' + '[CDATA[ second]',
      ']',
      '></shell>',
    ])
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('first second')
  })

  it('15) CDATA close where ]] ends a chunk and > starts the next chunk is assembled correctly', () => {
    const events = parseChunks([
      '<shell><!' + '[CDATA[mid chunk close]]',
      '></shell>',
    ])
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('mid chunk close')
  })

  it('16) single ] at chunk boundary that is not part of a close remains content', () => {
    const events = parseChunks([
      '<shell><!' + '[CDATA[edge]',
      'case]]></shell>',
    ])
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('edge]case')
  })

  it('17) content ending with ]] before the real close is preserved', () => {
    const events = parseChunks([
      '<shell><!' + '[CDATA[tricky]]',
      'tail]]></shell>',
    ])
    const body = byTag(events, 'BodyChunk').map(e => e.text).join('')
    expect(body).toBe('tricky]]tail')
  })
})
