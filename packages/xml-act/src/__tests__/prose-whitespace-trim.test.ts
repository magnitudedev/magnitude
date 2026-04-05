import { describe, expect, it } from 'bun:test'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'
import { LENSES_OPEN, LENSES_CLOSE, TURN_CONTROL_IDLE } from '../constants'

const TASK_OPEN = '<task id="t2">'
const TASK_CLOSE = '</task>'

function parse(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser()
  return [...parser.processChunk(xml), ...parser.flush()]
}

function parseChunks(chunks: string[]): ParseEvent[] {
  const parser = createStreamingXmlParser()
  const events: ParseEvent[] = []
  for (const chunk of chunks) events.push(...parser.processChunk(chunk))
  events.push(...parser.flush())
  return events
}

describe('prose whitespace trimming', () => {
  it('no whitespace-only ProseChunk between lenses and task blocks', () => {
    const xml = [
      LENSES_OPEN,
      '\n<lens name="intent">Just a greeting.</lens>\n',
      LENSES_CLOSE,
      '\n',
      TASK_OPEN,
      '\n<message>Hey!</message>\n',
      TASK_CLOSE,
      '\n',
      TURN_CONTROL_IDLE,
    ].join('')

    const events = parse(xml)
    const proseChunks = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'ProseChunk' }> =>
        e._tag === 'ProseChunk' && e.patternId === 'prose',
    )
    const whitespaceOnlyProse = proseChunks.filter(e => e.text.trim() === '')

    expect(whitespaceOnlyProse).toEqual([])
  })

  it('no ProseEnd for whitespace-only content between structural tags', () => {
    const xml = [
      LENSES_OPEN,
      '\n<lens name="intent">Just a greeting.</lens>\n',
      LENSES_CLOSE,
      '\n',
      TASK_OPEN,
      '\n<message>Hey!</message>\n',
      TASK_CLOSE,
      '\n',
      TURN_CONTROL_IDLE,
    ].join('')

    const events = parse(xml)
    const proseEnds = events.filter(e => e._tag === 'ProseEnd')

    expect(proseEnds).toEqual([])
  })

  it('actual prose content before structural tags is preserved', () => {
    const xml = [
      'Hello world\n',
      LENSES_OPEN,
      '\n<lens name="intent">thinking</lens>\n',
      LENSES_CLOSE,
    ].join('')

    const events = parse(xml)
    const proseChunks = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'ProseChunk' }> => e._tag === 'ProseChunk',
    )
    const combined = proseChunks.map(e => e.text).join('')

    expect(combined).toContain('Hello world')
  })

  it('whitespace-only prose between lenses and task blocks across chunks', () => {
    const chunks = [
      LENSES_OPEN + '\n<lens name="turn">planning</lens>\n' + LENSES_CLOSE,
      '\n' + TASK_OPEN + '\n<message>Hey Anders! What can',
      ' I help you with?</message>\n' + TASK_CLOSE + '\n' + TURN_CONTROL_IDLE,
    ]

    const events = parseChunks(chunks)
    const proseChunks = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'ProseChunk' }> =>
        e._tag === 'ProseChunk' && e.patternId === 'prose',
    )
    const whitespaceOnlyProse = proseChunks.filter(e => e.text.trim() === '')

    expect(whitespaceOnlyProse).toEqual([])
  })
})
