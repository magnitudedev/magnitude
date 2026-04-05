import { describe, it, expect } from 'vitest'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'

const knownTags = new Set(['shell'])
const childTagMap = new Map<string, Set<string>>()

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

function splitAtEveryPosition(xml: string): string[][] {
  const chunks: string[][] = [[xml]]
  for (let i = 1; i < xml.length; i++) {
    chunks.push([xml.slice(0, i), xml.slice(i)])
  }
  return chunks
}

function normalizeEvents(events: ParseEvent[]): unknown {
  const toolCallIds = new Map<string, string>()
  const messageIds = new Map<string, string>()
  let nextToolCallId = 0
  let nextMessageId = 0

  function normalizeToolCallId(id: string): string {
    let normalized = toolCallIds.get(id)
    if (!normalized) {
      normalized = `tool-${nextToolCallId++}`
      toolCallIds.set(id, normalized)
    }
    return normalized
  }

  function normalizeMessageId(id: string): string {
    let normalized = messageIds.get(id)
    if (!normalized) {
      normalized = `message-${nextMessageId++}`
      messageIds.set(id, normalized)
    }
    return normalized
  }

  const normalized = events
    .map((event) => {
    if ('toolCallId' in event && typeof event.toolCallId === 'string') {
      return {
        ...event,
        toolCallId: normalizeToolCallId(event.toolCallId),
        ...('element' in event && event.element && typeof event.element === 'object' && 'toolCallId' in event.element
          ? {
              element: {
                ...event.element,
                toolCallId: normalizeToolCallId(event.element.toolCallId),
              },
            }
          : {}),
      }
    }

    if ('id' in event && typeof event.id === 'string') {
      return {
        ...event,
        id: normalizeMessageId(event.id),
      }
    }

    return event
  })

  const coalesced: unknown[] = []
  for (const event of normalized) {
    const prev = coalesced[coalesced.length - 1] as Record<string, unknown> | undefined
    const current = event as Record<string, unknown>

    if (
      prev
      && prev._tag === 'MessageChunk'
      && current._tag === 'MessageChunk'
      && prev.id === current.id
    ) {
      prev.text = String(prev.text) + String(current.text)
      continue
    }

    if (
      prev
      && prev._tag === 'BodyChunk'
      && current._tag === 'BodyChunk'
      && prev.toolCallId === current.toolCallId
    ) {
      prev.text = String(prev.text) + String(current.text)
      continue
    }

    if (
      prev
      && prev._tag === 'ProseChunk'
      && current._tag === 'ProseChunk'
      && prev.patternId === current.patternId
    ) {
      prev.text = String(prev.text) + String(current.text)
      continue
    }

    coalesced.push(current)
  }

  return coalesced
}

function expectSameEventsForAllSingleSplits(xml: string): void {
  const expected = normalizeEvents(parse(xml))
  for (const chunks of splitAtEveryPosition(xml)) {
    expect(normalizeEvents(parseChunks(chunks))).toEqual(expected)
  }
}

function messages(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'MessageChunk' }> => e._tag === 'MessageChunk')
}

function lenses(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'LensEnd' }> => e._tag === 'LensEnd')
}

describe('structural tag depth tracking', () => {
  it('actions nested inside actions should not close the outer actions block early', () => {
    const events = parse(
      [
        '',
        '<shell>before</shell>',
        '',
        'inner literal body',
        '',
        '<shell>after</shell>',
        '',
      ].join('\n') + '\n',
    )

    const actionOpens = events.filter(e => e._tag === 'TagOpened')
    const actionCloses = events.filter(e => e._tag === 'TagClosed')
    const shells = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'shell',
    )

    expect(actionOpens).toHaveLength(2)
    expect(actionCloses).toHaveLength(2)
    expect(shells).toHaveLength(2)
    expect(shells.map(e => e.element.body)).toEqual(['before', 'after'])
    expect(events.filter(e => e._tag === 'ParseError')).toHaveLength(0)
  })


  it.skip('comms nested inside comms should not close the outer comms block early', () => {
    const xml = [
      '',
      '<message>before</message>',
      '',
      '',
      '<message>after</message>',
      '',
      '',
      '<shell>done</shell>',
      '',
    ].join('\n') + '\n'

    const events = parse(xml)

    const commsOpens = events.filter((e): e is Extract<ParseEvent, { _tag: 'TagOpened' }> => e._tag === 'TagOpened')
    const commsCloses = events.filter((e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> => e._tag === 'TagClosed')
    const messageOpens = events.filter(e => e._tag === 'MessageStart')
    const messageCloses = events.filter(e => e._tag === 'MessageEnd')
    const closeIndex = events.findIndex(e => e._tag === 'TagClosed')
    const secondMessageIndex = events.findIndex(
      (e, i) => i > 0 && e._tag === 'MessageChunk' && e.text.includes('after'),
    )

    expect(commsOpens.length).toBeGreaterThanOrEqual(2)
    expect(commsCloses.length).toBeGreaterThanOrEqual(2)
    expect(messageOpens).toHaveLength(2)
    expect(messageCloses).toHaveLength(2)
    expect(secondMessageIndex).toBeGreaterThan(-1)
    expect(closeIndex).toBeGreaterThan(-1)
    expect(events.filter(e => e._tag === 'ParseError')).toHaveLength(0)
    expectSameEventsForAllSingleSplits(xml)
  })

  it('lens nested inside lens should not close the outer lens early', () => {
    const events = parse(
      [
        '<lens name="outer">',
        'before',
        '<lens name="inner">',
        '</lens>',
        'after',
        '</lens>',
        '',
        '<shell>done</shell>',
        '',
      ].join('\n') + '\n',
    )

    const shells = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'shell',
    )

    expect(lenses(events)).toEqual([{ _tag: 'LensEnd', name: 'outer', content: 'before\n<lens name="inner">\n</lens>\nafter' }])
    expect(shells).toHaveLength(1)
    expect(shells[0].element.body).toBe('done')
    expect(events.filter(e => e._tag === 'ParseError')).toHaveLength(0)
  })

  it('message nested inside message should not close the outer message early', () => {
    const xml = [
      '',
      '<message>',
      'before',
      '<message>',
      '</message>',
      'after',
      '</message>',
      '',
      '',
      '<shell>done</shell>',
      '',
    ].join('\n') + '\n'

    const events = parse(xml)

    const messageOpens = events.filter(e => e._tag === 'MessageStart')
    const messageCloses = events.filter(e => e._tag === 'MessageEnd')
    const shells = events.filter(
      (e): e is Extract<ParseEvent, { _tag: 'TagClosed' }> =>
        e._tag === 'TagClosed' && e.tagName === 'shell',
    )

    expect(messageOpens).toHaveLength(1)
    expect(messageCloses).toHaveLength(1)
    expect(messages(events).map(e => e.text).join('')).toBe('before\n<message>\n</message>\nafter')
    expect(shells).toHaveLength(1)
    expect(shells[0].element.body).toBe('done')
    expect(events.filter(e => e._tag === 'ParseError')).toHaveLength(0)
    expectSameEventsForAllSingleSplits(xml)
  })
})