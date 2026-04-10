import { describe, it, expect } from 'vitest'
import { createStreamingXmlParser } from '../parser'
import type { ParseEvent } from '../format/types'

const knownTags = new Set(['shell', 'read', 'message'])
const childTagMap = new Map<string, Set<string>>()

function parse(xml: string): ParseEvent[] {
  const parser = createStreamingXmlParser(knownTags, childTagMap)
  return [...parser.processChunk(xml), ...parser.flush()]
}

function turnControls(events: ParseEvent[]) {
  return events.filter((e): e is Extract<ParseEvent, { _tag: 'TurnControl' }> => e._tag === 'TurnControl')
}

describe('end-turn parsing behavior', () => {
  it('idle/continue inside end-turn are parsed structurally', () => {
    const events = parse('<end-turn>\n<idle/>\n</end-turn>')
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
  })

  it('continue decision is captured', () => {
    const events = parse('<end-turn>\n<continue/>\n</end-turn>')
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
  })

  it('end-turn with no decision defaults to idle', () => {
    const events = parse('<end-turn>\n</end-turn>')
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
  })

  it('self-closing end-turn defaults to idle', () => {
    const events = parse('<end-turn/>')
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
  })

  it('observing mode is entered after </end-turn> — trailing content is not parsed', () => {
    const input = '<end-turn>\n<idle/>\n</end-turn>\n<message to="user">trailing</message>'
    const events = parse(input)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('idle')
    // No MessageStart should appear — trailing content is observed, not parsed
    const messages = events.filter(e => e._tag === 'MessageStart')
    expect(messages).toHaveLength(0)
  })

  it('observing mode is entered after <end-turn/> — trailing content is not parsed', () => {
    const input = '<end-turn/>\n<message to="user">trailing</message>'
    const events = parse(input)
    const tc = turnControls(events)
    expect(tc).toHaveLength(1)
    // No MessageStart should appear
    const messages = events.filter(e => e._tag === 'MessageStart')
    expect(messages).toHaveLength(0)
  })

  it('nested end-turn inside end-turn block is parsed structurally (model misbehavior)', () => {
    // When the model emits a second <end-turn><continue/></end-turn> inside
    // the first <end-turn> block, the inner one takes over because the parser
    // is still active inside the block. This is expected parser behavior —
    // the model shouldn't put arbitrary content inside <end-turn>.
    const input = [
      '<end-turn>',
      '<message to="user">hello world</message>',
      '<end-turn><continue/></end-turn>',
    ].join('\n')

    const events = parse(input)
    const tc = turnControls(events)

    // The nested <end-turn><continue/></end-turn> is parsed structurally
    // and becomes the actual turn control with decision 'continue'
    expect(tc).toHaveLength(1)
    expect(tc[0].decision).toBe('continue')
  })
})
