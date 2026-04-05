import { describe, expect, it } from 'vitest'
import { createStreamingXmlParser } from '../parser'
import type { XmlActEvent } from '../format/types'

function eventsOf<T extends XmlActEvent['_tag']>(events: XmlActEvent[], tag: T) {
  return events.filter((e): e is Extract<XmlActEvent, { _tag: T }> => e._tag === tag)
}

describe('parent tool close tag inside child body is passthrough', () => {
  it('single chunk: literal tool close tag inside child body is passthrough', () => {
    const knownTags = new Set(['edit'])
    const childTagMap = new Map([['edit', new Set(['old', 'new'])]])
    const parser = createStreamingXmlParser(knownTags, childTagMap)

    // Build input with literal </edit> inside <old> body
    // Using concatenation so our own parser doesn't consume it
    const input = '<task id="t1">\n<edit path="foo.ts">\n<old>before <' + '/edit> after<' + '/old>\n<new>replacement<' + '/new>\n<' + '/edit>\n<' + '/task>\n<idle/>'

    const events = [...parser.processChunk(input), ...parser.flush()]

    expect(eventsOf(events, 'ParseError')).toHaveLength(0)
    expect(eventsOf(events, 'TagOpened').some(e => e.tagName === 'edit')).toBe(true)

    const oldComplete = eventsOf(events, 'ChildComplete').find(e => e.childTagName === 'old')
    expect(oldComplete?.body).toBe('before <' + '/edit> after')

    const newComplete = eventsOf(events, 'ChildComplete').find(e => e.childTagName === 'new')
    expect(newComplete?.body).toBe('replacement')

    expect(eventsOf(events, 'TagClosed').filter(e => e.tagName === 'edit')).toHaveLength(1)
  })

  it('streaming chunks: tool close tag arrives as separate chunk inside child body', () => {
    const knownTags = new Set(['edit'])
    const childTagMap = new Map([['edit', new Set(['old', 'new'])]])
    const parser = createStreamingXmlParser(knownTags, childTagMap)

    const chunks = [
      '<task id="t1">\n<edit path="foo.ts">\n<old>before ',
      '<' + '/edit>',
      ' after<' + '/old>\n<new>replacement<' + '/new>\n<' + '/edit>\n<' + '/task>\n<idle/>',
    ]

    let events: XmlActEvent[] = []
    for (const chunk of chunks) {
      events.push(...parser.processChunk(chunk))
    }
    events.push(...parser.flush())

    expect(eventsOf(events, 'ParseError')).toHaveLength(0)

    const oldComplete = eventsOf(events, 'ChildComplete').find(e => e.childTagName === 'old')
    expect(oldComplete?.body).toBe('before <' + '/edit> after')

    const newComplete = eventsOf(events, 'ChildComplete').find(e => e.childTagName === 'new')
    expect(newComplete?.body).toBe('replacement')

    expect(eventsOf(events, 'TagClosed').filter(e => e.tagName === 'edit')).toHaveLength(1)
  })

  it('multiple tool close tags inside child body all passthrough', () => {
    const knownTags = new Set(['edit', 'write'])
    const childTagMap = new Map([['edit', new Set(['old', 'new'])]])
    const parser = createStreamingXmlParser(knownTags, childTagMap)

    const input = '<task id="t1">\n<edit path="foo.ts">\n<old>has <' + '/edit> and <' + '/write> and <' + '/edit> inside<' + '/old>\n<new>ok<' + '/new>\n<' + '/edit>\n<' + '/task>\n<idle/>'

    const events = [...parser.processChunk(input), ...parser.flush()]

    expect(eventsOf(events, 'ParseError')).toHaveLength(0)

    const oldComplete = eventsOf(events, 'ChildComplete').find(e => e.childTagName === 'old')
    expect(oldComplete?.body).toBe('has <' + '/edit> and <' + '/write> and <' + '/edit> inside')
  })
})
