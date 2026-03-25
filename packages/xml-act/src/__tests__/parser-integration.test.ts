import { describe, expect, it } from 'bun:test'
import { createStreamingXmlParser } from '../parser'

describe('parser integration', () => {
  it('parses basic prose', () => {
    const parser = createStreamingXmlParser()
    parser.push('hello world')
    parser.end()

    expect(parser.events.map((e) => e._tag)).toEqual(['ProseChunk'])
    expect(parser.events[0]).toEqual({ _tag: 'ProseChunk', patternId: 'prose', text: 'hello world' })
  })

  it('parses container open/close flow with turn control', () => {
    const parser = createStreamingXmlParser(new Set(['shell']))
    parser.push('\n<actions>\n<shell>ls</shell>\n</actions>\n<yield/>')
    parser.end()

    const tags = parser.events.map((e) => e._tag)
    expect(tags).toContain('ContainerOpen')
    expect(tags).toContain('TagOpened')
    expect(tags).toContain('BodyChunk')
    expect(tags).toContain('TagClosed')
    expect(tags).toContain('ContainerClose')
    expect(tags).toContain('TurnControl')
  })

  it('parses think block', () => {
    const parser = createStreamingXmlParser()
    parser.push('\n<think>\nreasoning here\n</think>\n<yield/>')
    parser.end()

    const thinkEvents = parser.events.filter(
      (e) => (e._tag === 'ProseChunk' && e.patternId === 'think') || (e._tag === 'ProseEnd' && e.patternId === 'think'),
    )

    expect(thinkEvents).toEqual([
      { _tag: 'ProseChunk', patternId: 'think', text: '\nreasoning here\n' },
      { _tag: 'ProseEnd', patternId: 'think', content: '\nreasoning here\n', about: null },
    ])
    expect(parser.events.at(-1)).toEqual({ _tag: 'TurnControl', decision: 'yield' })
  })

  it('parses lenses block', () => {
    const parser = createStreamingXmlParser()
    parser.push('\n<lenses>\n<lens name="task">content</lens>\n</lenses>\n<yield/>')
    parser.end()

    const tags = parser.events.map((e) => e._tag)
    expect(tags).toContain('LensStart')
    expect(tags).toContain('LensChunk')
    expect(tags).toContain('LensEnd')
    expect(tags).toContain('TurnControl')
  })

  it('parses comms message and preserves embedded conversation context content', () => {
    const parser = createStreamingXmlParser()
    parser.push(`\n<comms>
<message to="user">hello
</message>
<conversation_context>
<message role="user">
hello
</message>
</conversation_context>
</comms>
<yield/>`)
    parser.end()

    const tags = parser.events.map((e) => e._tag)
    expect(tags).toContain('MessageStart')
    expect(tags).toContain('MessageChunk')
    expect(tags).toContain('MessageEnd')
    expect(tags).toContain('ContainerClose')
    expect(tags).toContain('TurnControl')
  })
})
