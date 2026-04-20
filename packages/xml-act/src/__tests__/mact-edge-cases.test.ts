
import { describe, it, expect } from 'vitest'
import { createTokenizer } from '../tokenizer'
import { createParser, type ParserEvent } from '../parser'

// Helper to parse input with configurable chunk size
function parseWithChunkSize(input: string, chunkSize: number): ParserEvent[] {
  const events: ParserEvent[] = []
  const parser = createParser()
  const tokenizer = createTokenizer((token) => {
    parser.pushToken(token)
  })
  
  for (let i = 0; i < input.length; i += chunkSize) {
    tokenizer.push(input.slice(i, i + chunkSize))
    events.push(...parser.drain())
  }
  tokenizer.end()
  events.push(...parser.drain())
  parser.end()
  events.push(...parser.drain())
  return events
}

// Test at multiple chunk sizes to catch boundary issues
const CHUNK_SIZES = [1, 2, 3, 5, 10, Infinity]

// Helper to run a test at all chunk sizes
function testAllChunkSizes(name: string, input: string, expectedEvents: (events: ParserEvent[]) => void) {
  for (const chunkSize of CHUNK_SIZES) {
    it(`${name} (chunkSize=${chunkSize})`, () => {
      const events = parseWithChunkSize(input, chunkSize)
      expectedEvents(events)
    })
  }
}

describe('Mact Edge Cases', () => {
  describe('Basic tag parsing', () => {
    testAllChunkSizes(
      'Simple message tag',
      '<|message:user>Hello<message|>',
      (events) => {
        expect(events).toHaveLength(4)
        expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
        expect(events[1]).toMatchObject({ _tag: 'MessageChunk', text: 'Hello' })
        expect(events[2]).toMatchObject({ _tag: 'MessageEnd' })
        expect(events[3]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )

    testAllChunkSizes(
      'Simple think tag',
      '<|think:alignment>Thinking<think|>',
      (events) => {
        expect(events).toHaveLength(4)
        expect(events[0]).toMatchObject({ _tag: 'LensStart', name: 'alignment' })
        expect(events[1]).toMatchObject({ _tag: 'LensChunk', text: 'Thinking' })
        expect(events[2]).toMatchObject({ _tag: 'LensEnd', name: 'alignment', content: 'Thinking' })
        expect(events[3]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )

    testAllChunkSizes(
      'Yield self-close',
      '<|yield:user|>',
      (events) => {
        expect(events).toHaveLength(2)
        expect(events[0]).toMatchObject({ _tag: 'TurnControl', target: 'user' })
        expect(events[1]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )

    testAllChunkSizes(
      'Yield variants',
      '<|yield:tool|><|yield:worker|><|yield:parent|>',
      (events) => {
        expect(events).toHaveLength(4)
        expect(events[0]).toMatchObject({ _tag: 'TurnControl', target: 'tool' })
        expect(events[1]).toMatchObject({ _tag: 'TurnControl', target: 'worker' })
        expect(events[2]).toMatchObject({ _tag: 'TurnControl', target: 'parent' })
        expect(events[3]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )
  })

  describe('Chunk boundary at every position', () => {
    // Test splitting <|message:user> at every character position
    it('handles <| at chunk boundary', () => {
      const input = '<|message:user>Hello<message|>'
      // Split after '<'
      const events1 = parseWithChunkSize(input, 1)
      expect(events1).toHaveLength(4)
      expect(events1[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    })

    it('handles <|m at chunk boundary', () => {
      const input = '<|message:user>Hello<message|>'
      // First chunk: '<|m', second: 'essage:user>...'
      const events = parseWithChunkSize(input, 3)
      expect(events).toHaveLength(4)
      expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    })

    it('handles colon at chunk boundary', () => {
      const input = '<|message:user>Hello<message|>'
      // Split at colon
      const events = parseWithChunkSize(input, 10)
      expect(events).toHaveLength(4)
      expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    })

    it('handles variant at chunk boundary', () => {
      const input = '<|message:user>Hello<message|>'
      // Split in middle of variant
      const events = parseWithChunkSize(input, 12)
      expect(events).toHaveLength(4)
      expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    })

    it('handles > at chunk boundary', () => {
      const input = '<|message:user>Hello<message|>'
      // Split right after >
      const events = parseWithChunkSize(input, 15)
      expect(events).toHaveLength(4)
      expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    })

    it('handles close tag < at chunk boundary', () => {
      const input = '<|message:user>Hello<message|>'
      // Split at < of close tag
      const events = parseWithChunkSize(input, 20)
      expect(events).toHaveLength(4)
      expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    })

    it('handles close tag | at chunk boundary', () => {
      const input = '<|message:user>Hello<message|>'
      // Split at | of close tag
      const events = parseWithChunkSize(input, 26)
      expect(events).toHaveLength(4)
      expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    })

    it('handles close tag > at chunk boundary', () => {
      const input = '<|message:user>Hello<message|>'
      // Split at > of close tag
      const events = parseWithChunkSize(input, 27)
      expect(events).toHaveLength(4)
      expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    })

    it('handles self-close | at chunk boundary', () => {
      const input = '<|yield:user|>'
      // Split at first |
      const events = parseWithChunkSize(input, 7)
      expect(events).toHaveLength(2)
      expect(events[0]).toMatchObject({ _tag: 'TurnControl', target: 'user' })
    })

    it('handles self-close > at chunk boundary', () => {
      const input = '<|yield:user|>'
      // Split at >
      const events = parseWithChunkSize(input, 14)
      expect(events).toHaveLength(2)
      expect(events[0]).toMatchObject({ _tag: 'TurnControl', target: 'user' })
    })
  })

  describe('Coalescing / content merging', () => {
    testAllChunkSizes(
      'Multiple content chunks merge',
      'Hello world how are you',
      (events) => {
        expect(events).toHaveLength(2)
        expect(events[0]).toMatchObject({ _tag: 'ProseChunk', text: 'Hello world how are you' })
        expect(events[1]).toMatchObject({ _tag: 'ProseEnd', content: 'Hello world how are you' })
      }
    )

    testAllChunkSizes(
      'Content between tags',
      'Before<|message:user>Inside<message|>After',
      (events) => {
        expect(events).toHaveLength(6)
        expect(events[0]).toMatchObject({ _tag: 'ProseChunk', text: 'Before' })
        expect(events[1]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
        expect(events[2]).toMatchObject({ _tag: 'MessageChunk', text: 'Inside' })
        expect(events[3]).toMatchObject({ _tag: 'MessageEnd' })
        expect(events[4]).toMatchObject({ _tag: 'ProseChunk', text: 'After' })
        expect(events[5]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )

    testAllChunkSizes(
      'Empty content inside tags',
      '<|message:user><message|>',
      (events) => {
        expect(events).toHaveLength(3)
        expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
        expect(events[1]).toMatchObject({ _tag: 'MessageEnd' })
        expect(events[2]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )

    testAllChunkSizes(
      'Newlines in content',
      '<|message:user>Line1\nLine2<message|>',
      (events) => {
        expect(events).toHaveLength(4)
        expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
        expect(events[1]).toMatchObject({ _tag: 'MessageChunk', text: 'Line1\nLine2' })
        expect(events[2]).toMatchObject({ _tag: 'MessageEnd' })
        expect(events[3]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )

    testAllChunkSizes(
      'Content that looks like partial tags',
      'This is < not a tag and this is | also not',
      (events) => {
        expect(events).toHaveLength(2)
        expect(events[0]).toMatchObject({ _tag: 'ProseChunk', text: 'This is < not a tag and this is | also not' })
        expect(events[1]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )
  })

  describe('Nested structures', () => {
    testAllChunkSizes(
      'Think block with angle brackets in content',
      '<|think:code>const x = <div>test</div><think|>',
      (events) => {
        expect(events).toHaveLength(4)
        expect(events[0]).toMatchObject({ _tag: 'LensStart', name: 'code' })
        expect(events[1]).toMatchObject({ _tag: 'LensChunk', text: 'const x = <div>test</div>' })
        expect(events[2]).toMatchObject({ _tag: 'LensEnd', name: 'code' })
        expect(events[3]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )

    testAllChunkSizes(
      'Message containing < characters',
      '<|message:user>Compare a < b and c > d<message|>',
      (events) => {
        expect(events).toHaveLength(4)
        expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
        expect(events[1]).toMatchObject({ _tag: 'MessageChunk', text: 'Compare a < b and c > d' })
        expect(events[2]).toMatchObject({ _tag: 'MessageEnd' })
        expect(events[3]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )

    testAllChunkSizes(
      'Multiple think blocks in sequence',
      '<|think:first>First think<think|><|think:second>Second think<think|>',
      (events) => {
        expect(events).toHaveLength(7)
        expect(events[0]).toMatchObject({ _tag: 'LensStart', name: 'first' })
        expect(events[1]).toMatchObject({ _tag: 'LensChunk', text: 'First think' })
        expect(events[2]).toMatchObject({ _tag: 'LensEnd', name: 'first' })
        expect(events[3]).toMatchObject({ _tag: 'LensStart', name: 'second' })
        expect(events[4]).toMatchObject({ _tag: 'LensChunk', text: 'Second think' })
        expect(events[5]).toMatchObject({ _tag: 'LensEnd', name: 'second' })
        expect(events[6]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )

    testAllChunkSizes(
      'Think followed by message followed by yield',
      '<|think:alignment>Thinking<think|><|message:user>Hello<message|><|yield:user|>',
      (events) => {
        expect(events).toHaveLength(8)
        expect(events[0]).toMatchObject({ _tag: 'LensStart', name: 'alignment' })
        expect(events[1]).toMatchObject({ _tag: 'LensChunk', text: 'Thinking' })
        expect(events[2]).toMatchObject({ _tag: 'LensEnd', name: 'alignment' })
        expect(events[3]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
        expect(events[4]).toMatchObject({ _tag: 'MessageChunk', text: 'Hello' })
        expect(events[5]).toMatchObject({ _tag: 'MessageEnd' })
        expect(events[6]).toMatchObject({ _tag: 'TurnControl', target: 'user' })
        expect(events[7]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )
  })

  describe('Malformed input', () => {
    testAllChunkSizes(
      'Unclosed open tag at end of stream',
      '<|message:user>Hello',
      (events) => {
        // Should treat as content since message never closes
        expect(events.length).toBeGreaterThan(0)
        const lastEvent = events[events.length - 1]
        expect(lastEvent).toMatchObject({ _tag: 'ProseEnd' })
        // The content should be preserved
        const proseEvents = events.filter(e => e._tag === 'ProseChunk')
        const combinedText = proseEvents.map(e => (e as any).text).join('')
        expect(combinedText).toContain('<|message:user>Hello')
      }
    )

    testAllChunkSizes(
      'Close tag without matching open',
      'Hello<message|>',
      (events) => {
        // Should treat close tag as content
        expect(events).toHaveLength(2)
        expect(events[0]).toMatchObject({ _tag: 'ProseChunk', text: 'Hello<message|>' })
        expect(events[1]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )

    testAllChunkSizes(
      'Self-close where open is expected',
      '<|message:user|>Hello',
      (events) => {
        // Self-close message is invalid, should be treated as content
        expect(events.length).toBeGreaterThan(0)
        const proseEvents = events.filter(e => e._tag === 'ProseChunk')
        const combinedText = proseEvents.map(e => (e as any).text).join('')
        expect(combinedText).toContain('<|message:user|>Hello')
      }
    )

    testAllChunkSizes(
      'Tag with invalid characters in name',
      '<|123invalid>content<123invalid|>',
      (events) => {
        // Should treat as content since name can't start with number
        expect(events).toHaveLength(2)
        expect(events[0]).toMatchObject({ _tag: 'ProseChunk', text: '<|123invalid>content<123invalid|>' })
        expect(events[1]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )

    testAllChunkSizes(
      'Incomplete tag at end of stream',
      '<|message',
      (events) => {
        // Should treat as content
        expect(events).toHaveLength(2)
        expect(events[0]).toMatchObject({ _tag: 'ProseChunk', text: '<|message' })
        expect(events[1]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )

    testAllChunkSizes(
      'Incomplete close tag at end of stream',
      '<|message:user>Hello<message',
      (events) => {
        // Should treat incomplete close as content
        expect(events.length).toBeGreaterThan(0)
        const lastEvent = events[events.length - 1]
        expect(lastEvent).toMatchObject({ _tag: 'ProseEnd' })
      }
    )
  })

  describe('Streaming specifics', () => {
    it('handles 1-char chunks', () => {
      const input = '<|message:user>Hello<message|>'
      const events = parseWithChunkSize(input, 1)
      expect(events).toHaveLength(4)
      expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    })

    it('handles 2-char chunks', () => {
      const input = '<|message:user>Hello<message|>'
      const events = parseWithChunkSize(input, 2)
      expect(events).toHaveLength(4)
      expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    })

    it('handles entire input in one chunk', () => {
      const input = '<|message:user>Hello<message|>'
      const events = parseWithChunkSize(input, Infinity)
      expect(events).toHaveLength(4)
      expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    })

    it('handles chunk that is exactly a tag boundary', () => {
      const input = '<|message:user>Hello<message|>'
      // Chunk ends right at >
      const events = parseWithChunkSize(input, 15)
      expect(events).toHaveLength(4)
      expect(events[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
    })

    it('handles chunk that starts mid-tag', () => {
      const input = 'prefix<|message:user>Hello<message|>suffix'
      // Start at position 10 (mid-tag)
      const events1 = parseWithChunkSize(input.slice(0, 10), 10)
      const events2 = parseWithChunkSize(input.slice(10), 5)
      const combined = [...events1, ...events2]
      expect(combined.length).toBeGreaterThan(0)
    })
  })

  describe('Close tag with pipe (filter)', () => {
    testAllChunkSizes(
      'Piped close for invoke with filter',
      '<|invoke:browser>url<invoke|filter>query<filter|>',
      (events) => {
        expect(events.length).toBeGreaterThanOrEqual(6)
        expect(events[0]).toMatchObject({ _tag: 'InvokeStarted', toolTag: 'browser', toolName: 'browser' })
        expect(events[1]).toMatchObject({ _tag: 'ParameterStarted', parameterName: 'url' })
        expect(events[2]).toMatchObject({ _tag: 'ParameterComplete', parameterName: 'url', value: 'url' })
        expect(events.some(e => e._tag === 'FilterStarted')).toBe(true)
        expect(events.some(e => e._tag === 'FilterComplete')).toBe(true)
        expect(events.some(e => e._tag === 'InvokeComplete')).toBe(true)
      }
    )

    testAllChunkSizes(
      'Filter content across chunk boundaries',
      '<|invoke:browser>url<invoke|filter>long filter content here<filter|>',
      (events) => {
        expect(events.some(e => e._tag === 'FilterStarted')).toBe(true)
        expect(events.some(e => e._tag === 'FilterChunk')).toBe(true)
        expect(events.some(e => e._tag === 'FilterComplete')).toBe(true)
      }
    )
  })

  describe('Parameter tags', () => {
    testAllChunkSizes(
      'Parameter open and close',
      '<|invoke:browser><|parameter:url>https://example.com<parameter|><invoke|>',
      (events) => {
        expect(events.some(e => e._tag === 'InvokeStarted')).toBe(true)
        expect(events.some(e => e._tag === 'ParameterStarted' && e.parameterName === 'url')).toBe(true)
        expect(events.some(e => e._tag === 'ParameterChunk' && e.text === 'https://example.com')).toBe(true)
        expect(events.some(e => e._tag === 'ParameterComplete')).toBe(true)
        expect(events.some(e => e._tag === 'InvokeComplete')).toBe(true)
      }
    )

    testAllChunkSizes(
      'Multiple parameters',
      '<|invoke:browser><|parameter:url>https://example.com<parameter|><|parameter:method>GET<parameter|><invoke|>',
      (events) => {
        const paramStarts = events.filter(e => e._tag === 'ParameterStarted')
        expect(paramStarts).toHaveLength(2)
        expect(paramStarts[0]).toMatchObject({ parameterName: 'url' })
        expect(paramStarts[1]).toMatchObject({ parameterName: 'method' })
      }
    )
  })

  describe('Complex real-world scenarios', () => {
    testAllChunkSizes(
      'Full agent response with think, message, and yield',
      `<|think:alignment>
The user wants me to help with a task. I should acknowledge and ask for details.
<think|>
<|message:user>
I'm ready to help! What would you like me to work on?
<message|>
<|yield:user|>`,
      (events) => {
        // Should have lens, message, and turn control - no raw syntax in prose
        const proseEvents = events.filter(e => e._tag === 'ProseChunk' || e._tag === 'ProseEnd')
        for (const e of proseEvents) {
          const text = (e as any).text || (e as any).content || ''
          expect(text).not.toContain('<|')
          expect(text).not.toContain('|>')
        }
        
        expect(events.some(e => e._tag === 'LensStart')).toBe(true)
        expect(events.some(e => e._tag === 'MessageStart')).toBe(true)
        expect(events.some(e => e._tag === 'TurnControl')).toBe(true)
      }
    )

    testAllChunkSizes(
      'Code with lots of angle brackets',
      `<|think:code>
function example() {
  return <div className="test">
    {items.map(item => <Item key={item.id} />)}
  </div>
}
<think|>`,
      (events) => {
        expect(events.some(e => e._tag === 'LensStart' && e.name === 'code')).toBe(true)
        const lensChunks = events.filter(e => e._tag === 'LensChunk')
        const combined = lensChunks.map(e => (e as any).text).join('')
        expect(combined).toContain('<div className="test">')
        expect(combined).toContain('{items.map')
      }
    )

    testAllChunkSizes(
      'Nested angle brackets that look like tags but arent',
      '<|message:user>Here is some pseudo-code: <|not_a_tag>content<not_a_tag|><message|>',
      (events) => {
        // The pseudo-code tags should be preserved as content inside the message
        const msgChunks = events.filter(e => e._tag === 'MessageChunk')
        const combined = msgChunks.map(e => (e as any).text).join('')
        expect(combined).toContain('<|not_a_tag>')
        expect(combined).toContain('<not_a_tag|>')
      }
    )
  })

  describe('Whitespace handling', () => {
    testAllChunkSizes(
      'Whitespace before tag',
      '   <|message:user>Hello<message|>',
      (events) => {
        expect(events[0]).toMatchObject({ _tag: 'ProseChunk', text: '   ' })
        expect(events[1]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
      }
    )

    testAllChunkSizes(
      'Whitespace after tag',
      '<|message:user>Hello<message|>   ',
      (events) => {
        const proseEvents = events.filter(e => e._tag === 'ProseChunk')
        const lastProse = proseEvents[proseEvents.length - 1]
        expect((lastProse as any).text).toBe('   ')
      }
    )

    testAllChunkSizes(
      'Newlines around tags',
      '\n<|message:user>Hello<message|>\n',
      (events) => {
        expect(events[0]).toMatchObject({ _tag: 'ProseChunk', text: '\n' })
        expect(events[events.length - 2]).toMatchObject({ _tag: 'ProseChunk', text: '\n' })
      }
    )
  })

  describe('Empty and minimal inputs', () => {
    it('handles empty input', () => {
      const events = parseWithChunkSize('', 1)
      expect(events).toHaveLength(1)
      expect(events[0]).toMatchObject({ _tag: 'ProseEnd', content: '' })
    })

    testAllChunkSizes(
      'Single character content',
      'a',
      (events) => {
        expect(events).toHaveLength(2)
        expect(events[0]).toMatchObject({ _tag: 'ProseChunk', text: 'a' })
        expect(events[1]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )

    testAllChunkSizes(
      'Only whitespace',
      '   \n\t  ',
      (events) => {
        expect(events).toHaveLength(2)
        expect(events[0]).toMatchObject({ _tag: 'ProseChunk', text: '   \n\t  ' })
        expect(events[1]).toMatchObject({ _tag: 'ProseEnd' })
      }
    )
  })

  describe('Special characters in content', () => {
    testAllChunkSizes(
      'Unicode characters',
      '<|message:user>Hello 世界 🌍<message|>',
      (events) => {
        const msgChunk = events.find(e => e._tag === 'MessageChunk')
        expect(msgChunk).toMatchObject({ text: 'Hello 世界 🌍' })
      }
    )

    testAllChunkSizes(
      'Special XML characters',
      '<|message:user>A & B < C > D "quotes"<message|>',
      (events) => {
        const msgChunk = events.find(e => e._tag === 'MessageChunk')
        expect((msgChunk as any).text).toContain('&')
        expect((msgChunk as any).text).toContain('<')
        expect((msgChunk as any).text).toContain('>')
        expect((msgChunk as any).text).toContain('"')
      }
    )

    testAllChunkSizes(
      'Backslashes and escapes',
      '<|message:user>Path: C:\\Users\\test<message|>',
      (events) => {
        const msgChunk = events.find(e => e._tag === 'MessageChunk')
        expect((msgChunk as any).text).toContain('C:\\Users\\test')
      }
    )
  })

  describe('Tag name edge cases', () => {
    testAllChunkSizes(
      'Tag with hyphen in name',
      '<|my-tag:variant>content<my-tag|>',
      (events) => {
        // Hyphens are allowed in names
        expect(events.some(e => e._tag === 'ProseChunk')).toBe(true)
      }
    )

    testAllChunkSizes(
      'Tag with underscore',
      '<|my_tag:variant>content<my_tag|>',
      (events) => {
        // Underscores are allowed in names
        expect(events.some(e => e._tag === 'ProseChunk')).toBe(true)
      }
    )

    testAllChunkSizes(
      'Tag with numbers (after first char)',
      '<|tag123:var456>content<tag123|>',
      (events) => {
        // Numbers allowed after first char
        expect(events.some(e => e._tag === 'ProseChunk')).toBe(true)
      }
    )

    testAllChunkSizes(
      'Single char tag name',
      '<|x:y>content<x|>',
      (events) => {
        expect(events.some(e => e._tag === 'ProseChunk')).toBe(true)
      }
    )
  })

  describe('Protocol syntax should never leak', () => {
    const testCases = [
      '<|message:user>Hello<message|>',
      '<|think:alignment>Thinking<think|>',
      '<|yield:user|>',
      '<|invoke:browser>url<invoke|>',
      '<|parameter:name>value<parameter|>',
      '<|invoke:tool>param<invoke|filter>query<filter|>',
    ]

    for (const input of testCases) {
      for (const chunkSize of [1, 2, 3, 5, 10]) {
        it(`no leak: "${input.slice(0, 30)}..." at chunkSize=${chunkSize}`, () => {
          const events = parseWithChunkSize(input, chunkSize)
          const proseEvents = events.filter(e => e._tag === 'ProseChunk' || e._tag === 'ProseEnd')
          for (const e of proseEvents) {
            const text = (e as any).text || (e as any).content || ''
            // Protocol syntax should never appear in prose
            expect(text).not.toMatch(/<\|/)
            expect(text).not.toMatch(/\|>/)
          }
        })
      }
    }
  })
})
