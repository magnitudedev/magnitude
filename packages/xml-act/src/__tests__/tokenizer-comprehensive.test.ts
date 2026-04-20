
import { describe, expect, it } from 'vitest'
import { createTokenizer } from '../tokenizer'

function collect(input: string | string[]): any[] {
  const out: any[] = []
  const tokenizer = createTokenizer((token) => {
    // Strip raw field for cleaner test output
    const { raw, ...rest } = token as any
    out.push(rest)
  })
  if (Array.isArray(input)) {
    for (const chunk of input) tokenizer.push(chunk)
  } else {
    tokenizer.push(input)
  }
  tokenizer.end()
  return out
}

describe('Mact tokenizer comprehensive tests', () => {
  describe('basic tags', () => {
    it('parses simple open tag', () => {
      const tokens = collect('<|message>')
      expect(tokens).toEqual([{ _tag: 'Open', name: 'message' }])
    })

    it('parses open tag with variant', () => {
      const tokens = collect('<|message:user>')
      expect(tokens).toEqual([{ _tag: 'Open', name: 'message', variant: 'user' }])
    })

    it('parses close tag', () => {
      const tokens = collect('<message|>')
      expect(tokens).toEqual([{ _tag: 'Close', name: 'message' }])
    })

    it('parses self-close tag', () => {
      const tokens = collect('<|yield:user|>')
      expect(tokens).toEqual([{ _tag: 'SelfClose', name: 'yield', variant: 'user' }])
    })

    it('parses self-close without variant', () => {
      const tokens = collect('<|tool|>')
      expect(tokens).toEqual([{ _tag: 'SelfClose', name: 'tool' }])
    })
  })

  describe('parameter tags', () => {
    it('parses parameter open', () => {
      const tokens = collect('<|parameter:path>')
      expect(tokens).toEqual([{ _tag: 'Parameter', name: 'path' }])
    })

    it('parses parameter close as regular close tag', () => {
      // In Mact format, <parameter|> is just a close tag
      // The parser distinguishes parameter close based on context
      const tokens = collect('<parameter|>')
      expect(tokens).toEqual([{ _tag: 'Close', name: 'parameter', pipe: undefined }])
    })
  })

  describe('content', () => {
    it('parses plain text', () => {
      const tokens = collect('Hello world')
      expect(tokens).toEqual([{ _tag: 'Content', text: 'Hello world' }])
    })

    it('parses mixed content and tags', () => {
      const tokens = collect('Hello <|message:user>world<message|>!')
      expect(tokens).toEqual([
        { _tag: 'Content', text: 'Hello ' },
        { _tag: 'Open', name: 'message', variant: 'user' },
        { _tag: 'Content', text: 'world' },
        { _tag: 'Close', name: 'message' },
        { _tag: 'Content', text: '!' }
      ])
    })
  })

  describe('chunk boundary handling', () => {
    it('handles open tag split at < boundary', () => {
      const tokens = collect(['<', '|message:user>'])
      expect(tokens).toEqual([{ _tag: 'Open', name: 'message', variant: 'user' }])
    })

    it('handles open tag split at <| boundary', () => {
      const tokens = collect(['<|', 'message:user>'])
      expect(tokens).toEqual([{ _tag: 'Open', name: 'message', variant: 'user' }])
    })

    it('handles open tag split in name', () => {
      const tokens = collect(['<|mess', 'age:user>'])
      expect(tokens).toEqual([{ _tag: 'Open', name: 'message', variant: 'user' }])
    })

    it('handles open tag split at colon', () => {
      const tokens = collect(['<|message', ':user>'])
      expect(tokens).toEqual([{ _tag: 'Open', name: 'message', variant: 'user' }])
    })

    it('handles open tag split in variant', () => {
      const tokens = collect(['<|message:us', 'er>'])
      expect(tokens).toEqual([{ _tag: 'Open', name: 'message', variant: 'user' }])
    })

    it('handles close tag split at < boundary', () => {
      const tokens = collect(['<', 'message|>'])
      expect(tokens).toEqual([{ _tag: 'Close', name: 'message' }])
    })

    it('handles close tag split in name', () => {
      const tokens = collect(['<mess', 'age|>'])
      expect(tokens).toEqual([{ _tag: 'Close', name: 'message' }])
    })

    it('handles close tag split at |', () => {
      const tokens = collect(['<message|', '>'])
      expect(tokens).toEqual([{ _tag: 'Close', name: 'message' }])
    })

    it('handles self-close split at |', () => {
      const tokens = collect(['<|yield:user|', '>'])
      expect(tokens).toEqual([{ _tag: 'SelfClose', name: 'yield', variant: 'user' }])
    })

    it('handles char-by-char streaming', () => {
      const input = '<|message:user>hello<message|>'
      const chunks = input.split('')
      const tokens = collect(chunks)
      expect(tokens).toEqual([
        { _tag: 'Open', name: 'message', variant: 'user' },
        { _tag: 'Content', text: 'hello' },
        { _tag: 'Close', name: 'message' }
      ])
    })

    it('handles 2-char chunks (simulating LLM streaming)', () => {
      const input = `<|think:alignment>
Some reasoning here
<think|>

<|message:user>
I'm ready and waiting for a task
<message|>

<|yield:user|>`
      const chunks: string[] = []
      for (let i = 0; i < input.length; i += 2) {
        chunks.push(input.slice(i, i + 2))
      }
      const tokens = collect(chunks)
      
      // Should have all tags parsed, no raw syntax in content
      const contentTokens = tokens.filter(t => t._tag === 'Content')
      for (const ct of contentTokens) {
        expect(ct.text).not.toContain('<|')
        expect(ct.text).not.toContain('<think')
        expect(ct.text).not.toContain('<message')
      }
      
      // Should have the right tags
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'think' && t.variant === 'alignment')).toBe(true)
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'think')).toBe(true)
      expect(tokens.some(t => t._tag === 'Open' && t.name === 'message' && t.variant === 'user')).toBe(true)
      expect(tokens.some(t => t._tag === 'Close' && t.name === 'message')).toBe(true)
      expect(tokens.some(t => t._tag === 'SelfClose' && t.name === 'yield' && t.variant === 'user')).toBe(true)
    })
  })

  describe('edge cases', () => {
    it('handles incomplete tag at end (treats as content)', () => {
      const tokens = collect('<|message')
      expect(tokens).toEqual([{ _tag: 'Content', text: '<|message' }])
    })

    it('handles lone < as content', () => {
      const tokens = collect('hello < world')
      expect(tokens).toEqual([{ _tag: 'Content', text: 'hello < world' }])
    })

    it('handles multiple tags', () => {
      const tokens = collect('<|a><|b:two><c|><d|>')
      expect(tokens).toEqual([
        { _tag: 'Open', name: 'a' },
        { _tag: 'Open', name: 'b', variant: 'two' },
        { _tag: 'Close', name: 'c' },
        { _tag: 'Close', name: 'd' }
      ])
    })

    it('handles whitespace after tag name (invalid, treated as content)', () => {
      // Whitespace in tag names is invalid in Mact format
      // The entire thing is treated as content
      const tokens = collect('<|message >')
      expect(tokens).toEqual([{ _tag: 'Content', text: '<|message >' }])
    })
  })
})
