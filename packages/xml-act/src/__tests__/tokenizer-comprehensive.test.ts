
import { describe, expect, it } from 'vitest'
import { createTokenizer } from '../tokenizer'

const STD_OPTIONS = { strictNewlines: true, toolKeyword: 'invoke' } as const

function collect(input: string | string[]): any[] {
  const out: any[] = []
  const tokenizer = createTokenizer((token) => {
    // Strip raw field for cleaner test output
    const { raw, ...rest } = token as any
    out.push(rest)
  }, new Set(), STD_OPTIONS)
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
      const tokens = collect('Hello \n<|message:user>world\n<message|>!')
      expect(tokens).toEqual([
        { _tag: 'Content', text: 'Hello \n' },
        { _tag: 'Open', name: 'message', variant: 'user' },
        { _tag: 'Content', text: 'world\n' },
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
      const tokens = collect(['\n<', 'message|>'])
      expect(tokens).toEqual([{ _tag: 'Content', text: '\n' }, { _tag: 'Close', name: 'message' }])
    })

    it('handles close tag split in name', () => {
      const tokens = collect(['\n<mess', 'age|>'])
      expect(tokens).toEqual([{ _tag: 'Content', text: '\n' }, { _tag: 'Close', name: 'message' }])
    })

    it('handles close tag split at |', () => {
      const tokens = collect(['\n<message|', '>'])
      expect(tokens).toEqual([{ _tag: 'Content', text: '\n' }, { _tag: 'Close', name: 'message' }])
    })

    it('handles self-close split at |', () => {
      const tokens = collect(['\n<|yield:user|', '>'])
      expect(tokens).toEqual([{ _tag: 'Content', text: '\n' }, { _tag: 'SelfClose', name: 'yield', variant: 'user' }])
    })

    it('handles char-by-char streaming (produces per-char content tokens)', () => {
      const input = '<|message:user>hello\n<message|>'
      const chunks = input.split('')
      const tokens = collect(chunks)
      // Char-by-char produces individual content tokens (coalescing is parser-level)
      const opens = tokens.filter(t => t._tag === 'Open')
      const closes = tokens.filter(t => t._tag === 'Close')
      const contentText = tokens.filter(t => t._tag === 'Content').map(t => t.text).join('')
      expect(opens).toEqual([{ _tag: 'Open', name: 'message', variant: 'user' }])
      expect(closes).toEqual([{ _tag: 'Close', name: 'message' }])
      expect(contentText).toBe('hello\n')
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

  describe('Unit A: slash-prefix close tag leniency', () => {
    function collectWithKnown(input: string): any[] {
      const out: any[] = []
      const tokenizer = createTokenizer((token) => {
        const { raw, ...rest } = token as any
        out.push(rest)
      }, new Set(), STD_OPTIONS)
      tokenizer.push(input)
      tokenizer.end()
      return out
    }

    it('accepts </think|> as close tag (Mode 1)', () => {
      const tokens = collectWithKnown('</think|>')
      expect(tokens).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('accepts </message|> as close tag (Mode 1)', () => {
      const tokens = collectWithKnown('</message|>')
      expect(tokens).toEqual([{ _tag: 'Close', name: 'message', pipe: undefined }])
    })

    it('accepts </think> as close tag (Mode 2)', () => {
      const tokens = collectWithKnown('</think>')
      expect(tokens).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('accepts </message> as close tag (Mode 2)', () => {
      const tokens = collectWithKnown('</message>')
      expect(tokens).toEqual([{ _tag: 'Close', name: 'message', pipe: undefined }])
    })

    it('handles </think|> split across chunk boundary', () => {
      const out: any[] = []
      const tokenizer = createTokenizer((token) => {
        const { raw, ...rest } = token as any
        out.push(rest)
      }, new Set(), STD_OPTIONS)
      tokenizer.push('</')
      tokenizer.push('think|>')
      tokenizer.end()
      expect(out).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('handles </think> split: < then /think>', () => {
      const out: any[] = []
      const tokenizer = createTokenizer((token) => {
        const { raw, ...rest } = token as any
        out.push(rest)
      }, new Set(), STD_OPTIONS)
      tokenizer.push('<')
      tokenizer.push('/think>')
      tokenizer.end()
      expect(out).toEqual([{ _tag: 'Close', name: 'think', pipe: undefined }])
    })

    it('treats invalid slash-prefix as content (no name after /)', () => {
      const tokens = collectWithKnown('</ >')
      // Should be treated as content since no valid name follows /
      expect(tokens).toEqual([{ _tag: 'Content', text: '</ >' }])
    })
  })

  describe('Unit B: invoke-without-keyword leniency', () => {
    function collectWithTools(input: string, tools: string[]): any[] {
      const out: any[] = []
      const tokenizer = createTokenizer(
        (token) => {
          const { raw, ...rest } = token as any
          out.push(rest)
        },
        new Set(tools),
        STD_OPTIONS,
      )
      tokenizer.push(input)
      tokenizer.end()
      return out
    }

    it('rewrites <|shell> to invoke:shell when shell is a known tool', () => {
      const tokens = collectWithTools('<|shell>', ['shell'])
      expect(tokens).toEqual([{ _tag: 'Open', name: 'invoke', variant: 'shell' }])
    })

    it('rewrites <|spawn-worker> to invoke:spawn-worker when known', () => {
      const tokens = collectWithTools('<|spawn-worker>', ['spawn-worker'])
      expect(tokens).toEqual([{ _tag: 'Open', name: 'invoke', variant: 'spawn-worker' }])
    })

    it('does NOT rewrite unknown tags', () => {
      const tokens = collectWithTools('<|unknown>', ['shell'])
      expect(tokens).toEqual([{ _tag: 'Open', name: 'unknown', variant: undefined }])
    })

    it('does NOT rewrite when variant is already present', () => {
      // <|invoke:shell> should remain as-is
      const tokens = collectWithTools('<|invoke:shell>', ['shell'])
      expect(tokens).toEqual([{ _tag: 'Open', name: 'invoke', variant: 'shell' }])
    })

    it('uses custom toolKeyword when provided', () => {
      const out: any[] = []
      const tokenizer = createTokenizer(
        (token) => {
          const { raw, ...rest } = token as any
          out.push(rest)
        },
        new Set(['shell']),
        { strictNewlines: true, toolKeyword: 'tool' },
      )
      tokenizer.push('<|shell>')
      tokenizer.end()
      expect(out).toEqual([{ _tag: 'Open', name: 'tool', variant: 'shell' }])
    })
  })

  describe('Unit C: newline enforcement for top-level tags (always on)', () => {
    function collectStrict(input: string): any[] {
      const out: any[] = []
      const tokenizer = createTokenizer(
        (token) => {
          const { raw, ...rest } = token as any
          out.push(rest)
        },
        new Set(),
        STD_OPTIONS,
      )
      tokenizer.push(input)
      tokenizer.end()
      return out
    }

    it('accepts top-level open tag preceded by newline', () => {
      const tokens = collectStrict('\n<|think>')
      expect(tokens).toEqual([
        { _tag: 'Content', text: '\n' },
        { _tag: 'Open', name: 'think', variant: undefined },
      ])
    })

    it('rejects top-level open tag NOT preceded by newline', () => {
      const tokens = collectStrict('text<|think>')
      // 'text' is flushed before tag parsing, then tag becomes content
      expect(tokens).toEqual([
        { _tag: 'Content', text: 'text' },
        { _tag: 'Content', text: '<|think>' },
      ])
    })

    it('accepts top-level close tag preceded by newline', () => {
      const tokens = collectStrict('\n<think|>')
      expect(tokens).toEqual([
        { _tag: 'Content', text: '\n' },
        { _tag: 'Close', name: 'think', pipe: undefined },
      ])
    })

    it('rejects top-level close tag NOT preceded by newline', () => {
      const tokens = collectStrict('text<think|>')
      // 'text' is flushed before tag parsing, then tag becomes content
      expect(tokens).toEqual([
        { _tag: 'Content', text: 'text' },
        { _tag: 'Content', text: '<think|>' },
      ])
    })

    it('does NOT enforce newline for non-top-level tags', () => {
      // parameter tags don't need preceding newline
      const tokens = collectStrict('text<|parameter:foo>')
      expect(tokens).toEqual([
        { _tag: 'Content', text: 'text' },
        { _tag: 'Parameter', name: 'foo' },
      ])
    })

    it('accepts message open after newline', () => {
      const tokens = collectStrict('\n<|message:user>')
      expect(tokens).toEqual([
        { _tag: 'Content', text: '\n' },
        { _tag: 'Open', name: 'message', variant: 'user' },
      ])
    })

    it('rejects message open without preceding newline', () => {
      const tokens = collectStrict('foo<|message:user>')
      expect(tokens).toEqual([
        { _tag: 'Content', text: 'foo' },
        { _tag: 'Content', text: '<|message:user>' },
      ])
    })
  })
})
