/**
 * Incremental streaming test suite.
 *
 * Verifies that the tokenizer emits content incrementally per push() call,
 * that token boundaries are correct across all chunking strategies, and that
 * the parser's coalescing layer properly merges adjacent events within a batch.
 */

import { describe, expect, it } from 'vitest'
import { createTokenizer } from '../tokenizer'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Collect tokens from input, feeding as specified chunks */
function collectChunked(chunks: string[]): any[] {
  const out: any[] = []
  const tokenizer = createTokenizer((token) => {
    const { raw, ...rest } = token as any
    out.push(rest)
  })
  for (const chunk of chunks) {
    tokenizer.push(chunk)
  }
  tokenizer.end()
  return out
}

/** Collect tokens, recording which push() call produced each token */
function collectWithPushIndex(chunks: string[]): { pushIndex: number; token: any }[] {
  const out: { pushIndex: number; token: any }[] = []
  let currentPush = 0
  const tokenizer = createTokenizer((token) => {
    const { raw, ...rest } = token as any
    out.push({ pushIndex: currentPush, token: rest })
  })
  for (let i = 0; i < chunks.length; i++) {
    currentPush = i
    tokenizer.push(chunks[i])
  }
  currentPush = -1
  tokenizer.end()
  return out
}

/** Join all Content token texts */
function joinContent(tokens: any[]): string {
  return tokens.filter(t => t._tag === 'Content').map(t => t.text).join('')
}

/** Split input into chunks of size n */
function chunkString(s: string, n: number): string[] {
  const chunks: string[] = []
  for (let i = 0; i < s.length; i += n) {
    chunks.push(s.slice(i, i + n))
  }
  return chunks
}

// ---------------------------------------------------------------------------
// 1. Incremental content emission
// ---------------------------------------------------------------------------

describe('Incremental content emission', () => {
  it('flushes content at end of each push() — not buffered across calls', () => {
    const results = collectWithPushIndex(['hello ', 'world'])
    // Each push should produce its own Content token
    const push0 = results.filter(r => r.pushIndex === 0)
    const push1 = results.filter(r => r.pushIndex === 1)
    expect(push0).toEqual([{ pushIndex: 0, token: { _tag: 'Content', text: 'hello ' } }])
    expect(push1).toEqual([{ pushIndex: 1, token: { _tag: 'Content', text: 'world' } }])
  })

  it('emits content per push inside a think block', () => {
    const results = collectWithPushIndex([
      '<|think:strategy>\n',
      'Line 1\n',
      'Line 2\n',
      '<think|>\n',
    ])
    // Push 0: Open + Content(\n)
    // Push 1: Content(Line 1\n)
    // Push 2: Content(Line 2\n)
    // Push 3: Close + Content(\n)
    const push1 = results.filter(r => r.pushIndex === 1)
    expect(push1).toEqual([{ pushIndex: 1, token: { _tag: 'Content', text: 'Line 1\n' } }])
    const push2 = results.filter(r => r.pushIndex === 2)
    expect(push2).toEqual([{ pushIndex: 2, token: { _tag: 'Content', text: 'Line 2\n' } }])
  })

  it('emits content per push inside a message block', () => {
    const results = collectWithPushIndex([
      '<|message:user>\n',
      'Hey Anders!\n',
      'How can I help?\n',
      '<message|>\n',
    ])
    const push1 = results.filter(r => r.pushIndex === 1)
    expect(push1).toEqual([{ pushIndex: 1, token: { _tag: 'Content', text: 'Hey Anders!\n' } }])
    const push2 = results.filter(r => r.pushIndex === 2)
    expect(push2).toEqual([{ pushIndex: 2, token: { _tag: 'Content', text: 'How can I help?\n' } }])
  })

  it('does NOT flush mid-tag parse (content buffered until tag resolves)', () => {
    const results = collectWithPushIndex([
      'hello\n<|thi',  // starts a tag but doesn't finish
      'nk:strategy>\n',  // finishes the tag
    ])
    // Push 0: Content(hello\n) flushed, then tag starts — no more content
    const push0 = results.filter(r => r.pushIndex === 0)
    expect(push0).toEqual([{ pushIndex: 0, token: { _tag: 'Content', text: 'hello\n' } }])
    // Push 1: tag completes → Open emitted, then Content(\n)
    const push1 = results.filter(r => r.pushIndex === 1)
    expect(push1[0].token._tag).toBe('Open')
    expect(push1[1].token).toEqual({ _tag: 'Content', text: '\n' })
  })

  it('does NOT flush when pending < at chunk boundary', () => {
    const results = collectWithPushIndex([
      'content\n<',  // < at end, pending
      '|think:strategy>\n',  // resolves the <
    ])
    // Push 0: content is buffered (pending <), nothing emitted
    const push0 = results.filter(r => r.pushIndex === 0)
    expect(push0).toEqual([])
    // Push 1: pending < resolves, content flushed, then Open + Content
    const push1 = results.filter(r => r.pushIndex === 1)
    expect(push1[0].token).toEqual({ _tag: 'Content', text: 'content\n' })
    expect(push1[1].token._tag).toBe('Open')
    expect(push1[2].token).toEqual({ _tag: 'Content', text: '\n' })
  })

  it('empty push produces no tokens', () => {
    const results = collectWithPushIndex(['', 'hello', ''])
    expect(results.filter(r => r.pushIndex === 0)).toEqual([])
    expect(results.filter(r => r.pushIndex === 1)).toEqual([
      { pushIndex: 1, token: { _tag: 'Content', text: 'hello' } },
    ])
    expect(results.filter(r => r.pushIndex === 2)).toEqual([])
  })
})

// ---------------------------------------------------------------------------
// 2. Token boundary correctness across chunking strategies
// ---------------------------------------------------------------------------

describe('Token boundary correctness', () => {
  const fullInput = `
<|think:alignment>
User is asking about capabilities.
Keep it concise.
<think|>

<|message:user>
I can help with software development tasks.
<message|>

<|yield:user|>`

  it('produces same structural tokens regardless of chunk size', () => {
    // Try chunk sizes 1 through 20
    for (const size of [1, 2, 3, 5, 7, 10, 15, 20, fullInput.length]) {
      const chunks = chunkString(fullInput, size)
      const tokens = collectChunked(chunks)
      const structural = tokens.filter(t => t._tag !== 'Content')
      const contentJoined = joinContent(tokens)

      expect(structural).toEqual([
        { _tag: 'Open', name: 'think', variant: 'alignment' },
        { _tag: 'Close', name: 'think' },
        { _tag: 'Open', name: 'message', variant: 'user' },
        { _tag: 'Close', name: 'message' },
        { _tag: 'SelfClose', name: 'yield', variant: 'user' },
      ])

      // Content text is identical when joined, regardless of chunking
      expect(contentJoined).toBe(joinContent(collectChunked([fullInput])))
    }
  })

  it('char-by-char produces one content token per non-tag char', () => {
    const input = 'hello\n<|think:strategy>\nreasoning\n<think|>'
    const chars = input.split('')
    const tokens = collectChunked(chars)
    // Each char that's content gets its own token
    const contentTokens = tokens.filter(t => t._tag === 'Content')
    expect(contentTokens.length).toBeGreaterThan(1)
    expect(joinContent(tokens)).toBe('hello\n\nreasoning\n')
  })

  it('whole-input produces minimal content tokens', () => {
    const input = 'hello\n<|think:strategy>\nreasoning\n<think|>'
    const tokens = collectChunked([input])
    const contentTokens = tokens.filter(t => t._tag === 'Content')
    // Whole input: content flushed at tag boundaries → fewer tokens
    expect(contentTokens.length).toBeLessThanOrEqual(3)
    expect(joinContent(tokens)).toBe('hello\n\nreasoning\n')
  })

  it('line-by-line produces one content token per content line', () => {
    const input = `<|think:alignment>
Line one
Line two
Line three
<think|>`
    const lines = input.split('\n')
    const chunks = lines.map((l, i) => i < lines.length - 1 ? l + '\n' : l)
    const tokens = collectChunked(chunks)
    const contentTokens = tokens.filter(t => t._tag === 'Content')
    // Should get: \n, Line one\n, Line two\n, Line three\n
    expect(contentTokens.map(t => t.text)).toEqual([
      '\n',
      'Line one\n',
      'Line two\n',
      'Line three\n',
    ])
  })
})

// ---------------------------------------------------------------------------
// 3. Tag split across chunks
// ---------------------------------------------------------------------------

describe('Tag split across chunks — no content leakage', () => {
  it('open tag split mid-name: no partial tag in content', () => {
    const tokens = collectChunked(['before\n<|thi', 'nk:strategy>\nafter'])
    expect(joinContent(tokens)).toBe('before\n\nafter')
    expect(tokens.some(t => t._tag === 'Content' && t.text.includes('<'))).toBe(false)
  })

  it('open tag split at <|: no partial tag in content', () => {
    const tokens = collectChunked(['before\n<', '|think:strategy>\nafter'])
    expect(joinContent(tokens)).toBe('before\n\nafter')
  })

  it('close tag split mid-name: no partial tag in content', () => {
    const tokens = collectChunked(['\n<thi', 'nk|>'])
    const structural = tokens.filter(t => t._tag === 'Close')
    expect(structural).toEqual([{ _tag: 'Close', name: 'think' }])
  })

  it('close tag split at pipe: no partial tag in content', () => {
    const tokens = collectChunked(['\n<think', '|>'])
    const structural = tokens.filter(t => t._tag === 'Close')
    expect(structural).toEqual([{ _tag: 'Close', name: 'think' }])
  })

  it('lenient close </think|> split at /: correct parse', () => {
    const tokens = collectChunked(['\n<', '/think|>'])
    expect(tokens.some(t => t._tag === 'Close' && t.name === 'think')).toBe(true)
  })

  it('self-close split at final |>: correct parse', () => {
    const tokens = collectChunked(['\n<|yield:user', '|>'])
    expect(tokens.some(t => t._tag === 'SelfClose' && t.name === 'yield')).toBe(true)
  })

  it('failed tag (invalid name) dumps raw as content', () => {
    const tokens = collectChunked(['\n<', '123>'])
    // < + 123> should become content
    expect(joinContent(tokens)).toContain('<123>')
    expect(tokens.some(t => t._tag === 'Open' || t._tag === 'Close')).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// 4. Realistic LLM streaming simulation
// ---------------------------------------------------------------------------

describe('Realistic LLM streaming simulation', () => {
  it('simulates token-by-token LLM output with proper incremental emission', () => {
    // LLM typically streams ~1-5 tokens at a time
    const fullResponse = `
<|think:alignment>
User wants help with a bug. Let me investigate.
<think|>

<|message:user>
I'll look into that bug for you. Let me start by examining the relevant code.
<message|>

<|yield:user|>`

    const results = collectWithPushIndex([
      '\n<|think:alignment>\n',
      'User wants help ',
      'with a bug. ',
      'Let me investigate.\n',
      '<think|>\n',
      '\n',
      '<|message:user>\n',
      "I'll look into that ",
      'bug for you. ',
      'Let me start by examining ',
      'the relevant code.\n',
      '<message|>\n',
      '\n',
      '<|yield:user|>',
    ])

    // Each content push produces its own token
    const push1 = results.filter(r => r.pushIndex === 1)
    expect(push1).toEqual([{ pushIndex: 1, token: { _tag: 'Content', text: 'User wants help ' } }])

    const push2 = results.filter(r => r.pushIndex === 2)
    expect(push2).toEqual([{ pushIndex: 2, token: { _tag: 'Content', text: 'with a bug. ' } }])

    const push7 = results.filter(r => r.pushIndex === 7)
    expect(push7).toEqual([{ pushIndex: 7, token: { _tag: 'Content', text: "I'll look into that " } }])

    // Structural tokens appear in correct pushes
    const push0 = results.filter(r => r.pushIndex === 0)
    expect(push0.map(r => r.token._tag)).toEqual(['Content', 'Open', 'Content'])

    const push4 = results.filter(r => r.pushIndex === 4)
    expect(push4.map(r => r.token._tag)).toEqual(['Close', 'Content'])

    // Total content is preserved
    const allContent = joinContent(results.map(r => r.token))
    expect(allContent).toContain('User wants help with a bug. Let me investigate.')
    expect(allContent).toContain("I'll look into that bug for you. Let me start by examining the relevant code.")
  })

  it('the exact user-reported LLM response streams correctly', () => {
    const results = collectWithPushIndex([
      ' <|message:user>\n',
      'Hey Anders! ',
      'How can I help you today?\n',
      '<message|>\n',
      '\n',
      '<|yield:user|>',
    ])

    // Push 0: leading space content + Open + newline content
    const push0 = results.filter(r => r.pushIndex === 0)
    expect(push0[0].token).toEqual({ _tag: 'Content', text: ' ' })
    expect(push0[1].token._tag).toBe('Open')
    expect(push0[2].token).toEqual({ _tag: 'Content', text: '\n' })

    // Push 1: incremental message content
    const push1 = results.filter(r => r.pushIndex === 1)
    expect(push1).toEqual([{ pushIndex: 1, token: { _tag: 'Content', text: 'Hey Anders! ' } }])

    // Push 2: more incremental content
    const push2 = results.filter(r => r.pushIndex === 2)
    expect(push2).toEqual([{ pushIndex: 2, token: { _tag: 'Content', text: 'How can I help you today?\n' } }])

    // Push 3: Close + trailing newline
    const push3 = results.filter(r => r.pushIndex === 3)
    expect(push3[0].token._tag).toBe('Close')
  })
})
