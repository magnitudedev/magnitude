/**
 * Incremental streaming test suite (XML format).
 *
 * Verifies that the tokenizer emits content incrementally per push() call,
 * that token boundaries are correct across all chunking strategies.
 */

import { describe, expect, it } from 'vitest'
import { createTokenizer } from '../tokenizer'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function normalizeToken(token: any): any {
  const { raw, afterNewline, ...rest } = token
  const name = rest.tagName ?? rest.name
  const attrsRaw = rest.attrs
  const attrs = attrsRaw instanceof Map ? Object.fromEntries(attrsRaw) : attrsRaw
  const base: any = { _tag: rest._tag, name }
  if (rest._tag === 'Open' || rest._tag === 'SelfClose') base.attrs = attrs
  if (rest._tag === 'Content') base.text = rest.text
  return base
}

function collectChunked(chunks: string[]): any[] {
  const out: any[] = []
  const tokenizer = createTokenizer((token) => {
    out.push(normalizeToken(token))
  })
  for (const chunk of chunks) {
    tokenizer.push(chunk)
  }
  tokenizer.end()
  return out
}

function collectWithPushIndex(chunks: string[]): { pushIndex: number; token: any }[] {
  const out: { pushIndex: number; token: any }[] = []
  let currentPush = 0
  const tokenizer = createTokenizer((token) => {
    out.push({ pushIndex: currentPush, token: normalizeToken(token) })
  })
  for (let i = 0; i < chunks.length; i++) {
    currentPush = i
    tokenizer.push(chunks[i])
  }
  currentPush = -1
  tokenizer.end()
  return out
}

function joinContent(tokens: any[]): string {
  return tokens.filter(t => t._tag === 'Content').map(t => t.text).join('')
}

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
    const push0 = results.filter(r => r.pushIndex === 0)
    const push1 = results.filter(r => r.pushIndex === 1)
    expect(push0).toEqual([{ pushIndex: 0, token: { _tag: 'Content', text: 'hello ' } }])
    expect(push1).toEqual([{ pushIndex: 1, token: { _tag: 'Content', text: 'world' } }])
  })

  it('emits content per push inside a think block', () => {
    const results = collectWithPushIndex([
      '<magnitude:think about="strategy">\n',
      'Line 1\n',
      'Line 2\n',
      '</magnitude:think>\n',
    ])
    const push1 = results.filter(r => r.pushIndex === 1)
    expect(push1).toEqual([{ pushIndex: 1, token: { _tag: 'Content', text: 'Line 1\n' } }])
    const push2 = results.filter(r => r.pushIndex === 2)
    expect(push2).toEqual([{ pushIndex: 2, token: { _tag: 'Content', text: 'Line 2\n' } }])
  })

  it('emits content per push inside a message block', () => {
    const results = collectWithPushIndex([
      '<magnitude:message to="user">\n',
      'Hey Anders!\n',
      'How can I help?\n',
      '</magnitude:message>\n',
    ])
    const push1 = results.filter(r => r.pushIndex === 1)
    expect(push1).toEqual([{ pushIndex: 1, token: { _tag: 'Content', text: 'Hey Anders!\n' } }])
    const push2 = results.filter(r => r.pushIndex === 2)
    expect(push2).toEqual([{ pushIndex: 2, token: { _tag: 'Content', text: 'How can I help?\n' } }])
  })

  it('does NOT flush mid-tag parse (content buffered until tag resolves)', () => {
    const results = collectWithPushIndex([
      'hello\n<magnitude:think',
      ' about="strategy">\n',
    ])
    const push0 = results.filter(r => r.pushIndex === 0)
    expect(push0).toEqual([{ pushIndex: 0, token: { _tag: 'Content', text: 'hello\n' } }])
    const push1 = results.filter(r => r.pushIndex === 1)
    expect(push1[0].token._tag).toBe('Open')
  })

  it('does NOT flush when pending < at chunk boundary', () => {
    const results = collectWithPushIndex([
      'content\n<',
      'think about="strategy">\n',
    ])
    const push0 = results.filter(r => r.pushIndex === 0)
    // < pending — nothing emitted yet
    expect(push0).toEqual([])
    const push1 = results.filter(r => r.pushIndex === 1)
    expect(push1[0].token).toEqual({ _tag: 'Content', text: 'content\n' })
    expect(push1[1].token._tag).toBe('Open')
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
  // Trailing newline ensures close tags are confirmed
  const fullInput = `<magnitude:think about="alignment">
User is asking about capabilities.
Keep it concise.
</magnitude:think>
<magnitude:message to="user">
I can help with software development tasks.
</magnitude:message>
<magnitude:yield_user/>
`

  it('produces same structural tokens regardless of chunk size', () => {
    for (const size of [1, 2, 3, 5, 7, 10, 15, 20, fullInput.length]) {
      const chunks = chunkString(fullInput, size)
      const tokens = collectChunked(chunks)
      const structural = tokens.filter(t => t._tag !== 'Content')
      const contentJoined = joinContent(tokens)

      expect(structural).toEqual([
        { _tag: 'Open', name: 'magnitude:think', attrs: { about: 'alignment' } },
        { _tag: 'Close', name: 'magnitude:think' },
        { _tag: 'Open', name: 'magnitude:message', attrs: { to: 'user' } },
        { _tag: 'Close', name: 'magnitude:message' },
        { _tag: 'SelfClose', name: 'magnitude:yield_user', attrs: {} },
      ])

      expect(contentJoined).toBe(joinContent(collectChunked([fullInput])))
    }
  })

  it('char-by-char produces one content token per non-tag char', () => {
    const input = 'hello\n<magnitude:think about="strategy">\nreasoning\n</magnitude:think>\n'
    const chars = input.split('')
    const tokens = collectChunked(chars)
    const contentTokens = tokens.filter(t => t._tag === 'Content')
    expect(contentTokens.length).toBeGreaterThan(1)
    // trailing \n after </magnitude:think> is emitted as content (confirmation character)
    expect(joinContent(tokens)).toBe('hello\n\nreasoning\n\n')
  })

  it('whole-input produces minimal content tokens', () => {
    const input = 'hello\n<magnitude:think about="strategy">\nreasoning\n</magnitude:think>\n'
    const tokens = collectChunked([input])
    const contentTokens = tokens.filter(t => t._tag === 'Content')
    expect(contentTokens.length).toBeLessThanOrEqual(3)
    // trailing \n after </magnitude:think> is emitted as content (confirmation character)
    expect(joinContent(tokens)).toBe('hello\n\nreasoning\n\n')
  })

  it('line-by-line produces one content token per content line', () => {
    const input = `<magnitude:think about="alignment">
Line one
Line two
Line three
</magnitude:think>
`
    const lines = input.split('\n')
    const chunks = lines.map((l, i) => i < lines.length - 1 ? l + '\n' : l)
    const tokens = collectChunked(chunks)
    const contentTokens = tokens.filter(t => t._tag === 'Content')
    expect(contentTokens.map(t => t.text)).toEqual([
      '\n',
      'Line one\n',
      'Line two\n',
      'Line three\n',
      '\n', // trailing \n after </magnitude:think> (confirmation character)
    ])
  })
})

// ---------------------------------------------------------------------------
// 3. Tag split across chunks
// ---------------------------------------------------------------------------

describe('Tag split across chunks — no content leakage', () => {
  it('open tag split mid-name: no partial tag in content', () => {
    const tokens = collectChunked(['before\n<rea', 'son about="strategy">\nafter'])
    expect(joinContent(tokens)).toBe('before\n\nafter')
    expect(tokens.some(t => t._tag === 'Content' && t.text.includes('<'))).toBe(false)
  })

  it('open tag split at <: no partial tag in content', () => {
    const tokens = collectChunked(['before\n<magnitude:', 'think about="strategy">\nafter'])
  })

  it('close tag split mid-name: no partial tag in content', () => {
    const tokens = collectChunked(['\n</magnitude:thi', 'nk>\n'])
    const structural = tokens.filter(t => t._tag === 'Close')
    expect(structural).toEqual([{ _tag: 'Close', name: 'magnitude:think' }])
  })

  it('close tag split at /: no partial tag in content', () => {
    const tokens = collectChunked(['\n</', 'magnitude:think>\n'])
    const structural = tokens.filter(t => t._tag === 'Close')
    expect(structural).toEqual([{ _tag: 'Close', name: 'magnitude:think' }])
  })

  it('self-close <magnitude:yield_user/> split mid-name: correct parse', () => {
    const tokens = collectChunked(['\n<yield_u', 'ser/>\n'])
    expect(tokens.some(t => t._tag === 'SelfClose' && t.name === 'yield_user')).toBe(true)
  })

  it('failed tag (invalid name) dumps raw as content', () => {
    const tokens = collectChunked(['\n<', '123>'])
    expect(joinContent(tokens)).toContain('<123>')
    expect(tokens.some(t => t._tag === 'Open' || t._tag === 'Close')).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// 4. Realistic LLM streaming simulation
// ---------------------------------------------------------------------------

describe('Realistic LLM streaming simulation', () => {
  it('simulates token-by-token LLM output with proper incremental emission', () => {
    const results = collectWithPushIndex([
      '<magnitude:think about="alignment">\n',
      'User wants help ',
      'with a bug. ',
      'Let me investigate.\n',
      '</magnitude:think>\n',
      '\n',
      '<magnitude:message to="user">\n',
      "I'll look into that ",
      'bug for you.\n',
      '</magnitude:message>\n',
      '\n',
      '<magnitude:yield_user/>',
    ])

    const push1 = results.filter(r => r.pushIndex === 1)
    expect(push1).toEqual([{ pushIndex: 1, token: { _tag: 'Content', text: 'User wants help ' } }])

    const push2 = results.filter(r => r.pushIndex === 2)
    expect(push2).toEqual([{ pushIndex: 2, token: { _tag: 'Content', text: 'with a bug. ' } }])

    const push7 = results.filter(r => r.pushIndex === 7)
    expect(push7).toEqual([{ pushIndex: 7, token: { _tag: 'Content', text: "I'll look into that " } }])

    const push0 = results.filter(r => r.pushIndex === 0)
    expect(push0.map(r => r.token._tag)).toEqual(['Open', 'Content'])

    const allContent = joinContent(results.map(r => r.token))
    expect(allContent).toContain('User wants help with a bug. Let me investigate.')
    expect(allContent).toContain("I'll look into that bug for you.")
  })

  it('the exact user-reported LLM response streams correctly', () => {
    const results = collectWithPushIndex([
      ' <magnitude:message to="user">\n',
      'Hey Anders! ',
      'How can I help you today?\n',
      '</magnitude:message>\n',
      '\n',
      '<magnitude:yield_user/>',
    ])

    const push0 = results.filter(r => r.pushIndex === 0)
    expect(push0[0].token).toEqual({ _tag: 'Content', text: ' ' })
    expect(push0[1].token._tag).toBe('Open')
    expect(push0[2].token).toEqual({ _tag: 'Content', text: '\n' })

    const push1 = results.filter(r => r.pushIndex === 1)
    expect(push1).toEqual([{ pushIndex: 1, token: { _tag: 'Content', text: 'Hey Anders! ' } }])

    const push2 = results.filter(r => r.pushIndex === 2)
    expect(push2).toEqual([{ pushIndex: 2, token: { _tag: 'Content', text: 'How can I help you today?\n' } }])

    // Close emitted immediately on push 3 (tokenizer emits Close immediately now)
    const push3 = results.filter(r => r.pushIndex === 3)
    expect(push3[0].token._tag).toBe('Close')
  })
})
