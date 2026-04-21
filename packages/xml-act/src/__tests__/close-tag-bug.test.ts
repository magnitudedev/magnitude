/**
 * Failing tests for close tag handling bug.
 *
 * BUG: Arbitrary unknown tags like <skill-name> are being incorrectly
 * parsed as Close tokens and reconstructed as <skill-name|> (with pipe).
 *
 * Expected behavior:
 * 1. Tokenizer: <name> should only produce a Close token for KNOWN structural
 *    tags (think, message, invoke, parameter, filter). Unknown names → Content.
 * 2. tokenRaw: Close tokens with pipe:undefined should reconstruct as <name>,
 *    not <name|>.
 * 3. Integration: message content containing <skill-name> should be preserved
 *    literally, not mangled to <skill-name|>.
 */

import { describe, expect, it } from 'vitest'
import { createTokenizer } from '../tokenizer'
import { tokenRaw } from '../parser/resolve'
import type { Token } from '../types'

const STD_OPTIONS = { strictNewlines: true, toolKeyword: 'invoke' } as const

function collect(input: string | string[]): any[] {
  const out: any[] = []
  const tokenizer = createTokenizer((token) => {
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

// ---------------------------------------------------------------------------
// Tokenizer: unknown names in <name> form should be Content, not Close
// ---------------------------------------------------------------------------

describe('Tokenizer: bare <name> form for unknown tags', () => {
  it('<skill-name> should produce Content token, not Close', () => {
    const tokens = collect('hello <skill-name> world')
    // Should be a single content token (or content tokens around the text)
    // — no Close token with name 'skill-name'
    const closeTokens = tokens.filter(t => t._tag === 'Close' && t.name === 'skill-name')
    expect(closeTokens).toHaveLength(0)
    // The text should be preserved
    const allText = tokens.filter(t => t._tag === 'Content').map(t => t.text).join('')
    expect(allText).toContain('<skill-name>')
  })

  it('<div> should produce Content token, not Close', () => {
    const tokens = collect('\n<div>')
    const closeTokens = tokens.filter(t => t._tag === 'Close' && t.name === 'div')
    expect(closeTokens).toHaveLength(0)
    const allText = tokens.filter(t => t._tag === 'Content').map(t => t.text).join('')
    expect(allText).toContain('<div>')
  })

  it('<foo-bar> should produce Content token, not Close', () => {
    const tokens = collect('\n<foo-bar>')
    const closeTokens = tokens.filter(t => t._tag === 'Close' && t.name === 'foo-bar')
    expect(closeTokens).toHaveLength(0)
    const allText = tokens.filter(t => t._tag === 'Content').map(t => t.text).join('')
    expect(allText).toContain('<foo-bar>')
  })

  it('<unknown123> should produce Content token, not Close', () => {
    const tokens = collect('\n<unknown123>')
    const closeTokens = tokens.filter(t => t._tag === 'Close' && t.name === 'unknown123')
    expect(closeTokens).toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// Tokenizer: known structural tags should STILL produce Close tokens
// ---------------------------------------------------------------------------

describe('Tokenizer: bare <name> form for known structural tags', () => {
  it('<parameter> should still produce a Close token (known structural tag)', () => {
    // parameter doesn't require newline enforcement
    const tokens = collect('<parameter>')
    const closeToken = tokens.find(t => t._tag === 'Close' && t.name === 'parameter')
    expect(closeToken).toBeDefined()
  })

  it('<think> should still produce a Close token when on its own line', () => {
    const tokens = collect('\n<think>')
    const closeToken = tokens.find(t => t._tag === 'Close' && t.name === 'think')
    expect(closeToken).toBeDefined()
  })

  it('<message> should still produce a Close token when on its own line', () => {
    const tokens = collect('\n<message>')
    const closeToken = tokens.find(t => t._tag === 'Close' && t.name === 'message')
    expect(closeToken).toBeDefined()
  })

  it('<invoke> should still produce a Close token when on its own line', () => {
    const tokens = collect('\n<invoke>')
    const closeToken = tokens.find(t => t._tag === 'Close' && t.name === 'invoke')
    expect(closeToken).toBeDefined()
  })

  it('<filter> should still produce a Close token (no newline requirement)', () => {
    const tokens = collect('<filter>')
    const closeToken = tokens.find(t => t._tag === 'Close' && t.name === 'filter')
    expect(closeToken).toBeDefined()
  })
})

// ---------------------------------------------------------------------------
// tokenRaw: Close token reconstruction fidelity
// ---------------------------------------------------------------------------

describe('tokenRaw: Close token reconstruction', () => {
  it('Close with pipe:undefined reconstructs as <name|> (canonical form)', () => {
    // tokenRaw always uses canonical pipe form for Close tokens.
    // Unknown tags never reach tokenRaw because the tokenizer filters them.
    const token: Token = { _tag: 'Close', name: 'skill-name', pipe: undefined }
    expect(tokenRaw(token)).toBe('<skill-name|>')
  })

  it('Close with pipe:undefined for known tag reconstructs as <name|>', () => {
    const token: Token = { _tag: 'Close', name: 'message', pipe: undefined }
    expect(tokenRaw(token)).toBe('<message|>')
  })

  it('Close with pipe:"" (empty string) should reconstruct as <name|>', () => {
    const token: Token = { _tag: 'Close', name: 'message', pipe: '' }
    expect(tokenRaw(token)).toBe('<message|>')
  })

  it('Close with pipe:"filter" should reconstruct as <name|filter>', () => {
    const token: Token = { _tag: 'Close', name: 'invoke', pipe: 'filter' }
    expect(tokenRaw(token)).toBe('<invoke|filter>')
  })
})

// ---------------------------------------------------------------------------
// Integration: full parse should preserve unknown tags literally
// ---------------------------------------------------------------------------

import { createParser } from '../parser/index'

function parseInput(input: string) {
  const p = createParser({ tools: new Map() })
  const tokenizer = createTokenizer(
    (token) => p.pushToken(token),
    new Set(),
  )
  tokenizer.push(input)
  const fromPush = p.drain()
  tokenizer.end()
  p.end()
  const fromEnd = p.drain()
  return [...fromPush, ...fromEnd]
}

describe('Integration: unknown tags preserved in message content', () => {
  it('message body containing <skill-name> should not mangle it to <skill-name|>', () => {
    const input = [
      '<|message:user>',
      'See `packages/skills/builtin/<skill-name>/SKILL.md` for details.',
      '<message|>',
    ].join('\n')

    const events = parseInput(input)
    const allContent = JSON.stringify(events)
    // The text <skill-name> should be preserved literally
    expect(allContent).toContain('<skill-name>')
    // It must NOT be mangled to <skill-name|>
    expect(allContent).not.toContain('<skill-name|>')
  })

  it('message body containing multiple unknown tags should preserve them all', () => {
    const input = [
      '<|message:user>',
      'Use <div> and <span> and <skill-name> tags.',
      '<message|>',
    ].join('\n')

    const events = parseInput(input)
    const allContent = JSON.stringify(events)
    expect(allContent).toContain('<div>')
    expect(allContent).toContain('<span>')
    expect(allContent).toContain('<skill-name>')
    expect(allContent).not.toContain('<div|>')
    expect(allContent).not.toContain('<span|>')
    expect(allContent).not.toContain('<skill-name|>')
  })
})
