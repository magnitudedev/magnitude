/**
 * Close tag handling tests (XML format).
 * Unknown tags in content should be preserved literally.
 */

import { describe, expect, it } from 'vitest'
import { createTokenizer } from '../tokenizer'
import { tokenRaw } from '../parser/resolve'
import type { Token, SourcePos, SourceSpan } from '../types'

const ZERO_POS: SourcePos = { offset: 0, line: 1, col: 1 }
const ZERO_SPAN: SourceSpan = { start: ZERO_POS, end: ZERO_POS }

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

function collect(input: string | string[]): any[] {
  const out: any[] = []
  const tokenizer = createTokenizer((token) => {
    out.push(normalizeToken(token))
  }, new Set())
  if (Array.isArray(input)) {
    for (const chunk of input) tokenizer.push(chunk)
  } else {
    tokenizer.push(input)
  }
  tokenizer.end()
  return out
}

describe('unknown tags in content', () => {
  it('unknown open tag <skill-name> → Open token (parser treats as content)', () => {
    const tokens = collect('<skill-name>')
    // Tokenizer emits Open for well-formed tags; parser resolves to content based on validTags
    expect(tokens).toEqual([{ _tag: 'Open', name: 'skill-name', attrs: {} }])
  })

  it('unknown close tag </skill-name> → Content token', () => {
    const tokens = collect('</skill-name>')
    expect(tokens).toEqual([{ _tag: 'Content', text: '</skill-name>' }])
  })

  it('tokenRaw for Close reconstructs as </name>', () => {
    const token: Token = { _tag: 'Close', span: ZERO_SPAN, tagName: 'reason', afterNewline: false }
    expect(tokenRaw(token)).toBe('</reason>')
  })

  it('tokenRaw for Open reconstructs as <name attr="val">', () => {
    const token: Token = { _tag: 'Open', span: ZERO_SPAN, tagName: 'invoke', attrs: new Map([['tool', 'shell']]), afterNewline: false }
    expect(tokenRaw(token)).toBe('<invoke tool="shell">')
  })

  it('tokenRaw for SelfClose reconstructs as <name/>', () => {
    const token: Token = { _tag: 'SelfClose', span: ZERO_SPAN, tagName: 'yield_user', attrs: new Map(), afterNewline: false }
    expect(tokenRaw(token)).toBe('<yield_user/>')
  })
})
