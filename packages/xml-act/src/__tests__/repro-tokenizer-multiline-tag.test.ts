import { describe, expect, it } from 'bun:test'
import { createTokenizer, type Token } from '../tokenizer'

function collect(input: string | string[]): Token[] {
  const out: Token[] = []
  const tokenizer = createTokenizer((signal) => out.push(signal))
  if (Array.isArray(input)) {
    for (const chunk of input) tokenizer.push(chunk)
  } else {
    tokenizer.push(input)
  }
  tokenizer.end()
  return out.map(s => { const { raw, ...rest } = s as any; return rest }) as Token[]
}

describe('repro: tokenizer multiline open tags', () => {
  it('single-line tag with attrs works (baseline)', () => {
    const tokens = collect('<agent-create id="foo" type="explorer">')
    expect(tokens.map((t) => t.type)).toEqual(['open'])
  })

  it('newline between tag name and first attr should still emit open', () => {
    const tokens = collect('<agent-create\nid="foo">')
    expect(tokens.map((t) => t.type)).toEqual(['open'])
  })

  it('newline between attrs should still emit open', () => {
    const tokens = collect('<agent-create id="foo"\ntype="explorer">')
    expect(tokens.map((t) => t.type)).toEqual(['open'])
  })

  it('newline after last attr before closing should still emit open', () => {
    const tokens = collect('<agent-create id="foo"\n>')
    expect(tokens.map((t) => t.type)).toEqual(['open'])
  })

  it('CRLF between attrs should still emit open', () => {
    const tokens = collect('<agent-create id="foo"\r\ntype="explorer">')
    expect(tokens.map((t) => t.type)).toEqual(['open'])
  })

  it('self-closing multiline tag should emit selfClose', () => {
    const tokens = collect('<foo\nbar="1"\n/>')
    expect(tokens.map((t) => t.type)).toEqual(['selfClose'])
  })
})
