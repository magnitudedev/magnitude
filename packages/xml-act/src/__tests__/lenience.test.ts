/**
 * XML tokenizer behavior tests.
 *
 * Verifies correct tokenization of XML-format tags, attribute parsing,
 * self-closing tags, CDATA, and content preservation for unknown tags.
 */

import { describe, expect, it } from 'vitest'
import { createTokenizer } from '../tokenizer'

function normalizeToken(token: any): any {
  const { raw, afterNewline, ...rest } = token
  // Normalize tagName → name, Map attrs → plain object
  const name = rest.tagName ?? rest.name
  const attrsRaw = rest.attrs
  const attrs = attrsRaw instanceof Map
    ? Object.fromEntries(attrsRaw)
    : attrsRaw
  const base: any = { _tag: rest._tag, name }
  if (rest._tag === 'Open' || rest._tag === 'SelfClose') base.attrs = attrs
  if (rest._tag === 'Content') base.text = rest.text
  return base
}

function collect(input: string | string[], knownToolTags?: Set<string>): any[] {
  const out: any[] = []
  const tokenizer = createTokenizer((token) => {
    out.push(normalizeToken(token))
  }, knownToolTags ?? new Set())
  if (Array.isArray(input)) {
    for (const chunk of input) tokenizer.push(chunk)
  } else {
    tokenizer.push(input)
  }
  tokenizer.end()
  return out
}

function joinContent(tokens: any[]): string {
  return tokens.filter(t => t._tag === 'Content').map(t => t.text).join('')
}

// ---------------------------------------------------------------------------
// Open tags
// ---------------------------------------------------------------------------

describe('Open tags', () => {
  it('parses <reason about="turn">', () => {
    const tokens = collect('<reason about="turn">')
    expect(tokens).toEqual([{ _tag: 'Open', name: 'reason', attrs: { about: 'turn' } }])
  })

  it('parses <message to="user">', () => {
    const tokens = collect('<message to="user">')
    expect(tokens).toEqual([{ _tag: 'Open', name: 'message', attrs: { to: 'user' } }])
  })

  it('parses <invoke tool="shell">', () => {
    const tokens = collect('<invoke tool="shell">')
    expect(tokens).toEqual([{ _tag: 'Open', name: 'invoke', attrs: { tool: 'shell' } }])
  })

  it('parses <parameter name="command">', () => {
    const tokens = collect('<parameter name="command">')
    expect(tokens).toEqual([{ _tag: 'Open', name: 'parameter', attrs: { name: 'command' } }])
  })

  it('parses <filter>', () => {
    const tokens = collect('<filter>')
    expect(tokens).toEqual([{ _tag: 'Open', name: 'filter', attrs: {} }])
  })

  it('parses multiple attributes', () => {
    const tokens = collect('<invoke tool="shell" observe=".">')
    expect(tokens[0]._tag).toBe('Open')
    expect(tokens[0].attrs.tool).toBe('shell')
    expect(tokens[0].attrs.observe).toBe('.')
  })

  it('unknown open tag → Open token (parser decides structural vs content)', () => {
    const tokens = collect('<div>')
    // Tokenizer emits Open for any well-formed tag; parser resolves to content
    expect(tokens).toEqual([{ _tag: 'Open', name: 'div', attrs: {} }])
  })
})

// ---------------------------------------------------------------------------
// Close tags
// ---------------------------------------------------------------------------

describe('Close tags', () => {
  it('parses </reason>', () => {
    const tokens = collect('\n</reason>\n')
    expect(tokens.some(t => t._tag === 'Close' && t.name === 'reason')).toBe(true)
  })

  it('parses </message>', () => {
    const tokens = collect('\n</message>\n')
    expect(tokens.some(t => t._tag === 'Close' && t.name === 'message')).toBe(true)
  })

  it('parses </invoke>', () => {
    const tokens = collect('\n</invoke>\n')
    expect(tokens.some(t => t._tag === 'Close' && t.name === 'invoke')).toBe(true)
  })

  it('parses </parameter>', () => {
    const tokens = collect('\n</parameter>\n')
    expect(tokens.some(t => t._tag === 'Close' && t.name === 'parameter')).toBe(true)
  })

  it('parses </filter>', () => {
    const tokens = collect('\n</filter>\n')
    expect(tokens.some(t => t._tag === 'Close' && t.name === 'filter')).toBe(true)
  })

  it('unknown close tag → Content', () => {
    const tokens = collect('</div>')
    expect(tokens).toEqual([{ _tag: 'Content', text: '</div>' }])
  })

  it('unknown close tag with hyphen → Content', () => {
    const tokens = collect('</skill-name>')
    expect(tokens).toEqual([{ _tag: 'Content', text: '</skill-name>' }])
  })
})

// ---------------------------------------------------------------------------
// Self-closing yield tags
// ---------------------------------------------------------------------------

describe('Self-closing yield tags', () => {
  it('parses <yield_user/>', () => {
    const tokens = collect('<yield_user/>')
    expect(tokens).toEqual([{ _tag: 'SelfClose', name: 'yield_user', attrs: {} }])
  })

  it('parses <yield_invoke/>', () => {
    const tokens = collect('<yield_invoke/>')
    expect(tokens).toEqual([{ _tag: 'SelfClose', name: 'yield_invoke', attrs: {} }])
  })

  it('parses <yield_parent/>', () => {
    const tokens = collect('<yield_parent/>')
    expect(tokens).toEqual([{ _tag: 'SelfClose', name: 'yield_parent', attrs: {} }])
  })

  it('parses <yield_worker/>', () => {
    const tokens = collect('<yield_worker/>')
    expect(tokens).toEqual([{ _tag: 'SelfClose', name: 'yield_worker', attrs: {} }])
  })
})

// ---------------------------------------------------------------------------
// Attribute parsing edge cases
// ---------------------------------------------------------------------------

describe('Attribute parsing', () => {
  it('handles single-quoted attribute values', () => {
    const tokens = collect("<message to='user'>")
    expect(tokens[0]._tag).toBe('Open')
    expect(tokens[0].attrs.to).toBe('user')
  })

  it('handles attribute with special chars in value', () => {
    const tokens = collect('<invoke tool="my-tool:v2">')
    expect(tokens[0].attrs.tool).toBe('my-tool:v2')
  })

  it('handles tag with no attributes', () => {
    const tokens = collect('<filter>')
    expect(tokens[0]).toEqual({ _tag: 'Open', name: 'filter', attrs: {} })
  })
})

// ---------------------------------------------------------------------------
// Close-tag confirmation (bounded lookahead)
// ---------------------------------------------------------------------------

describe('Close-tag confirmation', () => {
  it('confirms close tag followed by newline', () => {
    const tokens = collect('content\n</reason>\n')
    expect(tokens.some(t => t._tag === 'Close' && t.name === 'reason')).toBe(true)
  })

  it('confirms close tag followed by next open tag', () => {
    const tokens = collect('content\n</reason><message to="user">')
    expect(tokens.some(t => t._tag === 'Close' && t.name === 'reason')).toBe(true)
    expect(tokens.some(t => t._tag === 'Open' && t.name === 'message')).toBe(true)
  })

  it('confirms close tag with whitespace then newline', () => {
    const tokens = collect('content\n</reason>  \n')
    expect(tokens.some(t => t._tag === 'Close' && t.name === 'reason')).toBe(true)
  })

  it('close tag followed by prose — Close emitted immediately by tokenizer', () => {
    // Tokenizer emits Close immediately — parser decides if it's structural
    const tokens = collect('content\n</reason> some prose here')
    // Tokenizer DOES emit a Close token (parser handles confirmation)
    expect(tokens.some(t => t._tag === 'Close' && t.name === 'reason')).toBe(true)
    // Prose text preserved as content
    expect(joinContent(tokens)).toContain('some prose here')
  })
})

// ---------------------------------------------------------------------------
// CDATA
// ---------------------------------------------------------------------------

describe('CDATA handling', () => {
  it('emits CDATA inner text as Content', () => {
    const tokens = collect('hello world')
    expect(joinContent(tokens)).toBe('hello world')
  })

  it('CDATA with < and > inside', () => {
    const tokens = collect('a < b && b > c')
    expect(joinContent(tokens)).toBe('a < b && b > c')
  })

  it('CDATA split across chunk boundary', () => {
    const tokens = collect(['hel', 'lo'])
    expect(joinContent(tokens)).toBe('hello')
  })
})

// ---------------------------------------------------------------------------
// Content preservation
// ---------------------------------------------------------------------------

describe('Content preservation', () => {
  it('preserves plain text', () => {
    expect(joinContent(collect('hello world'))).toBe('hello world')
  })

  it('preserves text with angle brackets that are not valid tags', () => {
    const tokens = collect('a < b and b > c')
    expect(joinContent(tokens)).toBe('a < b and b > c')
  })

  it('emits Open token for unknown tags (parser treats as content)', () => {
    const tokens = collect('text <unknown-tag> more text')
    // Tokenizer emits Open for any well-formed tag; parser resolves to content
    expect(tokens.some(t => t._tag === 'Open' && t.name === 'unknown-tag')).toBe(true)
    expect(joinContent(tokens)).toContain('text')
    expect(joinContent(tokens)).toContain('more text')
  })
})
