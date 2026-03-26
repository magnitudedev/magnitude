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
  // Strip `raw` field — it's an implementation detail for unknown tag reconstruction
  return out.map(s => { const { raw, ...rest } = s as any; return rest }) as Token[]
}

describe('tokenizer', () => {
  it('basic open tag', () => {
    expect(collect('<foo>')).toEqual([
      { type: 'open', tagName: 'foo', attrs: new Map(), afterNewline: true },
    ])
  })

  it('close tag', () => {
    expect(collect('</foo>')).toEqual([{ type: 'close', tagName: 'foo', afterNewline: true }])
  })

  it('self-close', () => {
    expect(collect('<foo/>')).toEqual([
      { type: 'selfClose', tagName: 'foo', attrs: new Map(), afterNewline: true },
    ])
  })

  it('attributes + quoting', () => {
    expect(collect(`<foo bar="baz" x='y'>`)).toEqual([
      {
        type: 'open',
        tagName: 'foo',
        attrs: new Map([
          ['bar', 'baz'],
          ['x', 'y'],
        ]),
        afterNewline: true,
      },
    ])
  })

  it('content batching between tags', () => {
    expect(collect('<a>hello world</a>')).toEqual([
      { type: 'open', tagName: 'a', attrs: new Map(), afterNewline: true },
      { type: 'content', text: 'hello world' },
      { type: 'close', tagName: 'a', afterNewline: false },
    ])
  })

  it('invalid < forms stay content', () => {
    expect(collect('<1foo> < foo>')).toEqual([{ type: 'content', text: '<1foo> < foo>' }])
  })

  it('afterNewline tracking', () => {
    expect(collect('x<foo>\n<bar>')).toEqual([
      { type: 'content', text: 'x' },
      { type: 'open', tagName: 'foo', attrs: new Map(), afterNewline: false },
      { type: 'content', text: '\n' },
      { type: 'open', tagName: 'bar', attrs: new Map(), afterNewline: true },
    ])
  })

  it('fence suppresses tags', () => {
    expect(collect('```\n<foo>\n```\n<bar>')).toEqual([
      { type: 'content', text: '```\n<foo>\n```\n' },
      { type: 'open', tagName: 'bar', attrs: new Map(), afterNewline: true },
    ])
  })

  it('cdata passthrough', () => {
    const input = '<' + '![CDATA[<foo>&</foo>]]>'
    expect(collect(input)).toEqual([{ type: 'content', text: '<foo>&</foo>' }])
  })

  it('partial chunks tag', () => {
    expect(collect(['<fo', 'o>'])).toEqual([
      { type: 'open', tagName: 'foo', attrs: new Map(), afterNewline: true },
    ])
  })

  it('end flushes partial tag buffer', () => {
    // New tokenizer completes partial tags at EOF as open signals (for incomplete tool detection)
    expect(collect('<foo')).toEqual([{ type: 'open', tagName: 'foo', attrs: new Map(), afterNewline: true }])
  })

  it('multiple tags in one chunk', () => {
    expect(collect('<a><b/></a>')).toEqual([
      { type: 'open', tagName: 'a', attrs: new Map(), afterNewline: true },
      { type: 'selfClose', tagName: 'b', attrs: new Map(), afterNewline: false },
      { type: 'close', tagName: 'a', afterNewline: false },
    ])
  })

  it('self-close with attrs', () => {
    expect(collect('<foo bar="1"/>')).toEqual([
      {
        type: 'selfClose',
        tagName: 'foo',
        attrs: new Map([['bar', '1']]),
        afterNewline: true,
      },
    ])
  })

  it('tag names allow hyphen', () => {
    expect(collect('<fs-read>')).toEqual([
      { type: 'open', tagName: 'fs-read', attrs: new Map(), afterNewline: true },
    ])
  })

  it('attr whitespace around equals', () => {
    expect(collect('<foo  bar = "baz" >')).toEqual([
      {
        type: 'open',
        tagName: 'foo',
        attrs: new Map([['bar', 'baz']]),
        afterNewline: true,
      },
    ])
  })

  it('boolean + unquoted attrs', () => {
    expect(collect('<foo enabled val=abc>')).toEqual([
      {
        type: 'open',
        tagName: 'foo',
        attrs: new Map([
          ['enabled', ''],
          ['val', 'abc'],
        ]),
        afterNewline: true,
      },
    ])
  })
})
