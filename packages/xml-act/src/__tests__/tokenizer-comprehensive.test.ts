import { describe, expect, it } from 'vitest'
import { createTokenizer } from '../tokenizer'

function collect(input: string | string[], knownToolTags?: ReadonlySet<string>): any[] {
  const out: any[] = []
  const tokenizer = createTokenizer((token) => {
    const { raw, span, ...rest } = token as any
    out.push(rest)
  }, knownToolTags)
  if (Array.isArray(input)) {
    for (const chunk of input) tokenizer.push(chunk)
  } else {
    tokenizer.push(input)
  }
  tokenizer.end()
  return out
}

describe('XML tokenizer tests', () => {
  describe('basic open tags', () => {
    it('parses simple open tag', () => {
      const tokens = collect('<magnitude:message>')
      expect(tokens).toEqual([{ _tag: 'Open', tagName: 'magnitude:message', attrs: new Map(), afterNewline: true }])
    })

    it('parses open tag with attribute', () => {
      const tokens = collect('<magnitude:message to="user">')
      expect(tokens).toEqual([{ _tag: 'Open', tagName: 'magnitude:message', attrs: new Map([['to', 'user']]), afterNewline: true }])
    })

    it('parses open tag with multiple attributes', () => {
      const tokens = collect('<magnitude:invoke tool="shell" extra="val">')
      expect(tokens).toEqual([{ _tag: 'Open', tagName: 'magnitude:invoke', attrs: new Map([['tool', 'shell'], ['extra', 'val']]), afterNewline: true }])
    })

    it('parses open tag with single-quoted attribute', () => {
      const tokens = collect("<magnitude:message to='user'>")
      expect(tokens).toEqual([{ _tag: 'Open', tagName: 'magnitude:message', attrs: new Map([['to', 'user']]), afterNewline: true }])
    })

    it('parses open tag with boolean attribute', () => {
      const tokens = collect('<tag disabled>')
      expect(tokens).toEqual([{ _tag: 'Open', tagName: 'tag', attrs: new Map([['disabled', '']]), afterNewline: true }])
    })
  })

  describe('close tags with confirmation', () => {
    it('confirms close tag on newline', () => {
      const tokens = collect('</magnitude:message>\n')
      expect(tokens).toEqual([
        { _tag: 'Close', tagName: 'magnitude:message', afterNewline: true },
        { _tag: 'Content', text: '\n' },
      ])
    })

    it('confirms close tag on following open tag', () => {
      const tokens = collect('</magnitude:message><magnitude:invoke tool="shell">')
      expect(tokens).toEqual([
        { _tag: 'Close', tagName: 'magnitude:message', afterNewline: true },
        { _tag: 'Open', tagName: 'magnitude:invoke', attrs: new Map([['tool', 'shell']]), afterNewline: false },
      ])
    })

    it('confirms close tag with spaces then newline (at EOF)', () => {
      const tokens = collect('</magnitude:message>   \n')
      expect(tokens).toEqual([
        { _tag: 'Close', tagName: 'magnitude:message', afterNewline: true },
        { _tag: 'Content', text: '   \n' },
      ])
    })

    it('confirms close tag with spaces then next tag', () => {
      const tokens = collect('</magnitude:message>  <magnitude:invoke tool="shell">')
      expect(tokens).toEqual([
        { _tag: 'Close', tagName: 'magnitude:message', afterNewline: true },
        { _tag: 'Content', text: '  ' },
        { _tag: 'Open', tagName: 'magnitude:invoke', attrs: new Map([['tool', 'shell']]), afterNewline: false },
      ])
    })

    it('confirms close tag with tab then newline (at EOF)', () => {
      const tokens = collect('</magnitude:message>\t\n')
      expect(tokens).toEqual([
        { _tag: 'Close', tagName: 'magnitude:message', afterNewline: true },
        { _tag: 'Content', text: '\t\n' },
      ])
    })

    it('close tag followed by prose — Close emitted immediately', () => {
      const tokens = collect('</magnitude:message> to end your block.')
      expect(tokens).toEqual([
        { _tag: 'Close', tagName: 'magnitude:message', afterNewline: true },
        { _tag: 'Content', text: ' to end your block.' },
      ])
    })

    it('close tag followed by non-confirming char — Close emitted immediately', () => {
      const tokens = collect('</magnitude:message>.')
      expect(tokens).toEqual([
        { _tag: 'Close', tagName: 'magnitude:message', afterNewline: true },
        { _tag: 'Content', text: '.' },
      ])
    })

    it('close tag with 5 spaces confirmed at EOF (unbounded ws)', () => {
      const tokens = collect('</magnitude:message>     \n')
      // unbounded ws: 5 spaces + \n buffered, EOF confirms
      expect(tokens).toEqual([
        { _tag: 'Close', tagName: 'magnitude:message', afterNewline: true },
        { _tag: 'Content', text: '     \n' },
      ])
    })

    it('confirms close tag at end of stream (EOF confirms)', () => {
      const tokens = collect('</magnitude:message>')
      expect(tokens).toEqual([
        { _tag: 'Close', tagName: 'magnitude:message', afterNewline: true },
      ])
    })

    it('handles close tag confirmation across chunk boundary', () => {
      const tokens = collect(['</magnitude:message>', '\n'])
      expect(tokens).toEqual([
        { _tag: 'Close', tagName: 'magnitude:message', afterNewline: true },
        { _tag: 'Content', text: '\n' },
      ])
    })

    it('handles close tag with spaces split across chunks', () => {
      const tokens = collect(['</magnitude:message> ', ' <magnitude:invoke tool="x">'])
      expect(tokens).toEqual([
        { _tag: 'Close', tagName: 'magnitude:message', afterNewline: true },
        { _tag: 'Content', text: ' ' },
        { _tag: 'Content', text: ' ' },
        { _tag: 'Open', tagName: 'magnitude:invoke', attrs: new Map([['tool', 'x']]), afterNewline: false },
      ])
    })
  })

  describe('self-closing tags', () => {
    it('parses self-close tag', () => {
      const tokens = collect('<magnitude:yield_user/>')
      expect(tokens).toEqual([{ _tag: 'SelfClose', tagName: 'magnitude:yield_user', attrs: new Map(), afterNewline: true }])
    })

    it('parses self-close with attribute', () => {
      const tokens = collect('<tag attr="val"/>')
      expect(tokens).toEqual([{ _tag: 'SelfClose', tagName: 'tag', attrs: new Map([['attr', 'val']]), afterNewline: true }])
    })
  })

  describe('content', () => {
    it('emits plain text as content', () => {
      const tokens = collect('hello world')
      expect(tokens).toEqual([{ _tag: 'Content', text: 'hello world' }])
    })

    it('emits angle bracket as content when not a tag', () => {
      const tokens = collect('a < b')
      expect(tokens).toEqual([{ _tag: 'Content', text: 'a < b' }])
    })

    it('handles mixed content and tags', () => {
      // Close emitted immediately — parser handles confirmation
      const tokens = collect('before\n<magnitude:message to="user">\nhello\n</magnitude:message>\nafter')
      expect(tokens).toEqual([
        { _tag: 'Content', text: 'before\n' },
        { _tag: 'Open', tagName: 'magnitude:message', attrs: new Map([['to', 'user']]), afterNewline: true },
        { _tag: 'Content', text: '\nhello\n' },
        { _tag: 'Close', tagName: 'magnitude:message', afterNewline: true },
        { _tag: 'Content', text: '\nafter' },
      ])
    })
  })

  describe('parameter tags', () => {
    it('parses parameter open tag', () => {
      const tokens = collect('<magnitude:parameter name="command">')
      expect(tokens).toEqual([{ _tag: 'Open', tagName: 'magnitude:parameter', attrs: new Map([['name', 'command']]), afterNewline: true }])
    })

    it('parses parameter close tag (confirmed by newline)', () => {
      const tokens = collect('</magnitude:parameter>\n')
      expect(tokens).toEqual([
        { _tag: 'Close', tagName: 'magnitude:parameter', afterNewline: true },
        { _tag: 'Content', text: '\n' },
      ])
    })
  })

  describe('CDATA', () => {
    it('parses CDATA section as content', () => {
      const tokens = collect('<![CDATA[hello world]]]>')
      expect(tokens).toEqual([{ _tag: 'Content', text: 'hello world]' }])
    })

    it('handles CDATA with angle brackets inside', () => {
      const tokens = collect('<![CDATA[<not-a-tag>]]]>')
      expect(tokens).toEqual([{ _tag: 'Content', text: '<not-a-tag>]' }])
    })

    it('handles CDATA split across chunks', () => {
      const tokens = collect(['<![CDATA[hel', 'lo]]]>'])
      expect(tokens).toEqual([{ _tag: 'Content', text: 'hello]' }])
    })

    it('handles CDATA close split across chunks', () => {
      const tokens = collect(['<![CDATA[hello]', ']]', '>'])
      expect(tokens).toEqual([{ _tag: 'Content', text: 'hello]' }])
    })

    it('treats unclosed CDATA as raw content at end', () => {
      const tokens = collect('<![CDATA[unclosed')
      expect(tokens).toEqual([{ _tag: 'Content', text: '<![CDATA[unclosed' }])
    })
  })

  describe('chunk boundary handling', () => {
    it('handles < at chunk boundary', () => {
      const tokens = collect(['<magnitude:', 'message to="user">\n</magnitude:message>\n'])
      expect(tokens).toEqual([
        { _tag: 'Open', tagName: 'magnitude:message', attrs: new Map([['to', 'user']]), afterNewline: true },
        { _tag: 'Content', text: '\n' },
        { _tag: 'Close', tagName: 'magnitude:message', afterNewline: true },
        { _tag: 'Content', text: '\n' },
      ])
    })

    it('handles tag split across many chunks', () => {
      const tokens = collect(['<magnitude:', 'invoke', ' to', 'ol', '="s', 'hell', '">\n'])
      expect(tokens).toEqual([
        { _tag: 'Open', tagName: 'magnitude:invoke', attrs: new Map([['tool', 'shell']]), afterNewline: true },
        { _tag: 'Content', text: '\n' },
      ])
    })

    it('handles < followed by non-tag char at chunk boundary', () => {
      const tokens = collect(['<', '3 is less'])
      expect(tokens).toEqual([{ _tag: 'Content', text: '<3 is less' }])
    })
  })

  describe('malformed invoke tags', () => {
    it('emits Open for malformed invoke so parser can surface error', () => {
      const tokens = collect('<magnitude:invoke @@@>', new Set(['magnitude:invoke']))
      expect(tokens).toEqual([
        { _tag: 'Open', tagName: 'magnitude:invoke', attrs: new Map(), afterNewline: true },
      ])
    })
  })

  describe('tag names with underscores', () => {
    it('parses yield_user self-close', () => {
      const tokens = collect('<magnitude:yield_user/>')
      expect(tokens).toEqual([{ _tag: 'SelfClose', tagName: 'magnitude:yield_user', attrs: new Map(), afterNewline: true }])
    })

    it('parses yield_parent self-close', () => {
      const tokens = collect('<magnitude:yield_parent/>')
      expect(tokens).toEqual([{ _tag: 'SelfClose', tagName: 'magnitude:yield_parent', attrs: new Map(), afterNewline: true }])
    })
  })

  describe('afterNewline tracking', () => {
    it('sets afterNewline=true at start', () => {
      const tokens = collect('<magnitude:message to="x">')
      expect(tokens[0]).toMatchObject({ _tag: 'Open', afterNewline: true })
    })

    it('sets afterNewline=false when no preceding newline', () => {
      const tokens = collect('text<magnitude:message to="x">')
      const open = tokens.find(t => t._tag === 'Open')
      expect(open?.afterNewline).toBe(false)
    })

    it('sets afterNewline=true after newline in content', () => {
      const tokens = collect('text\n<magnitude:message to="x">')
      const open = tokens.find(t => t._tag === 'Open')
      expect(open?.afterNewline).toBe(true)
    })
  })
})
