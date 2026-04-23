import { describe, it } from 'vitest'
import { shellValidator, buildValidator, SHELL_TOOL } from './helpers'

// Helpers to build full valid sequences
const YIELD = '<magnitude:yield_user/>'

/** Wrap content in a reason block + yield */
function withReason(body: string): string {
  return `<magnitude:reason about="alignment">\n${body}</magnitude:reason>\n${YIELD}`
}

/** Wrap content in a message block + yield */
function withMessage(body: string): string {
  return `<magnitude:message to="user">\n${body}</magnitude:message>\n${YIELD}`
}

/** Wrap content in a shell invoke with command param + yield */
function withShellCommand(body: string): string {
  return `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">\n${body}</magnitude:parameter>\n</magnitude:invoke>\n${YIELD}`
}

describe('body DFA — content with < that is not a close tag', () => {
  it('accepts < followed by a space (unrelated char)', () => {
    const v = shellValidator()
    v.passes(withReason('foo < bar\n'))
  })

  it('accepts partial tag name that does not complete', () => {
    const v = shellValidator()
    v.passes(withReason('see <rea something here\n'))
  })

  it('accepts HTML-like content in reason body', () => {
    const v = shellValidator()
    v.passes(withReason('<div>hello</div>\n'))
  })

  it('accepts </foo> (wrong tag name) in parameter body', () => {
    const v = shellValidator()
    v.passes(withShellCommand('echo </foo> hello\n'))
  })

  it('accepts code with < comparison operators', () => {
    const v = shellValidator()
    v.passes(withReason('if a < b then do something\n'))
  })

  it('accepts </rea (partial close tag name) in reason body', () => {
    const v = shellValidator()
    v.passes(withReason('</rea partial\n'))
  })

  it('accepts < at end of line (followed by newline)', () => {
    const v = shellValidator()
    v.passes(withReason('line ending with <\n'))
  })
})

describe('body DFA — false close tag rejection', () => {
  it('close tag followed by non-ws char is treated as body content', () => {
    // </magnitude:reason>` — backtick at tw0 matches [^ \t\n<] → back to s0
    const v = shellValidator()
    v.passes(withReason('content\n</magnitude:reason>`more content\n'))
  })

  it('close tag followed by letter is treated as body content', () => {
    const v = shellValidator()
    v.passes(withMessage('hello\n</magnitude:message>foo\n'))
  })

  it('close tag followed by space then letter is treated as body content', () => {
    const v = shellValidator()
    v.passes(withMessage('hello\n</magnitude:message> to end your message\n'))
  })

  it('false close tag in prose: real close later', () => {
    const v = shellValidator()
    v.passes(`<magnitude:message to="user">\nhello</magnitude:message> to end your message</magnitude:message>\n${YIELD}`)
  })
})

describe('body DFA — multiline content', () => {
  it('accepts content with blank lines in reason body', () => {
    const v = shellValidator()
    v.passes(withReason('line one\n\nline two\n\nline three\n'))
  })

  it('accepts content with blank lines in message body', () => {
    const v = shellValidator()
    v.passes(withMessage('paragraph one\n\nparagraph two\n'))
  })

  it('accepts content with blank lines in parameter body', () => {
    const v = shellValidator()
    v.passes(withShellCommand('line one\n\nline two\n'))
  })
})

describe('body DFA — trailing whitespace window', () => {
  it('0 trailing spaces after close tag: confirmed by next tag', () => {
    const v = shellValidator()
    v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message><magnitude:yield_user/>`)
  })

  it('1 trailing space after close tag: confirmed by newline', () => {
    const v = shellValidator()
    v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message> \n${YIELD}`)
  })

  it('4 trailing spaces after close tag: confirmed by <', () => {
    const v = shellValidator()
    v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>    <magnitude:yield_user/>`)
  })

  it('5 trailing spaces: accepted (unbounded ws)', () => {
    const v = shellValidator()
    v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>     \n${YIELD}`)
  })

  it('4 trailing tabs: confirmed by newline', () => {
    const v = shellValidator()
    v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>\t\t\t\t\n${YIELD}`)
  })

  it('5 trailing tabs: close tag treated as content (tab escapes to s0 at tw4)', () => {
    // 5th tab at tw4 matches [^ \n<] → back to s0, close tag becomes content
    // Need real close after
    const v = shellValidator()
    v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>\t\t\t\t\tmore content\n</magnitude:message>\n${YIELD}`)
  })
})
