import { describe, it } from 'vitest'
import { shellValidator } from './helpers'

// Helpers to build full valid sequences
const YIELD = '\n<|yield:user|>'

/** Wrap content in a think block + yield */
function withThink(body: string): string {
  return `\n<|think:alignment>\n${body}<think|>\n${YIELD}`
}

/**
 * Wrap content in a shell invoke with command param + yield.
 *
 * After param-body DFA exits (consuming close tag + trailing \n),
 * the invoke-close rule is `hws invoke-end hws "\n"` — no leading \n.
 */
function withShellCommand(body: string): string {
  return `\n<|invoke:shell>\n<|parameter:command>\n${body}<parameter|>\n<invoke|>\n${YIELD}`
}

describe('body DFA — content with < that is not a close tag', () => {
  it('accepts < followed by a space (unrelated char)', () => {
    const v = shellValidator()
    v.passes(withThink('foo < bar\n'))
  })

  it('accepts partial tag name that does not complete', () => {
    // <para is a prefix of <parameter but doesn't complete it — goes back to s0
    const v = shellValidator()
    v.passes(withThink('see <para something here\n'))
  })

  it('accepts HTML-like content in think body', () => {
    // </div> — "d" is not "t" so DFA returns to s0 after "/"
    const v = shellValidator()
    v.passes(withThink('<div>hello</div>\n'))
  })

  it('accepts </foo> (wrong tag name) in parameter body', () => {
    // </foo> — "f" is not "p" so DFA returns to s0 after "/"
    const v = shellValidator()
    v.passes(withShellCommand('echo </foo> hello\n'))
  })

  it('accepts code with < comparison operators', () => {
    const v = shellValidator()
    v.passes(withThink('if a < b then do something\n'))
  })

  it('accepts the word "parameter" without angle brackets', () => {
    const v = shellValidator()
    v.passes(withShellCommand('# parameter value here\n'))
  })

  it('accepts </shell> (wrong tag name) in parameter body', () => {
    // </shell> — "s" is not "p" so DFA returns to s0 after "/"
    const v = shellValidator()
    v.passes(withShellCommand('echo </shell> done\n'))
  })

  it('accepts <thinx (wrong char after "thin") in think body', () => {
    // <thinx — "x" is not "k" so DFA returns to s0
    const v = shellValidator()
    v.passes(withThink('<thinx is not a close tag\n'))
  })

  it('accepts << in think body (s1 loops back to s1 on <)', () => {
    const v = shellValidator()
    v.passes(withThink('<<< >>>\n'))
  })

  it('accepts << in parameter body (s1 loops back to s1 on <)', () => {
    const v = shellValidator()
    v.passes(withShellCommand('a << b\n'))
  })
})

describe('body DFA — close tag variants (think body)', () => {
  it('terminates think body with canonical <think|>', () => {
    const v = shellValidator()
    v.passes(`\n<|think:alignment>\nhello\n<think|>\n${YIELD}`)
  })

  it('terminates think body with </think|> (Mode 1 — slash)', () => {
    const v = shellValidator()
    v.passes(`\n<|think:alignment>\nhello\n</think|>\n${YIELD}`)
  })

  it('terminates think body with </think> (Mode 2 — slash no pipe)', () => {
    const v = shellValidator()
    v.passes(`\n<|think:alignment>\nhello\n</think>\n${YIELD}`)
  })

  it('terminates think body with <think> (Mode 3 — no pipe)', () => {
    const v = shellValidator()
    v.passes(`\n<|think:alignment>\nhello\n<think>\n${YIELD}`)
  })
})

describe('body DFA — close tag variants (parameter body)', () => {
  it('terminates parameter body with canonical <parameter|>', () => {
    const v = shellValidator()
    v.passes(`\n<|invoke:shell>\n<|parameter:command>\necho hi\n<parameter|>\n<invoke|>\n${YIELD}`)
  })

  it('terminates parameter body with </parameter|> (Mode 1 — slash)', () => {
    const v = shellValidator()
    v.passes(`\n<|invoke:shell>\n<|parameter:command>\necho hi\n</parameter|>\n<invoke|>\n${YIELD}`)
  })

  it('terminates parameter body with </parameter> (Mode 2 — slash no pipe)', () => {
    const v = shellValidator()
    v.passes(`\n<|invoke:shell>\n<|parameter:command>\necho hi\n</parameter>\n<invoke|>\n${YIELD}`)
  })

  it('terminates parameter body with <parameter> (Mode 3 — no pipe)', () => {
    const v = shellValidator()
    v.passes(`\n<|invoke:shell>\n<|parameter:command>\necho hi\n<parameter>\n<invoke|>\n${YIELD}`)
  })
})

describe('body DFA — empty body (close tag immediately after open)', () => {
  it('accepts empty think body (just close tag)', () => {
    const v = shellValidator()
    v.passes(`\n<|think:alignment>\n<think|>\n${YIELD}`)
  })

  it('accepts empty parameter body (just close tag)', () => {
    const v = shellValidator()
    v.passes(`\n<|invoke:shell>\n<|parameter:command>\n<parameter|>\n<invoke|>\n${YIELD}`)
  })
})

describe('body DFA — trailing whitespace after close tag (tw0/tw1/tw2)', () => {
  it('accepts close tag followed directly by newline', () => {
    const v = shellValidator()
    v.passes(`\n<|think:alignment>\nhello\n<think|>\n${YIELD}`)
  })

  it('accepts close tag with one trailing space before newline', () => {
    const v = shellValidator()
    v.passes(`\n<|think:alignment>\nhello\n<think|> \n${YIELD}`)
  })

  it('accepts close tag with one trailing tab before newline', () => {
    const v = shellValidator()
    v.passes(`\n<|think:alignment>\nhello\n<think|>\t\n${YIELD}`)
  })

  it('accepts close tag with two trailing spaces before newline', () => {
    const v = shellValidator()
    v.passes(`\n<|think:alignment>\nhello\n<think|>  \n${YIELD}`)
  })

  it('accepts close tag with three trailing spaces before newline', () => {
    const v = shellValidator()
    v.passes(`\n<|think:alignment>\nhello\n<think|>   \n${YIELD}`)
  })

  it('accepts close tag with four trailing spaces before newline', () => {
    const v = shellValidator()
    v.passes(`\n<|think:alignment>\nhello\n<think|>    \n${YIELD}`)
  })

  it('rejects close tag with five trailing spaces before newline', () => {
    const v = shellValidator()
    v.rejects(`\n<|think:alignment>\nhello\n<think|>     \n${YIELD}`)
  })
})

describe('body DFA — open tag requires newline before body', () => {
  it('accepts parameter value on same line as open tag (inline)', () => {
    // Parameters are inline — content immediately after open tag
    const v = shellValidator()
    v.passes(`\n<|invoke:shell>\n<|parameter:command>echo hi<parameter|>\n<invoke|>\n${YIELD}`)
  })

  it('accepts parameter value starting on next line', () => {
    const v = shellValidator()
    v.passes(`\n<|invoke:shell>\n<|parameter:command>\necho hi\n<parameter|>\n<invoke|>\n${YIELD}`)
  })
})

describe('body DFA — multiline content', () => {
  it('accepts multiline think content', () => {
    const v = shellValidator()
    v.passes(withThink('line one\nline two\nline three\n'))
  })

  it('accepts multiline parameter content', () => {
    const v = shellValidator()
    v.passes(withShellCommand('line one\nline two\nline three\n'))
  })

  it('accepts content with blank lines in think body', () => {
    const v = shellValidator()
    v.passes(withThink('first paragraph\n\nsecond paragraph\n'))
  })

  it('accepts content with blank lines in parameter body', () => {
    const v = shellValidator()
    v.passes(withShellCommand('first line\n\nthird line\n'))
  })
})
