import { describe, it } from 'vitest'
import { buildValidator, shellValidator, SHELL_TOOL } from './helpers'

// ─── Helpers ──────────────────────────────────────────────────────────────────

const thinkBlock = (close: string) =>
  `\n<|think:alignment>\nsome thought\n${close}\n\n<|yield:user|>`

const messageBlock = (close: string) =>
  `\n<|message:user>\nhello world\n${close}\n\n<|yield:user|>`

const invokeBlock = (paramClose: string, invokeClose: string) =>
  `\n<|invoke:shell>\n<|parameter:command>\nls -la\n${paramClose}\n${invokeClose}\n\n<|yield:user|>`

// ─── Think block close tag variants ──────────────────────────────────────────

describe('lenient close tags — think block', () => {
  it('canonical: <think|>', () => {
    const v = buildValidator([SHELL_TOOL])
    v.passes(thinkBlock('<think|>'))
  })

  it('mode 1 (slash+pipe): </think|>', () => {
    const v = buildValidator([SHELL_TOOL])
    v.passes(thinkBlock('</think|>'))
  })

  it('mode 2 (slash, no pipe): </think>', () => {
    const v = buildValidator([SHELL_TOOL])
    v.passes(thinkBlock('</think>'))
  })

  it('mode 3 (bare): <think>', () => {
    const v = buildValidator([SHELL_TOOL])
    v.passes(thinkBlock('<think>'))
  })
})

// ─── Message block close tag variants ────────────────────────────────────────

describe('lenient close tags — message block', () => {
  it('canonical: <message|>', () => {
    const v = buildValidator([SHELL_TOOL])
    v.passes(messageBlock('<message|>'))
  })

  it('mode 1 (slash+pipe): </message|>', () => {
    const v = buildValidator([SHELL_TOOL])
    v.passes(messageBlock('</message|>'))
  })

  it('mode 2 (slash, no pipe): </message>', () => {
    const v = buildValidator([SHELL_TOOL])
    v.passes(messageBlock('</message>'))
  })

  it('mode 3 (bare): <message>', () => {
    const v = buildValidator([SHELL_TOOL])
    v.passes(messageBlock('<message>'))
  })

  // Invalid close forms are treated as body content — DFA does not match them as close tags.
  // The DFA then finds the real close tag later in the sequence and terminates normally.
  it('invalid close form <message/> treated as body content — passes with real close following', () => {
    const v = buildValidator([SHELL_TOOL])
    // <message/> is body content; the real <message|> close comes after
    v.passes(`\n<|message:user>\nhello world\n<message/>\nmore content\n<message|>\n\n<|yield:user|>`)
  })

  it('invalid close form <MESSAGE|> treated as body content — passes with real close following', () => {
    const v = buildValidator([SHELL_TOOL])
    v.passes(`\n<|message:user>\nhello world\n<MESSAGE|>\nmore content\n<message|>\n\n<|yield:user|>`)
  })

  it('invalid close form <|message> treated as body content — passes with real close following', () => {
    const v = buildValidator([SHELL_TOOL])
    v.passes(`\n<|message:user>\nhello world\n<|message>\nmore content\n<message|>\n\n<|yield:user|>`)
  })
})

// ─── Parameter close tag variants ────────────────────────────────────────────

describe('lenient close tags — parameter body', () => {
  it('canonical: <parameter|>', () => {
    const v = shellValidator()
    v.passes(invokeBlock('<parameter|>', '<invoke|>'))
  })

  it('mode 1 (slash+pipe): </parameter|>', () => {
    const v = shellValidator()
    v.passes(invokeBlock('</parameter|>', '<invoke|>'))
  })

  it('mode 2 (slash, no pipe): </parameter>', () => {
    const v = shellValidator()
    v.passes(invokeBlock('</parameter>', '<invoke|>'))
  })

  it('mode 3 (bare): <parameter>', () => {
    const v = shellValidator()
    v.passes(invokeBlock('<parameter>', '<invoke|>'))
  })

  // Invalid close forms are treated as body content — DFA does not match them as close tags.
  it('invalid close form <parameter/> treated as body content — passes with real close following', () => {
    const v = shellValidator()
    v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<parameter/>\nmore content\n<parameter|>\n<invoke|>\n\n<|yield:user|>`)
  })

  it('invalid close form <PARAMETER|> treated as body content — passes with real close following', () => {
    const v = shellValidator()
    v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<PARAMETER|>\nmore content\n<parameter|>\n<invoke|>\n\n<|yield:user|>`)
  })

  it('invalid close form <|parameter> treated as body content — passes with real close following', () => {
    const v = shellValidator()
    v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<|parameter>\nmore content\n<parameter|>\n<invoke|>\n\n<|yield:user|>`)
  })
})

// ─── Invoke close tag variants ────────────────────────────────────────────────

describe('lenient close tags — invoke close', () => {
  it('canonical: <invoke|>', () => {
    const v = shellValidator()
    v.passes(invokeBlock('<parameter|>', '<invoke|>'))
  })

  it('mode 1 (slash+pipe): </invoke|>', () => {
    const v = shellValidator()
    v.passes(invokeBlock('<parameter|>', '</invoke|>'))
  })

  it('mode 2 (slash, no pipe): </invoke>', () => {
    const v = shellValidator()
    v.passes(invokeBlock('<parameter|>', '</invoke>'))
  })

  it('mode 3 (bare): <invoke>', () => {
    const v = shellValidator()
    v.passes(invokeBlock('<parameter|>', '<invoke>'))
  })

  // Invalid invoke close forms — the invoke-end rule is an explicit literal union, not a DFA body.
  // These forms are NOT in the union and will be rejected by the grammar.
  it('invalid invoke close <invoke/> is rejected', () => {
    const v = shellValidator()
    v.rejects(invokeBlock('<parameter|>', '<invoke/>'))
  })

  it('invalid invoke close <INVOKE|> is rejected', () => {
    const v = shellValidator()
    v.rejects(invokeBlock('<parameter|>', '<INVOKE|>'))
  })

  it('invalid invoke close <|invoke> is rejected', () => {
    const v = shellValidator()
    v.rejects(invokeBlock('<parameter|>', '<|invoke>'))
  })
})

// ─── Cross-combinations ───────────────────────────────────────────────────────

describe('lenient close tags — cross-combinations', () => {
  it('mode 1 param close + mode 1 invoke close', () => {
    const v = shellValidator()
    v.passes(invokeBlock('</parameter|>', '</invoke|>'))
  })

  it('mode 2 param close + mode 2 invoke close', () => {
    const v = shellValidator()
    v.passes(invokeBlock('</parameter>', '</invoke>'))
  })

  it('mode 3 param close + mode 3 invoke close', () => {
    const v = shellValidator()
    v.passes(invokeBlock('<parameter>', '<invoke>'))
  })

  it('mode 1 think close + mode 2 think close in sequence', () => {
    const v = buildValidator([SHELL_TOOL])
    // Two think blocks: first with mode 1, second with mode 2
    v.passes(
      `\n<|think:alignment>\nfirst thought\n</think|>\n\n<|think:skills>\nsecond thought\n</think>\n\n<|yield:user|>`
    )
  })

  it('mixed: mode 2 param close + canonical invoke close', () => {
    const v = shellValidator()
    v.passes(invokeBlock('</parameter>', '<invoke|>'))
  })

  it('mixed: mode 3 param close + mode 1 invoke close', () => {
    const v = shellValidator()
    v.passes(invokeBlock('<parameter>', '</invoke|>'))
  })
})

// ─── Trailing whitespace with lenient close tags ──────────────────────────────

describe('lenient close tags — trailing whitespace', () => {
  it('mode 1 think close with 1 trailing space', () => {
    const v = buildValidator([SHELL_TOOL])
    v.passes(`\n<|think:alignment>\nsome thought\n</think|> \n\n<|yield:user|>`)
  })

  it('mode 2 think close with 2 trailing spaces', () => {
    const v = buildValidator([SHELL_TOOL])
    v.passes(`\n<|think:alignment>\nsome thought\n</think> \n\n<|yield:user|>`)
  })

  it('mode 1 parameter close with 1 trailing space', () => {
    const v = shellValidator()
    v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls\n</parameter|> \n<invoke|>\n\n<|yield:user|>`)
  })

  it('mode 2 parameter close with 2 trailing spaces', () => {
    const v = shellValidator()
    v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls\n</parameter>  \n<invoke|>\n\n<|yield:user|>`)
  })

  it('mode 3 parameter close with 3+ trailing spaces is rejected', () => {
    const v = shellValidator()
    v.rejects(`\n<|invoke:shell>\n<|parameter:command>\nls\n<parameter>   \n<invoke|>\n\n<|yield:user|>`)
  })
})
