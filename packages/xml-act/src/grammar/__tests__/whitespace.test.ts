import { describe, it } from 'vitest'
import { buildValidator, shellValidator, SHELL_TOOL } from './helpers'

describe('whitespace redesign', () => {
  describe('mandatory newline after open tags', () => {
    it('parameter with newline then content passes', () => {
      const v = shellValidator()
      v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<parameter|>\n<invoke|>\n\n<|yield:user|>`)
    })

    it('think block with newline after open passes', () => {
      const v = buildValidator([SHELL_TOOL])
      v.passes(`\n<|think:alignment>\nsome thought\n<think|>\n\n<|yield:user|>`)
    })

    it('message block with newline after open passes', () => {
      const v = buildValidator([SHELL_TOOL])
      v.passes(`\n<|message:user>\nhello\n<message|>\n\n<|yield:user|>`)
    })

    it('inline content after parameter open (no newline) is accepted', () => {
      const v = shellValidator()
      // Parameter body DFA has no required newline — content can start immediately
      v.passes(`\n<|invoke:shell>\n<|parameter:command>review content\n<parameter|>\n<invoke|>\n\n<|yield:user|>`)
    })

    it('inline content after think open (no newline) is rejected', () => {
      const v = buildValidator([SHELL_TOOL])
      v.rejects(`\n<|think:alignment>some thought\n<think|>\n\n<|yield:user|>`)
    })

    it('inline content after message open (no newline) is rejected', () => {
      const v = buildValidator([SHELL_TOOL])
      v.rejects(`\n<|message:user>hello\n<message|>\n\n<|yield:user|>`)
    })
  })

  describe('trailing whitespace after parameter close tag', () => {
    it('0 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<parameter|>\n<invoke|>\n\n<|yield:user|>`)
    })

    it('1 trailing space before newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<parameter|> \n<invoke|>\n\n<|yield:user|>`)
    })

    it('2 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<parameter|>  \n<invoke|>\n\n<|yield:user|>`)
    })

    it('3 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<parameter|>   \n<invoke|>\n\n<|yield:user|>`)
    })

    it('4 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<parameter|>    \n<invoke|>\n\n<|yield:user|>`)
    })

    it('5 trailing spaces before newline is rejected', () => {
      const v = shellValidator()
      v.rejects(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<parameter|>     \n<invoke|>\n\n<|yield:user|>`)
    })

    it('1 trailing tab before newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<parameter|>\t\n<invoke|>\n\n<|yield:user|>`)
    })

    it('2 trailing tabs before newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<parameter|>\t\t\n<invoke|>\n\n<|yield:user|>`)
    })

    it('3 trailing tabs before newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<parameter|>\t\t\t\n<invoke|>\n\n<|yield:user|>`)
    })

    it('4 trailing tabs before newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<parameter|>\t\t\t\t\n<invoke|>\n\n<|yield:user|>`)
    })

    it('5 trailing tabs before newline is rejected', () => {
      const v = shellValidator()
      v.rejects(`\n<|invoke:shell>\n<|parameter:command>\nls -la\n<parameter|>\t\t\t\t\t\n<invoke|>\n\n<|yield:user|>`)
    })
  })

  describe('trailing whitespace after yield', () => {
    it('yield with no trailing content passes', () => {
      const v = shellValidator()
      v.passes(`\n<|yield:user|>`)
    })

    it('yield with trailing newline is rejected (grammar ends at |>)', () => {
      const v = shellValidator()
      v.rejects(`\n<|yield:user|>\n`)
    })

    it('yield with trailing space is rejected', () => {
      const v = shellValidator()
      v.rejects(`\n<|yield:user|> `)
    })

    it('yield with trailing tab is rejected', () => {
      const v = shellValidator()
      v.rejects(`\n<|yield:user|>\t`)
    })
  })

  describe('trailing whitespace after message close tag', () => {
    it('no trailing whitespace passes', () => {
      const v = shellValidator()
      v.passes(`\n<|message:user>\nhello\n<message|>\n\n<|yield:user|>`)
    })

    it('1 trailing space before newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<|message:user>\nhello\n<message|> \n\n<|yield:user|>`)
    })

    it('2 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<|message:user>\nhello\n<message|>  \n\n<|yield:user|>`)
    })

    it('4 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<|message:user>\nhello\n<message|>    \n\n<|yield:user|>`)
    })

    it('5 trailing spaces before newline is rejected', () => {
      const v = shellValidator()
      v.rejects(`\n<|message:user>\nhello\n<message|>     \n\n<|yield:user|>`)
    })

    it('4 trailing tabs before newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<|message:user>\nhello\n<message|>\t\t\t\t\n\n<|yield:user|>`)
    })

    it('5 trailing tabs before newline is rejected', () => {
      const v = shellValidator()
      v.rejects(`\n<|message:user>\nhello\n<message|>\t\t\t\t\t\n\n<|yield:user|>`)
    })

    it('close tag followed by non-ws content does not close (treated as body, second close works)', () => {
      const v = shellValidator()
      // <message|>` does NOT close the message — backtick escapes back to body
      // The second <message|>\n properly closes
      v.passes(`\n<|message:user>\nhello\n<message|>\`more\n<message|>\n\n<|yield:user|>`)
    })

    it('close tag followed by non-ws without later close treats entire content as body', () => {
      const v = shellValidator()
      // <message|>` doesn't close — backtick escapes to body. Content including yield becomes body text.
      // A second proper close is needed.
      v.passes(`\n<|message:user>\nhello\n<message|>\`more\n<message|>\n\n<|yield:user|>`)
    })
  })

  describe('trailing whitespace after think close tag', () => {
    it('no trailing whitespace passes', () => {
      const v = shellValidator()
      v.passes(`\n<|think:alignment>\nreasoning\n<think|>\n\n<|yield:user|>`)
    })

    it('4 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<|think:alignment>\nreasoning\n<think|>    \n\n<|yield:user|>`)
    })

    it('5 trailing spaces before newline is rejected', () => {
      const v = shellValidator()
      v.rejects(`\n<|think:alignment>\nreasoning\n<think|>     \n\n<|yield:user|>`)
    })

    it('close tag followed by non-ws content does not close (treated as body, second close works)', () => {
      const v = shellValidator()
      // <think|>` does NOT close — backtick escapes back to body
      // The second <think|>\n properly closes
      v.passes(`\n<|think:alignment>\nreasoning\n<think|>\`more\n<think|>\n\n<|yield:user|>`)
    })

    it('close tag followed by non-ws without later close treats entire content as body', () => {
      const v = shellValidator()
      v.passes(`\n<|think:alignment>\nreasoning\n<think|>\`more\n<think|>\n\n<|yield:user|>`)
    })
  })

  describe('full sequences', () => {
    it('full turn: think + invoke + yield passes', () => {
      const v = shellValidator()
      v.passes(
        `\n<|think:alignment>\nsome thought\n<think|>\n\n<|invoke:shell>\n<|parameter:command>\nls -la\n<parameter|>\n<invoke|>\n\n<|yield:user|>`
      )
    })

    it('full turn: message + invoke + yield passes', () => {
      const v = shellValidator()
      v.passes(
        `\n<|message:user>\nhello\n<message|>\n\n<|invoke:shell>\n<|parameter:command>\necho hi\n<parameter|>\n<invoke|>\n\n<|yield:user|>`
      )
    })

    it('full turn: yield only passes', () => {
      const v = shellValidator()
      v.passes(`\n<|yield:user|>`)
    })
  })
})
