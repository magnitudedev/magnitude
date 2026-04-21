import { describe, it } from 'vitest'
import { buildValidator, shellValidator, SHELL_TOOL } from './helpers'

describe('yield tags', () => {
  describe('default yield tags', () => {
    it('yield:user passes', () => {
      const v = shellValidator()
      v.passes(`\n<|yield:user|>`)
    })

    it('yield:invoke passes', () => {
      const v = shellValidator()
      v.passes(`\n<|yield:invoke|>`)
    })

    it('yield:worker passes', () => {
      const v = shellValidator()
      v.passes(`\n<|yield:worker|>`)
    })
  })

  describe('indentation before yield', () => {
    it('yield with leading spaces passes', () => {
      const v = shellValidator()
      v.passes(`\n  <|yield:user|>`)
    })

    it('yield with leading tab passes', () => {
      const v = shellValidator()
      v.passes(`\n\t<|yield:user|>`)
    })
  })

  describe('trailing whitespace after yield', () => {
    it('yield with 1 trailing space is rejected (no trailing content)', () => {
      const v = shellValidator()
      v.rejects(`\n<|yield:user|> `)
    })

    it('yield with trailing newline is rejected (no trailing content)', () => {
      const v = shellValidator()
      v.rejects(`\n<|yield:user|>\n`)
    })

    it('yield with trailing tab is rejected (no trailing content)', () => {
      const v = shellValidator()
      v.rejects(`\n<|yield:user|>\t`)
    })
  })

  describe('yield after other blocks', () => {
    it('yield after think block passes', () => {
      const v = buildValidator([SHELL_TOOL])
      v.passes(`\n<|think:turn>\nsome thought\n<think|>\n\n<|yield:user|>`)
    })

    it('yield after message block passes', () => {
      const v = buildValidator([SHELL_TOOL])
      v.passes(`\n<|message:user>\nhello\n<message|>\n\n<|yield:user|>`)
    })

    it('yield after invoke passes', () => {
      const v = shellValidator()
      v.passes(`\n<|invoke:shell>\n<|parameter:command>\nls\n<parameter|>\n<invoke|>\n\n<|yield:user|>`)
    })

    it('yield after think + invoke passes', () => {
      const v = shellValidator()
      v.passes(
        `\n<|think:turn>\nsome thought\n<think|>\n\n<|invoke:shell>\n<|parameter:command>\nls\n<parameter|>\n<invoke|>\n\n<|yield:user|>`
      )
    })
  })

  describe('custom yield tags', () => {
    it('custom yield tag idle passes', () => {
      const v = buildValidator([SHELL_TOOL], b => b.withYieldTags(['idle', 'continue']))
      v.passes(`\n<|yield:idle|>`)
    })

    it('custom yield tag continue passes', () => {
      const v = buildValidator([SHELL_TOOL], b => b.withYieldTags(['idle', 'continue']))
      v.passes(`\n<|yield:continue|>`)
    })

    it('default yield tag rejected when custom tags set', () => {
      const v = buildValidator([SHELL_TOOL], b => b.withYieldTags(['idle', 'continue']))
      v.rejects(`\n<|yield:user|>`)
    })
  })

  describe('forbidden sequences', () => {
    it('unknown yield tag is rejected', () => {
      const v = shellValidator()
      v.rejects(`\n<|yield:unknown|>`)
    })

    it('content after yield is rejected', () => {
      const v = shellValidator()
      v.rejects(`\n<|yield:user|>\nextra content`)
    })
  })
})
