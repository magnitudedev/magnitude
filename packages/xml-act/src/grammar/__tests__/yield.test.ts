import { describe, it } from 'vitest'
import { buildValidator, shellValidator, SHELL_TOOL } from './helpers'

describe('yield tags', () => {
  describe('default yield tags', () => {
    it('yield_user passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:yield_user/>`)
    })

    it('yield_invoke passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:yield_invoke/>`)
    })

    it('yield_worker passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:yield_worker/>`)
    })
  })

  describe('whitespace before yield', () => {
    it('yield with leading spaces passes (ws rule)', () => {
      const v = shellValidator()
      v.passes(`  <magnitude:yield_user/>`)
    })

    it('yield with leading newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<magnitude:yield_user/>`)
    })

    it('yield with leading tab passes', () => {
      const v = shellValidator()
      v.passes(`\t<magnitude:yield_user/>`)
    })
  })

  describe('trailing content after yield', () => {
    it('yield with trailing space is rejected', () => {
      const v = shellValidator()
      v.rejects(`<magnitude:yield_user/> `)
    })

    it('yield with trailing newline is rejected', () => {
      const v = shellValidator()
      v.rejects(`<magnitude:yield_user/>\n`)
    })

    it('yield with trailing tab is rejected', () => {
      const v = shellValidator()
      v.rejects(`<magnitude:yield_user/>\t`)
    })
  })

  describe('yield after other blocks', () => {
    it('yield after reason block passes', () => {
      const v = buildValidator([SHELL_TOOL])
      v.passes(`<magnitude:reason about="turn">\nsome thought\n</magnitude:reason>\n<magnitude:yield_user/>`)
    })

    it('yield after message block passes', () => {
      const v = buildValidator([SHELL_TOOL])
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>\n<magnitude:yield_user/>`)
    })

    it('yield after invoke passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n<magnitude:yield_user/>`)
    })

    it('yield after reason + invoke passes', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:reason about="turn">\nsome thought\n</magnitude:reason>\n` +
        `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n` +
        `<magnitude:yield_user/>`
      )
    })
  })

  describe('custom yield tags', () => {
    it('custom yield tag passes', () => {
      const v = buildValidator([SHELL_TOOL], b => b.withYieldTags(['yield_idle', 'yield_continue']))
      v.passes(`<yield_idle/>`)
    })

    it('second custom yield tag passes', () => {
      const v = buildValidator([SHELL_TOOL], b => b.withYieldTags(['yield_idle', 'yield_continue']))
      v.passes(`<yield_continue/>`)
    })

    it('default yield tag rejected when custom tags set', () => {
      const v = buildValidator([SHELL_TOOL], b => b.withYieldTags(['yield_idle', 'yield_continue']))
      v.rejects(`<magnitude:yield_user/>`)
    })
  })

  describe('forbidden sequences', () => {
    it('content after yield is rejected', () => {
      const v = shellValidator()
      v.rejects(`<magnitude:yield_user/>\nextra content`)
    })

    it('two yields is rejected', () => {
      const v = shellValidator()
      v.rejects(`<magnitude:yield_user/><magnitude:yield_user/>`)
    })
  })
})
