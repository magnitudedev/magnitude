import { describe, it } from 'vitest'
import { buildValidator, shellValidator, SHELL_TOOL } from './helpers'

describe('yield tags', () => {
  describe('default yield tags', () => {
    it('yield_user passes', () => {
      const v = shellValidator()
      v.passes(`<yield_user/>`)
    })

    it('yield_invoke passes', () => {
      const v = shellValidator()
      v.passes(`<yield_invoke/>`)
    })

    it('yield_worker passes', () => {
      const v = shellValidator()
      v.passes(`<yield_worker/>`)
    })
  })

  describe('whitespace before yield', () => {
    it('yield with leading spaces passes (ws rule)', () => {
      const v = shellValidator()
      v.passes(`  <yield_user/>`)
    })

    it('yield with leading newline passes', () => {
      const v = shellValidator()
      v.passes(`\n<yield_user/>`)
    })

    it('yield with leading tab passes', () => {
      const v = shellValidator()
      v.passes(`\t<yield_user/>`)
    })
  })

  describe('trailing content after yield', () => {
    it('yield with trailing space is rejected', () => {
      const v = shellValidator()
      v.rejects(`<yield_user/> `)
    })

    it('yield with trailing newline is rejected', () => {
      const v = shellValidator()
      v.rejects(`<yield_user/>\n`)
    })

    it('yield with trailing tab is rejected', () => {
      const v = shellValidator()
      v.rejects(`<yield_user/>\t`)
    })
  })

  describe('yield after other blocks', () => {
    it('yield after reason block passes', () => {
      const v = buildValidator([SHELL_TOOL])
      v.passes(`<reason about="turn">\nsome thought\n</reason>\n<yield_user/>`)
    })

    it('yield after message block passes', () => {
      const v = buildValidator([SHELL_TOOL])
      v.passes(`<message to="user">\nhello\n</message>\n<yield_user/>`)
    })

    it('yield after invoke passes', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n<yield_user/>`)
    })

    it('yield after reason + invoke passes', () => {
      const v = shellValidator()
      v.passes(
        `<reason about="turn">\nsome thought\n</reason>\n` +
        `<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n` +
        `<yield_user/>`
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
      v.rejects(`<yield_user/>`)
    })
  })

  describe('forbidden sequences', () => {
    it('content after yield is rejected', () => {
      const v = shellValidator()
      v.rejects(`<yield_user/>\nextra content`)
    })

    it('two yields is rejected', () => {
      const v = shellValidator()
      v.rejects(`<yield_user/><yield_user/>`)
    })
  })
})
