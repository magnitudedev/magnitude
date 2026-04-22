import { describe, it } from 'vitest'
import { buildValidator, shellValidator, SHELL_TOOL } from './helpers'

describe('whitespace handling', () => {
  describe('whitespace between tags', () => {
    it('newline between close and next open passes', () => {
      const v = shellValidator()
      v.passes(`<reason about="turn">\nreasoning\n</reason>\n<message to="user">\nhello\n</message>\n<yield_user/>`)
    })

    it('no whitespace between close and next open passes', () => {
      const v = shellValidator()
      v.passes(`<reason about="turn">\nreasoning\n</reason><message to="user">\nhello\n</message><yield_user/>`)
    })

    it('spaces between close and next open passes (up to 4)', () => {
      const v = shellValidator()
      v.passes(`<reason about="turn">\nreasoning\n</reason>    <message to="user">\nhello\n</message>\n<yield_user/>`)
    })

    it('tab between close and next open passes', () => {
      const v = shellValidator()
      v.passes(`<reason about="turn">\nreasoning\n</reason>\t<message to="user">\nhello\n</message>\n<yield_user/>`)
    })

    it('blank lines between tags pass (ws handles multiple newlines)', () => {
      const v = shellValidator()
      v.passes(`<reason about="turn">\nreasoning\n</reason>\n\n<message to="user">\nhello\n</message>\n<yield_user/>`)
    })
  })

  describe('trailing whitespace after close tags', () => {
    it('0 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n</message>\n<yield_user/>`)
    })

    it('1 trailing space before newline passes', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n</message> \n<yield_user/>`)
    })

    it('2 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n</message>  \n<yield_user/>`)
    })

    it('4 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n</message>    \n<yield_user/>`)
    })

    it('5 trailing spaces before newline is rejected (exceeds tw window)', () => {
      const v = shellValidator()
      v.rejects(`<message to="user">\nhello\n</message>     \n<yield_user/>`)
    })

    it('4 trailing tabs before newline passes', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n</message>\t\t\t\t\n<yield_user/>`)
    })

    it('5 trailing tabs causes close tag to be treated as content (5th tab escapes to s0)', () => {
      // 5th tab at tw4 matches [^ \n<] → back to s0, close tag becomes content
      // A real close tag is needed later
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n</message>\t\t\t\t\t<invoke tool="x">\n</invoke>\n</message>\n<yield_user/>`)
    })

    it('4 trailing spaces after </reason> passes', () => {
      const v = shellValidator()
      v.passes(`<reason about="turn">\nreasoning\n</reason>    \n<yield_user/>`)
    })

    it('5 trailing spaces after </reason> is rejected', () => {
      const v = shellValidator()
      v.rejects(`<reason about="turn">\nreasoning\n</reason>     \n<yield_user/>`)
    })
  })

  describe('trailing whitespace after yield', () => {
    it('yield with no trailing content passes', () => {
      const v = shellValidator()
      v.passes(`<yield_user/>`)
    })

    it('yield with trailing newline is rejected', () => {
      const v = shellValidator()
      v.rejects(`<yield_user/>\n`)
    })

    it('yield with trailing space is rejected', () => {
      const v = shellValidator()
      v.rejects(`<yield_user/> `)
    })
  })

  describe('false close tag in body', () => {
    it('close tag followed by non-ws does not close (treated as body)', () => {
      const v = shellValidator()
      // </message>` — backtick at tw0 matches [^ \t\n<] → back to s0
      // Real close later
      v.passes(`<message to="user">\nhello\n</message>\`more\n</message>\n<yield_user/>`)
    })

    it('close tag in prose confirmed by newline (false positive)', () => {
      // </message> in prose followed by \n → confirmed as close tag
      // This is the known false-positive edge case
      const v = shellValidator()
      v.passes(`<message to="user">\nThe tag </message>\n<yield_user/>`)
    })
  })

  describe('parameter trailing whitespace', () => {
    it('4 trailing spaces after </parameter> passes', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n<parameter name="command">ls</parameter>    \n</invoke>\n<yield_user/>`)
    })

    it('5 trailing spaces after </parameter> is rejected', () => {
      const v = shellValidator()
      v.rejects(`<invoke tool="shell">\n<parameter name="command">ls</parameter>     \n</invoke>\n<yield_user/>`)
    })

    it('4 trailing tabs after </parameter> passes', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n<parameter name="command">ls</parameter>\t\t\t\t\n</invoke>\n<yield_user/>`)
    })

    it('5 trailing tabs after </parameter> causes it to be treated as content', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n<parameter name="command">ls</parameter>\t\t\t\t\t<parameter name="x">y</parameter>\n</invoke>\n<yield_user/>`)
    })
  })

  describe('full sequences', () => {
    it('full turn: reason + invoke + yield passes', () => {
      const v = shellValidator()
      v.passes(
        `<reason about="alignment">\nsome thought\n</reason>\n` +
        `<invoke tool="shell">\n<parameter name="command">ls -la</parameter>\n</invoke>\n` +
        `<yield_user/>`
      )
    })

    it('full turn: message + invoke + yield passes', () => {
      const v = shellValidator()
      v.passes(
        `<message to="user">\nhello\n</message>\n` +
        `<invoke tool="shell">\n<parameter name="command">echo hi</parameter>\n</invoke>\n` +
        `<yield_user/>`
      )
    })

    it('full turn: yield only passes', () => {
      const v = shellValidator()
      v.passes(`<yield_user/>`)
    })
  })
})
