import { describe, it } from 'vitest'
import { buildValidator, shellValidator, SHELL_TOOL } from './helpers'

describe('whitespace handling', () => {
  describe('whitespace between tags', () => {
    it('newline between close and next open passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:reason about="turn">\nreasoning\n</magnitude:reason>\n<magnitude:message to="user">\nhello\n</magnitude:message>\n<magnitude:yield_user/>`)
    })

    it('no whitespace between close and next open passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:reason about="turn">\nreasoning\n</magnitude:reason><magnitude:message to="user">\nhello\n</magnitude:message><magnitude:yield_user/>`)
    })

    it('spaces between close and next open passes (up to 4)', () => {
      const v = shellValidator()
      v.passes(`<magnitude:reason about="turn">\nreasoning\n</magnitude:reason>    <magnitude:message to="user">\nhello\n</magnitude:message>\n<magnitude:yield_user/>`)
    })

    it('tab between close and next open passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:reason about="turn">\nreasoning\n</magnitude:reason>\t<magnitude:message to="user">\nhello\n</magnitude:message>\n<magnitude:yield_user/>`)
    })

    it('blank lines between tags pass (ws handles multiple newlines)', () => {
      const v = shellValidator()
      v.passes(`<magnitude:reason about="turn">\nreasoning\n</magnitude:reason>\n\n<magnitude:message to="user">\nhello\n</magnitude:message>\n<magnitude:yield_user/>`)
    })
  })

  describe('trailing whitespace after close tags', () => {
    it('0 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>\n<magnitude:yield_user/>`)
    })

    it('1 trailing space before newline passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message> \n<magnitude:yield_user/>`)
    })

    it('2 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>  \n<magnitude:yield_user/>`)
    })

    it('4 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>    \n<magnitude:yield_user/>`)
    })

    it('5 trailing spaces before newline is accepted (unbounded ws)', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>     \n<magnitude:yield_user/>`)
    })

    it('4 trailing tabs before newline passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>\t\t\t\t\n<magnitude:yield_user/>`)
    })

    it('5 trailing tabs causes close tag to be treated as content (5th tab escapes to s0)', () => {
      // 5th tab at tw4 matches [^ \n<] → back to s0, close tag becomes content
      // A real close tag is needed later
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>\t\t\t\t\t<magnitude:invoke tool="x">\n</magnitude:invoke>\n</magnitude:message>\n<magnitude:yield_user/>`)
    })

    it('4 trailing spaces after </magnitude:reason> passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:reason about="turn">\nreasoning\n</magnitude:reason>    \n<magnitude:yield_user/>`)
    })

    it('5 trailing spaces after </magnitude:reason> is accepted (unbounded ws)', () => {
      const v = shellValidator()
      v.passes(`<magnitude:reason about="turn">\nreasoning\n</magnitude:reason>     \n<magnitude:yield_user/>`)
    })
  })

  describe('trailing whitespace after yield', () => {
    it('yield with no trailing content passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:yield_user/>`)
    })

    it('yield with trailing newline is rejected', () => {
      const v = shellValidator()
      v.rejects(`<magnitude:yield_user/>\n`)
    })

    it('yield with trailing space is rejected', () => {
      const v = shellValidator()
      v.rejects(`<magnitude:yield_user/> `)
    })
  })

  describe('false close tag in body', () => {
    it('close tag followed by non-ws does not close (treated as body)', () => {
      const v = shellValidator()
      // </magnitude:message>` — backtick at tw0 matches [^ \t\n<] → back to s0
      // Real close later
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>\`more\n</magnitude:message>\n<magnitude:yield_user/>`)
    })

    it('close tag in prose confirmed by newline (false positive)', () => {
      // </magnitude:message> in prose followed by \n → confirmed as close tag
      // This is the known false-positive edge case
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nThe tag </magnitude:message>\n<magnitude:yield_user/>`)
    })
  })

  describe('parameter trailing whitespace', () => {
    it('4 trailing spaces after </magnitude:parameter> passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>    \n</magnitude:invoke>\n<magnitude:yield_user/>`)
    })

    // Under greedy last-match, whitespace after close tag is valid content
    // or structural whitespace — both paths are live.
    it('5 trailing spaces after </magnitude:parameter> is accepted (greedy)', () => {
      const v = shellValidator()
      v.passes(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>     \n</magnitude:invoke>\n<magnitude:yield_user/>`)
    })

    it('4 trailing tabs after </magnitude:parameter> passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\t\t\t\t\n</magnitude:invoke>\n<magnitude:yield_user/>`)
    })

    it('5 trailing tabs after </magnitude:parameter> causes it to be treated as content', () => {
      const v = shellValidator()
      v.passes(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\t\t\t\t\t<magnitude:parameter name="x">y</magnitude:parameter>\n</magnitude:invoke>\n<magnitude:yield_user/>`)
    })
  })

  describe('full sequences', () => {
    it('full turn: reason + invoke + yield passes', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:reason about="alignment">\nsome thought\n</magnitude:reason>\n` +
        `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls -la</magnitude:parameter>\n</magnitude:invoke>\n` +
        `<magnitude:yield_user/>`
      )
    })

    it('full turn: message + invoke + yield passes', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:message to="user">\nhello\n</magnitude:message>\n` +
        `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hi</magnitude:parameter>\n</magnitude:invoke>\n` +
        `<magnitude:yield_user/>`
      )
    })

    it('full turn: yield only passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:yield_user/>`)
    })
  })
})
