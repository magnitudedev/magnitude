import { describe, it } from 'vitest'
import { buildValidator, shellValidator, SHELL_TOOL } from './helpers'

const YIELD = '<magnitude:yield_user/>'

describe('message blocks', () => {
  describe('passing sequences', () => {
    it('accepts a simple message to user', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello world\n</magnitude:message>\n${YIELD}`)
    })

    it('accepts a message to parent', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="parent">\nhello\n</magnitude:message>\n${YIELD}`)
    })

    it('accepts a message to task-123', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="task-123">\nsome content\n</magnitude:message>\n${YIELD}`)
    })

    it('accepts a multi-line body', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nline one\nline two\nline three\n</magnitude:message>\n${YIELD}`)
    })

    it('accepts a body containing <', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nuse <b>bold</b> text here\n</magnitude:message>\n${YIELD}`)
    })

    it('accepts multiple messages before yield', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:message to="user">\nfirst\n</magnitude:message>\n` +
        `<magnitude:message to="parent">\nsecond\n</magnitude:message>\n` +
        YIELD
      )
    })

    it('accepts message followed by invoke then yield', () => {
      const v = buildValidator([SHELL_TOOL])
      v.passes(
        `<magnitude:message to="user">\nrunning now\n</magnitude:message>\n` +
        `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hi</magnitude:parameter>\n</magnitude:invoke>\n` +
        YIELD
      )
    })

    it('accepts message with no newline before next tag', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message><magnitude:yield_user/>`)
    })

    describe('requiredMessageTo', () => {
      it('accepts message to required recipient', () => {
        const v = buildValidator([SHELL_TOOL], b => b.requireMessageTo('parent'))
        v.passes(`<magnitude:message to="parent">\nhello\n</magnitude:message>\n${YIELD}`)
      })

      it('accepts required message followed by another message', () => {
        const v = buildValidator([SHELL_TOOL], b => b.requireMessageTo('parent'))
        v.passes(
          `<magnitude:message to="parent">\nrequired\n</magnitude:message>\n` +
          `<magnitude:message to="user">\ninfo\n</magnitude:message>\n` +
          YIELD
        )
      })

      it('accepts reason before required message', () => {
        const v = buildValidator([SHELL_TOOL], b => b.requireMessageTo('parent'))
        v.passes(
          `<magnitude:reason about="turn">\nplanning\n</magnitude:reason>\n` +
          `<magnitude:message to="parent">\nhello\n</magnitude:message>\n` +
          YIELD
        )
      })
    })
  })

  describe('forbidden sequences', () => {
    it('rejects sequence missing required message (yield only)', () => {
      const v = buildValidator([SHELL_TOOL], b => b.requireMessageTo('parent'))
      v.rejects(YIELD)
    })

    it('rejects required message to wrong recipient', () => {
      const v = buildValidator([SHELL_TOOL], b => b.requireMessageTo('parent'))
      v.rejects(`<magnitude:message to="user">\nhello\n</magnitude:message>\n${YIELD}`)
    })
  })
})
