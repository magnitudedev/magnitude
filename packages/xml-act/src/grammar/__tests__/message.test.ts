import { describe, it } from 'vitest'
import { buildValidator, shellValidator, SHELL_TOOL } from './helpers'

const YIELD = '<yield_user/>'

describe('message blocks', () => {
  describe('passing sequences', () => {
    it('accepts a simple message to user', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nhello world\n</message>\n${YIELD}`)
    })

    it('accepts a message to parent', () => {
      const v = shellValidator()
      v.passes(`<message to="parent">\nhello\n</message>\n${YIELD}`)
    })

    it('accepts a message to task-123', () => {
      const v = shellValidator()
      v.passes(`<message to="task-123">\nsome content\n</message>\n${YIELD}`)
    })

    it('accepts a multi-line body', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nline one\nline two\nline three\n</message>\n${YIELD}`)
    })

    it('accepts a body containing <', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nuse <b>bold</b> text here\n</message>\n${YIELD}`)
    })

    it('accepts multiple messages before yield', () => {
      const v = shellValidator()
      v.passes(
        `<message to="user">\nfirst\n</message>\n` +
        `<message to="parent">\nsecond\n</message>\n` +
        YIELD
      )
    })

    it('accepts message followed by invoke then yield', () => {
      const v = buildValidator([SHELL_TOOL])
      v.passes(
        `<message to="user">\nrunning now\n</message>\n` +
        `<invoke tool="shell">\n<parameter name="command">echo hi</parameter>\n</invoke>\n` +
        YIELD
      )
    })

    it('accepts message with no newline before next tag', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n</message><yield_user/>`)
    })

    describe('requiredMessageTo', () => {
      it('accepts message to required recipient', () => {
        const v = buildValidator([SHELL_TOOL], b => b.requireMessageTo('parent'))
        v.passes(`<message to="parent">\nhello\n</message>\n${YIELD}`)
      })

      it('accepts required message followed by another message', () => {
        const v = buildValidator([SHELL_TOOL], b => b.requireMessageTo('parent'))
        v.passes(
          `<message to="parent">\nrequired\n</message>\n` +
          `<message to="user">\ninfo\n</message>\n` +
          YIELD
        )
      })

      it('accepts reason before required message', () => {
        const v = buildValidator([SHELL_TOOL], b => b.requireMessageTo('parent'))
        v.passes(
          `<reason about="turn">\nplanning\n</reason>\n` +
          `<message to="parent">\nhello\n</message>\n` +
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
      v.rejects(`<message to="user">\nhello\n</message>\n${YIELD}`)
    })
  })
})
