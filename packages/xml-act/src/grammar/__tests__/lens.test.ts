import { describe, it } from 'vitest'
import { shellValidator, buildValidator, SHELL_TOOL } from './helpers'

const YIELD = '<yield_user/>'

describe('lens/reason blocks', () => {
  describe('passing sequences', () => {
    it('minimal reason block with about attribute', () => {
      const v = shellValidator()
      v.passes(`<reason about="alignment">\nsome thought\n</reason>\n${YIELD}`)
    })

    it('any lens name is accepted (free-form)', () => {
      const v = shellValidator()
      v.passes(`<reason about="tasks">\nthink about tasks\n</reason>\n${YIELD}`)
    })

    it('unknown lens name also accepted (not enumerated)', () => {
      const v = shellValidator()
      v.passes(`<reason about="anything-at-all">\nsome thought\n</reason>\n${YIELD}`)
    })

    it('reason without about attribute (optional)', () => {
      const v = shellValidator()
      v.passes(`<reason>\nsome thought\n</reason>\n${YIELD}`)
    })

    it('multiple reasons before yield', () => {
      const v = shellValidator()
      v.passes(
        `<reason about="alignment">\nfirst thought\n</reason>\n` +
        `<reason about="turn">\nsecond thought\n</reason>\n` +
        YIELD
      )
    })

    it('reason with multi-line content', () => {
      const v = shellValidator()
      v.passes(
        `<reason about="alignment">\n` +
        `line one\n` +
        `line two\n` +
        `line three\n` +
        `</reason>\n` +
        YIELD
      )
    })

    it('reason body with < that does not form close tag', () => {
      const v = shellValidator()
      v.passes(`<reason about="turn">\nfoo < bar\n</reason>\n${YIELD}`)
    })

    it('reason with no newline before next tag', () => {
      const v = shellValidator()
      v.passes(`<reason about="turn">\nreasoning\n</reason><reason about="alignment">\nmore\n</reason>\n${YIELD}`)
    })

    it('reason then message then yield', () => {
      const v = shellValidator()
      v.passes(`<reason about="turn">\nplan\n</reason>\n<message to="user">\nhello\n</message>\n${YIELD}`)
    })
  })

  describe('minLenses: 1', () => {
    it('reason first with minLenses:1 passes', () => {
      const v = buildValidator([SHELL_TOOL], b => b.withMinLenses(1))
      v.passes(`<reason about="turn">\nthinking\n</reason>\n${YIELD}`)
    })

    it('yield only with minLenses:1 is rejected (reason required first)', () => {
      const v = buildValidator([SHELL_TOOL], b => b.withMinLenses(1))
      v.rejects(YIELD)
    })

    it('message without reason with minLenses:1 is rejected', () => {
      const v = buildValidator([SHELL_TOOL], b => b.withMinLenses(1))
      v.rejects(`<message to="user">\nhello\n</message>\n${YIELD}`)
    })
  })

  describe('ordering enforcement', () => {
    it('reason after message is rejected (post-lens phase)', () => {
      const v = shellValidator()
      v.rejects(
        `<message to="user">\nhello\n</message>\n` +
        `<reason about="turn">\nthinking\n</reason>\n` +
        YIELD
      )
    })
  })
})
