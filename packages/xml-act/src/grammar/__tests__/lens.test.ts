import { describe, it } from 'vitest'
import { shellValidator, buildValidator, SHELL_TOOL } from './helpers'

const YIELD = '<magnitude:yield_user/>'

describe('lens/think blocks', () => {
  describe('passing sequences', () => {
    it('minimal think block with about attribute', () => {
      const v = shellValidator()
      v.passes(`<magnitude:think about="alignment">\nsome thought\n</magnitude:think>\n${YIELD}`)
    })

    it('any lens name is accepted (free-form)', () => {
      const v = shellValidator()
      v.passes(`<magnitude:think about="tasks">\nthink about tasks\n</magnitude:think>\n${YIELD}`)
    })

    it('unknown lens name also accepted (not enumerated)', () => {
      const v = shellValidator()
      v.passes(`<magnitude:think about="anything-at-all">\nsome thought\n</magnitude:think>\n${YIELD}`)
    })

    it('think without about attribute (optional)', () => {
      const v = shellValidator()
      v.passes(`<magnitude:think>\nsome thought\n</magnitude:think>\n${YIELD}`)
    })

    it('multiple thinks before yield', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:think about="alignment">\nfirst thought\n</magnitude:think>\n` +
        `<magnitude:think about="turn">\nsecond thought\n</magnitude:think>\n` +
        YIELD
      )
    })

    it('think with multi-line content', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:think about="alignment">\n` +
        `line one\n` +
        `line two\n` +
        `line three\n` +
        `</magnitude:think>\n` +
        YIELD
      )
    })

    it('think body with < that does not form close tag', () => {
      const v = shellValidator()
      v.passes(`<magnitude:think about="turn">\nfoo < bar\n</magnitude:think>\n${YIELD}`)
    })

    it('think with no newline before next tag', () => {
      const v = shellValidator()
      v.passes(`<magnitude:think about="turn">\nreasoning\n</magnitude:think><magnitude:think about="alignment">\nmore\n</magnitude:think>\n${YIELD}`)
    })

    it('think then message then yield', () => {
      const v = shellValidator()
      v.passes(`<magnitude:think about="turn">\nplan\n</magnitude:think>\n<magnitude:message to="user">\nhello\n</magnitude:message>\n${YIELD}`)
    })
  })

  describe('minLenses: 1', () => {
    it('think first with minLenses:1 passes', () => {
      const v = buildValidator([SHELL_TOOL], b => b.withMinLenses(1))
      v.passes(`<magnitude:think about="turn">\nthinking\n</magnitude:think>\n${YIELD}`)
    })

    it('yield only with minLenses:1 is rejected (think required first)', () => {
      const v = buildValidator([SHELL_TOOL], b => b.withMinLenses(1))
      v.rejects(YIELD)
    })

    it('message without think with minLenses:1 is rejected', () => {
      const v = buildValidator([SHELL_TOOL], b => b.withMinLenses(1))
      v.rejects(`<magnitude:message to="user">\nhello\n</magnitude:message>\n${YIELD}`)
    })
  })

  describe('ordering enforcement', () => {
    it('think after message is rejected (post-lens phase)', () => {
      const v = shellValidator()
      v.rejects(
        `<magnitude:message to="user">\nhello\n</magnitude:message>\n` +
        `<magnitude:think about="turn">\nthinking\n</magnitude:think>\n` +
        YIELD
      )
    })
  })
})
