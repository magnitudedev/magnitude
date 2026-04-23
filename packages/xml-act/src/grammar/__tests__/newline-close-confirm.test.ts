/**
 * Tests for newline-based close confirmation of top-level tags.
 *
 * After a closing tag (</magnitude:reason>, </magnitude:message>), the grammar should accept
 * EITHER a newline OR a left-angle-bracket as valid close confirmation indicators.
 *
 * These tests are currently FAILING — they document the desired behavior.
 * Once the grammar fix is applied, they should all pass.
 */
import { describe, it } from 'vitest'
import { shellValidator } from './helpers'

const YIELD = '<magnitude:yield_user/>'

describe('newline close confirmation (currently failing — desired behavior)', () => {
  describe('</magnitude:reason> + newline + structural tag or yield', () => {
    it('01: </magnitude:reason> + \\n + yield — newline should confirm the close', () => {
      const v = shellValidator()
      // Currently fails: grammar requires < immediately after </magnitude:reason>, not \n
      v.passes(`<magnitude:reason about="turn">\nthinking\n</magnitude:reason>\n${YIELD}`)
    })

    it('02: </magnitude:reason> + \\n + \\n + yield — multiple newlines should still confirm', () => {
      const v = shellValidator()
      v.passes(`<magnitude:reason about="turn">\nthinking\n</magnitude:reason>\n\n${YIELD}`)
    })

    it('03: </magnitude:reason> + space + \\n + yield — horizontal whitespace then newline', () => {
      const v = shellValidator()
      v.passes(`<magnitude:reason about="turn">\nthinking\n</magnitude:reason>  \n${YIELD}`)
    })

    it('04: </magnitude:reason> + \\n + <magnitude:message ...> — newline confirm then message tag', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:reason about="turn">\nplan\n</magnitude:reason>\n` +
        `<magnitude:message to="user">\nhello\n</magnitude:message>\n` +
        YIELD
      )
    })

    it('05: </magnitude:reason> + \\n + <magnitude:invoke> — newline confirm then invoke tag', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:reason about="turn">\nplan\n</magnitude:reason>\n` +
        `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n` +
        YIELD
      )
    })

    it('06: multiple reasons separated by newlines only', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:reason about="alignment">\nfirst\n</magnitude:reason>\n` +
        `<magnitude:reason about="turn">\nsecond\n</magnitude:reason>\n` +
        YIELD
      )
    })
  })

  describe('</magnitude:message> + newline + yield', () => {
    it('07: </magnitude:message> + \\n + yield — newline should confirm the close', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>\n${YIELD}`)
    })

    it('08: </magnitude:message> + \\n + \\n + yield — multiple newlines', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>\n\n${YIELD}`)
    })

    it('09: reason then message with newline separators', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:reason about="turn">\nthinking\n</magnitude:reason>\n` +
        `<magnitude:message to="user">\nhello\n</magnitude:message>\n` +
        YIELD
      )
    })
  })

  describe('existing < confirmation still works (should stay passing)', () => {
    it('10: </magnitude:reason> + < + yield — angle bracket still confirms (regression guard)', () => {
      const v = shellValidator()
      v.passes(`<magnitude:reason about="turn">\nthinking\n</magnitude:reason>${YIELD}`)
    })

    it('11: </magnitude:reason> + < + message — angle bracket confirms into message (regression guard)', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:reason about="turn">\nplan\n</magnitude:reason>` +
        `<magnitude:message to="user">\nhello\n</magnitude:message>\n` +
        YIELD
      )
    })
  })
})