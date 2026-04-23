/**
 * Tests for newline-based close confirmation of top-level tags.
 *
 * After a closing tag (</reason>, </message>), the grammar should accept
 * EITHER a newline OR a left-angle-bracket as valid close confirmation indicators.
 *
 * These tests are currently FAILING — they document the desired behavior.
 * Once the grammar fix is applied, they should all pass.
 */
import { describe, it } from 'vitest'
import { shellValidator } from './helpers'

const YIELD = '<yield_user/>'

describe('newline close confirmation (currently failing — desired behavior)', () => {
  describe('</reason> + newline + structural tag or yield', () => {
    it('01: </reason> + \\n + yield — newline should confirm the close', () => {
      const v = shellValidator()
      // Currently fails: grammar requires < immediately after </reason>, not \n
      v.passes(`<reason about="turn">\nthinking\n</reason>\n${YIELD}`)
    })

    it('02: </reason> + \\n + \\n + yield — multiple newlines should still confirm', () => {
      const v = shellValidator()
      v.passes(`<reason about="turn">\nthinking\n</reason>\n\n${YIELD}`)
    })

    it('03: </reason> + space + \\n + yield — horizontal whitespace then newline', () => {
      const v = shellValidator()
      v.passes(`<reason about="turn">\nthinking\n</reason>  \n${YIELD}`)
    })

    it('04: </reason> + \\n + <message ...> — newline confirm then message tag', () => {
      const v = shellValidator()
      v.passes(
        `<reason about="turn">\nplan\n</reason>\n` +
        `<message to="user">\nhello\n</message>\n` +
        YIELD
      )
    })

    it('05: </reason> + \\n + <invoke> — newline confirm then invoke tag', () => {
      const v = shellValidator()
      v.passes(
        `<reason about="turn">\nplan\n</reason>\n` +
        `<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n` +
        YIELD
      )
    })

    it('06: multiple reasons separated by newlines only', () => {
      const v = shellValidator()
      v.passes(
        `<reason about="alignment">\nfirst\n</reason>\n` +
        `<reason about="turn">\nsecond\n</reason>\n` +
        YIELD
      )
    })
  })

  describe('</message> + newline + yield', () => {
    it('07: </message> + \\n + yield — newline should confirm the close', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n</message>\n${YIELD}`)
    })

    it('08: </message> + \\n + \\n + yield — multiple newlines', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n</message>\n\n${YIELD}`)
    })

    it('09: reason then message with newline separators', () => {
      const v = shellValidator()
      v.passes(
        `<reason about="turn">\nthinking\n</reason>\n` +
        `<message to="user">\nhello\n</message>\n` +
        YIELD
      )
    })
  })

  describe('existing < confirmation still works (should stay passing)', () => {
    it('10: </reason> + < + yield — angle bracket still confirms (regression guard)', () => {
      const v = shellValidator()
      v.passes(`<reason about="turn">\nthinking\n</reason>${YIELD}`)
    })

    it('11: </reason> + < + message — angle bracket confirms into message (regression guard)', () => {
      const v = shellValidator()
      v.passes(
        `<reason about="turn">\nplan\n</reason>` +
        `<message to="user">\nhello\n</message>\n` +
        YIELD
      )
    })
  })
})