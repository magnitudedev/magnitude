/**
 * Category 10: Body content edge cases
 *
 * Thorough testing of what's allowed inside body content for each element type.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, collectLensChunks, collectMessageChunks,
  getToolInput, hasEvent, YIELD_USER,
} from './helpers'

const v = () => grammarValidator()
const Y = YIELD_USER

describe('Category 10: body content edge cases', () => {
  // =========================================================================
  // Think body content
  // =========================================================================

  describe('think body', () => {
    it('01: body with multiple < chars', () => {
      const input = `<magnitude:think about="t">a < b < c</magnitude:think>\n${Y}`
      v().passes(input)
      expect(collectLensChunks(parse(input))).toBe('a < b < c')
    })

    it('02: body with HTML tags', () => {
      const input = `<magnitude:think about="t"><div><span>x</span></div></magnitude:think>\n${Y}`
      v().passes(input)
      expect(collectLensChunks(parse(input))).toContain('<div>')
    })

    it('03: body with code block containing close tag text', () => {
      // The close tag text is just text — but with first-close-wins, the BUC stops at </magnitude:think>
      // This means we can't have literal </magnitude:think> in think body anymore
      const input = `<magnitude:think about="t">Use </magnitude:think> to close</magnitude:think>\n${Y}`
      // First close wins — body is "Use ", then grammar rejects " to close"
      v().rejects(input)
    })

    it('04: body with partial close prefix </magnitude:reax — REJECT (pre-existing: BUC stops at </magnitude: prefix)', () => {
      // Current BUC uses excludeClosePrefix: 'magnitude:' — stops at </magnitude:
      // After changes: think-body won't use excludeClosePrefix, so this would PASS
      v().rejects(`<magnitude:think about="t">text </magnitude:reax more</magnitude:think>\n${Y}`)
    })

    it('05: body with </magnitude:message> in think body — REJECT (pre-existing: BUC stops at </magnitude: prefix)', () => {
      // Current BUC stops at </magnitude: — then expects "think>" but sees "message>"
      // After changes: think-body won't use excludeClosePrefix, so this would PASS
      v().rejects(`<magnitude:think about="t">text </magnitude:message> more</magnitude:think>\n${Y}`)
    })

    it('06: body with newlines', () => {
      const input = `<magnitude:think about="t">line1\nline2\nline3</magnitude:think>\n${Y}`
      v().passes(input)
      expect(collectLensChunks(parse(input))).toContain('line1')
    })

    it('07: body with only newlines', () => {
      v().passes(`<magnitude:think about="t">\n\n\n</magnitude:think>\n${Y}`)
    })

    it('08: very long body', () => {
      const longText = 'x'.repeat(1000)
      v().passes(`<magnitude:think about="t">${longText}</magnitude:think>\n${Y}`)
    })

    it('09: body with backticks and code', () => {
      const input = `<magnitude:think about="t">Run \`ls -la\` to check</magnitude:think>\n${Y}`
      v().passes(input)
    })

    it('10: body with quotes', () => {
      const input = `<magnitude:think about="t">He said "hello" and 'goodbye'</magnitude:think>\n${Y}`
      v().passes(input)
    })
  })

  // =========================================================================
  // Message body content
  // =========================================================================

  describe('message body', () => {
    it('11: body with HTML', () => {
      const input = `<magnitude:message to="u"><b>bold</b></magnitude:message>\n${Y}`
      v().passes(input)
    })

    it('12: body with < chars', () => {
      const input = `<magnitude:message to="u">a < b > c</magnitude:message>\n${Y}`
      v().passes(input)
    })

    it('13: body with markdown', () => {
      const input = `<magnitude:message to="u">## Title\n- item1\n- item2\n</magnitude:message>\n${Y}`
      v().passes(input)
    })

    it('14: false close in message body → REJECT', () => {
      v().rejects(`<magnitude:message to="u">text</magnitude:message>more</magnitude:message>\n${Y}`)
    })
  })

  // =========================================================================
  // Parameter body content
  // =========================================================================

  describe('parameter body', () => {
    it('15: body with shell command', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo "hello world" | grep hello</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
      expect(getToolInput(parse(input))?.command).toContain('grep hello')
    })

    it('16: body with multiline content', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">cat << 'EOF'\nline1\nline2\nEOF</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
    })

    it('17: body with < and > chars', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo "<div>hi</div>"</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
    })

    it('18: body with non-magnitude close tag', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo "</div>"</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
    })

    it('19: false close in param body → REJECT', () => {
      v().rejects(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">x</magnitude:parameter>y</magnitude:parameter>\n</magnitude:invoke>\n${Y}`)
    })

    it('20: param body with </magnitude:invoke> text', () => {
      // param-body uses excludeClosePrefix: 'magnitude:' so it stops at </magnitude:invoke>
      // This means </magnitude:invoke> in param body causes the BUC to stop
      // Then the param-close rule tries to match — it won't match </magnitude:invoke>
      // Grammar rejects
      v().rejects(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo </magnitude:invoke></magnitude:parameter>\n</magnitude:invoke>\n${Y}`)
    })
  })
})
