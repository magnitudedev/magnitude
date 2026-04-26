/**
 * Category 5: Escape removal
 *
 * <magnitude:escape> is completely removed from grammar and parser.
 * Grammar rejects it everywhere. Parser treats it as InvalidMagnitudeOpen.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, collectLensChunks,
  collectMessageChunks, YIELD_USER,
} from './helpers'

const v = () => grammarValidator()
const Y = YIELD_USER

describe('Category 5: escape removal', () => {
  // =========================================================================
  // Grammar rejection at every position
  // =========================================================================

  describe('grammar rejects escape everywhere', () => {
    it('01: escape at top level (before yield)', () => {
      v().rejects(`<magnitude:escape>content</magnitude:escape>\n${Y}`)
    })

    it('02: escape at top level between reason and yield', () => {
      v().rejects(`<magnitude:reason about="t">r</magnitude:reason>\n<magnitude:escape>x</magnitude:escape>\n${Y}`)
    })

    it('03: escape in reason body', () => {
      v().rejects(`<magnitude:reason about="t">text <magnitude:escape>inner</magnitude:escape> more</magnitude:reason>\n${Y}`)
    })

    it('04: escape in message body', () => {
      v().rejects(`<magnitude:message to="u">text <magnitude:escape>inner</magnitude:escape> more</magnitude:message>\n${Y}`)
    })

    it('05: escape in parameter body', () => {
      v().rejects(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo <magnitude:escape>x</magnitude:escape></magnitude:parameter>\n</magnitude:invoke>\n${Y}`)
    })

    it('06: escape as invoke child (pre-existing rejection)', () => {
      v().rejects(`<magnitude:invoke tool="shell">\n<magnitude:escape>x</magnitude:escape>\n</magnitude:invoke>\n${Y}`)
    })

    it('07: escape wrapping a full invoke', () => {
      v().rejects(`<magnitude:escape><magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke></magnitude:escape>\n${Y}`)
    })

    it('08: nested escape', () => {
      v().rejects(`<magnitude:escape><magnitude:escape>x</magnitude:escape></magnitude:escape>\n${Y}`)
    })
  })

  // =========================================================================
  // Parser behavior after escape removal
  // =========================================================================

  describe('parser treats escape as InvalidMagnitudeOpen', () => {
    it('09: escape open in reason body → StructuralParseError with InvalidMagnitudeOpen', () => {
      const input = `<magnitude:reason about="t">text <magnitude:escape>inner</magnitude:escape> more</magnitude:reason>\n${Y}`
      const events = parse(input)
      const errors = events.filter((e): e is any => e._tag === 'StructuralParseError')
      expect(errors.some((e: any) => e.error._tag === 'InvalidMagnitudeOpen' && e.error.tagName === 'magnitude:escape')).toBe(true)
      expect(hasEvent(events, 'LensEnd')).toBe(true)
    })

    it('10: escape open in message body → StructuralParseError with InvalidMagnitudeOpen', () => {
      const input = `<magnitude:message to="u">text <magnitude:escape>inner</magnitude:escape> more</magnitude:message>\n${Y}`
      const events = parse(input)
      const errors = events.filter((e): e is any => e._tag === 'StructuralParseError')
      expect(errors.some((e: any) => e.error._tag === 'InvalidMagnitudeOpen' && e.error.tagName === 'magnitude:escape')).toBe(true)
      expect(hasEvent(events, 'MessageEnd')).toBe(true)
    })
  })
})
