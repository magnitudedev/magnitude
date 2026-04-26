import { describe, it, expect } from 'vitest'
import { parse, YIELD_USER, collectMessageChunks, collectLensChunks } from './helpers'

/**
 * Category 12: Mismatch recovery
 *
 * When a mismatched magnitude close tag appears on a newline boundary,
 * the parser should recover by treating it as the correct close tag.
 * This behavior was NOT part of the changes — it must be preserved.
 */
describe('Category 12: mismatch recovery', () => {
  const Y = YIELD_USER

  describe('message closed by wrong close tag on newline', () => {
    it('01: message closed by reason close on newline → recovers', () => {
      const input = `<magnitude:message to="user">hello\n</magnitude:reason>\n<magnitude:message to="user">world</magnitude:message>\n${Y}`
      const events = parse(input)
      // Recovery should produce two separate messages
      const messageStarts = events.filter((e: any) => e._tag === 'MessageStart')
      const messageEnds = events.filter((e: any) => e._tag === 'MessageEnd')
      expect(messageStarts.length).toBe(2)
      expect(messageEnds.length).toBe(2)
    })

    it('02: message closed by invoke close on newline → recovers', () => {
      const input = `<magnitude:message to="user">hello\n</magnitude:invoke>\n<magnitude:message to="user">world</magnitude:message>\n${Y}`
      const events = parse(input)
      const messageStarts = events.filter((e: any) => e._tag === 'MessageStart')
      const messageEnds = events.filter((e: any) => e._tag === 'MessageEnd')
      expect(messageStarts.length).toBe(2)
      expect(messageEnds.length).toBe(2)
    })
  })

  describe('reason closed by wrong close tag on newline', () => {
    it('03: reason closed by message close on newline → recovers', () => {
      const input = `<magnitude:reason about="t">thinking\n</magnitude:message>\n<magnitude:message to="user">hi</magnitude:message>\n${Y}`
      const events = parse(input)
      const body = collectLensChunks(events)
      expect(body).toContain('thinking')
    })

    it('04: reason closed by parameter close on newline → recovers', () => {
      const input = `<magnitude:reason about="t">thinking\n</magnitude:parameter>\n<magnitude:message to="user">hi</magnitude:message>\n${Y}`
      const events = parse(input)
      const body = collectLensChunks(events)
      expect(body).toContain('thinking')
    })
  })

  describe('same-line mismatch emits AmbiguousMagnitudeClose', () => {
    it('05: message closed by reason close same-line → error + content', () => {
      const input = `<magnitude:message to="user">hello</magnitude:reason>more</magnitude:message>\n${Y}`
      const events = parse(input)
      const errors = events.filter((e: any) => e._tag === 'StructuralParseError')
      const ambiguous = errors.filter((e: any) => e.error._tag === 'AmbiguousMagnitudeClose')
      expect(ambiguous.length).toBeGreaterThanOrEqual(1)
      // The mismatched close should be dumped as content, message continues
      const body = collectMessageChunks(events)
      expect(body).toContain('</magnitude:reason>')
    })

    it('06: reason closed by message close same-line → error + content', () => {
      const input = `<magnitude:reason about="t">thinking</magnitude:message>more</magnitude:reason>\n<magnitude:message to="user">hi</magnitude:message>\n${Y}`
      const events = parse(input)
      const errors = events.filter((e: any) => e._tag === 'StructuralParseError')
      const ambiguous = errors.filter((e: any) => e.error._tag === 'AmbiguousMagnitudeClose')
      expect(ambiguous.length).toBeGreaterThanOrEqual(1)
      const body = collectLensChunks(events)
      expect(body).toContain('</magnitude:message>')
    })
  })

  describe('mismatch recovery does not apply to matching closes', () => {
    it('07: correct close tag always closes immediately (first-close-wins)', () => {
      const input = `<magnitude:message to="user">hello</magnitude:message>extra</magnitude:message>\n${Y}`
      const events = parse(input)
      const body = collectMessageChunks(events)
      // First matching close wins — body is just 'hello'
      expect(body).toBe('hello')
    })
  })
})
