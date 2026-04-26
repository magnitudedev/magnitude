/**
 * Category 4: Alias closes
 *
 * Grammar recognizes tool-name and param-name as close tags.
 * Parser already supports this (resolve.ts).
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, getToolInput, hasEvent, YIELD_USER,
} from './helpers'

const v = () => grammarValidator()
const Y = YIELD_USER

describe('Category 4: alias closes', () => {
  // =========================================================================
  // Invoke alias closes
  // =========================================================================

  describe('invoke alias closes', () => {
    it('01: </magnitude:shell> closes shell invoke', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:shell>\n${Y}`
      v().passes(input)
      expect(getToolInput(parse(input))?.command).toBe('ls')
    })

    it('02: </magnitude:shell> + immediate yield (zero ws)', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:shell>${Y}`
      v().passes(input)
    })

    it('03: </magnitude:shell> + space + yield', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:shell> ${Y}`
      v().passes(input)
    })

    it('04: </magnitude:edit> closes edit invoke', () => {
      const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\n<magnitude:parameter name="old">x</magnitude:parameter>\n<magnitude:parameter name="new">y</magnitude:parameter>\n</magnitude:edit>\n${Y}`
      v().passes(input)
      expect(getToolInput(parse(input))?.path).toBe('f')
    })

    it('05: wrong alias — </magnitude:edit> does not close shell invoke', () => {
      v().rejects(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:edit>\n${Y}`)
    })

    it('06: wrong alias — </magnitude:shell> does not close edit invoke', () => {
      v().rejects(`<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\n<magnitude:parameter name="old">x</magnitude:parameter>\n<magnitude:parameter name="new">y</magnitude:parameter>\n</magnitude:shell>\n${Y}`)
    })

    it('07: canonical </magnitude:invoke> still works for shell', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
      expect(getToolInput(parse(input))?.command).toBe('ls')
    })

    it('08: canonical </magnitude:invoke> still works for edit', () => {
      const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\n<magnitude:parameter name="old">x</magnitude:parameter>\n<magnitude:parameter name="new">y</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
    })

    it('09: alias invoke close + chaining to message', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:shell>\n<magnitude:message to="u">done</magnitude:message>\n${Y}`
      v().passes(input)
      expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
      expect(hasEvent(parse(input), 'MessageEnd')).toBe(true)
    })

    it('10: alias invoke close + chaining to another invoke', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:shell>\n<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">pwd</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
    })
  })

  // =========================================================================
  // Parameter alias closes
  // =========================================================================

  describe('parameter alias closes', () => {
    it('11: </magnitude:command> closes command parameter', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:command>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
      expect(getToolInput(parse(input))?.command).toBe('ls')
    })

    it('12: </magnitude:path> closes path parameter in edit', () => {
      const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:path>\n<magnitude:parameter name="old">x</magnitude:parameter>\n<magnitude:parameter name="new">y</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
      expect(getToolInput(parse(input))?.path).toBe('f')
    })

    it('13: </magnitude:old> closes old parameter in edit', () => {
      const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\n<magnitude:parameter name="old">x</magnitude:old>\n<magnitude:parameter name="new">y</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
      expect(getToolInput(parse(input))?.old).toBe('x')
    })

    it('14: </magnitude:new> closes new parameter in edit', () => {
      const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter>\n<magnitude:parameter name="old">x</magnitude:parameter>\n<magnitude:parameter name="new">y</magnitude:new>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
      expect(getToolInput(parse(input))?.new).toBe('y')
    })

    it('15: canonical </magnitude:parameter> still works', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
    })

    it('16: wrong param alias — </magnitude:path> does not close command in shell', () => {
      // shell only has "command" param, so </magnitude:path> is not a valid alias
      v().rejects(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:path>\n</magnitude:invoke>\n${Y}`)
    })
  })

  // =========================================================================
  // Mixed alias closes
  // =========================================================================

  describe('mixed alias closes', () => {
    it('17: alias param close + alias invoke close', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:command>\n</magnitude:shell>\n${Y}`
      v().passes(input)
      expect(getToolInput(parse(input))?.command).toBe('ls')
    })

    it('18: alias param close + canonical invoke close', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:command>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
    })

    it('19: canonical param close + alias invoke close', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:shell>\n${Y}`
      v().passes(input)
    })

    it('20: all alias closes in edit (3 params + invoke)', () => {
      const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:path>\n<magnitude:parameter name="old">x</magnitude:old>\n<magnitude:parameter name="new">y</magnitude:new>\n</magnitude:edit>\n${Y}`
      v().passes(input)
      const tool = getToolInput(parse(input))
      expect(tool?.path).toBe('f')
      expect(tool?.old).toBe('x')
      expect(tool?.new).toBe('y')
    })
  })

  // =========================================================================
  // Alias + first-close-wins interaction
  // =========================================================================

  describe('alias + first-close-wins', () => {
    it('21: first </magnitude:command> closes param immediately', () => {
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo </magnitude:command>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
      expect(getToolInput(parse(input))?.command).toBe('echo ')
    })

    it('22: alias close in body text — BUC stops at </magnitude: prefix', () => {
      // param-body BUC stops at </magnitude: so </magnitude:command> is not consumed as body
      const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">text</magnitude:command>\n</magnitude:invoke>\n${Y}`
      v().passes(input)
      expect(getToolInput(parse(input))?.command).toBe('text')
    })
  })
})
