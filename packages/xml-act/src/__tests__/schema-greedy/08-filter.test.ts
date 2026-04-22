/**
 * Category 8: Filter Handling
 *
 * Filter body uses greedy matching. Filter always closes the invoke
 * (it's the last child), so it gets deep confirmation through invoke close.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, getToolInput, YIELD,
} from './helpers'

const v = () => grammarValidator()

describe('filter handling', () => {
  it('01: filter after single param', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">ls</parameter><filter>$.stdout</filter></invoke><${YIELD.slice(1)}`
    v().passes(input)
    expect(getToolInput(parse(input))?.command).toBe('ls')
  })

  it('02: filter only (no params)', () => {
    const input = `<invoke tool="shell">\n<filter>$.result</filter></invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('03: filter after multiple params', () => {
    const input = `<invoke tool="edit">\n<parameter name="path">f</parameter><parameter name="old">x</parameter><filter>//item</filter></invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('04: filter with </filter> in content (greedy match)', () => {
    const input = `<invoke tool="shell">\n<parameter name="command">ls</parameter><filter>$.</filter>more</filter></invoke><${YIELD.slice(1)}`
    v().passes(input)
  })

  it('05: filter with whitespace before </invoke>', () => {
    const input = `<invoke tool="shell">\n<filter>$.x</filter>\n</invoke><${YIELD.slice(1)}`
    v().passes(input)
  })
})
