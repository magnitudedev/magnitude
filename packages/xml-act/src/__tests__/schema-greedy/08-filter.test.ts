/**
 * Category 8: Filter Handling
 *
 * Filter behavior has tightened under the new grammar contract.
 * These cases document the current rejected forms in schema-greedy fixtures.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, getToolInput, YIELD,
} from './helpers'

const v = () => grammarValidator()

describe('filter handling', () => {
  it('01: filter after single param', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter><magnitude:filter>$.stdout</magnitude:filter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
    expect(hasEvent(parse(input), 'ToolInputReady')).toBe(true)
  })

  it('02: filter only (no params)', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:filter>$.result</magnitude:filter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('03: filter after multiple params', () => {
    const input = `<magnitude:invoke tool="edit">\n<magnitude:parameter name="path">f</magnitude:parameter><magnitude:parameter name="old">x</magnitude:parameter><magnitude:filter>//item</magnitude:filter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('04: filter with </magnitude:filter> in content (greedy match)', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter><magnitude:filter>$.</magnitude:filter>more</magnitude:filter></magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })

  it('05: filter with whitespace before </magnitude:invoke>', () => {
    const input = `<magnitude:invoke tool="shell">\n<magnitude:filter>$.x</magnitude:filter>\n</magnitude:invoke><${YIELD.slice(1)}`
    v().rejects(input)
  })
})
