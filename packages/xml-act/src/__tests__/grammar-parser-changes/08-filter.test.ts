/**
 * Category 8: Filter behavior with first-close-wins
 *
 * Filter is currently rejected by the grammar for most tool configs.
 * These tests verify pre-existing behavior is preserved.
 */
import { describe, it } from 'vitest'
import { grammarValidator, YIELD_USER } from './helpers'

const v = () => grammarValidator()
const Y = YIELD_USER

describe('Category 8: filter behavior', () => {
  it('01: filter after param — grammar rejects (pre-existing)', () => {
    v().rejects(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n<magnitude:filter>$.stdout</magnitude:filter>\n</magnitude:invoke>\n${Y}`)
  })

  it('02: filter only (no params) — grammar rejects (pre-existing)', () => {
    v().rejects(`<magnitude:invoke tool="shell">\n<magnitude:filter>$.result</magnitude:filter>\n</magnitude:invoke>\n${Y}`)
  })
})
