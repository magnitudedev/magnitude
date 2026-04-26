/**
 * Category 7: Forced message mode
 *
 * When requiredMessageTo is set, the grammar forces think(s) -> message.
 * First-close-wins applies to think bodies in forced mode too.
 */
import { describe, it, expect } from 'vitest'
import {
  grammarValidator, parse, hasEvent, collectLensChunks,
  collectMessageChunks, YIELD_USER,
} from './helpers'
import { SHELL_TOOL, MULTI_PARAM_TOOL, buildValidator } from '../../grammar/__tests__/helpers'
import type { GrammarBuilder } from '../../grammar/grammar-builder'

const Y = YIELD_USER

function forcedValidator(maxLenses?: number) {
  return buildValidator([SHELL_TOOL, MULTI_PARAM_TOOL], (b: GrammarBuilder) => {
    let r = b.requireMessageTo('user')
    if (maxLenses !== undefined) r = r.withMaxLenses(maxLenses)
    return r
  })
}

describe('Category 7: forced message mode', () => {
  // =========================================================================
  // Basic forced message → ACCEPT
  // =========================================================================

  it('01: message only (no think)', () => {
    const v = forcedValidator()
    v.passes(`<magnitude:message to="user">hi</magnitude:message>\n${Y}`)
  })

  it('02: think then message', () => {
    const v = forcedValidator()
    v.passes(`<magnitude:think about="t">think</magnitude:think>\n<magnitude:message to="user">hi</magnitude:message>\n${Y}`)
  })

  it('03: two thinks then message', () => {
    const v = forcedValidator()
    v.passes(`<magnitude:think about="a">1</magnitude:think>\n<magnitude:think about="b">2</magnitude:think>\n<magnitude:message to="user">hi</magnitude:message>\n${Y}`)
  })

  it('04: maxLenses=2, two thinks then message', () => {
    const v = forcedValidator(2)
    v.passes(`<magnitude:think about="a">1</magnitude:think>\n<magnitude:think about="b">2</magnitude:think>\n<magnitude:message to="user">hi</magnitude:message>\n${Y}`)
  })

  it('05: maxLenses=1, one think then message', () => {
    const v = forcedValidator(1)
    v.passes(`<magnitude:think about="t">think</magnitude:think>\n<magnitude:message to="user">hi</magnitude:message>\n${Y}`)
  })

  // =========================================================================
  // Forced mode rejections
  // =========================================================================

  it('06: wrong recipient → REJECT', () => {
    const v = forcedValidator()
    v.rejects(`<magnitude:message to="other">hi</magnitude:message>\n${Y}`)
  })

  it('07: invoke without message → REJECT', () => {
    const v = forcedValidator()
    v.rejects(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n${Y}`)
  })

  it('08: yield without message → REJECT', () => {
    const v = forcedValidator()
    v.rejects(Y)
  })

  // =========================================================================
  // First-close-wins in forced mode
  // =========================================================================

  it('09: false close in forced think → REJECT', () => {
    const v = forcedValidator()
    v.rejects(`<magnitude:think about="t">text</magnitude:think>more</magnitude:think>\n<magnitude:message to="user">hi</magnitude:message>\n${Y}`)
  })

  it('10: false close in forced message → REJECT', () => {
    const v = forcedValidator()
    v.rejects(`<magnitude:message to="user">text</magnitude:message>more</magnitude:message>\n${Y}`)
  })
})
