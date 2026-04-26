/**
 * Category 11: minLenses=1 mode
 */
import { describe, it } from 'vitest'
import { YIELD_USER } from './helpers'
import { buildValidator, SHELL_TOOL, MULTI_PARAM_TOOL } from '../../grammar/__tests__/helpers'
import type { GrammarBuilder } from '../../grammar/grammar-builder'

const Y = YIELD_USER

function minLensValidator() {
  return buildValidator([SHELL_TOOL, MULTI_PARAM_TOOL], (b: GrammarBuilder) => b.withMinLenses(1))
}

describe('Category 11: minLenses=1 mode', () => {
  it('01: reason then yield', () => {
    minLensValidator().passes(`<magnitude:reason about="t">think</magnitude:reason>\n${Y}`)
  })

  it('02: reason then message then yield', () => {
    minLensValidator().passes(`<magnitude:reason about="t">think</magnitude:reason>\n<magnitude:message to="u">hi</magnitude:message>\n${Y}`)
  })

  it('03: yield only → REJECT (need at least one reason)', () => {
    minLensValidator().rejects(Y)
  })

  it('04: message only → REJECT (need reason first)', () => {
    minLensValidator().rejects(`<magnitude:message to="u">hi</magnitude:message>\n${Y}`)
  })

  it('05: false close in reason → REJECT', () => {
    minLensValidator().rejects(`<magnitude:reason about="t">text</magnitude:reason>more</magnitude:reason>\n${Y}`)
  })

  it('06: reason then invoke with alias close', () => {
    minLensValidator().passes(`<magnitude:reason about="t">think</magnitude:reason>\n<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:shell>\n${Y}`)
  })
})
