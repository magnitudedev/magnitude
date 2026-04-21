import { describe, it } from 'vitest'
import { shellValidator } from './helpers'

const YIELD = '\n<|yield:user|>'

describe('lens/think blocks', () => {
  describe('passing sequences', () => {
    it('minimal lens with alignment', () => {
      const v = shellValidator()
      v.passes('\n<|think:alignment>\nsome thought\n<think|>\n' + YIELD)
    })

    it('lens name: tasks', () => {
      const v = shellValidator()
      v.passes('\n<|think:tasks>\nthink about tasks\n<think|>\n' + YIELD)
    })

    it('lens name: diligence', () => {
      const v = shellValidator()
      v.passes('\n<|think:diligence>\nbe diligent\n<think|>\n' + YIELD)
    })

    it('lens name: skills', () => {
      const v = shellValidator()
      v.passes('\n<|think:skills>\ncheck skills\n<think|>\n' + YIELD)
    })

    it('lens name: turn', () => {
      const v = shellValidator()
      v.passes('\n<|think:turn>\nplan the turn\n<think|>\n' + YIELD)
    })

    it('lens name: pivot', () => {
      const v = shellValidator()
      v.passes('\n<|think:pivot>\npivot strategy\n<think|>\n' + YIELD)
    })

    it('multiple lenses before yield', () => {
      const v = shellValidator()
      v.passes(
        '\n<|think:alignment>\nfirst thought\n<think|>\n' +
        '\n<|think:turn>\nsecond thought\n<think|>\n' +
        YIELD
      )
    })

    it('lens with multi-line content', () => {
      const v = shellValidator()
      v.passes(
        '\n<|think:alignment>\n' +
        'line one\n' +
        'line two\n' +
        'line three\n' +
        '<think|>\n' +
        YIELD
      )
    })

    it('lens body with < that does not form close tag', () => {
      const v = shellValidator()
      v.passes('\n<|think:turn>\nfoo < bar\n<think|>\n' + YIELD)
    })

    it('lens with indentation before open tag', () => {
      const v = shellValidator()
      v.passes('\n  <|think:turn>\ncontent\n<think|>\n' + YIELD)
    })

    it('lenient close tag: </think|>', () => {
      const v = shellValidator()
      v.passes('\n<|think:alignment>\nsome thought\n</think|>\n' + YIELD)
    })

    it('lenient close tag: </think>', () => {
      const v = shellValidator()
      v.passes('\n<|think:alignment>\nsome thought\n</think>\n' + YIELD)
    })

    it('lenient close tag: <think>', () => {
      const v = shellValidator()
      v.passes('\n<|think:alignment>\nsome thought\n<think>\n' + YIELD)
    })
  })

  describe('forbidden sequences', () => {
    it('unknown lens name rejected', () => {
      const v = shellValidator()
      v.rejects('\n<|think:unknown>\nsome thought\n<think|>\n' + YIELD)
    })


  })
})
