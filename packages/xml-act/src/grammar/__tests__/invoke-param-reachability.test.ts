import { describe, it, expect } from 'vitest'
import { buildValidator, SHELL_TOOL, SKILL_TOOL, MULTI_PARAM_TOOL, NO_PARAM_TOOL } from './helpers'

/**
 * These tests validate that the grammar allows models to reach parameters
 * after an invoke open tag. This was the core bug: the close rule consumed
 * the newline after the open tag, preventing the model from ever starting
 * a parameter.
 */

describe('invoke parameter reachability', () => {
  describe('single-param tool (skill)', () => {
    it('accepts invoke with parameter', () => {
      const v = buildValidator([SKILL_TOOL])
      v.passes('\n<|invoke:skill>\n<|parameter:name>review<parameter|>\n<invoke|>\n\n<|yield:user|>')
    })

    it('accepts invoke with no parameters (skip to close)', () => {
      const v = buildValidator([SKILL_TOOL])
      v.passes('\n<|invoke:skill>\n<invoke|>\n\n<|yield:user|>')
    })

    it('after invoke open + newline, parameter open is valid', () => {
      const v = buildValidator([SKILL_TOOL])
      const rules = v.validAfter('\n<|invoke:skill>\n')
      const validChars = rules.flatMap((r: any) => {
        if (r.type === 'char') return r.value.map((v: number) => String.fromCharCode(v))
        if (r.type === 'char_exclude') return ['[exclude]']
        return []
      })
      // '<' must be valid (for both parameter and close paths)
      expect(validChars).toContain('<')
    })

    it('after invoke open + newline + <, both | and i are valid', () => {
      const v = buildValidator([SKILL_TOOL])
      const rules = v.validAfter('\n<|invoke:skill>\n<')
      const validChars = rules.flatMap((r: any) => {
        if (r.type === 'char') return r.value.map((v: number) => String.fromCharCode(v))
        if (r.type === 'char_exclude') return ['[exclude]']
        return []
      })
      // '|' for parameter path, 'i' for canonical close, '/' for lenient close
      expect(validChars).toContain('|')  // <|parameter:...
      expect(validChars).toContain('i')  // <invoke|>
      expect(validChars).toContain('/')  // </invoke|>
    })
  })

  describe('multi-param tool (edit with path, old, new)', () => {
    it('accepts invoke with all 3 parameters', () => {
      const v = buildValidator([MULTI_PARAM_TOOL])
      v.passes(
        '\n<|invoke:edit>\n' +
        '<|parameter:path>foo.ts<parameter|>\n' +
        '<|parameter:old>bar<parameter|>\n' +
        '<|parameter:new>baz<parameter|>\n' +
        '<invoke|>\n' +
        '\n<|yield:user|>'
      )
    })

    it('accepts invoke with subset of parameters', () => {
      const v = buildValidator([MULTI_PARAM_TOOL])
      v.passes(
        '\n<|invoke:edit>\n' +
        '<|parameter:path>foo.ts<parameter|>\n' +
        '<invoke|>\n' +
        '\n<|yield:user|>'
      )
    })

    it('accepts invoke with parameters in different order', () => {
      const v = buildValidator([MULTI_PARAM_TOOL])
      v.passes(
        '\n<|invoke:edit>\n' +
        '<|parameter:new>baz<parameter|>\n' +
        '<|parameter:old>bar<parameter|>\n' +
        '<|parameter:path>foo.ts<parameter|>\n' +
        '<invoke|>\n' +
        '\n<|yield:user|>'
      )
    })
  })

  describe('no-param tool', () => {
    it('accepts invoke with no parameters', () => {
      const v = buildValidator([NO_PARAM_TOOL])
      v.passes('\n<|invoke:tree>\n<invoke|>\n\n<|yield:user|>')
    })
  })

  describe('parameter is inline (no newline after open tag)', () => {
    it('accepts value immediately after parameter open tag', () => {
      const v = buildValidator([SKILL_TOOL])
      v.passes('\n<|invoke:skill>\n<|parameter:name>review<parameter|>\n<invoke|>\n\n<|yield:user|>')
    })

    it('after parameter open tag, content chars are valid', () => {
      const v = buildValidator([SKILL_TOOL])
      const rules = v.validAfter('\n<|invoke:skill>\n<|parameter:name>')
      const hasContentChars = rules.some((r: any) => {
        if (r.type === 'char_exclude') return true  // [^<] means any non-< char
        if (r.type === 'char') return r.value.some((v: number) => v >= 33 && v <= 126 && v !== 60)
        return false
      })
      expect(hasContentChars).toBe(true)
    })
  })

  describe('bounded parameter repetition', () => {
    it('rejects more parameters than the tool defines', () => {
      const v = buildValidator([SKILL_TOOL])  // 1 param
      // Try to put 2 parameters — should fail since skill only has 1 param slot
      v.rejects(
        '\n<|invoke:skill>\n' +
        '<|parameter:name>review<parameter|>\n' +
        '<|parameter:name>again<parameter|>\n' +  // second occurrence — exceeds 1 slot
        '<invoke|>\n' +
        '\n<|yield:user|>'
      )
    })
  })

  describe('close tag variants after parameters', () => {
    it('canonical close: <invoke|>', () => {
      const v = buildValidator([SKILL_TOOL])
      v.passes('\n<|invoke:skill>\n<|parameter:name>review<parameter|>\n<invoke|>\n\n<|yield:user|>')
    })

    it('lenient close: </invoke|>', () => {
      const v = buildValidator([SKILL_TOOL])
      v.passes('\n<|invoke:skill>\n<|parameter:name>review<parameter|>\n</invoke|>\n\n<|yield:user|>')
    })

    it('lenient close: </invoke>', () => {
      const v = buildValidator([SKILL_TOOL])
      v.passes('\n<|invoke:skill>\n<|parameter:name>review<parameter|>\n</invoke>\n\n<|yield:user|>')
    })

    it('lenient close: <invoke>', () => {
      const v = buildValidator([SKILL_TOOL])
      v.passes('\n<|invoke:skill>\n<|parameter:name>review<parameter|>\n<invoke>\n\n<|yield:user|>')
    })
  })
})
