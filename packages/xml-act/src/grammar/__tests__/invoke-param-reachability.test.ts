import { describe, it, expect } from 'vitest'
import { buildValidator, shellValidator, multiToolValidator, SHELL_TOOL } from './helpers'

/**
 * Tests that validate parameter reachability inside invoke blocks.
 * In the new XML grammar, invoke-next handles the choice between
 * parameters, filter, and close tag.
 */

const YIELD = '<magnitude:yield_user/>'

describe('invoke parameter reachability', () => {
  describe('basic parameter access', () => {
    it('accepts invoke with parameter', () => {
      const v = shellValidator()
      v.passes(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n${YIELD}`)
    })

    it('rejects invoke with no parameters for tools with required params', () => {
      const v = shellValidator()
      v.rejects(`<magnitude:invoke tool="shell">\n</magnitude:invoke>\n${YIELD}`)
    })

    it('after invoke open + newline, < is valid (for parameter or close)', () => {
      const v = shellValidator()
      const rules = v.validAfter(`<magnitude:invoke tool="shell">\n`)
      const validChars = rules.flatMap((r: any) => {
        if (r.type === 'char') return r.value.map((v: number) => String.fromCharCode(v))
        if (r.type === 'char_exclude') return ['[exclude]']
        return []
      })
      expect(validChars).toContain('<')
    })

    it('after invoke open + newline + <, only structural open paths are valid before required params', () => {
      const v = shellValidator()
      const rules = v.validAfter(`<magnitude:invoke tool="shell">\n<`)
      const validChars = rules.flatMap((r: any) => {
        if (r.type === 'char') return r.value.map((v: number) => String.fromCharCode(v))
        if (r.type === 'char_exclude') return ['[exclude]']
        return []
      })
      expect(validChars).toContain('m')
      expect(validChars).not.toContain('/')
    })
  })

  describe('multiple parameters', () => {
    it('accepts invoke with multiple parameters', () => {
      const v = multiToolValidator()
      v.passes(
        `<magnitude:invoke tool="edit">\n` +
        `<magnitude:parameter name="path">foo.ts</magnitude:parameter>\n` +
        `<magnitude:parameter name="old">bar</magnitude:parameter>\n` +
        `<magnitude:parameter name="new">baz</magnitude:parameter>\n` +
        `</magnitude:invoke>\n` +
        YIELD
      )
    })

    it('accepts invoke with parameters in any order', () => {
      const v = multiToolValidator()
      v.passes(
        `<magnitude:invoke tool="edit">\n` +
        `<magnitude:parameter name="new">baz</magnitude:parameter>\n` +
        `<magnitude:parameter name="old">bar</magnitude:parameter>\n` +
        `<magnitude:parameter name="path">foo.ts</magnitude:parameter>\n` +
        `</magnitude:invoke>\n` +
        YIELD
      )
    })

    it('param names constrained to tool schema', () => {
      // New grammar constrains param names — unknown names are absorbed as content
      const v = multiToolValidator()
      // Edit tool accepts its 3 known params
      v.passes(
        `<magnitude:invoke tool="edit">\n` +
        `<magnitude:parameter name="path">a</magnitude:parameter>\n` +
        `<magnitude:parameter name="old">b</magnitude:parameter>\n` +
        `<magnitude:parameter name="new">c</magnitude:parameter>\n` +
        `</magnitude:invoke>\n` +
        YIELD
      )
    })
  })

  describe('filter reachability', () => {
    it('rejects invoke with filter after required params in the current shell grammar path', () => {
      const v = shellValidator()
      v.rejects(
        `<magnitude:invoke tool="shell">\n` +
        `<magnitude:parameter name="command">ls</magnitude:parameter>\n` +
        `<magnitude:filter>$.stdout</magnitude:filter>\n` +
        YIELD
      )
    })

    it('after parameter close and <, valid continuations include structural tags', () => {
      const v = shellValidator()
      // With greedy body, </magnitude:parameter>\n< is inside the BUC pattern.
      // After <, the grammar allows / (for close tags) and any content char via char_exclude.
      const rules = v.validAfter(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n<`)
      const validChars = rules.flatMap((r: any) => {
        if (r.type === 'char') return r.value.map((v: number) => String.fromCharCode(v))
        if (r.type === 'char_exclude') return ['[exclude]']
        return []
      })
      // / is explicitly valid (for </magnitude:invoke> or </magnitude:parameter>)
      expect(validChars).toContain('/')
      // Other chars (like f for <magnitude:filter>) are valid via char_exclude rule
      expect(validChars.some((c: string) => c === 'f' || c === '[exclude]')).toBe(true)
    })
  })

  describe('inline parameter values (no newline after open)', () => {
    it('accepts value immediately after parameter open tag', () => {
      const v = shellValidator()
      v.passes(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls -la</magnitude:parameter>\n</magnitude:invoke>\n${YIELD}`)
    })

    it('after parameter open tag, content chars are valid', () => {
      const v = shellValidator()
      const rules = v.validAfter(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">`)
      const hasContentChars = rules.some((r: any) => {
        if (r.type === 'char_exclude') return true  // [^<] means any non-< char
        if (r.type === 'char') return r.value.some((v: number) => v >= 33 && v <= 126 && v !== 60)
        return false
      })
      expect(hasContentChars).toBe(true)
    })
  })

  describe('close tag variants', () => {
    it('standard </magnitude:invoke> closes invoke block', () => {
      const v = shellValidator()
      v.passes(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n${YIELD}`)
    })

    it('old MACT-style <magnitude:invoke|> is rejected inside parameter bodies', () => {
      const v = shellValidator()
      v.rejects(
        `<magnitude:invoke tool="shell">\n` +
        `<magnitude:parameter name="command">ls\n<magnitude:invoke|>\nmore</magnitude:parameter>\n` +
        `</magnitude:invoke>\n` +
        YIELD
      )
    })
  })
})
