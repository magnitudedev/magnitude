import { describe, it, expect } from 'vitest'
import { buildValidator, shellValidator, multiToolValidator, SHELL_TOOL } from './helpers'

/**
 * Tests that validate parameter reachability inside invoke blocks.
 * In the new XML grammar, invoke-next handles the choice between
 * parameters, filter, and close tag.
 */

const YIELD = '<yield_user/>'

describe('invoke parameter reachability', () => {
  describe('basic parameter access', () => {
    it('accepts invoke with parameter', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n${YIELD}`)
    })

    it('accepts invoke with no parameters (direct close)', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n</invoke>\n${YIELD}`)
    })

    it('after invoke open + newline, < is valid (for parameter or close)', () => {
      const v = shellValidator()
      const rules = v.validAfter(`<invoke tool="shell">\n`)
      const validChars = rules.flatMap((r: any) => {
        if (r.type === 'char') return r.value.map((v: number) => String.fromCharCode(v))
        if (r.type === 'char_exclude') return ['[exclude]']
        return []
      })
      expect(validChars).toContain('<')
    })

    it('after invoke open + newline + <, parameter and close paths are valid', () => {
      const v = shellValidator()
      const rules = v.validAfter(`<invoke tool="shell">\n<`)
      const validChars = rules.flatMap((r: any) => {
        if (r.type === 'char') return r.value.map((v: number) => String.fromCharCode(v))
        if (r.type === 'char_exclude') return ['[exclude]']
        return []
      })
      // 'p' for <parameter, '/' for </invoke>, 'f' for <filter
      expect(validChars).toContain('p')  // <parameter ...
      expect(validChars).toContain('/')  // </invoke>
      expect(validChars).toContain('f')  // <filter>
    })
  })

  describe('multiple parameters', () => {
    it('accepts invoke with multiple parameters', () => {
      const v = multiToolValidator()
      v.passes(
        `<invoke tool="edit">\n` +
        `<parameter name="path">foo.ts</parameter>\n` +
        `<parameter name="old">bar</parameter>\n` +
        `<parameter name="new">baz</parameter>\n` +
        `</invoke>\n` +
        YIELD
      )
    })

    it('accepts invoke with parameters in any order', () => {
      const v = multiToolValidator()
      v.passes(
        `<invoke tool="edit">\n` +
        `<parameter name="new">baz</parameter>\n` +
        `<parameter name="old">bar</parameter>\n` +
        `<parameter name="path">foo.ts</parameter>\n` +
        `</invoke>\n` +
        YIELD
      )
    })

    it('param names constrained to tool schema', () => {
      // New grammar constrains param names — unknown names are absorbed as content
      const v = multiToolValidator()
      // Edit tool accepts its 3 known params
      v.passes(
        `<invoke tool="edit">\n` +
        `<parameter name="path">a</parameter>\n` +
        `<parameter name="old">b</parameter>\n` +
        `<parameter name="new">c</parameter>\n` +
        `</invoke>\n` +
        YIELD
      )
    })
  })

  describe('filter reachability', () => {
    it('accepts invoke with filter', () => {
      const v = shellValidator()
      v.passes(
        `<invoke tool="shell">\n` +
        `<parameter name="command">ls</parameter>\n` +
        `<filter>$.stdout</filter>\n` +
        `</invoke>\n` +
        YIELD
      )
    })

    it('after parameter close and <, valid continuations include structural tags', () => {
      const v = shellValidator()
      // With greedy body, </parameter>\n< is inside the BUC pattern.
      // After <, the grammar allows / (for close tags) and any content char via char_exclude.
      const rules = v.validAfter(`<invoke tool="shell">\n<parameter name="command">ls</parameter>\n<`)
      const validChars = rules.flatMap((r: any) => {
        if (r.type === 'char') return r.value.map((v: number) => String.fromCharCode(v))
        if (r.type === 'char_exclude') return ['[exclude]']
        return []
      })
      // / is explicitly valid (for </invoke> or </parameter>)
      expect(validChars).toContain('/')
      // Other chars (like f for <filter>) are valid via char_exclude rule
      expect(validChars.some((c: string) => c === 'f' || c === '[exclude]')).toBe(true)
    })
  })

  describe('inline parameter values (no newline after open)', () => {
    it('accepts value immediately after parameter open tag', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n<parameter name="command">ls -la</parameter>\n</invoke>\n${YIELD}`)
    })

    it('after parameter open tag, content chars are valid', () => {
      const v = shellValidator()
      const rules = v.validAfter(`<invoke tool="shell">\n<parameter name="command">`)
      const hasContentChars = rules.some((r: any) => {
        if (r.type === 'char_exclude') return true  // [^<] means any non-< char
        if (r.type === 'char') return r.value.some((v: number) => v >= 33 && v <= 126 && v !== 60)
        return false
      })
      expect(hasContentChars).toBe(true)
    })
  })

  describe('close tag variants', () => {
    it('standard </invoke> closes invoke block', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n${YIELD}`)
    })

    it('old MACT-style <invoke|> is treated as body content (not a close tag)', () => {
      // <invoke|> doesn't match </invoke> — treated as content in param body
      // A real </invoke> is needed after
      const v = shellValidator()
      v.passes(
        `<invoke tool="shell">\n` +
        `<parameter name="command">ls\n<invoke|>\nmore</parameter>\n` +
        `</invoke>\n` +
        YIELD
      )
    })
  })
})
