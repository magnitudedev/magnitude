import { describe, it } from 'vitest'
import { buildValidator, shellValidator, multiToolValidator, SHELL_TOOL } from './helpers'

const YIELD = '<yield_user/>'

describe('invoke / tool call blocks', () => {
  describe('single-parameter tool', () => {
    it('invoke shell with command parameter', () => {
      const v = shellValidator()
      v.passes(
        `<invoke tool="shell">\n` +
        `<parameter name="command">ls</parameter>\n` +
        `</invoke>\n` +
        YIELD
      )
    })

    it('invoke shell with multi-word command', () => {
      const v = shellValidator()
      v.passes(
        `<invoke tool="shell">\n` +
        `<parameter name="command">git diff --stat --cached</parameter>\n` +
        `</invoke>\n` +
        YIELD
      )
    })

    it('invoke shell with multi-line command', () => {
      const v = shellValidator()
      v.passes(
        `<invoke tool="shell">\n` +
        `<parameter name="command">cd /tmp\nls -la</parameter>\n` +
        `</invoke>\n` +
        YIELD
      )
    })

    it('any tool name is accepted (free-form)', () => {
      const v = shellValidator()
      v.passes(
        `<invoke tool="anything">\n` +
        `<parameter name="foo">bar</parameter>\n` +
        `</invoke>\n` +
        YIELD
      )
    })
  })

  describe('no-parameter invoke', () => {
    it('invoke with no parameters (direct close)', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n</invoke>\n${YIELD}`)
    })
  })

  describe('multiple parameters', () => {
    it('invoke with multiple parameters', () => {
      const v = shellValidator()
      v.passes(
        `<invoke tool="edit">\n` +
        `<parameter name="path">src/foo.ts</parameter>\n` +
        `<parameter name="old">const x = 1</parameter>\n` +
        `<parameter name="new">const x = 2</parameter>\n` +
        `</invoke>\n` +
        YIELD
      )
    })

    it('parameters in any order', () => {
      const v = shellValidator()
      v.passes(
        `<invoke tool="edit">\n` +
        `<parameter name="new">const x = 2</parameter>\n` +
        `<parameter name="path">src/foo.ts</parameter>\n` +
        `<parameter name="old">const x = 1</parameter>\n` +
        `</invoke>\n` +
        YIELD
      )
    })
  })

  describe('invoke with filter', () => {
    it('invoke with parameter and filter', () => {
      const v = shellValidator()
      v.passes(
        `<invoke tool="shell">\n` +
        `<parameter name="command">ls</parameter>\n` +
        `<filter>$.stdout</filter>\n` +
        `</invoke>\n` +
        YIELD
      )
    })

    it('invoke with filter only (no parameter)', () => {
      const v = shellValidator()
      v.passes(
        `<invoke tool="shell">\n` +
        `<filter>$.result</filter>\n` +
        `</invoke>\n` +
        YIELD
      )
    })
  })

  describe('multiple invokes', () => {
    it('two invokes before yield', () => {
      const v = shellValidator()
      v.passes(
        `<invoke tool="shell">\n<parameter name="command">echo hello</parameter>\n</invoke>\n` +
        `<invoke tool="tree">\n</invoke>\n` +
        YIELD
      )
    })

    it('three invokes before yield', () => {
      const v = shellValidator()
      v.passes(
        `<invoke tool="shell">\n<parameter name="command">pwd</parameter>\n</invoke>\n` +
        `<invoke tool="skill">\n<parameter name="name">review</parameter>\n</invoke>\n` +
        `<invoke tool="tree">\n</invoke>\n` +
        YIELD
      )
    })
  })

  describe('no-newline between tags', () => {
    it('parameter immediately followed by another parameter (no newline)', () => {
      const v = shellValidator()
      v.passes(
        `<invoke tool="edit">\n` +
        `<parameter name="a">val</parameter><parameter name="b">val2</parameter>\n` +
        `</invoke>\n` +
        YIELD
      )
    })

    it('parameter followed by close invoke (no newline)', () => {
      const v = shellValidator()
      v.passes(
        `<invoke tool="shell">\n` +
        `<parameter name="command">ls</parameter></invoke>\n` +
        YIELD
      )
    })
  })

  describe('trailing whitespace after close tags', () => {
    it('0 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n${YIELD}`)
    })

    it('4 trailing spaces after </parameter> passes', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n<parameter name="command">ls</parameter>    \n</invoke>\n${YIELD}`)
    })

    // Under greedy last-match, whitespace after close tag is valid content
    // or structural whitespace — both paths are live.
    it('5 trailing spaces after </parameter> is accepted (greedy)', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n<parameter name="command">ls</parameter>     \n</invoke>\n${YIELD}`)
    })
  })

  describe('forbidden sequences', () => {
    it('invoke after yield is rejected', () => {
      const v = shellValidator()
      v.rejects(`<yield_user/><invoke tool="shell">\n</invoke>\n`)
    })
  })
})
