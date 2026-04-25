import { describe, it } from 'vitest'
import { buildValidator, shellValidator, multiToolValidator, SHELL_TOOL } from './helpers'

const YIELD = '<magnitude:yield_user/>'

describe('invoke / tool call blocks', () => {
  describe('single-parameter tool', () => {
    it('invoke shell with command parameter', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:invoke tool="shell">\n` +
        `<magnitude:parameter name="command">ls</magnitude:parameter>\n` +
        `</magnitude:invoke>\n` +
        YIELD
      )
    })

    it('invoke shell with multi-word command', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:invoke tool="shell">\n` +
        `<magnitude:parameter name="command">git diff --stat --cached</magnitude:parameter>\n` +
        `</magnitude:invoke>\n` +
        YIELD
      )
    })

    it('invoke shell with multi-line command', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:invoke tool="shell">\n` +
        `<magnitude:parameter name="command">cd /tmp\nls -la</magnitude:parameter>\n` +
        `</magnitude:invoke>\n` +
        YIELD
      )
    })

    it('unknown tool name is rejected (constrained)', () => {
      const v = shellValidator()
      v.rejects(
        `<magnitude:invoke tool="anything">\n` +
        `<magnitude:parameter name="foo">bar</magnitude:parameter>\n` +
        `</magnitude:invoke>\n` +
        YIELD
      )
    })
  })

  describe('no-parameter invoke', () => {
    it('rejects direct close for tools with required params', () => {
      const v = shellValidator()
      v.rejects(`<magnitude:invoke tool="shell">\n</magnitude:invoke>\n${YIELD}`)
    })
  })

  describe('multiple parameters', () => {
    it('invoke with multiple parameters', () => {
      const v = multiToolValidator()
      v.passes(
        `<magnitude:invoke tool="edit">\n` +
        `<magnitude:parameter name="path">src/foo.ts</magnitude:parameter>\n` +
        `<magnitude:parameter name="old">const x = 1</magnitude:parameter>\n` +
        `<magnitude:parameter name="new">const x = 2</magnitude:parameter>\n` +
        `</magnitude:invoke>\n` +
        YIELD
      )
    })

    it('parameters in any order', () => {
      const v = multiToolValidator()
      v.passes(
        `<magnitude:invoke tool="edit">\n` +
        `<magnitude:parameter name="new">const x = 2</magnitude:parameter>\n` +
        `<magnitude:parameter name="path">src/foo.ts</magnitude:parameter>\n` +
        `<magnitude:parameter name="old">const x = 1</magnitude:parameter>\n` +
        `</magnitude:invoke>\n` +
        YIELD
      )
    })
  })

  describe('invoke with filter', () => {
    it('rejects invoke with filter in the current shell grammar path', () => {
      const v = shellValidator()
      v.rejects(
        `<magnitude:invoke tool="shell">\n` +
        `<magnitude:parameter name="command">ls</magnitude:parameter>\n` +
        `<magnitude:filter>$.stdout</magnitude:filter>\n` +
        YIELD
      )
    })

    it('rejects filter-only invoke when required params are missing', () => {
      const v = shellValidator()
      v.rejects(
        `<magnitude:invoke tool="shell">\n` +
        `<magnitude:filter>$.result</magnitude:filter>\n` +
        `</magnitude:invoke>\n` +
        YIELD
      )
    })
  })

  describe('multiple invokes', () => {
    it('two invokes before yield', () => {
      const v = multiToolValidator()
      v.passes(
        `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hello</magnitude:parameter>\n</magnitude:invoke>\n` +
        `<magnitude:invoke tool="tree">\n</magnitude:invoke>\n` +
        YIELD
      )
    })

    it('three invokes before yield', () => {
      const v = multiToolValidator()
      v.passes(
        `<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">pwd</magnitude:parameter>\n</magnitude:invoke>\n` +
        `<magnitude:invoke tool="skill">\n<magnitude:parameter name="name">review</magnitude:parameter>\n</magnitude:invoke>\n` +
        `<magnitude:invoke tool="tree">\n</magnitude:invoke>\n` +
        YIELD
      )
    })
  })

  describe('no-newline between tags', () => {
    it('parameter immediately followed by another parameter (no newline)', () => {
      const v = multiToolValidator()
      v.passes(
        `<magnitude:invoke tool="edit">\n` +
        `<magnitude:parameter name="path">val</magnitude:parameter><magnitude:parameter name="old">val2</magnitude:parameter><magnitude:parameter name="new">val3</magnitude:parameter>\n` +
        `</magnitude:invoke>\n` +
        YIELD
      )
    })

    it('parameter followed by close invoke (no newline)', () => {
      const v = shellValidator()
      v.passes(
        `<magnitude:invoke tool="shell">\n` +
        `<magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke>\n` +
        YIELD
      )
    })
  })

  describe('trailing whitespace after close tags', () => {
    it('0 trailing spaces before newline passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n${YIELD}`)
    })

    it('4 trailing spaces after </magnitude:parameter> passes', () => {
      const v = shellValidator()
      v.passes(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>    \n</magnitude:invoke>\n${YIELD}`)
    })

    // Under greedy last-match, whitespace after close tag is valid content
    // or structural whitespace — both paths are live.
    it('5 trailing spaces after </magnitude:parameter> is accepted (greedy)', () => {
      const v = shellValidator()
      v.passes(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>     \n</magnitude:invoke>\n${YIELD}`)
    })
  })

  describe('forbidden sequences', () => {
    it('invoke after yield is rejected', () => {
      const v = shellValidator()
      v.rejects(`<magnitude:yield_user/><magnitude:invoke tool="shell">\n</magnitude:invoke>\n`)
    })
  })
})
