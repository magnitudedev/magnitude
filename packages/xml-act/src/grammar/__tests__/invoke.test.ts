import { describe, it } from 'vitest'
import { multiToolValidator, buildValidator, SHELL_TOOL, NO_PARAM_TOOL, MULTI_PARAM_TOOL } from './helpers'

const YIELD = '\n<|yield:user|>'

describe('invoke / tool call blocks', () => {
  describe('single-parameter tool', () => {
    it('invoke shell with command parameter', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:shell>\n' +
        '<|parameter:command>\n' +
        'ls\n' +
        '<parameter|>\n' +
        '<invoke|>\n' +
        YIELD
      )
    })

    it('invoke shell with multi-word command', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:shell>\n' +
        '<|parameter:command>\n' +
        'git diff --stat --cached\n' +
        '<parameter|>\n' +
        '<invoke|>\n' +
        YIELD
      )
    })

    it('invoke shell with multi-line command', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:shell>\n' +
        '<|parameter:command>\n' +
        'cd /tmp\nls -la\n' +
        '<parameter|>\n' +
        '<invoke|>\n' +
        YIELD
      )
    })

    it('invoke skill with name parameter', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:skill>\n' +
        '<|parameter:name>\n' +
        'review\n' +
        '<parameter|>\n' +
        '<invoke|>\n' +
        YIELD
      )
    })
  })

  describe('no-parameter tool', () => {
    it('invoke tree with no parameters', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:tree>\n' +
        '<invoke|>\n' +
        YIELD
      )
    })
  })

  describe('multi-parameter tool', () => {
    it('invoke edit with all three parameters', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:edit>\n' +
        '<|parameter:path>\n' +
        'src/foo.ts\n' +
        '<parameter|>\n' +
        '<|parameter:old>\n' +
        'const x = 1\n' +
        '<parameter|>\n' +
        '<|parameter:new>\n' +
        'const x = 2\n' +
        '<parameter|>\n' +
        '<invoke|>\n' +
        YIELD
      )
    })

    it('invoke edit with parameters in different order (unordered)', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:edit>\n' +
        '<|parameter:new>\n' +
        'const x = 2\n' +
        '<parameter|>\n' +
        '<|parameter:path>\n' +
        'src/foo.ts\n' +
        '<parameter|>\n' +
        '<|parameter:old>\n' +
        'const x = 1\n' +
        '<parameter|>\n' +
        '<invoke|>\n' +
        YIELD
      )
    })

    it('invoke edit with only one parameter provided', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:edit>\n' +
        '<|parameter:path>\n' +
        'src/foo.ts\n' +
        '<parameter|>\n' +
        '<invoke|>\n' +
        YIELD
      )
    })
  })

  describe('multiple invokes', () => {
    it('two invokes before yield', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:shell>\n' +
        '<|parameter:command>\n' +
        'echo hello\n' +
        '<parameter|>\n' +
        '<invoke|>\n' +
        '\n<|invoke:tree>\n' +
        '<invoke|>\n' +
        YIELD
      )
    })

    it('three invokes before yield', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:shell>\n' +
        '<|parameter:command>\n' +
        'pwd\n' +
        '<parameter|>\n' +
        '<invoke|>\n' +
        '\n<|invoke:skill>\n' +
        '<|parameter:name>\n' +
        'review\n' +
        '<parameter|>\n' +
        '<invoke|>\n' +
        '\n<|invoke:tree>\n' +
        '<invoke|>\n' +
        YIELD
      )
    })
  })

  describe('lenient invoke close tag forms', () => {
    it('close with </invoke|>', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:shell>\n' +
        '<|parameter:command>\n' +
        'ls\n' +
        '<parameter|>\n' +
        '</invoke|>\n' +
        YIELD
      )
    })

    it('close with </invoke>', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:shell>\n' +
        '<|parameter:command>\n' +
        'ls\n' +
        '<parameter|>\n' +
        '</invoke>\n' +
        YIELD
      )
    })

    it('close with <invoke>', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:shell>\n' +
        '<|parameter:command>\n' +
        'ls\n' +
        '<parameter|>\n' +
        '<invoke>\n' +
        YIELD
      )
    })
  })

  describe('invoke close with trailing spaces', () => {
    it('close with one trailing space', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:shell>\n' +
        '<|parameter:command>\n' +
        'ls\n' +
        '<parameter|>\n' +
        '<invoke|> \n' +
        YIELD
      )
    })

    it('close with two trailing spaces', () => {
      const v = multiToolValidator()
      v.passes(
        '\n<|invoke:shell>\n' +
        '<|parameter:command>\n' +
        'ls\n' +
        '<parameter|>\n' +
        '<invoke|>  \n' +
        YIELD
      )
    })
  })

  describe('forbidden sequences', () => {
    it('unknown tool name rejected', () => {
      const v = multiToolValidator()
      v.rejects(
        '\n<|invoke:unknown-tool>\n' +
        '<invoke|>\n' +
        YIELD
      )
    })



    it('unknown parameter name rejected', () => {
      const v = multiToolValidator()
      v.rejects(
        '\n<|invoke:shell>\n' +
        '<|parameter:unknown>\n' +
        'value\n' +
        '<parameter|>\n' +
        '<invoke|>\n' +
        YIELD
      )
    })
  })
})
