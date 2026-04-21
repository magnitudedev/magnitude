import { describe, it } from 'vitest'
import { buildValidator, shellValidator, multiToolValidator, SHELL_TOOL, SKILL_TOOL, MULTI_PARAM_TOOL, NO_PARAM_TOOL } from './helpers'

// Convenience snippets for building turn sequences
const THINK = (name: string, content: string) =>
  `\n<|think:${name}>\n${content}\n<think|>\n`

const MSG = (recipient: string, content: string) =>
  `\n<|message:${recipient}>\n${content}\n<message|>\n`

const INVOKE_SHELL = (cmd: string) =>
  `\n<|invoke:shell>\n<|parameter:command>\n${cmd}\n<parameter|>\n<invoke|>\n`

const INVOKE_SKILL = (name: string) =>
  `\n<|invoke:skill>\n<|parameter:name>\n${name}\n<parameter|>\n<invoke|>\n`

const INVOKE_TREE = () =>
  `\n<|invoke:tree>\n<invoke|>\n`

const YIELD = (tag = 'user') => `\n<|yield:${tag}|>`

describe('full turn sequences', () => {
  describe('yield only', () => {
    it('yield:user alone passes', () => {
      const v = shellValidator()
      v.passes(YIELD('user'))
    })

    it('yield:invoke alone passes', () => {
      const v = shellValidator()
      v.passes(YIELD('invoke'))
    })

    it('yield:worker alone passes', () => {
      const v = shellValidator()
      v.passes(YIELD('worker'))
    })
  })

  describe('lens → yield', () => {
    it('single think block then yield passes', () => {
      const v = shellValidator()
      v.passes(THINK('turn', 'some thought') + YIELD())
    })

    it('multiple think blocks then yield passes', () => {
      const v = shellValidator()
      v.passes(
        THINK('turn', 'first thought') +
        THINK('alignment', 'second thought') +
        YIELD()
      )
    })

    it('think with multi-line content then yield passes', () => {
      const v = shellValidator()
      v.passes(THINK('turn', 'line one\nline two\nline three') + YIELD())
    })
  })

  describe('lens → message → yield', () => {
    it('think then message then yield passes', () => {
      const v = shellValidator()
      v.passes(
        THINK('turn', 'I should respond') +
        MSG('user', 'Hello there') +
        YIELD()
      )
    })

    it('think then forced message then yield passes', () => {
      const v = buildValidator([SHELL_TOOL], b => b.requireMessageTo('user'))
      v.passes(
        THINK('turn', 'planning') +
        MSG('user', 'Here is my response') +
        YIELD()
      )
    })
  })

  describe('lens → invoke → yield', () => {
    it('think then invoke then yield passes', () => {
      const v = shellValidator()
      v.passes(
        THINK('turn', 'I will run ls') +
        INVOKE_SHELL('ls -la') +
        YIELD('invoke')
      )
    })

    it('think then no-param invoke then yield passes', () => {
      const v = multiToolValidator()
      v.passes(
        THINK('turn', 'checking tree') +
        INVOKE_TREE() +
        YIELD('invoke')
      )
    })
  })

  describe('multiple invokes', () => {
    it('two invokes then yield passes', () => {
      const v = multiToolValidator()
      v.passes(
        INVOKE_SHELL('echo hello') +
        INVOKE_SKILL('review') +
        YIELD('invoke')
      )
    })

    it('three invokes then yield passes', () => {
      const v = multiToolValidator()
      v.passes(
        INVOKE_SHELL('ls') +
        INVOKE_SKILL('review') +
        INVOKE_TREE() +
        YIELD('invoke')
      )
    })
  })

  describe('mixed lens + message + invoke', () => {
    it('lens → message → invoke → yield passes', () => {
      const v = shellValidator()
      v.passes(
        THINK('turn', 'planning') +
        MSG('user', 'Running a command') +
        INVOKE_SHELL('ls') +
        YIELD('invoke')
      )
    })

    it('lens → invoke → message → yield passes', () => {
      const v = shellValidator()
      v.passes(
        THINK('turn', 'planning') +
        INVOKE_SHELL('ls') +
        MSG('user', 'Done') +
        YIELD()
      )
    })

    it('multiple lenses → message → invoke → yield passes', () => {
      const v = multiToolValidator()
      v.passes(
        THINK('turn', 'first') +
        THINK('alignment', 'second') +
        MSG('user', 'Doing things') +
        INVOKE_SHELL('echo hi') +
        YIELD('invoke')
      )
    })

    it('lens → message → invoke → message → yield passes', () => {
      const v = shellValidator()
      v.passes(
        THINK('turn', 'plan') +
        MSG('user', 'Starting') +
        INVOKE_SHELL('ls') +
        MSG('user', 'Done') +
        YIELD()
      )
    })

    it('full complex sequence passes', () => {
      const v = multiToolValidator()
      v.passes(
        THINK('turn', 'I need to do several things') +
        THINK('diligence', 'check quality') +
        MSG('user', 'Starting work') +
        INVOKE_SHELL('ls -la') +
        INVOKE_SKILL('review') +
        MSG('user', 'All done') +
        YIELD()
      )
    })
  })

  describe('multi-param invokes in sequences', () => {
    it('multi-param invoke in sequence passes', () => {
      const v = multiToolValidator()
      v.passes(
        THINK('turn', 'editing a file') +
        `\n<|invoke:edit>\n<|parameter:path>\nfoo.ts\n<parameter|>\n<|parameter:old>\nold content\n<parameter|>\n<|parameter:new>\nnew content\n<parameter|>\n<invoke|>\n` +
        YIELD('invoke')
      )
    })

    it('multi-param invoke with subset of params passes', () => {
      const v = multiToolValidator()
      v.passes(
        `\n<|invoke:edit>\n<|parameter:path>\nfoo.ts\n<parameter|>\n<invoke|>\n` +
        YIELD('invoke')
      )
    })
  })

  describe('forbidden sequences', () => {
    // Note: the gbnf library checks prefix validity. Sequences that are valid
    // prefixes but can never be completed are tested by checking that adding
    // the next expected character fails. Full-sequence rejection tests only
    // apply to sequences that fail mid-stream.

    it('yield before invoke is rejected', () => {
      const v = shellValidator()
      // After yield, the grammar is done — cannot start an invoke
      v.rejects(YIELD() + INVOKE_SHELL('ls'))
    })

    it('yield before message is rejected', () => {
      const v = shellValidator()
      v.rejects(YIELD() + MSG('user', 'hello'))
    })

    it('two yields is rejected', () => {
      const v = shellValidator()
      v.rejects(YIELD() + YIELD())
    })

    it('content after yield is rejected', () => {
      const v = shellValidator()
      v.rejects(YIELD() + THINK('turn', 'extra'))
    })

    it('invoking unknown tool is rejected', () => {
      const v = shellValidator()
      v.rejects(`\n<|invoke:unknown>\n\n<invoke|>\n` + YIELD('invoke'))
    })

    it('yield tag with wrong name is rejected', () => {
      const v = shellValidator()
      v.rejects(`\n<|yield:badtag|>\n`)
    })
  })
})
