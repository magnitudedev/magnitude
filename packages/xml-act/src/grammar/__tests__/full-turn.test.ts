import { describe, it } from 'vitest'
import { buildValidator, shellValidator, SHELL_TOOL } from './helpers'

const REASON = (name: string, content: string) =>
  `<magnitude:think about="${name}">\n${content}\n</magnitude:think>\n`

const MSG = (recipient: string, content: string) =>
  `<magnitude:message to="${recipient}">\n${content}\n</magnitude:message>\n`

const INVOKE = (tool: string, params: Record<string, string> = {}) => {
  const paramLines = Object.entries(params)
    .map(([k, v]) => `<magnitude:parameter name="${k}">${v}</magnitude:parameter>`)
    .join('\n')
  return `<magnitude:invoke tool="${tool}">\n${paramLines ? paramLines + '\n' : ''}</magnitude:invoke>\n`
}

const YIELD = (tag = 'user') => `<magnitude:yield_${tag}/>`

describe('full turn sequences', () => {
  describe('yield only', () => {
    it('yield_user alone passes', () => {
      const v = shellValidator()
      v.passes(YIELD('user'))
    })

    it('yield_invoke alone passes', () => {
      const v = shellValidator()
      v.passes(YIELD('invoke'))
    })

    it('yield_worker alone passes', () => {
      const v = shellValidator()
      v.passes(YIELD('worker'))
    })
  })

  describe('think → yield', () => {
    it('single think block then yield passes', () => {
      const v = shellValidator()
      v.passes(REASON('turn', 'some thought') + YIELD())
    })

    it('multiple think blocks then yield passes', () => {
      const v = shellValidator()
      v.passes(
        REASON('turn', 'first thought') +
        REASON('alignment', 'second thought') +
        YIELD()
      )
    })

    it('think with multi-line content then yield passes', () => {
      const v = shellValidator()
      v.passes(REASON('turn', 'line one\nline two\nline three') + YIELD())
    })
  })

  describe('think → message → yield', () => {
    it('think then message then yield passes', () => {
      const v = shellValidator()
      v.passes(
        REASON('turn', 'I should respond') +
        MSG('user', 'Hello there') +
        YIELD()
      )
    })

    it('think then forced message then yield passes', () => {
      const v = buildValidator([SHELL_TOOL], b => b.requireMessageTo('user'))
      v.passes(
        REASON('turn', 'planning') +
        MSG('user', 'Here is my response') +
        YIELD()
      )
    })
  })

  describe('think → invoke → yield', () => {
    it('think then invoke then yield passes', () => {
      const v = shellValidator()
      v.passes(
        REASON('turn', 'I will run ls') +
        INVOKE('shell', { command: 'ls -la' }) +
        YIELD('invoke')
      )
    })

    it('think then no-param invoke then yield passes', () => {
      const v = buildValidator([SHELL_TOOL, { tagName: 'tree', parameters: [] }])
      v.passes(
        REASON('turn', 'checking tree') +
        INVOKE('tree') +
        YIELD('invoke')
      )
    })
  })

  describe('multiple invokes', () => {
    it('two invokes then yield passes', () => {
      const v = buildValidator([
        SHELL_TOOL,
        { tagName: 'skill', parameters: [{ name: 'name', field: 'name', type: 'scalar', required: true }] },
      ])
      v.passes(
        INVOKE('shell', { command: 'echo hello' }) +
        INVOKE('skill', { name: 'review' }) +
        YIELD('invoke')
      )
    })

    it('three invokes then yield passes', () => {
      const v = buildValidator([
        SHELL_TOOL,
        { tagName: 'skill', parameters: [{ name: 'name', field: 'name', type: 'scalar', required: true }] },
        { tagName: 'tree', parameters: [] },
      ])
      v.passes(
        INVOKE('shell', { command: 'ls' }) +
        INVOKE('skill', { name: 'review' }) +
        INVOKE('tree') +
        YIELD('invoke')
      )
    })
  })

  describe('mixed think + message + invoke', () => {
    it('think → message → invoke → yield passes', () => {
      const v = shellValidator()
      v.passes(
        REASON('turn', 'planning') +
        MSG('user', 'Running a command') +
        INVOKE('shell', { command: 'ls' }) +
        YIELD('invoke')
      )
    })

    it('think → invoke → message → yield passes', () => {
      const v = shellValidator()
      v.passes(
        REASON('turn', 'planning') +
        INVOKE('shell', { command: 'ls' }) +
        MSG('user', 'Done') +
        YIELD()
      )
    })

    it('multiple thinks → message → invoke → yield passes', () => {
      const v = shellValidator()
      v.passes(
        REASON('turn', 'first') +
        REASON('alignment', 'second') +
        MSG('user', 'Doing things') +
        INVOKE('shell', { command: 'echo hi' }) +
        YIELD('invoke')
      )
    })

    it('think → message → invoke → message → yield passes', () => {
      const v = shellValidator()
      v.passes(
        REASON('turn', 'plan') +
        MSG('user', 'Starting') +
        INVOKE('shell', { command: 'ls' }) +
        MSG('user', 'Done') +
        YIELD()
      )
    })

    it('full complex sequence passes', () => {
      const v = buildValidator([
        SHELL_TOOL,
        { tagName: 'skill', parameters: [{ name: 'name', field: 'name', type: 'scalar', required: true }] },
      ])
      v.passes(
        REASON('turn', 'I need to do several things') +
        REASON('diligence', 'check quality') +
        MSG('user', 'Starting work') +
        INVOKE('shell', { command: 'ls -la' }) +
        INVOKE('skill', { name: 'review' }) +
        MSG('user', 'All done') +
        YIELD()
      )
    })
  })

  describe('forbidden sequences', () => {
    it('yield before invoke is rejected', () => {
      const v = shellValidator()
      v.rejects(YIELD() + INVOKE('shell', { command: 'ls' }))
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
      v.rejects(YIELD() + REASON('turn', 'extra'))
    })

    it('think after message is rejected (ordering)', () => {
      const v = shellValidator()
      v.rejects(
        MSG('user', 'hello') +
        REASON('turn', 'thinking') +
        YIELD()
      )
    })
  })
})
