import { describe, it } from 'vitest'
import { shellValidator, multiToolValidator } from './helpers'

const EO = '<magnitude:escape>'
const EC = '</magnitude:escape>'

const REASON = (name: string, content: string) =>
  `<magnitude:reason about="${name}">${content}</magnitude:reason>\n`

const MSG = (recipient: string, content: string) =>
  `<magnitude:message to="${recipient}">${content}</magnitude:message>\n`

const INVOKE = (tool: string, params: Record<string, string> = {}) => {
  const paramLines = Object.entries(params)
    .map(([k, v]) => `<magnitude:parameter name="${k}">${v}</magnitude:parameter>`)
    .join('\n')
  return `<magnitude:invoke tool="${tool}">\n${paramLines ? paramLines + '\n' : ''}</magnitude:invoke>\n`
}

const YIELD_INVOKE = '<magnitude:yield_invoke/>'
const YIELD_USER = '<magnitude:yield_user/>'
const YIELD_WORKER = '<magnitude:yield_worker/>'

const ESC = (content: string) => EO + content + EC

describe('magnitude:escape grammar tests', () => {

  describe('top-level positions', () => {
    it('escape as only element before yield_user', () => {
      const v = shellValidator()
      v.passes(ESC('content') + '\n' + YIELD_USER)
    })

    it('escape as only element before yield_invoke', () => {
      const v = shellValidator()
      v.passes(INVOKE('shell', { command: 'ls' }) + ESC('content') + '\n' + YIELD_INVOKE)
    })

    it('escape as only element before yield_worker', () => {
      const v = shellValidator()
      v.passes(ESC('content') + '\n' + YIELD_WORKER)
    })

    it('escape after single reason', () => {
      const v = shellValidator()
      v.passes(REASON('turn', 'thinking') + ESC('content') + '\n' + YIELD_USER)
    })

    it('escape after multiple reasons', () => {
      const v = shellValidator()
      v.passes(REASON('turn', 'a') + REASON('skills', 'b') + ESC('content') + '\n' + YIELD_USER)
    })

    it('escape after message', () => {
      const v = shellValidator()
      v.passes(MSG('user', 'hi') + ESC('content') + '\n' + YIELD_USER)
    })

    it('escape after invoke', () => {
      const v = shellValidator()
      v.passes(INVOKE('shell', { command: 'ls' }) + ESC('c') + '\n' + YIELD_USER)
    })

    it('escape between two invokes', () => {
      const v = shellValidator()
      v.passes(INVOKE('shell', { command: 'ls' }) + ESC('middle') + '\n' + INVOKE('shell', { command: 'pwd' }) + YIELD_INVOKE)
    })

    it('escape between message and invoke', () => {
      const v = shellValidator()
      v.passes(MSG('user', 'starting') + ESC('escaped') + '\n' + INVOKE('shell', { command: 'ls' }) + YIELD_INVOKE)
    })

    it('escape before message', () => {
      const v = shellValidator()
      v.passes(ESC('preamble') + '\n' + MSG('user', 'hello') + YIELD_USER)
    })

    it('multiple escape blocks in one turn', () => {
      const v = shellValidator()
      v.passes(ESC('first') + '\n' + ESC('second') + '\n' + YIELD_USER)
    })

    it('three consecutive escape blocks', () => {
      const v = shellValidator()
      v.passes(ESC('a') + '\n' + ESC('b') + '\n' + ESC('c') + '\n' + YIELD_USER)
    })

    it('escape with reason, message, invoke, and yield', () => {
      const v = shellValidator()
      v.passes(REASON('turn', 'plan') + MSG('user', 'status') + ESC('escaped') + '\n' + INVOKE('shell', { command: 'ls' }) + YIELD_INVOKE)
    })

    it('escape between reason and message', () => {
      const v = shellValidator()
      v.passes(REASON('turn', 'thinking') + ESC('raw') + '\n' + MSG('user', 'hi') + YIELD_USER)
    })

    it('escape between two reasons', () => {
      const v = shellValidator()
      v.passes(REASON('turn', 'a') + ESC('raw') + '\n' + REASON('skills', 'b') + MSG('user', 'hi') + YIELD_USER)
    })

  })

  describe('inline escape in message body', () => {
    it('escape block inside message body', () => {
      const v = shellValidator()
      v.passes(MSG('user', 'before ' + ESC('raw content') + ' after') + YIELD_USER)
    })

    it('escape containing close tag of parent message', () => {
      const v = shellValidator()
      v.passes(MSG('user', 'text ' + ESC('</magnitude:message>') + ' more') + YIELD_USER)
    })

    it('escape containing open tag of parent message', () => {
      const v = shellValidator()
      v.passes(MSG('user', 'text ' + ESC('<magnitude:message to="x">') + ' more') + YIELD_USER)
    })

    it('escape containing full message block', () => {
      const v = shellValidator()
      v.passes(MSG('user', ESC('<magnitude:message to="x">inner</magnitude:message>')) + YIELD_USER)
    })

    it('escape containing invoke tags', () => {
      const v = shellValidator()
      v.passes(MSG('user', ESC('<magnitude:invoke tool="shell"><magnitude:parameter name="cmd">ls</magnitude:parameter></magnitude:invoke>')) + YIELD_USER)
    })

    it('escape containing reason tags', () => {
      const v = shellValidator()
      v.passes(MSG('user', ESC('<magnitude:reason about="x">thought</magnitude:reason>')) + YIELD_USER)
    })

    it('escape containing yield tags', () => {
      const v = shellValidator()
      v.passes(MSG('user', ESC('<magnitude:yield_user/>')) + YIELD_USER)
    })

    it('multiple escape blocks in message body', () => {
      const v = shellValidator()
      v.passes(MSG('user', ESC('first') + ' middle ' + ESC('second')) + YIELD_USER)
    })

    it('escape at start of message body', () => {
      const v = shellValidator()
      v.passes(MSG('user', ESC('start') + ' rest') + YIELD_USER)
    })

    it('escape at end of message body', () => {
      const v = shellValidator()
      v.passes(MSG('user', 'start ' + ESC('end')) + YIELD_USER)
    })

    it('escape as only content in message body', () => {
      const v = shellValidator()
      v.passes(MSG('user', ESC('only')) + YIELD_USER)
    })

    it('empty escape in message body', () => {
      const v = shellValidator()
      v.passes(MSG('user', 'text ' + ESC('') + ' more') + YIELD_USER)
    })

  })

  describe('inline escape in reason body', () => {
    it('escape block inside reason body', () => {
      const v = shellValidator()
      v.passes(REASON('turn', 'before ' + ESC('raw') + ' after') + MSG('user', 'hi') + YIELD_USER)
    })

    it('escape containing close tag of parent reason', () => {
      const v = shellValidator()
      v.passes(REASON('turn', 'text ' + ESC('</magnitude:reason>') + ' more') + MSG('user', 'hi') + YIELD_USER)
    })

    it('escape containing full reason block', () => {
      const v = shellValidator()
      v.passes(REASON('turn', ESC('<magnitude:reason about="x">inner</magnitude:reason>')) + MSG('user', 'hi') + YIELD_USER)
    })

    it('multiple escape blocks in reason body', () => {
      const v = shellValidator()
      v.passes(REASON('turn', ESC('a') + ' mid ' + ESC('b')) + MSG('user', 'hi') + YIELD_USER)
    })

  })

  describe('inline escape in parameter body', () => {
    it('escape block inside parameter value', () => {
      const v = shellValidator()
      v.passes(INVOKE('shell', { command: ESC('raw content') }) + YIELD_INVOKE)
    })

    it('escape containing close tag of parent parameter', () => {
      const v = shellValidator()
      v.passes(INVOKE('shell', { command: ESC('</magnitude:parameter>') }) + YIELD_INVOKE)
    })

    it('escape containing invoke close tag', () => {
      const v = shellValidator()
      v.passes(INVOKE('shell', { command: ESC('</magnitude:invoke>') }) + YIELD_INVOKE)
    })

    it('escape containing parameter tags', () => {
      const v = shellValidator()
      v.passes(INVOKE('shell', { command: ESC('<magnitude:parameter name="x">val</magnitude:parameter>') }) + YIELD_INVOKE)
    })

    it('text before and after escape in parameter', () => {
      const v = shellValidator()
      v.passes(INVOKE('shell', { command: 'before ' + ESC('mid') + ' after' }) + YIELD_INVOKE)
    })

    it('multiple escapes in parameter value', () => {
      const v = shellValidator()
      v.passes(INVOKE('shell', { command: ESC('a') + ' ' + ESC('b') }) + YIELD_INVOKE)
    })

    it('empty escape in parameter value', () => {
      const v = shellValidator()
      v.passes(INVOKE('shell', { command: 'cmd ' + ESC('') + ' rest' }) + YIELD_INVOKE)
    })

  })

  describe('inner content treated as literal text', () => {
    it('inner invoke open tag', () => {
      const v = shellValidator()
      v.passes(ESC('<magnitude:invoke tool="shell">') + '\n' + YIELD_USER)
    })

    it('inner invoke close tag', () => {
      const v = shellValidator()
      v.passes(ESC('</magnitude:invoke>') + '\n' + YIELD_USER)
    })

    it('inner parameter tags', () => {
      const v = shellValidator()
      v.passes(ESC('<magnitude:parameter name="x">val</magnitude:parameter>') + '\n' + YIELD_USER)
    })

    it('inner reason tags', () => {
      const v = shellValidator()
      v.passes(ESC('<magnitude:reason about="x">thought</magnitude:reason>') + '\n' + YIELD_USER)
    })

    it('inner message tags', () => {
      const v = shellValidator()
      v.passes(ESC('<magnitude:message to="user">hello</magnitude:message>') + '\n' + YIELD_USER)
    })

    it('inner filter tags', () => {
      const v = shellValidator()
      v.passes(ESC('<magnitude:filter>$.stdout</magnitude:filter>') + '\n' + YIELD_USER)
    })

    it('inner yield tags', () => {
      const v = shellValidator()
      v.passes(ESC('<magnitude:yield_user/>') + '\n' + YIELD_USER)
    })

    it('full invoke block', () => {
      const v = shellValidator()
      v.passes(ESC('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>') + '\n' + YIELD_USER)
    })

    it('multiple different tags', () => {
      const v = shellValidator()
      v.passes(ESC('<magnitude:reason about="x">a</magnitude:reason><magnitude:message to="u">b</magnitude:message>') + '\n' + YIELD_USER)
    })

  })

  describe('edge cases', () => {
    it('empty escape block', () => {
      const v = shellValidator()
      v.passes(ESC('') + '\n' + YIELD_USER)
    })

    it('escape with only whitespace', () => {
      const v = shellValidator()
      v.passes(ESC('   \n  \n  ') + '\n' + YIELD_USER)
    })

    it('escape with newlines', () => {
      const v = shellValidator()
      v.passes(ESC('line1\nline2\nline3') + '\n' + YIELD_USER)
    })

    it('escape with special characters', () => {
      const v = shellValidator()
      v.passes(ESC('!@#$%^&*(){}[]|') + '\n' + YIELD_USER)
    })

    it('escape with angle brackets', () => {
      const v = shellValidator()
      v.passes(ESC('a < b > c << d >> e') + '\n' + YIELD_USER)
    })

    it('escape with partial close tag prefix', () => {
      const v = shellValidator()
      v.passes(ESC('</magnitud') + '\n' + YIELD_USER)
    })

    it('escape with close tag missing >', () => {
      const v = shellValidator()
      v.passes(ESC('</magnitude:escape') + '\n' + YIELD_USER)
    })

    it('escape with close tag extra space', () => {
      const v = shellValidator()
      v.passes(ESC('</magnitude:escape >') + '\n' + YIELD_USER)
    })

    it('escape with wrong case close', () => {
      const v = shellValidator()
      v.passes(ESC('</MAGNITUDE:ESCAPE>') + '\n' + YIELD_USER)
    })

  })

  describe('no nesting', () => {
    it('inner escape open tag is just content, first close ends block', () => {
      const v = shellValidator()
      v.passes(ESC(EO + 'nested?') + '\n' + YIELD_USER)
    })

    it('grammar rejects content between two close tags at top level', () => {
      const v = shellValidator()
      v.rejects(EO + 'inner' + EC + 'more' + EC + '\n' + YIELD_USER)
    })
  })

  describe('compound interactions', () => {
    it('escape in message then escape at top level', () => {
      const v = shellValidator()
      v.passes(MSG('user', 'has ' + ESC('inner') + ' text') + ESC('top level') + '\n' + YIELD_USER)
    })

    it('escape in reason then escape in message', () => {
      const v = shellValidator()
      v.passes(REASON('turn', ESC('raw')) + MSG('user', ESC('also raw')) + YIELD_USER)
    })

    it('escape in parameter then escape at top level', () => {
      const v = shellValidator()
      v.passes(INVOKE('shell', { command: ESC('raw') }) + ESC('top') + '\n' + YIELD_USER)
    })

    it('reason with escape, then message with escape, then invoke with escape param, then top-level escape', () => {
      const v = shellValidator()
      v.passes(REASON('turn', 'a ' + ESC('b') + ' c') + MSG('user', 'd ' + ESC('e') + ' f') + INVOKE('shell', { command: ESC('g') }) + ESC('h') + '\n' + YIELD_INVOKE)
    })

  })

  describe('escape before each yield type', () => {
    it('escape before yield_user', () => {
      const v = shellValidator()
      v.passes(ESC('content') + '\n' + YIELD_USER)
    })

    it('escape before yield_invoke (with preceding invoke)', () => {
      const v = shellValidator()
      v.passes(INVOKE('shell', { command: 'ls' }) + ESC('c') + '\n' + YIELD_INVOKE)
    })

    it('escape before yield_worker', () => {
      const v = shellValidator()
      v.passes(ESC('content') + '\n' + YIELD_WORKER)
    })
  })
})
