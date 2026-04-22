import { describe, it } from 'vitest'
import { shellValidator } from './helpers'

/**
 * Close tag behavior in the new XML grammar.
 *
 * Only standard XML close tags are recognized: </tagname>
 * The old MACT-style 4-variant lenience (<tag|>, </tag|>, <tag>) is removed.
 */

const YIELD = '<yield_user/>'

describe('close tag behavior', () => {
  describe('standard close tags accepted', () => {
    it('</reason> closes reason block', () => {
      const v = shellValidator()
      v.passes(`<reason about="turn">\nsome thought\n</reason>\n${YIELD}`)
    })

    it('</message> closes message block', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nhello world\n</message>\n${YIELD}`)
    })

    it('</parameter> closes parameter block', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n${YIELD}`)
    })

    it('</invoke> closes invoke block', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n</invoke>\n${YIELD}`)
    })

    it('</filter> closes filter block', () => {
      const v = shellValidator()
      v.passes(`<invoke tool="shell">\n<filter>$.stdout</filter>\n</invoke>\n${YIELD}`)
    })
  })

  describe('old MACT-style close variants treated as body content', () => {
    it('<reason|> in reason body is treated as content (not a close tag)', () => {
      // <reason|> does not match </reason> — treated as content, real close follows
      const v = shellValidator()
      v.passes(`<reason about="turn">\nsome thought\n<reason|>\nmore content\n</reason>\n${YIELD}`)
    })

    it('<message|> in message body is treated as content', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n<message|>\nmore\n</message>\n${YIELD}`)
    })

    it('</message|> in message body is treated as content', () => {
      // </message| → matches </message up to "|", then "|" at gt state is not ">" → back to s0
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n</message|>\nmore\n</message>\n${YIELD}`)
    })
  })

  describe('close tag confirmation', () => {
    it('close tag confirmed by newline', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n</message>\n${YIELD}`)
    })

    it('close tag confirmed by next tag open (no newline)', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n</message><yield_user/>`)
    })

    it('close tag confirmed by next tag open after spaces', () => {
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n</message>  <yield_user/>`)
    })

    it('false close tag in prose rejected by trailing non-ws', () => {
      // </message> followed by prose → close tag treated as content
      const v = shellValidator()
      v.passes(`<message to="user">\nhello\n</message> to end your message\n</message>\n${YIELD}`)
    })
  })
})
