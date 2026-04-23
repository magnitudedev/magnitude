import { describe, it } from 'vitest'
import { shellValidator } from './helpers'

/**
 * Close tag behavior in the new XML grammar.
 *
 * Only standard XML close tags are recognized: </tagname>
 * The old MACT-style 4-variant lenience (<tag|>, </tag|>, <tag>) is removed.
 */

const YIELD = '<magnitude:yield_user/>'

describe('close tag behavior', () => {
  describe('standard close tags accepted', () => {
    it('</magnitude:reason> closes reason block', () => {
      const v = shellValidator()
      v.passes(`<magnitude:reason about="turn">\nsome thought\n</magnitude:reason>\n${YIELD}`)
    })

    it('</magnitude:message> closes message block', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello world\n</magnitude:message>\n${YIELD}`)
    })

    it('</magnitude:parameter> closes parameter block', () => {
      const v = shellValidator()
      v.passes(`<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>\n${YIELD}`)
    })

    it('</magnitude:invoke> closes invoke block', () => {
      const v = shellValidator()
      v.passes(`<magnitude:invoke tool="shell">\n</magnitude:invoke>\n${YIELD}`)
    })

    it('</magnitude:filter> closes filter block', () => {
      const v = shellValidator()
      v.passes(`<magnitude:invoke tool="shell">\n<magnitude:filter>$.stdout</magnitude:filter>\n</magnitude:invoke>\n${YIELD}`)
    })
  })

  describe('old MACT-style close variants treated as body content', () => {
    it('<magnitude:reason|> in reason body is treated as content (not a close tag)', () => {
      // <magnitude:reason|> does not match </magnitude:reason> — treated as content, real close follows
      const v = shellValidator()
      v.passes(`<magnitude:reason about="turn">\nsome thought\n<magnitude:reason|>\nmore content\n</magnitude:reason>\n${YIELD}`)
    })

    it('<magnitude:message|> in message body is treated as content', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n<magnitude:message|>\nmore\n</magnitude:message>\n${YIELD}`)
    })

    it('</message|> in message body is treated as content', () => {
      // </message| → matches </message up to "|", then "|" at gt state is not ">" → back to s0
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</message|>\nmore\n</magnitude:message>\n${YIELD}`)
    })
  })

  describe('close tag confirmation', () => {
    it('close tag confirmed by newline', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>\n${YIELD}`)
    })

    it('close tag confirmed by next tag open (no newline)', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message><magnitude:yield_user/>`)
    })

    it('close tag confirmed by next tag open after spaces', () => {
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message>  <magnitude:yield_user/>`)
    })

    it('false close tag in prose rejected by trailing non-ws', () => {
      // </magnitude:message> followed by prose → close tag treated as content
      const v = shellValidator()
      v.passes(`<magnitude:message to="user">\nhello\n</magnitude:message> to end your message\n</magnitude:message>\n${YIELD}`)
    })
  })
})
