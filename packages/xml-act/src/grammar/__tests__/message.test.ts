import { describe, it } from 'vitest'
import { buildValidator, shellValidator, SHELL_TOOL } from './helpers'

// A minimal valid yield suffix to complete the grammar's root rule
const YIELD = '\n<|yield:user|>'

describe('message blocks', () => {
  describe('passing sequences', () => {
    it('accepts a simple message to user', () => {
      const v = shellValidator()
      v.passes('\n<|message:user>\nhello world\n<message|>\n' + YIELD)
    })

    it('accepts a message to parent', () => {
      const v = shellValidator()
      v.passes('\n<|message:parent>\nhello\n<message|>\n' + YIELD)
    })

    it('accepts a message to task-123', () => {
      const v = shellValidator()
      v.passes('\n<|message:task-123>\nsome content\n<message|>\n' + YIELD)
    })

    it('accepts a multi-line body', () => {
      const v = shellValidator()
      v.passes('\n<|message:user>\nline one\nline two\nline three\n<message|>\n' + YIELD)
    })

    it('accepts a body containing <', () => {
      const v = shellValidator()
      v.passes('\n<|message:user>\nuse <b>bold</b> text here\n<message|>\n' + YIELD)
    })

    it('accepts multiple messages before yield', () => {
      const v = shellValidator()
      v.passes(
        '\n<|message:user>\nfirst\n<message|>\n' +
        '\n<|message:parent>\nsecond\n<message|>\n' +
        YIELD
      )
    })

    it('accepts message followed by invoke then yield', () => {
      const v = buildValidator([SHELL_TOOL])
      v.passes(
        '\n<|message:user>\nrunning now\n<message|>\n' +
        '\n<|invoke:shell>\n<|parameter:command>\necho hi\n<parameter|>\n<invoke|>\n' +
        YIELD
      )
    })

    it('accepts inline content immediately after open tag newline', () => {
      const v = shellValidator()
      v.passes('\n<|message:user>\ncontent right away\n<message|>\n' + YIELD)
    })

    describe('requiredMessageTo', () => {
      it('accepts message to required recipient', () => {
        const v = buildValidator([SHELL_TOOL], b => b.requireMessageTo('parent'))
        v.passes('\n<|message:parent>\nhello\n<message|>\n' + YIELD)
      })

      it('accepts required message followed by another message', () => {
        const v = buildValidator([SHELL_TOOL], b => b.requireMessageTo('parent'))
        v.passes(
          '\n<|message:parent>\nrequired\n<message|>\n' +
          '\n<|message:user>\ninfo\n<message|>\n' +
          YIELD
        )
      })
    })
  })

  describe('forbidden sequences', () => {


    it('rejects sequence missing required message', () => {
      const v = buildValidator([SHELL_TOOL], b => b.requireMessageTo('parent'))
      // Only a yield with no message to parent
      v.rejects(YIELD)
    })

    it('rejects required message to wrong recipient', () => {
      const v = buildValidator([SHELL_TOOL], b => b.requireMessageTo('parent'))
      // Message to user but parent is required
      v.rejects('\n<|message:user>\nhello\n<message|>\n' + YIELD)
    })
  })
})
