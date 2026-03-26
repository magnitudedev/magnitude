import { done, emit, pop, push } from '../ops'
import { endTopProse } from '../prose'
import type { TagHandler, XmlActEvent, XmlActFrame } from '../types'
import { findFrame } from '../types'

export function finishHandler(): TagHandler<XmlActFrame, XmlActEvent> {
  return {
    open(ctx) {
      return [...endTopProse(ctx.stack), push({ type: 'body-capture', tag: ctx.tagName, body: '', tags: new Map() })]
    },
    close(ctx) {
      const frame = findFrame(ctx.stack, 'body-capture')
      if (!frame) return []
      return [
        emit({ _tag: 'TurnControl', decision: 'finish', evidence: frame.body.trim() }),
        pop,
        done,
      ]
    },
    selfClose() {
      return [emit({ _tag: 'ParseError', error: { _tag: 'FinishWithoutEvidence' } }), done]
    },
  }
}